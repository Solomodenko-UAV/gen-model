import os

from cnn.pkg.layers.conv.layer import Convolution, col2im, im2col
from helper import helper

on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'
if on_cpu:
    import numpy as cp
else:
    import cupy as cp


class DepthwiseConv(Convolution):
    def __init__(self,
                 input_channels: int,
                 filter_size: int,
                 num_filters: int,
                 stride: int,
                 padding: int,
                 l2_lambda: float = 0.0001,
                 clip_value: float = 5.0,
                 momentum=0.8):

        # don't call the parent's __init__ for filters because we override them.
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.l2_lambda = l2_lambda
        self.clip_value = clip_value
        self.momentum = momentum
        self.eps = 1e-7

        std_depthwise = cp.sqrt(2 / (input_channels * filter_size**2)).astype(cp.float32)
        # don't have orthogonal 3d matrix creator
        self.depthwise_kernel = helper.create_orthogonal_3d_matrix((filter_size, filter_size, input_channels), std_depthwise)

        std_pointwise = cp.sqrt(2 / (input_channels)).astype(cp.float32)
        self.pointwise_kernel = helper.create_orthogonal_4d_matrix((1, 1, input_channels, num_filters), std_pointwise)  # shape (1, 1, input_channels, num_filters)

        self.biases = cp.full((1, 1, 1, num_filters), 1.2).astype(cp.float32)

        self.gamma = cp.ones((1, 1, 1, num_filters), dtype=cp.float32)
        self.beta = cp.zeros((1, 1, 1, num_filters), dtype=cp.float32)
        self.running_mean = cp.zeros((1, 1, 1, num_filters), dtype=cp.float32)
        self.running_variance = cp.ones((1, 1, 1, num_filters), dtype=cp.float32)
        self.weight_norms = []

        self.rho = 0.95
        self.E_g_depthwise = cp.zeros_like(self.depthwise_kernel)
        self.E_delta_depthwise = cp.zeros_like(self.depthwise_kernel)
        self.E_g_pointwise = cp.zeros_like(self.pointwise_kernel)
        self.E_delta_pointwise = cp.zeros_like(self.pointwise_kernel)
        self.E_g_biases = cp.zeros_like(self.biases)
        self.E_delta_biases = cp.zeros_like(self.biases)
        self.E_g_gamma = cp.zeros_like(self.gamma)
        self.E_delta_gamma = cp.zeros_like(self.gamma)
        self.E_g_beta = cp.zeros_like(self.beta)
        self.E_delta_beta = cp.zeros_like(self.beta)

        self.cache = {}

    def convolve_forward(self, X: cp.ndarray):
        """
        Forward pass using depthwise separable convolution.
        Args:
            X (cp.ndarray): Input tensor of shape (m, height, width, input_channels)
        Returns:
            conv_out (cp.ndarray): Output tensor of shape (m, output_height, output_width, input_channels)
        """
        _, prev_height, prev_width, _ = X.shape

        output_height = int((prev_height - self.filter_size + 2 * self.padding) / self.stride) + 1
        output_width = int((prev_width - self.filter_size + 2 * self.padding) / self.stride) + 1

        depthwise_out = self._depthwise_conv_forward(X, output_height, output_width)

        conv_out = self._pointwise_conv_forward(depthwise_out)  # shape (m, output_height, output_width, num_filters)

        conv_out, batch_cache = self._normalize_forward(conv_out)

        self.cache = ({'X': X, 'depthwise_out': depthwise_out}, batch_cache)

        return conv_out

    def convolve_backward(self, dZ: cp.ndarray):
        """
        Backward pass for depthwise separable convolution.
        Args:
            dZ (cp.ndarray): Gradient with respect to output, shape (m, output_height, output_width, num_filters)
            learning_rate (float): Learning rate (if needed for parameter update; note that AdaDelta may not use it directly)
        Returns:
            dX (cp.ndarray): Gradient with respect to input X.
        """
        dZ, dgamma, dbeta = self._normalize_backward(dZ)

        dZ_pointwise, dW_pointwise, db = self._pointwise_conv_backward(dZ)  # shape (m, output_height, output_width, input_channels)
        dX, dW_depthwise = self._depthwise_conv_backward(dZ_pointwise)  # shape (m, height, width, input_channels)

        helper.adadelta_update(self.depthwise_kernel, dW_depthwise, self.E_g_depthwise, self.E_delta_depthwise, self.rho, self.eps)
        helper.adadelta_update(self.pointwise_kernel, dW_pointwise, self.E_g_pointwise, self.E_delta_pointwise, self.rho, self.eps)
        helper.adadelta_update(self.biases, db, self.E_g_biases, self.E_delta_biases, self.rho, self.eps)
        helper.adadelta_update(self.gamma, dgamma, self.E_g_gamma, self.E_delta_gamma, self.rho, self.eps)
        helper.adadelta_update(self.beta, dbeta, self.E_g_beta, self.E_delta_beta, self.rho, self.eps)

        return dX

    def _depthwise_conv_forward(self, X: cp.ndarray, output_height: float, output_width: float):
        """
        Forward pass for depthwise convolution.
        Args:
            X (cp.ndarray): Input tensor of shape (m, height, width, input_channels)
        Returns:
            conv_out (cp.ndarray): Output tensor of shape (m, output_height, output_width, input_channels)
        """
        m, _, _, input_channels = X.shape

        # im2col: expected shape (self.filter_size * self.filter_size * self.input_channels, m * output_height * output_width)
        X_cols = im2col(X, self.filter_size, self.filter_size, self.padding, self.stride)
        # Reshape X_cols into shape (m * output_height * output_width, filter_size * filter_size * input_channels)
        X_cols = X_cols.reshape(self.filter_size*self.filter_size, input_channels, -1).transpose(2, 0, 1)  # shape: (m * output_height * output_width, filter_size * filter_size, input_channels)
        # Reshape depthwise kernel into shape (k*k, C)

        depthwise_kernel_flat = self.depthwise_kernel.reshape(self.filter_size*self.filter_size, input_channels)  # shape: (self.filter_size*self.filter_size, input_channels)

        # Elementwise multiply and sum over the filter dimension.
        out = cp.sum(X_cols * depthwise_kernel_flat[None, :, :], axis=1)  # shape: (m * output_height * output_width, input_channels)

        # Reshape to (m, output_height, output_width, input_channels)
        conv_out = out.reshape(m, output_height, output_width, input_channels)
        return conv_out

    def _pointwise_conv_forward(self, depthwise_out: cp.ndarray):
        """
        Forward pass for pointwise convolution.
        Args:
            depthwise_out (cp.ndarray): Output tensor from depthwise convolution, shape (m, output_height, output_width, input_channels)
        Returns:
            conv_out (cp.ndarray): Output tensor of shape (m, output_height, output_width, num_filters)
        """
        m, output_height, output_width, _ = depthwise_out.shape

        d_out_flat = depthwise_out.reshape(-1, self.input_channels)  # shape (m * output_height * output_width, input_channels)

        pointwise_kernel_flat = self.pointwise_kernel.reshape(self.input_channels, self.num_filters)  # shape (input_channels, num_filters)
        conv_flat = cp.dot(d_out_flat, pointwise_kernel_flat) + self.biases.reshape(1, self.num_filters)  # shape (m * output_height * output_width, num_filters)

        return conv_flat.reshape(m, output_height, output_width, self.num_filters)  # shape (m, output_height, output_width, num_filters)

    def _depthwise_conv_backward(self, dZ: cp.ndarray):
        """
        Backward pass for depthwise convolution.
        Args:
            dZ (cp.ndarray): Gradient with respect to output, shape (m, output_height, output_width, num_filters)
        Returns:
            dX (cp.ndarray): Gradient with respect to input X.
        """
        X = self.cache[0]['X']
        m, output_height, output_width, num_filters = dZ.shape

        # Use im2col to extract patches from X.
        X_cols = im2col(X, self.filter_size, self.filter_size, self.padding, self.stride)
        X_cols = X_cols.reshape(self.filter_size * self.filter_size, num_filters, -1).transpose(2, 0, 1)  # shape: (m * output_H * output_W, filter_size * filter_size, num_filters)

        # Reshape dZ to (m * output_H * output_W, num_filters)
        dZ_flat = dZ.reshape(m * output_height * output_width, num_filters)  # shape (m * output_H * output_W, num_filters)

        # Compute gradient w.r.t. depthwise kernel per channel:
        # For each channel c, we need to multiply each patch (shape: (self.filter_size * self.filter_size)) by the corresponding dZ (scalar) and sum over positions.
        # Use einsum: 'ijk,ik->jk'
        dW_flat = cp.einsum('ijk,ik->jk', X_cols, dZ_flat)
        dW_depthwise = dW_flat.reshape(self.filter_size, self.filter_size, num_filters)

        # Now, compute gradient w.r.t. input X.
        # The gradient in column space is computed by multiplying the gradient for each channel by the kernel.
        # Reshape depthwise kernel to (self.filter_size * self.filter_size, num_filters)
        depthwise_kernel_flat = self.depthwise_kernel.reshape(self.filter_size * self.filter_size, num_filters)

        dX_cols = (depthwise_kernel_flat[None, :, :] * dZ_flat[:, None, :]).reshape(self.filter_size * self.filter_size * num_filters, -1)
        dX = col2im(dX_cols, X.shape, self.filter_size, self.filter_size, self.padding, self.stride)

        return dX, dW_depthwise

    def _pointwise_conv_backward(self, dZ: cp.ndarray):
        """
        Backward pass for pointwise convolution.
        Args:
            dZ (cp.ndarray): Gradient with respect to output, shape (m, output_height, output_width, num_filters)
        Returns:
            dX (cp.ndarray): Gradient with respect to input X.
        """
        depthwise_out = self.cache[0]['depthwise_out']
        m, output_height, output_width, num_filters = dZ.shape

        # Flatten dZ and depthwise_out
        dZ_flat = dZ.reshape(-1, num_filters)  # shape (m * output_height * output_width, num_filters)
        depthwise_out_flat = depthwise_out.reshape(-1, self.input_channels)  # shape (m * output_height * output_width, input_channels)

        d_pointwise_kernel = cp.dot(
            depthwise_out_flat.T,
            dZ_flat
        ).reshape(self.pointwise_kernel.shape)  # shape (1, 1, input_channels, num_filters)

        d_biases = cp.sum(dZ, axis=(0, 1, 2), keepdims=True)  # shape (1, 1, 1, num_filters)

        # Gradient w.r.t. depthwise_out (input to pointwise conv)
        d_depthwise_flat = cp.dot(
            dZ_flat,
            self.pointwise_kernel.reshape(self.input_channels, self.num_filters).T
        )  # shape (m * output_height * output_width, input_channels)

        d_depthwise = d_depthwise_flat.reshape(m, output_height, output_width, self.input_channels)

        return d_depthwise, d_pointwise_kernel, d_biases
