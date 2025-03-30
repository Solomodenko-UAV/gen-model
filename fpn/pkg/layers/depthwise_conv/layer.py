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
        depthwise_temp = helper.create_orthogonal_4d_matrix((filter_size, filter_size, input_channels, 1), std_depthwise)
        self.depthwise_kernel = cp.squeeze(depthwise_temp, axis=3)  # shape (filter_size, filter_size, input_channels)

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
        helper.adadelta_update(self.gamma, dgamma, self.E_g_biases, self.E_delta_biases, self.rho, self.eps)
        helper.adadelta_update(self.beta, dbeta, self.E_g_biases, self.E_delta_biases, self.rho, self.eps)

        return dX

    def _depthwise_conv_forward(self, X: cp.ndarray, output_height: float, output_width: float):
        """
        Forward pass for depthwise convolution.
        Args:
            X (cp.ndarray): Input tensor of shape (m, height, width, input_channels)
        Returns:
            conv_out (cp.ndarray): Output tensor of shape (m, output_height, output_width, input_channels)
        """
        m, _, _, _ = X.shape
        k = self.filter_size

        shape = (m, output_height, output_width, k, k, self.input_channels)
        strides = (X.strides[0],
                   X.strides[1] * self.stride,
                   X.strides[2] * self.stride,
                   X.strides[1],
                   X.strides[2],
                   X.strides[3])
        X_windows = cp.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        # Depthwise convolution: multiply each patch by the corresponding depthwise kernel.
        # self.depthwise_kernel has shape (k, k, input_channels); broadcast it to (1,1,1,k,k,input_channels).
        return cp.sum(X_windows * self.depthwise_kernel[None, None, None, :, :, :], axis=(3, 4))  # shape (m, output_height, output_width, input_channels)

    def _pointwise_conv_forward(self, depthwise_out: cp.ndarray):
        """
        Forward pass for pointwise convolution.
        Args:
            depthwise_out (cp.ndarray): Output tensor from depthwise convolution, shape (m, output_height, output_width, input_channels)
        Returns:
            conv_out (cp.ndarray): Output tensor of shape (m, output_height, output_width, num_filters)
        """
        m, _, _, _ = depthwise_out.shape

        d_out_flat = depthwise_out.reshape(-1, self.input_channels)  # shape (m * output_height * output_width, input_channels)

        pointwise_kernel_flat = self.pointwise_kernel.reshape(self.input_channels, self.num_filters)  # shape (input_channels, num_filters)
        conv_flat = cp.dot(d_out_flat, pointwise_kernel_flat) + self.biases.reshape(1, self.num_filters)  # shape (m * output_height * output_width, num_filters)

        return conv_flat.reshape(m, -1, self.num_filters)  # shape (m, output_height, output_width, num_filters)

    def _depthwise_conv_backward(self, dZ: cp.ndarray):
        """
        Backward pass for depthwise convolution.
        Args:
            dZ (cp.ndarray): Gradient with respect to output, shape (m, output_height, output_width, num_filters)
        Returns:
            dX (cp.ndarray): Gradient with respect to input X.
        """
        X = self.cache[0]['X']
        filter_height, filter_width, _ = self.depthwise_kernel.shape

        X_cols = im2col(X, filter_height, filter_width, self.padding, self.stride)  # shape (m * output_height * output_width, filter_height * filter_width * input_channels)

        X_cols = X_cols.reshape(filter_height * filter_width, self.input_channels, -1)  # shape (filter_height * filter_width, input_channels, m * output_height * output_width)

        dZ_reshaped = dZ.transpose(0, 3, 1, 2).reshape(self.input_channels, -1)  # shape (input_channels, m * output_height * output_width)

        dW_flat = cp.einsum('ijk,ik->ij', X_cols, dZ_reshaped)
        dW = dW_flat.reshape(filter_height, filter_width, self.input_channels)  # shape (filter_height, filter_width, input_channels)

        depthwise_kernel_flat = self.depthwise_kernel.reshape(filter_height, filter_width, self.input_channels)

        # For each channel c, the contribution is: depthwise_kernel_flat[:, c] * dZ_reshaped[c, :]
        # We can compute this by:
        dX_cols = (depthwise_kernel_flat[:, :, None] * dZ_reshaped[None, :, :]).reshape(filter_height*filter_width * self.input_channels, -1)

        dX = col2im(dX_cols, X.shape, filter_height, filter_width, self.padding, self.stride)

        return dX, dW

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

        dZ_flat = dZ.reshape(-1, num_filters)  # shape (m * output_height * output_width, num_filters)
        depthwise_out_flat = depthwise_out.reshape(-1, self.input_channels)  # shape (m * output_height * output_width, input_channels)

        # Gradient with respect to pointwise kernel
        d_pointwise_kernel = cp.tensordot(depthwise_out_flat, dZ_flat, axes=([0], [0])).reshape(self.pointwise_kernel.shape)

        # Gradient with respect to biases.
        d_biases = cp.sum(dZ, axis=(0, 1, 2), keepdims=True)

        # Gradient w.r.t. depthwise_out.
        d_depthwise_flat = cp.dot(dZ_flat, self.pointwise_kernel.reshape(self.input_channels, self.num_filters).T)
        d_depthwise = d_depthwise_flat.reshape(m, output_height, output_width, self.input_channels)

        return d_depthwise, d_pointwise_kernel, d_biases
