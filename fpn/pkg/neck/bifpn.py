from fpn.pkg.layers.depthwise_conv.layer import DepthwiseConv
import os

from helper import helper


on_cpu = os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True'

if on_cpu:
    import numpy as cp
else:
    import cupy as cp


class BiFPN:
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-6):
        """
        BiFPN module

        Args:
            in_channels (int): number of channels in the input feature map.
            out_channels (int): number of channels for the fused features.
        """
        self.lateral_high = DepthwiseConv(input_channels=in_channels, filter_size=1, num_filters=out_channels, stride=1, padding=0)

        self.downsample_initial = DepthwiseConv(input_channels=in_channels, filter_size=3, num_filters=in_channels, stride=2, padding=1)
        self.lateral_low = DepthwiseConv(input_channels=in_channels, filter_size=1, num_filters=out_channels, stride=1, padding=0)

        self.antialias_top = DepthwiseConv(input_channels=out_channels, filter_size=3, num_filters=out_channels, stride=1, padding=1)

        self.downsample_bottom = DepthwiseConv(input_channels=out_channels, filter_size=3, num_filters=out_channels, stride=2, padding=1)
        self.antialias_bottom = DepthwiseConv(input_channels=out_channels, filter_size=3, num_filters=out_channels, stride=1, padding=1)

        self.eps = eps

        # top-down weights
        self.weights1_high = cp.array([1.0], dtype=cp.float32)
        self.weights1_low = cp.array([1.0], dtype=cp.float32)

        # bottom-up weights
        self.weights2_high = cp.array([1.0], dtype=cp.float32)
        self.weights2_low = cp.array([1.0], dtype=cp.float32)

        # AdaDelta hyperparameters
        self.rho = 0.95
        self.E_g_weights1_high = cp.zeros_like(self.weights1_high)
        self.E_delta_weights1_high = cp.zeros_like(self.weights1_high)
        self.E_g_weights1_low = cp.zeros_like(self.weights1_low)
        self.E_delta_weights1_low = cp.zeros_like(self.weights1_low)

        self.E_g_weights2_high = cp.zeros_like(self.weights2_high)
        self.E_delta_weights2_high = cp.zeros_like(self.weights2_high)
        self.E_g_weights2_low = cp.zeros_like(self.weights2_low)
        self.E_delta_weights2_low = cp.zeros_like(self.weights2_low)

    def forward(self, X: cp.ndarray):
        """
        Forward pass of the BiFPN module

        Args:
            X (cp.ndarray): output from the backbone with shape (m, height, width, in_channels)


        Returns:
            tuple: high resolution and low resolution feature maps with shape (m, height, width, out_channels)            
        """

        # separate backbone output into high and low resolution features
        X_high = self.lateral_high.convolve_forward(X)  # (m, height, width, out_channels)

        # initial downsampling to low resolution
        X_low_initial = self.downsample_initial.convolve_forward(X)  # (m, height/2, width/2, in_channels)
        X_low = self.lateral_low.convolve_forward(X_low_initial)  # (m, height/2, width/2, out_channels)

        # top-down fusion
        X_low_upsampled = self._upsample(X_low)  # (m, height, width, out_channels)
        fused_high = self._fuse_features_forward(X_high, X_low_upsampled, "weights1")  # (m, height, width, out_channels)
        X_high_fused = self.antialias_top.convolve_forward(fused_high)  # (m, height, width, out_channels)

        # bottom-up fusion
        X_high_downsampled = self.downsample_bottom.convolve_forward(X_high_fused)  # (m, height/2, width/2, out_channels)
        fused_low = self._fuse_features_forward(X_low, X_high_downsampled, "weights2")  # (m, height/2, width/2, out_channels)
        X_low_fused = self.antialias_bottom.convolve_forward(fused_low)

        self.cache = {'X_high': X_high, 'X_low': X_low, 'X_low_upsampled': X_low_upsampled, 'X_high_downsampled': X_high_downsampled}
        return X_high_fused, X_low_fused

    def backward(self, dX_high_fused, dX_low_fused: cp.ndarray):
        """
        Backward pass of the BiFPN module

        Args:
            dX_high_fused (cp.ndarray): gradient of the loss with respect to the high resolution feature map
            dX_low_fused (cp.ndarray): gradient of the loss with respect to the low resolution feature map
        """

        # backprop bottom-up fusion
        dX_low_fused = self.antialias_bottom.convolve_backward(dX_low_fused)
        dX_high_downsampled = self.downsample_bottom.convolve_backward(dX_low_fused)

        # backprop top-down fusion
        dX_high_fused_total = self.antialias_top.convolve_backward(dX_high_fused + dX_high_downsampled)
        dX_low_upsampled = self._upsample_backward(dX_high_fused_total)

        # backprop initial layers
        dX_low_combined = dX_low_fused + dX_low_upsampled
        dX_low_initial = self.lateral_low.convolve_backward(dX_low_combined)
        dX_initial = self.downsample_initial.convolve_backward(dX_low_initial)

        dX_high = self.lateral_high.convolve_backward(dX_high_fused)

        dX = dX_initial + dX_high  # combine gradients from high and low resolution paths

        # top-down
        X_high = self.cache['X_high']
        X_low_upsampled = self.cache['X_low_upsampled']

        total_weights1 = self.weights1_high + self.weights1_low + self.eps
        dW1_high = cp.sum(dX_high_fused * (X_high * total_weights1 - (self.weights1_high * X_high + self.weights1_low * X_low_upsampled)) / total_weights1**2)
        dW1_low = cp.sum(dX_high_fused * (X_low_upsampled * total_weights1 - (self.weights1_high * X_high + self.weights1_low * X_low_upsampled)) / total_weights1**2)

        # bottom-up
        X_high_downsampled = self.cache['X_high_downsampled']
        X_low = self.cache['X_low']

        total_weights2 = self.weights2_high + self.weights2_low + self.eps
        dW2_high = cp.sum(dX_low_fused * (X_low * total_weights2 - (self.weights2_high * X_low + self.weights2_low * X_high_downsampled)) / total_weights2**2)
        dW2_low = cp.sum(dX_low_fused * (X_high_downsampled * total_weights2 - (self.weights2_high * X_low + self.weights2_low * X_high_downsampled)) / total_weights2**2)

        helper.adadelta_update(self.weights1_high, dW1_high, self.E_g_weights1_high, self.E_delta_weights1_high, self.rho)
        helper.adadelta_update(self.weights1_low, dW1_low, self.E_g_weights1_low, self.E_delta_weights1_low, self.rho)
        helper.adadelta_update(self.weights2_high, dW2_high, self.E_g_weights2_high, self.E_delta_weights2_high, self.rho)
        helper.adadelta_update(self.weights2_low, dW2_low, self.E_g_weights2_low, self.E_delta_weights2_low, self.rho)

        return dX

    def _upsample(self, X: cp.ndarray, scale_factor: int = 2):
        """
        Nearest-neighbor upsampling

        Args:
            X (cp.ndarray): input feature map to be upsampled with shape (m, height, width, channels)
            scale_factor (int, optional): upsampling scale factor. Defaults to 2.

        Returns:
            cp.ndarray: upsampled feature map with shape (m, height * scale_factor, width * scale_factor, channels)
        """

        return cp.repeat(cp.repeat(X, scale_factor, axis=1), scale_factor, axis=2)

    def _upsample_backward(self, dX_up: cp.ndarray, scale_factor: int = 2) -> cp.ndarray:
        """
        Backward pass for nearest-neighbor upsampling.
        Sums gradients from each repeated block.

        Args:
            dX_up (cp.ndarray): Gradient from upsampled feature, shape (m, H_up, W_up, channels).
            scale_factor (int): Upsampling factor.

        Returns:
            cp.ndarray: Gradient with respect to the original input, shape (m, H, W, channels).
        """
        m, H_up, W_up, channels = dX_up.shape
        H = H_up // scale_factor
        W = W_up // scale_factor

        # reshape so that each block is grouped together, then sum over the upsample dimensions.
        dX = dX_up.reshape(m, H, scale_factor, W, scale_factor, channels).sum(axis=2).sum(axis=3)
        return dX

    def _fuse_features_forward(self, X_high: cp.ndarray, X_low: cp.ndarray, weights_num: str):
        """
        Fuses high and low resolution features using the weights

        Args:
            X_high (cp.ndarray): matrix of shape (m, height, width, out_channels) - high resolution features
            X_low (cp.ndarray): matrix of shape (m, height, width, out_channels) - low resolution features
            weights_num (str): weights1 or weights2 - which weights to use for fusion

        Returns:
            cp.ndarray: fused features with shape (m, height, width, out_channels)
        """

        weights_high = self.weights1_high if weights_num == 'weights1' else self.weights2_high
        weights_low = self.weights1_low if weights_num == 'weights1' else self.weights2_low

        total_weight = weights_high + weights_low + self.eps

        return (X_high * weights_high + X_low * weights_low) / total_weight
