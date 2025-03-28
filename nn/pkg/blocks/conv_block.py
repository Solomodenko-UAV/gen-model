import os

from nn.pkg.layers.activation.prelu import PReLU

if os.environ.get("USE_GPU") != False and os.environ.get("USE_GPU") != 'True':
    import numpy as cp
else:
    import cupy as cp

from nn.pkg.activations import activations
from nn.pkg.layers.conv.layer import Convolution
from nn.pkg.layers.maxpool.layer import MaxPool
import helper.helper as helper

conv_filter_size = 3
conv_stride = 1
conv_padding = 1
maxpool_stride = 2
maxpool_filter_size = 2


class ConvBlock:
    def __init__(self,
                 first_layer_num_of_filters: int,
                 second_layer_num_of_filters: int,
                 input_channels: int,
                 l2_lambda: float = 0.0001,
                 clip_value: float = 5.0,
                 momentum=0.8,
                 strategy='cmc'  # cmc or ccm
                 ):

        self.conv1 = Convolution(
            input_channels=input_channels,
            filter_size=conv_filter_size,
            num_filters=first_layer_num_of_filters,
            stride=conv_stride,
            padding=conv_stride,
            clip_value=clip_value,
            l2_lambda=l2_lambda,
            momentum=momentum
        )

        self.conv1_activation = PReLU(alpha=0.25)

        self.conv2 = Convolution(
            input_channels=first_layer_num_of_filters,
            filter_size=conv_filter_size,
            num_filters=second_layer_num_of_filters,
            stride=conv_stride,
            padding=conv_stride,
            clip_value=clip_value,
            l2_lambda=l2_lambda,
            momentum=momentum
        )

        self.conv2_activation = PReLU(alpha=0.25)

        self.maxpool = MaxPool(
            pool_size=maxpool_filter_size,
            stride=maxpool_stride,
        )

        self.activation_values = []
        self.strategy = strategy

    def forward(self, X: cp.ndarray):
        """
        forward pass of the convolution block

        Args:
            X (cp.ndarray): input to the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images

        Returns:
            Z (cp.ndarray): output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
        """

        if self.strategy == 'cmc':
            return self.cmc_forward(X)

        return self.ccm_forward(X)

    def backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward pass of the convolution block

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns:
            dA(cp.ndarray): gradient of the cost with respect to the input of the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images
        """

        if self.strategy == 'cmc':
            return self.cmc_backward(dZ, learning_rate)

        return self.ccm_backward(dZ, learning_rate)

    # conv1, conv2, maxpool
    def ccm_forward(self, X: cp.ndarray):
        """
        forward pass of the convolution block

        Args:
            X (cp.ndarray): input to the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images

        Returns:
            Z (cp.ndarray): output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
        """

        A_conv1 = self.conv1.convolve_forward_vectorized(X)
        Z_conv1 = self.conv1_activation.forward(A_conv1)

        self.activation_values.append(("conv1_mean", cp.mean(A_conv1), "conv1_median", cp.median(A_conv1)))

        A_conv2 = self.conv2.convolve_forward_vectorized(Z_conv1)
        Z_conv2 = self.conv2_activation.forward(A_conv2)

        self.activation_values.append(("conv2_mean", cp.mean(A_conv1), "conv2_median", cp.median(A_conv1)))

        Z = self.maxpool.pool_forward_vectorized(Z_conv2)

        return Z

    # conv1, conv2, maxpool
    def ccm_backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward pass of the convolution block

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns:
            dA(cp.ndarray): gradient of the cost with respect to the input of the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images
        """

        dZ_maxpool = self.maxpool.pool_backward_vectorized(dZ)

        dZ_conv2 = self.conv2_activation.backward(dZ_maxpool, learning_rate)
        dA_conv2, _ = self.conv2.convolve_backward_vectorized(dZ_conv2, learning_rate)

        dZ_conv1 = self.conv1_activation.backward(dA_conv2, learning_rate)
        dA, dW = self.conv1.convolve_backward_vectorized(dZ_conv1, learning_rate)

        return dA, dW

    # conv1, maxpool, conv2
    def cmc_forward(self, X: cp.ndarray):
        """
        forward pass of the convolution block

        Args:
            X (cp.ndarray): input to the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images

        Returns:
            Z (cp.ndarray): output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
        """

        A_conv1 = self.conv1.convolve_forward_vectorized(X)
        Z_conv1 = self.conv1_activation.forward(A_conv1)

        self.activation_values.append(("conv1_mean", cp.mean(A_conv1), "conv1_median", cp.median(A_conv1)))

        Z_maxpool = self.maxpool.pool_forward_vectorized(Z_conv1)

        A_conv2 = self.conv2.convolve_forward_vectorized(Z_maxpool)
        Z_conv2 = self.conv2_activation.forward(A_conv2)

        self.activation_values.append(("conv2_mean", cp.mean(A_conv1), "conv2_median", cp.median(A_conv1)))

        return Z_conv2

    # conv1, maxpool, conv2
    def cmc_backward(self, dZ: cp.ndarray, learning_rate: float):
        """
        backward pass of the convolution block

        Args:
            dZ (cp.ndarray): gradient of the cost with respect to the output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns:
            dA(cp.ndarray): gradient of the cost with respect to the input of the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images
        """

        dZ_conv2 = self.conv2_activation.backward(dZ, learning_rate)
        dA_conv2, _ = self.conv2.convolve_backward_vectorized(dZ_conv2, learning_rate)

        dA_maxpool = self.maxpool.pool_backward_vectorized(dA_conv2)

        dZ_conv1 = self.conv1_activation.backward(dA_maxpool, learning_rate)
        dA, dW = self.conv1.convolve_backward_vectorized(dZ_conv1, learning_rate)

        return dA, dW

    def calc_output_dims(self, image_wh: tuple):
        conv1_wh = helper.calc_conv_layer_out_dim(image_wh, padding=1, kernel_size=conv_filter_size, stride=1)
        conv2_wh = helper.calc_conv_layer_out_dim(conv1_wh, padding=1, kernel_size=conv_filter_size, stride=1)
        maxpool_wh = helper.calc_conv_layer_out_dim(conv2_wh, padding=0, kernel_size=maxpool_filter_size, stride=maxpool_stride)

        return maxpool_wh

    def get_params(self, params: dict, key: str):
        self.conv1.get_params(params, f'{key}_conv1_params')
        self.conv2.get_params(params, f'{key}_conv2_params')
        self.maxpool.get_params(params, f'{key}_pool_params')

    def set_params(self, params: dict, key: str):
        self.conv1.set_params(params, f'{key}_conv1_params')
        self.conv2.set_params(params, f'{key}_conv2_params')
        self.maxpool.set_params(params, f'{key}_pool_params')
