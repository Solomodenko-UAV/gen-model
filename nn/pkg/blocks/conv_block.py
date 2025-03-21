import numpy as np

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
                 momentum=0.8
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

        self.maxpool = MaxPool(
            pool_size=maxpool_filter_size,
            stride=maxpool_stride,
        )

    def forward(self, X: np.ndarray):
        """
        forward pass of the convolution block

        Args:
            X (np.ndarray): input to the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images

        Returns:
            Z (np.ndarray): output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
        """

        A_conv1 = self.conv1.convolve_forward(X)
        Z_conv1 = activations.relu(A_conv1)

        A_conv2 = self.conv2.convolve_forward(Z_conv1)
        Z_conv2 = activations.relu(A_conv2)

        Z = self.maxpool.pool_forward(Z_conv2)

        return Z

    def backward(self, dZ: np.ndarray, learning_rate: float):
        """
        backward pass of the convolution block

        Args:
            dZ (np.ndarray): gradient of the cost with respect to the output of the convolution block - matrix of shape (m, output_height, output_width, num_output_channels), represents a batch of m images
            learning_rate (float): learning rate to be used for updating the weights and biases

        Returns:
            dA(np.ndarray): gradient of the cost with respect to the input of the convolution block - matrix of shape (m, height, width, num_input_channels), represents a batch of m images
        """

        dZ_maxpool = self.maxpool.pool_backward(dZ)
        dZ_conv2 = activations.relu_derivative(dZ_maxpool)
        dA_conv2 = self.conv2.convolve_backward(dZ_conv2, learning_rate)

        dZ_conv1 = activations.relu_derivative(dA_conv2)
        dA = self.conv1.convolve_backward(dZ_conv1, learning_rate)

        return dA

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
