

import numpy as np
from nn.pkg.blocks.conv_block import ConvBlock
from nn.pkg.layers.fc.layer import FullyConnected
from nn.pkg.layers.yolo_output.layer import YoloOutput


class TinysimmoYOLOModel:
    def __init__(self,
                 model_path: str,
                 S: int, B: int, C: int,
                 ):
        """
        Creates a new instance of the model

        Args:
            model_path (str): path to the model
            S (int): number of grid cells
            B (int): number of bounding boxes per grid cell
            C (int): number of classes
        """
        self.model_path = model_path
        self.S = S
        self.B = B
        self.C = C

    def create_new_model(self,
                         image_size: tuple = (88, 88),
                         conv_l2_lambda=0.0001,
                         fc_l2_lambda=0.0001,
                         clip_value=5.0,
                         conv_momentum=0.8
                         ):
        self.conv_blocks = [
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, input_channels=3, l2_lambda=conv_l2_lambda, clip_value=clip_value, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, input_channels=16, l2_lambda=conv_l2_lambda, clip_value=clip_value, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, input_channels=32, l2_lambda=conv_l2_lambda, clip_value=clip_value, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, input_channels=64, l2_lambda=conv_l2_lambda, clip_value=clip_value, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, input_channels=64, l2_lambda=conv_l2_lambda, clip_value=clip_value, momentum=conv_momentum),
        ]

        for block in self.conv_blocks:
            image_size = block.calc_output_dims(image_size)

        self.fc_layer = FullyConnected(input_size=image_size[0]*image_size[1]*128, output_size=256, l2_lambda=fc_l2_lambda, clip_value=clip_value)  # 256 is a bit more than 4*4(2*5+1)
        self.output_layer = YoloOutput(input_size=256, S=self.S, B=self.B, C=self.C, l2_lambda=fc_l2_lambda, clip_value=clip_value)

    def forward(self, images: np.ndarray):
        """
        Forward pass of the model

        Args:
            images (np.ndarray): images, matrix of shape (m, height, width, num_channels), represents a batch of m images

        Returns:
            np.ndarray: output of the model, matrix of shape (m, S, S, B*5+C), represents a batch of m images
        """
        for block in self.conv_blocks:
            images = block.forward(images)

        # print(images)
        self.conv_output_shape = images.shape
        images = images.reshape(images.shape[0], -1)
        images = self.fc_layer.feed_forward(images)
        images = self.output_layer.feed_forward(images)

        return images

    def backward(self, grad_A: np.ndarray, learning_rate: float):
        """
        Backward pass of the model

        Args:
            grad_A (np.ndarray): gradient of the loss with respect to the output of the model
            learning_rate (float): learning rate to be used for updating the weights and biases
        """
        dZ = self.output_layer.feed_backward(grad_A, learning_rate)
        dZ = self.fc_layer.feed_backward(dZ, learning_rate)

        dZ = dZ.reshape(self.conv_output_shape)

        for block in reversed(self.conv_blocks):
            dZ = block.backward(dZ, learning_rate)

    def save_model(self):
        """
        Saves the model to the disk
        """
        params = {}
        for i, block in enumerate(self.conv_blocks):
            block.get_params(params, f'conv_block_{i}')

        self.fc_layer.get_params(params, 'fc_layer')
        self.output_layer.get_params(params, 'output_layer')
        np.savez(self.model_path, **params)

    def load_model(self):
        """
        Loads the model from the disk
        """
        params = np.load(self.model_path + ".npz", allow_pickle=True)
        for i, block in enumerate(self.conv_blocks):
            block.set_params(params, f'conv_block_{i}')

        self.fc_layer.set_params(params, 'fc_layer')
        self.output_layer.set_params(params, 'output_layer')
