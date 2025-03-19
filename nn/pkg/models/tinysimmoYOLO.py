

import numpy as np
import helper.helper as helper
from nn.pkg.blocks.conv_block import ConvBlock
from nn.pkg.layers.fc.layer import FullyConnected
from nn.pkg.layers.yolo_output.layer import YoloOutput


class TinysimmoYOLOModel:
    def __init__(self, 
                 model_path: str, 
                 S: int, B: int, C: int,):
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

    def create_new_model(self, image_size: tuple = (88, 88)):
        self.conv_blocks = [
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, input_channels=3),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, input_channels=16),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, input_channels=32),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, input_channels=64),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, input_channels=64),
        ]

        
        for block in self.conv_blocks:
            image_size = block.calc_output_dims(image_size)
        
        # self.fc_layer = FullyConnected(input_size=image_size[0]*image_size[1]*128, output_size=256)  # 256 is a bit more than 4*4(2*5+1)
        self.fc_layer = FullyConnected(input_size=image_size[0]*image_size[1]*128, output_size=1700)  # TODO: rollback
        # self.output_layer = YoloOutput(input_size=256, S=self.S, B=self.B, C=self.C)
        self.output_layer = YoloOutput(input_size=1700, S=self.S, B=self.B, C=self.C)  # TODO: rollback

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
        
        self.conv_output_shape = images.shape
        images = images.reshape(images.shape[0], -1)
        images = self.fc_layer.feed_forward(images)
        images = self.output_layer.feed_forward(images)

        return images
    
    def backward(self, loss:float, learning_rate: float):
        """
        Backward pass of the model

        Args:
            loss (float): loss of the model
            learning_rate (float): learning rate to be used for updating the weights and biases
        """
        dZ = self.output_layer.feed_backward(loss, learning_rate)
        dZ = self.fc_layer.feed_backward(dZ, learning_rate)
        
        dZ = dZ.reshape(self.conv_output_shape)

        for block in reversed(self.conv_blocks):
            dZ = block.backward(dZ, learning_rate)
            
        
