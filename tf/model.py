import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.python.keras.activations import leaky_relu
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from tf.conv_block import ConvBlock


class TfModel:
    def __init__(self, model):

    def create_new_model(self):
        self.model = Sequential()

        self.conv_blocks = [
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, input_channels=3),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, input_channels=16),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, input_channels=32),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, input_channels=64),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, input_channels=64),
        ]
        
        for block in self.conv_blocks:
            block.add_to_model(self.model)
        
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Dense(6, activation='softmax'))
