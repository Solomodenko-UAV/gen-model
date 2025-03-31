import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.python.keras.activations import leaky_relu
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

conv_filter_size = 3
conv_stride = 1
conv_padding = 1
maxpool_stride = 2
maxpool_filter_size = 2

class ConvBlock:
    def __init__(self,
                 first_layer_num_of_filters: int,
                 second_layer_num_of_filters: int,
                 l2_lambda: float = 0.0001,
                 ):

        self.conv1 = Conv2D(
            filters=first_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_lambda),
        )
        self.conv1_norm = tf.keras.layers.BatchNormalization
        self.conv1_activation = leaky_relu(alpha=0.25)

        self.max_pool1 = MaxPooling2D(
            pool_size=(maxpool_filter_size, maxpool_filter_size),
            strides=(maxpool_stride, maxpool_stride),
        )

        self.conv2 = Conv2D(
            filters=second_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_lambda),
        )
        self.max_pool2 = MaxPooling2D(
            pool_size=(maxpool_filter_size, maxpool_filter_size),
            strides=(maxpool_stride, maxpool_stride),
        )
        
        
    def add_to_model(self, model):
        model.add(self.conv1)
        model.add(self.max_pool1)
        model.add(self.conv2)
        model.add(self.max_pool2)
