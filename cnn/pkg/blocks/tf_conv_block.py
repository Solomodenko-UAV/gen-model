import keras
from keras.api.activations import leaky_relu
from keras.api.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras import layers
from keras.api.regularizers import l2
from keras.api.layers import Layer
from typing import Optional


conv_filter_size = 3
conv_stride = 1
conv_padding = 1
maxpool_stride = 2
maxpool_filter_size = 2


@keras.utils.register_keras_serializable()
class ConvBlock(Layer):
    def __init__(self,
                 first_layer_num_of_filters: int,
                 second_layer_num_of_filters: int,
                 l2_lambda: float = 0.0001,
                 conv1: Optional[Conv2D] = None,
                 batch_norm1: Optional[layers.BatchNormalization] = None,
                 activation1: Optional[LeakyReLU] = None,
                 max_pool1: Optional[MaxPooling2D] = None,
                 conv2: Optional[Conv2D] = None,
                 batch_norm2: Optional[layers.BatchNormalization] = None,
                 activation2: Optional[LeakyReLU] = None,
                 momentum: float = 0.8,
                 **kwargs,
                 ):

        super(ConvBlock, self).__init__(**kwargs)

        self.first_layer_num_of_filters = first_layer_num_of_filters
        self.second_layer_num_of_filters = second_layer_num_of_filters
        self.l2_lambda = l2_lambda
        self.momentum = momentum

        self.conv1 = conv1
        self.batch_norm1 = batch_norm1
        self.activation1 = activation1
        self.max_pool1 = max_pool1
        self.conv2 = conv2
        self.batch_norm2 = batch_norm2
        self.activation2 = activation2

    def create_new_block(self):
        self.conv1 = Conv2D(
            filters=self.first_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_initializer=keras.initializers.Orthogonal(),  # type: ignore
            bias_initializer='zeros',
            kernel_regularizer=l2(self.l2_lambda),
        )

        self.batch_norm1 = layers.BatchNormalization(
            momentum=self.momentum,
        )

        self.activation1 = LeakyReLU(
            # negative_slope=0.1,
        )

        self.max_pool1 = MaxPooling2D(
            pool_size=(maxpool_filter_size, maxpool_filter_size),
            strides=(maxpool_stride, maxpool_stride),
        )

        self.conv2 = Conv2D(
            filters=self.second_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_regularizer=l2(self.l2_lambda),
            kernel_initializer=keras.initializers.Orthogonal(),  # type: ignore
            bias_initializer='zeros',
        )

        self.batch_norm2 = layers.BatchNormalization(
            momentum=self.momentum,
        )

        self.activation2 = LeakyReLU(
            # negative_slope=0.1,
        )

    def call(self, input):
        X = self.conv1(input)  # type: ignore
        X = self.batch_norm1(X)  # type: ignore
        X = self.activation1(X)  # type: ignore
        X = self.max_pool1(X)  # type: ignore
        X = self.conv2(X)  # type: ignore
        X = self.batch_norm2(X)  # type: ignore
        X = self.activation2(X)  # type: ignore

        return X

    def get_config(self):
        base_config = super().get_config()
        config = {
            'first_layer_num_of_filters': self.first_layer_num_of_filters,
            'second_layer_num_of_filters': self.second_layer_num_of_filters,
            'l2_lambda': self.l2_lambda,
            'momentum': self.momentum,
            'name': self.name,
            'conv1': keras.saving.serialize_keras_object(self.conv1),
            'batch_norm1': keras.saving.serialize_keras_object(self.batch_norm1),
            'activation1': keras.saving.serialize_keras_object(self.activation1),
            'max_pool1': keras.saving.serialize_keras_object(self.max_pool1),
            'conv2': keras.saving.serialize_keras_object(self.conv2),
            'batch_norm2': keras.saving.serialize_keras_object(self.batch_norm2),
            'activation2': keras.saving.serialize_keras_object(self.activation2),
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        conv1 = keras.saving.deserialize_keras_object(config['conv1'])
        batch_norm1 = keras.saving.deserialize_keras_object(config['batch_norm1'])
        activation1 = keras.saving.deserialize_keras_object(config['activation1'])
        max_pool1 = keras.saving.deserialize_keras_object(config['max_pool1'])
        conv2 = keras.saving.deserialize_keras_object(config['conv2'])
        batch_norm2 = keras.saving.deserialize_keras_object(config['batch_norm2'])
        activation2 = keras.saving.deserialize_keras_object(config['activation2'])

        return cls(
            first_layer_num_of_filters=config['first_layer_num_of_filters'],
            second_layer_num_of_filters=config['second_layer_num_of_filters'],
            l2_lambda=config['l2_lambda'],
            momentum=config['momentum'],
            name=config['name'],
            conv1=conv1,
            batch_norm1=batch_norm1,
            activation1=activation1,
            max_pool1=max_pool1,
            conv2=conv2,
            batch_norm2=batch_norm2,
            activation2=activation2,
        )
