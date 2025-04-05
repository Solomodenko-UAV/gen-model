from keras.api.activations import leaky_relu
from keras.api.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras import layers
from keras.api.regularizers import l2
from keras.api.layers import Layer


conv_filter_size = 3
conv_stride = 1
conv_padding = 1
maxpool_stride = 2
maxpool_filter_size = 2


class ConvBlock(Layer):
    def __init__(self,
                 first_layer_num_of_filters: int,
                 second_layer_num_of_filters: int,
                 l2_lambda: float = 0.0001,
                 momentum=0.8,
                 **kwargs,
                 ):
        super(ConvBlock, self).__init__(**kwargs)

        self.conv1 = Conv2D(
            filters=first_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=l2(l2_lambda),
        )

        self.batch_norm1 = layers.BatchNormalization(
            momentum=momentum,
        )

        self.activation1 = LeakyReLU(
            # negative_slope=0.1,
        )

        self.max_pool1 = MaxPooling2D(
            pool_size=(maxpool_filter_size, maxpool_filter_size),
            strides=(maxpool_stride, maxpool_stride),
        )

        self.conv2 = Conv2D(
            filters=second_layer_num_of_filters,
            kernel_size=(conv_filter_size, conv_filter_size),
            strides=(conv_stride, conv_stride),
            padding='same',
            kernel_regularizer=l2(l2_lambda),
            kernel_initializer='he_normal',
            bias_initializer='zeros',
        )

        self.batch_norm2 = layers.BatchNormalization(
            momentum=momentum,
        )

        self.activation2 = LeakyReLU(
            # negative_slope=0.1,
        )

    def call(self, input):
        X = self.conv1(input)
        X = self.batch_norm1(X)
        X = self.activation1(X)
        X = self.max_pool1(X)

        X = self.conv2(X)
        X = self.batch_norm2(X)
        X = self.activation2(X)

        return X
