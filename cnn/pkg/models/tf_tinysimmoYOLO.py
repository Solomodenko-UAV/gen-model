
import tensorflow as tf
from blocks.tf_conv_block import ConvBlock
from layers.yolo_output.tf_layer import TFYoloOutput
from tensorflow.python.keras.layers import Dense, Flatten
from keras import initializers
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import Model


class TFTinysimmoYOLOModel(Model):
    def __init__(self,
                 model_path: str,
                 S: int, B: int, C: int,
                 ):
        super(TFTinysimmoYOLOModel, self).__init__()
        self.model_path = model_path
        self.S = S
        self.B = B
        self.C = C

    def create_new_model(self,
                         conv_l2_lambda=0.0001,
                         fc_l2_lambda=0.0001,
                         conv_momentum=0.8
                         ):
        self.conv_blocks = [
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, input_channels=3, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, input_channels=16, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, input_channels=32, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, input_channels=64, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, input_channels=64, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
        ]

        self.flatten = Flatten()

        self.fc_layer = Dense(
            units=256,
            kernel_initializer=initializers.Orthogonal(),
            bias_initializer=initializers.Zeros(),
            kernel_regularizer=l2(fc_l2_lambda),
        )

        self.output_layer = TFYoloOutput(
            S=self.S,
            B=self.B,
            C=self.C,
            l2_lambda=fc_l2_lambda,
        )

    def set_anchors(self, anchors: tf.Tensor):
        """
        set the anchors for the model

        Args:
            anchors (tf.Tensor): anchors for the model
        """
        self.output_layer.anchors = anchors

    def call(self, images: tf.Tensor, anchors: tf.Tensor = None):
        """
        forward pass of the model

        Args:
            images (tf.Tensor): input images
            anchors (tf.Tensor): anchors for the model

        Returns:
            tf.Tensor: output of the model
        """
        x = images
        for block in self.conv_blocks:
            x = block.call(x)

        x = self.flatten(x)
        x = self.fc_layer(x)
        x = self.output_layer.call(x, anchors)

        return x
