
import keras
import tensorflow as tf
from cnn.pkg.blocks.tf_conv_block import ConvBlock
from cnn.pkg.layers.yolo_output.tf_layer import TFYoloOutput
from keras.api.layers import Dense, Flatten
from keras import initializers
from keras.api.regularizers import l2
from keras.api import Model


class TFTinysimmoYOLOModel(Model):
    def __init__(self,
                 model_path: str,
                 S: int, B: int, C: int,
                 ):
        """
        Creates a new instance of the model

        Args:
            model_path (str): path to the model's weights & biases
            S (int): number of grid cells
            B (int): number of bounding boxes per grid cell
            C (int): number of classes
        """
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
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=16, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=16, second_layer_num_of_filters=32, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=32, second_layer_num_of_filters=64, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=64, second_layer_num_of_filters=64, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
            ConvBlock(first_layer_num_of_filters=128, second_layer_num_of_filters=128, l2_lambda=conv_l2_lambda, momentum=conv_momentum),
        ]

        self.flatten = Flatten()

        self.fc_layer = Dense(
            units=256,
            kernel_initializer=keras.initializers.Orthogonal(), # type: ignore
            bias_initializer='zeros',
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

    def get_anchors(self):
        return self.output_layer.anchors

    def call(self, images: tf.Tensor):
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
        x = self.output_layer.call(x)

        return x
    
    def build(self, input_shape):
        """
        build the model

        Args:
            input_shape (tuple): input shape of the model
        """
        if not hasattr(self, 'conv_blocks') or self.conv_blocks is None:
            raise ValueError("Model must be created using create_new_model() before building.")
        # Build the convolutional blocks
    
        super(TFTinysimmoYOLOModel, self).build(input_shape)
