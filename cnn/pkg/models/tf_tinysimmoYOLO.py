import os
from typing import Optional
import keras
import tensorflow as tf
from cnn.pkg.blocks.tf_conv_block import ConvBlock
from cnn.pkg.layers.yolo_output.tf_layer import TFYoloOutput
from keras.api.layers import Dense, Flatten
from keras import initializers
from keras.api.regularizers import l2
from keras.api import Model


@keras.utils.register_keras_serializable(package="custom")
class TFTinysimmoYOLOModel(Model):
    def __init__(self,
                 S: int, B: int, C: int,
                 conv_blocks: list = [],
                 flatten: Optional[Flatten] = None,
                 fc_layer: Optional[Dense] = None,
                 output_layer: Optional[TFYoloOutput] = None,
                 **kwargs
                 ):
        """
        Creates a new instance of the model

        Args:
            model_path (str): path to the model's weights & biases
            S (int): number of grid cells
            B (int): number of bounding boxes per grid cell
            C (int): number of classes
        """
        super(TFTinysimmoYOLOModel, self).__init__(**kwargs)
        self.S = S
        self.B = B
        self.C = C

        self.conv_blocks = conv_blocks
        self.flatten = flatten
        self.fc_layer = fc_layer
        self.output_layer = output_layer

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

        [conv_block.create_new_block() for conv_block in self.conv_blocks]

        self.flatten = Flatten()

        self.fc_layer = Dense(
            units=256,
            kernel_initializer=keras.initializers.Orthogonal(),  # type: ignore
            bias_initializer='zeros',
            kernel_regularizer=l2(fc_l2_lambda),
        )

        self.output_layer = TFYoloOutput(
            S=self.S,
            B=self.B,
            C=self.C,
            l2_lambda=fc_l2_lambda,
        )

        self.build((None, 88, 88, 3))  # type: ignore

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

        x = self.flatten(x)  # type: ignore
        x = self.fc_layer(x)  # type: ignore
        x = self.output_layer.call(x)  # type: ignore

        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            'S': self.S,
            'B': self.B,
            'C': self.C,
            'conv_blocks': keras.saving.serialize_keras_object(self.conv_blocks),
            'flatten': keras.saving.serialize_keras_object(self.flatten),
            'fc_layer': keras.saving.serialize_keras_object(self.fc_layer),
            'output_layer': keras.saving.serialize_keras_object(self.output_layer),
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        conv_blocks = keras.saving.deserialize_keras_object(config['conv_blocks'])
        flatten = keras.saving.deserialize_keras_object(config['flatten'])
        fc_layer = keras.saving.deserialize_keras_object(config['fc_layer'])
        output_layer = keras.saving.deserialize_keras_object(config['output_layer'])

        return cls(
            S=config['S'],
            B=config['B'],
            C=config['C'],
            conv_blocks=conv_blocks,
            flatten=flatten,
            fc_layer=fc_layer,
            output_layer=output_layer,
        )

    def save_model(self, folder_path: str, model_name: str):
        """
        save the model to the given path

        Args:
            save_path (str): path to save the model
        """

        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, model_name + '.keras')

        keras.models.save_model(
            self,
            file_path,
            overwrite=True,
        )

    def load_model(self, folder_path: str, model_name: str):
        """
        load the model from the given path

        Args:
            load_path (str): path to load the model
        """
        file_path = os.path.join(folder_path, model_name + '.keras')
        model = keras.models.load_model(
            file_path,
            custom_objects={
                'TFYoloOutput': TFYoloOutput,
                'TFTinysimmoYOLOModel': TFTinysimmoYOLOModel,
                'ConvBlock': ConvBlock,
            },
            compile=False,
        )

        self.conv_blocks = model.conv_blocks
        self.flatten = model.flatten
        self.fc_layer = model.fc_layer
        self.output_layer = model.output_layer
        self.S = model.S
        self.B = model.B
        self.C = model.C
