import sys
import tensorflow as tf
from keras import layers

from keras.api.layers import Layer, Dense
from keras.api.regularizers import l2
import keras


@keras.utils.register_keras_serializable()
class TFYoloOutput(Layer):
    def __init__(self, S: int, B: int, C: int, l2_lambda=0.0001, anchors: tf.Tensor = tf.Tensor(), **kwargs):
        out_dim = S * S * (C + B * 5)

        self.S = S
        self.B = B
        self.C = C
        self.out_per_cell = C + B * 5
        self.l2_lambda = l2_lambda

        self.fc = Dense(
            units=out_dim,
            kernel_initializer=keras.initializers.Orthogonal(),  # type: ignore
            bias_initializer=keras.initializers.Constant(-2.0),  # type: ignore
            kernel_regularizer=l2(l2_lambda),
        )

        self.reshaper = layers.Reshape((S, S, self.out_per_cell))
        self.anchors = anchors

        super(TFYoloOutput, self).__init__(**kwargs)

    def call(self, input: tf.Tensor):
        """
        forwrad pass of the layer

        Args:
            input (tf.Tensor): tensor of shape (m, S, S, B*5+C), represents a batch of m images
            m: number of images in the batch

        Returns:
            Z (tf.Tensor): tensor of shape (m, S, S, B*5+C), represents a batch of m images
        """
        A = self.fc(input)
        A = self.reshaper(A)

        box_outputs = A[..., :self.B*5]
        class_outputs = A[..., self.B*5:]

        boxes = tf.reshape(box_outputs, [-1, self.S, self.S, self.B, 5])
        x = boxes[..., 0:1]
        y = boxes[..., 1:2]
        w = boxes[..., 2:3]
        h = boxes[..., 3:4]
        conf = boxes[..., 4:5]

        x_sigmoid = tf.sigmoid(x)
        y_sigmoid = tf.sigmoid(y)
        conf_sigmoid = tf.sigmoid(conf)

        self.grid_x = tf.reshape(tf.range(self.S, dtype=tf.float32), (1, 1, self.S, 1, 1))  # (1, 1, S, 1, 1)
        self.grid_y = tf.reshape(tf.range(self.S, dtype=tf.float32), (1, self.S, 1, 1, 1))  # (1, S, 1, 1, 1)

        x_absolute = (x_sigmoid + self.grid_x) / self.S
        y_absolute = (y_sigmoid + self.grid_y) / self.S

        anchor_dims = tf.reshape(self.anchors, [1, 1, 1, self.B, 2])  # (1, 1, 1, B, 2)
        w_absolute = tf.exp(w) * anchor_dims[..., 0:1]
        h_absolute = tf.exp(h) * anchor_dims[..., 1:2]

        processed_boxes = tf.concat([
            x_absolute,
            y_absolute,
            w_absolute,
            h_absolute,
            conf_sigmoid
        ], axis=-1)

        processed_boxes = tf.reshape(processed_boxes, [-1, self.S, self.S, self.B*5])  # (m, S, S, B*5)
        class_probs = tf.nn.softmax(class_outputs, axis=-1)

        Z = tf.concat([processed_boxes, class_probs], axis=-1)  # (m, S, S, B*5+C)
        return Z

    def get_config(self):
        base_config = super().get_config()
        config = {
            "S": self.S,
            "B": self.B,
            "C": self.C,
            "anchors": self.anchors.numpy(),  # type: ignore
            "l2_lambda": self.l2_lambda,
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        S = config.pop("S")
        B = config.pop("B")
        C = config.pop("C")
        anchors = config.pop("anchors")
        l2_lambda = config.pop("l2_lambda")

        return cls(S=S, B=B, C=C, anchors=anchors, l2_lambda=l2_lambda, **config)
