import tensorflow as tf
from keras import layers

from keras.api.layers import Layer, Dense
from keras.api.regularizers import l2


class TFYoloOutput(Layer):
    def __init__(self, S: int, B: int, C: int, l2_lambda=0.0001, anchors: tf.Tensor = tf.Tensor(), **kwargs):
        super(TFYoloOutput, self).__init__(**kwargs)

        out_dim = S * S * (C + B * 5)

        self.S = S
        self.B = B
        self.C = C
        self.out_per_cell = C + B * 5

        self.fc = Dense(
            units=out_dim,
            kernel_initializer='orthogonal',#'he_normal',
            bias_initializer='zeros',
            kernel_regularizer=l2(l2_lambda),
        )

        self.reshaper = layers.Reshape((S, S, self.out_per_cell))
        self.anchors = anchors

    def call(self, input: tf.Tensor):
        A = self.fc(input)
        A = self.reshaper(A)

        processed_boxes = []
        for b in range(self.B):
            x_coordinate_idx = b * 5  # relative center x coordinate value index
            y_coordinate_idx = b * 5 + 1  # relative center y coordinate value index
            width_idx = b * 5 + 2  # relative width value index
            height_idx = b * 5 + 3  # relative height value index
            confidence_idx = b * 5 + 4  # confidence value index

            x_sigmoid = tf.sigmoid(A[:, :, :, x_coordinate_idx])
            y_sigmoid = tf.sigmoid(A[:, :, :, y_coordinate_idx])

            x_sigmoid = tf.expand_dims(x_sigmoid, axis=-1)  # shape (m, S, S, 1)
            y_sigmoid = tf.expand_dims(y_sigmoid, axis=-1)  # shape (m, S, S, 1)

            grid_x = tf.reshape(tf.range(self.S, dtype=tf.float32), (1, self.S, 1, 1))
            grid_y = tf.reshape(tf.range(self.S, dtype=tf.float32), (1, 1, self.S, 1))

            x = (x_sigmoid + grid_x) / self.S
            y = (y_sigmoid + grid_y) / self.S

            width = tf.expand_dims(A[:, :, :, width_idx], axis=-1)
            height = tf.expand_dims(A[:, :, :, height_idx], axis=-1)

            anchor_w, anchor_h = tf.gather(self.anchors, b)
            width_exp = anchor_w * tf.exp(width)
            height_exp = anchor_h * tf.exp(height)

            conf = tf.expand_dims(A[:, :, :, confidence_idx], axis=-1)
            confidence = tf.sigmoid(conf)

            box_processed = tf.concat([x, y, width_exp, height_exp, confidence], axis=-1)
            processed_boxes.append(box_processed)

        class_probs = tf.nn.softmax(A[:, :, :, self.B * 5:], axis=-1)

        Z = tf.concat(processed_boxes + [class_probs], axis=-1)

        return Z
