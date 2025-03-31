import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import adadelta_v2
from cnn.pkg.layers.data_pre_processing.tf_data_pre_processing import DataPreProcessor
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from cnn.pkg.layers.losses.tf_loss import YOLOLoss
from tensorflow.python.ops.image_ops import non_max_suppression
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from metadata import visDrone


class TFYOLOActorPhoto():
    def __init__(self,
                 data_folder: str,
                 annotation_folder: str,
                 model_input_img_res: tuple,
                 model: TFTinysimmoYOLOModel,
                 mini_batch_size: int
                 ):

        self.data_processor = DataPreProcessor(images_folder=data_folder, annotations_folder=annotation_folder)
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.input_shape = model_input_img_res

    def train_model(self):
        orig_image, downscaled_image, downscaled_annotations = self.data_processor.load_data_for_training(self.input_shape)

        factor_W = orig_image.shape[0] / self.input_shape[0]
        factor_H = orig_image.shape[1] / self.input_shape[1]
        grid_cell_size = (downscaled_image.shape[0] / self.model.S, downscaled_image.shape[1] / self.model.S)
        self._compile_model(factor_H, factor_W, grid_cell_size)

        X = keras.Input(shape=self.input_shape, batch_size=self.mini_batch_size)(downscaled_image)
        Y_true = keras.Input(shape=downscaled_annotations.shape, batch_size=self.mini_batch_size)(downscaled_annotations)

        history = self.model.fit(
            x=X,
            y=Y_true,
            batch_size=self.mini_batch_size,
            epochs=10,
            verbose=1,
        )

        return history

    def print_results(self, image_path: str):
        box_coordinates, scores, classes = self.predict(image_path=image_path)
        box_coordinates = self._upscale_bboxes(box_coordinates)
        image = self.data_processor.load_single_image(image_path=image_path)
        y1, x1, y2, x2 = box_coordinates[0]

        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.axis('off')

        ax.imshow(image)

        for i in range(len(box_coordinates)):
            y1, x1, y2, x2 = box_coordinates[i]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label = visDrone.categories.get()
            ax.text(x1, y1, f'{label} {scores[i]:.2f}', color='white', fontsize=12,)

    def predict(self, image_path: str, iou_threshold: float = 0.5):
        img, image_downsampled = self.data_processor.prepare_single_image(image_path, self.input_shape)

        self.factor_H = img.shape[0] / self.input_shape[0]
        self.factor_W = img.shape[1] / self.input_shape[1]

        image_downsampled = keras.Input(shape=self.input_shape, batch_size=1)(image_downsampled)

        predictions = self.model.predict(image_downsampled)

        return self._bboxes_from_predictions(predictions, iou_threshold=iou_threshold)

    def _compile_model(self, factor_H: float, factor_W: float, grid_cell_size: tuple):
        anchors = self.data_processor.find_out_anchors(factor_H, factor_W, grid_cell_size, self.model.B)
        self.model.set_anchors(anchors)

        loss = YOLOLoss(
            S=self.model.S,
            B=self.model.B,
            C=self.model.C,
            lambda_coord=5.0,
            lambda_noobj=0.5,
            focal_gamma=2.0,
            focal_alpha=0.25,
        )

        optimizer = adadelta_v2.Adadelta(
            learning_rate=0.001,
            rho=0.95,
            epsilon=1e-7,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'],
        )

    def _bboxes_from_predictions(self, predictions: np.ndarray, iou_threshold: float = 0.5):
        """
        Convert the predictions to bounding boxes.

        Args:
            predictions (np.ndarray): matrices of shape (m, S * S * B, 6)
            iou_threshold (float): IoU threshold for non-max suppression

        Returns:
            bboxes (np.ndarray): bounding boxes of shape (m, 4) [y1, x1, y2, x2]
            scores (np.ndarray): scores of shape (m, 1)
            classes (np.ndarray): classes of shape (m, 1)
        """

        m = predictions.shape[0]
        B = self.model.B
        S = self.model.S

        cell_width = self.model_input_img_res[0] / S
        cell_height = self.model_input_img_res[1] / S

        boxes = np.zeros((m, S, S, B, 6))

        boxes_coordinates = np.zeros((m, 4))
        boxes_scores = np.zeros((m, 1))
        boxes_classes = np.zeros((m, 1))

        for i in range(m):
            for b in range(B):
                x_idx = b * 5
                y_idx = b * 5 + 1
                w_idx = b * 5 + 2
                h_idx = b * 5 + 3
                conf_idx = b * 5 + 4

                x = predictions[i, :, :, x_idx]
                y = predictions[i, :, :, y_idx]
                w = predictions[i, :, :, w_idx]
                h = predictions[i, :, :, h_idx]
                confidence = predictions[i, :, :, conf_idx]

                class_probs = predictions[i, :, :, b * 5:]
                class_ids = np.argmax(class_probs, axis=-1)
                score = np.max(class_probs, axis=3) * confidence

                x1 = (x + np.arange(S)) * cell_width
                y1 = (y + np.arange(S)) * cell_height

                x2 = x1 + w
                y2 = y1 + h

                boxes_coordinates[i, 0] = y1
                boxes_coordinates[i, 1] = x1
                boxes_coordinates[i, 2] = y2
                boxes_coordinates[i, 3] = x2

                boxes_scores[i, 0] = score
                boxes_classes[i, 0] = class_ids

        indices = non_max_suppression(boxes_coordinates, boxes_scores, iou_threshold=iou_threshold)
        boxes_coordinates = tf.gather(boxes_coordinates, indices)
        boxes_scores = tf.gather(boxes_scores, indices)
        boxes_classes = tf.gather(boxes_classes, indices)

        return boxes_coordinates, boxes_scores, boxes_classes

    def _upscale_bboxes(self, bboxes: np.ndarray):
        """
        Upscale the bounding boxes to the original image size.

        Args:
            bboxes (np.ndarray): bounding boxes of shape (m, 4) [y1, x1, y2, x2]

        Returns:
            np.ndarray: upscaled bounding boxes of shape (m, 4)
        """
        # TODO: this will only work for one image or if all images are of the same size (that's not the case with visDrone dataset)
        bboxes[:, 0] *= self.factor_H
        bboxes[:, 1] *= self.factor_W
        bboxes[:, 2] *= self.factor_H
        bboxes[:, 3] *= self.factor_W

        return bboxes
