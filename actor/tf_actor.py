from operator import index
import numpy as np
import tensorflow as tf
from cnn.pkg.layers.data_pre_processing.tf_data_pre_processing import DataPreProcessor
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from cnn.pkg.layers.losses.tf_loss import YOLOLoss
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from keras.api.optimizers import Adadelta
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

    def train_model(self, loops: int = 1):
        orig_image, downscaled_image = self.data_processor.load_images_for_training(self.input_shape)

        epochs = loops * 1  # since it's a single image

        factor_H = orig_image.shape[0] / self.input_shape[0]
        factor_W = orig_image.shape[1] / self.input_shape[1]
        grid_cell_size = (self.input_shape[0] / self.model.S, self.input_shape[1] / self.model.S)
        self._compile_model(factor_H, factor_W, grid_cell_size)

        X = tf.expand_dims(downscaled_image, axis=0)  # Add batch dimension (1, ...)

        downscaled_annotations = self.data_processor.load_annotations_for_training(
            S=self.model.S,
            B=self.model.B,
            C=self.model.C,
            anchors=self.model.get_anchors(),
            annotation_file_name='0000006_00159_d_0000001.txt'
        )

        Y_true = tf.expand_dims(downscaled_annotations, axis=0)  # add batch dimension (1, ...)

        history = self.model.fit(
            x=X,
            y=Y_true,
            batch_size=self.mini_batch_size,
            epochs=epochs,
            verbose='1',
        )

        return history

    def print_results(self, image_path: str):
        box_coordinates, scores, classes = self.predict(image_path=image_path)
        box_coordinates = tf.convert_to_tensor(box_coordinates, dtype=tf.float32)
        scores = tf.convert_to_tensor(scores, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.int64)

        if box_coordinates.shape[0] == 0:
            print("No objects detected.")
            return

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
            label = visDrone.categories.get(int(classes[i].numpy()), "Unknown")
            ax.text(x1, y1, f'{label} {scores[i]:.2f}', color='white', fontsize=12,)

        plt.show()

    def predict(self, image_path: str, iou_threshold: float = 0.5):
        """
        predict for the single image
        """

        img, image_downsampled = self.data_processor.prepare_single_image(image_path, self.input_shape)

        self.factor_H = img.shape[0] / self.input_shape[0]
        self.factor_W = img.shape[1] / self.input_shape[1]

        image_downsampled = tf.expand_dims(image_downsampled, axis=0)  # add batch dimension (1, H, W, C)

        predictions = self.model.predict(image_downsampled)

        return self._bboxes_from_predictions(predictions, iou_threshold=iou_threshold)
        

    def plot_training_history(self, history):
        # Get all metrics from the history object
        metrics = history.history
        epochs_range = range(1, len(metrics['loss']) + 1)
        
        # Create a figure with subplots - one for loss, one for other metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs_range, metrics['loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics:
            ax1.plot(epochs_range, metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot other metrics if they exist (accuracy, mae, etc.)
        for metric in metrics:
            if metric != 'loss' and metric != 'val_loss':
                ax2.plot(epochs_range, metrics[metric], label=f'Training {metric}')
                # Check if there's a validation version of this metric
                val_metric = f'val_{metric}'
                if val_metric in metrics:
                    ax2.plot(epochs_range, metrics[val_metric], '--', label=f'Validation {metric}')
        
        ax2.set_title('Training and Validation Metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _compile_model(self, factor_H: float, factor_W: float, grid_cell_size: tuple):
        anchors = self.data_processor.find_out_anchors(factor_H, factor_W, grid_cell_size, self.model.B)
        self.model.set_anchors(tf.convert_to_tensor(anchors, dtype=tf.float32))

        loss = YOLOLoss(
            S=self.model.S,
            B=self.model.B,
            C=self.model.C,
            lambda_coord=5.0,
            lambda_noobj=0.5,
            focal_gamma=2.0,
            focal_alpha=0.25,
        )

        # optimizer = Adadelta(
        #     learning_rate=0.001,
        #     rho=0.95,
        #     epsilon=1e-7,
        # )

        self.model.compile(
            optimizer='adadelta',
            loss=loss,
            metrics=['accuracy'],
        )

    def _bboxes_from_predictions(self, predictions, iou_threshold=0.5, score_threshold=0.1):
        """
        Convert the predictions to bounding boxes.

        Args:
            predictions: tensor of shape (batch_size, S, S, B*5+C)
            iou_threshold: IoU threshold for non-max suppression

        Returns:
            boxes_coordinates: tensor of shape (num_boxes, 4) [y1, x1, y2, x2]
            boxes_scores: tensor of shape (num_boxes,)
            boxes_classes: tensor of shape (num_boxes,)
        """

        m = predictions.shape[0]
        S = self.model.S
        B = self.model.B

        cell_width = self.input_shape[1] / S
        cell_height = self.input_shape[0] / S

        all_boxes = []
        all_scores = []
        all_classes = []

        # Process each item in the batch
        for i in range(m):
            batch_pred = predictions[i]  # Shape: (S, S, B*5+C)

            # For each bounding box predictor
            for box_idx in range(B):
                start_idx = box_idx * 5

                # Extract predictions for this box
                x = batch_pred[:, :, start_idx]      # Already contains grid offset, normalized [0-1]
                y = batch_pred[:, :, start_idx + 1]  # Already contains grid offset, normalized [0-1]
                w = batch_pred[:, :, start_idx + 2]  # Width (normalized)
                h = batch_pred[:, :, start_idx + 3]  # Height (normalized)
                confidence = batch_pred[:, :, start_idx + 4]  # Confidence score

                # absolute coordinates
                abs_x = x * self.input_shape[1] 
                abs_y = y * self.input_shape[0]  
                abs_w = w * self.input_shape[1]
                abs_h = h * self.input_shape[0]

                # corner format [y1, x1, y2, x2]
                y1 = abs_y - abs_h / 2
                x1 = abs_x - abs_w / 2
                y2 = abs_y + abs_h / 2
                x2 = abs_x + abs_w / 2

                # class predictions
                class_scores = batch_pred[:, :, B*5:]  # Shape: (S, S, C)
                class_idx = tf.argmax(class_scores, axis=-1)  # Shape: (S, S)
                max_class_score = tf.reduce_max(class_scores, axis=-1)  # Shape: (S, S)

                # Final score is confidence * class score
                final_scores = confidence * max_class_score

                # Reshape everything to [S*S, ...]
                boxes = tf.stack([y1, x1, y2, x2], axis=-1)
                boxes = tf.reshape(boxes, [-1, 4])
                final_scores = tf.reshape(final_scores, [-1])
                class_idx = tf.reshape(class_idx, [-1])

                # Filter out low confidence boxes
                conf_mask = final_scores > score_threshold
                filtered_boxes = tf.boolean_mask(boxes, conf_mask)
                filtered_scores = tf.boolean_mask(final_scores, conf_mask)
                filtered_classes = tf.boolean_mask(class_idx, conf_mask)

                # Apply NMS
                selected_indices = tf.image.non_max_suppression(
                    filtered_boxes, filtered_scores, max_output_size=100,
                    iou_threshold=iou_threshold
                )

                selected_boxes = tf.gather(filtered_boxes, selected_indices)
                selected_scores = tf.gather(filtered_scores, selected_indices)
                selected_classes = tf.gather(filtered_classes, selected_indices)

                all_boxes.append(selected_boxes)
                all_scores.append(selected_scores)
                all_classes.append(selected_classes)

        # combine results from all batches
        if all_boxes:
            boxes_coordinates = tf.concat(all_boxes, axis=0)
            boxes_scores = tf.concat(all_scores, axis=0)
            boxes_classes = tf.concat(all_classes, axis=0)
            return boxes_coordinates, boxes_scores, boxes_classes
        else:
            # empty tensors if no boxes were found
            return tf.zeros((0, 4)), tf.zeros((0,)), tf.zeros((0,), dtype=tf.int64)

    def _upscale_bboxes(self, bboxes: tf.Tensor):
        """
        Upscale the bounding boxes to the original image size.

        Args:
            bboxes (tf.Tensor): bounding boxes of shape (m, 4) [y1, x1, y2, x2]

        Returns:
            tf.Tensor: upscaled bounding boxes of shape (m, 4)
        """
        # TODO: this will only work for one image or if all images are of the same size (that's not the case with visDrone dataset)
        scaling_factors = tf.constant([self.factor_H, self.factor_W, self.factor_H, self.factor_W], dtype=bboxes.dtype)
        bboxes = tf.multiply(bboxes, scaling_factors)

        return bboxes
