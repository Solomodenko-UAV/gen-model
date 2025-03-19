import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from metadata import visDrone
from nn.pkg.models.tinysimmoYOLO import TinysimmoYOLOModel


class YOLOActorPhoto():
    def __init__(self,
                 data_folder: str,
                 annotation_folder: str,
                 model_input_img_res: tuple,
                 model: TinysimmoYOLOModel,
                 mini_batch_size: int
                 ):

        self.data_folder = data_folder
        self.annotation_folder = annotation_folder
        self.model_input_img_res = model_input_img_res
        self.model = model
        self.mini_batch_size = mini_batch_size

    def load_data(self):
        entries = os.listdir(self.data_folder)
        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        # just show annotations and boxes on the image
        # for file_name, img_path in img_files.items():
        #     img, img_resized = self._prepare_single_image(img_path)
        #     annotations = _extract_visDrone_annotations(annotation_files[file_name])
        #     annotations_resized = _downscale_annotation(annotations, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

        #     fig, ax = plt.subplots(1)
        #     ax.imshow(img_resized)
        #     for i in range(len(annotations)):
        #         x, y, w, h, _, c, _, _ = annotations_resized[i]
        #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)
        #         label = visDrone.categories.get(c)
        #         ax.text(x, y - 20, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        #     plt.axis('off')
        #     plt.show()
            for file_name, img_path in img_files.items():
                img, img_resized = self._prepare_single_image(img_path)
                images = img_resized.reshape(1, *img_resized.shape)
                predicted_boxes = self.predict(images)

                boxes_coordinates = predicted_boxes.reshape(predicted_boxes.shape[0], -1, predicted_boxes.shape[-1])

                fig, ax = plt.subplots(1)
                ax.imshow(img)
                for i in range(len(predicted_boxes)):
                    boxes_per_image = boxes_coordinates[i]
                    for box in boxes_per_image:
                        x, y, w, h, obj_class = box
                        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                # plt.axis('off')
                plt.show()

    def calc_loss(self, A: np.ndarray, Y: np.ndarray, original_img_shape: tuple, model_input_img_shape: tuple):
        """
        Calculate the loss
        Args:
            A (np.ndarray): predicted values - matrix of shape (S, S, B*5+C), represents a batch of m images
            Y (np.ndarray): true values - matrix of shape (m, 8), represents a batch of m annotations
            original_img_shape (tuple): shape of the original image (height, width)
            model_input_img_shape (tuple): shape of the image that the model accepts (height, width)
        """

        Y_target = self._cook_annotations(Y, original_img_shape, model_input_img_shape)

    def _cook_annotations(self, Y: np.ndarray, original_img_shape: tuple, model_input_img_shape: tuple):
        """
        convert list of annotations to the model output shape

        Args:
            Y (np.ndarray): true values - matrix of shape (m, 8), represents a batch of m annotations
            original_img_shape (tuple): shape of the original image (height, width)
            model_input_img_shape (tuple): shape of the image that the model accepts (height, width)
        """
        m = Y.shape[0]
        W = original_img_shape[0] / model_input_img_shape[0]
        H = original_img_shape[1] / model_input_img_shape[1]

        Y_norm = Y.copy()
        Y_norm[:, visDrone.top_left_x_idx] /= W
        Y_norm[:, visDrone.top_left_y_idx] /= H
        Y_norm[:, visDrone.width_idx] /= W
        Y_norm[:, visDrone.height_idx] /= H

        # assign annotations to a grid cells
        X_center = (Y_norm[visDrone.top_left_x_idx] + Y_norm[visDrone.width_idx] / 2)
        Y_center = (Y_norm[visDrone.top_left_y_idx] + Y_norm[visDrone.height_idx] / 2)
        cell_height = model_input_img_shape[0] / self.model.S
        cell_width = model_input_img_shape[1] / self.model.S

        # assign bounding box to a cell
        cells_x = X_center // cell_width
        cells_y = Y_center // cell_height

        # calc offset of the x and y inside the cell where the center of the box is. Relative values [0, 1]
        X_cells_offset = (X_center - cells_x * cell_width) / cell_width
        Y_cells_offset = (Y_center - cells_y * cell_height) / cell_height

        # create one-hot encoding vector for the class
        class_targets = np.zeros((m, self.model.C))
        class_targets[Y_norm[:, visDrone.class_idx]] = 1

        # construct output matrix - aka reshape annotations to the model output shape
        Y_target = np.zeros((self.model.S, self.model.S, self.model.B * 5 + self.model.C))

        for i in range(m):
            cell_x = int(cells_x[i])
            cell_y = int(cells_y[i])

            # bounding box target
            bbox_target = [
                X_cells_offset[i],
                Y_cells_offset[i],
                Y_norm[i, visDrone.width_idx],
                Y_norm[i, visDrone.height_idx],
                1,
            ]

            # TODO: for now it's just 1st box, but I should choose the best somehow (IoU based?)
            bbox_slot = 0
            start_index = bbox_slot * 5
            Y_target[cell_y, cell_x, start_index:start_index+5] = bbox_target
            Y_target[cell_y, cell_x, self.model.B * 5:] = class_targets[i]

        return Y_target

    def _calc_loss_and_gradient_single_image(self, A: np.ndarray, Y_target: np.ndarray):
        """
        Calculate the loss and gradient

        Args:
            A (np.ndarray): predicted values - matrix of shape (m, S, S, B*5+C). For this function m == 1 is required
            Y_target (np.ndarray): true values - matrix of shape (S, S, B*5+C)
        """
        loss = 0.0
        grad_A = np.zeros_like(A)

        # Constants for loss weighting (set these as needed)
        lambda_coord = 5.0
        lambda_noobj = 0.5

        m, S, _, total = A.shape
        B = self.model.B
        C = self.model.C

        for i in range(m):
            for row in range(S):
                for col in range(S):
                    prediction = A[i, row, col]
                    target = Y_target[row, col]

                    # for each bounding box predictor in this cell
                    for b in range(B):
                        idx = b * 5
                        pred_bbox = prediction[idx:idx+5]  # [x, y, w, h, confidence]
                        target_bbox = target[idx:idx+5]

                        # if model thinks there is no object in this box
                        if target_bbox[visDrone.object_existence_idx] == 0:
                            conf_diff = pred_bbox[visDrone.object_existence_idx]
                            loss += lambda_noobj * (conf_diff ** 2)
                            grad_A[i, row, col, idx + visDrone.object_existence_idx] = 2 * lambda_noobj * conf_diff
                            continue
                        
                        # if model thinks there is an object in this box
                        # localization loss for x, y
                        for j in range(2):
                            diff = pred_bbox[j] - target_bbox[j]
                            loss += lambda_coord * (diff ** 2)
                            grad_A[i, row, col, idx + j] = 2 * lambda_coord * diff

    def predict(self, images: np.ndarray):
        """
        Predict the bounding boxes for the images

        Args:
            images (np.ndarray): images to predict the bounding boxes for - matrix of shape (m, height, width, num_channels), represents a batch of m images

        Returns:
            np.ndarray: predicted bounding boxes - matrix of shape (m, S, S, B, 5), represents a batch of m images. Each box is [x, y, w, h, class]
        """
        model_output = self.model.forward(images)

        m = model_output.shape[0]
        B = self.model.B
        S = self.model.S
        boxes_pred = model_output[..., :B * 5]  # shape (m, S, S, B*5)
        boxes_pred = boxes_pred.reshape(m, S, S, B, 5)

        classes_probs = model_output[..., B * 5:]  # shape (m, S, S, C)
        predicted_class_indices = np.argmax(classes_probs, axis=-1)  # shape (m, S, S)
        predicted_class_probabilities = np.max(classes_probs, axis=-1)  # shape (m, S, S)

        boxes = np.zeros((m, S, S, B, 5))

        cell_width = self.model_input_img_res[0] // S
        cell_height = self.model_input_img_res[1] // S

        for i in range(boxes_pred.shape[0]):
            for row in range(boxes_pred.shape[1]):
                for col in range(boxes_pred.shape[2]):
                    for b in range(B):
                        x, y, w, h, confidence = boxes_pred[i, row, col, b, :]

                        x_center_abs = (row + x) * cell_width
                        y_center_abs = (i + y) * cell_height

                        w_abs = w * self.model_input_img_res[0]
                        h_abs = h * self.model_input_img_res[1]

                        x1 = x_center_abs - w_abs / 2
                        y1 = y_center_abs - h_abs / 2

                        obj_class_probability = predicted_class_probabilities[i, row, col]
                        obj_class = predicted_class_indices[i, row, col]

                        boxes[i, row, col, b] = [x1, y1, w_abs, h_abs, obj_class]

        return boxes

    def _prepare_single_image(self, img_path: str):
        """
        Prepare the image for the model

        Args:
            img_path (str): path to the image

        Returns:
            tuple: tuple containing the original image and the resized image
        """

        img = Image.open(img_path)

        img_array = np.array(img)
        img_resized = _downscale_img(img_array, self.model_input_img_res)
        return (img_array, img_resized)


def _downscale_img(image: np.ndarray, new_shape: tuple):
    """
    Downscale the image to the new shape

    Args:
        image (np.ndarray): matrix of shape (height, width, num_channels), represents an image
        new_shape (np.ndarray): tuple (new_height, new_width), represents the new shape of the image

    Returns:
        np.ndarray: downsampled image of shape(new_height, new_width, num_channels)
    """

    (new_height, new_width) = new_shape

    factor_H = image.shape[0] // new_height
    factor_W = image.shape[1] // new_width

    image_cropped = image[:new_height * factor_H, :new_width * factor_W, :]
    reshaped = image_cropped.reshape(new_height, factor_H, new_width, factor_W, image.shape[2])

    return reshaped.mean(axis=(1, 3)).astype(np.uint8)


def _downscale_annotation(annotations: list, factor_H: int, factor_W: int):
    """
    Downscale the annotation to the new shape

    Args:
        annotation (list): list of annotations
        new_shape (np.ndarray): tuple (new_height, new_width), represents the new shape of the image

    Returns:
        list: downsampled annotation
    """

    annotation_downscaled = [list(annotation) for annotation in annotations]
    for i in range(len(annotations)):
        annotation_downscaled[i][visDrone.top_left_x_idx] //= factor_W
        annotation_downscaled[i][visDrone.top_left_y_idx] //= factor_H
        annotation_downscaled[i][visDrone.width_idx] //= factor_W
        annotation_downscaled[i][visDrone.height_idx] //= factor_H

    return annotation_downscaled


def _extract_visDrone_annotations(file_path: str):
    """
    Extract annotations from the file

    Returns:
        list: list of dictionaries, each dictionary contains the annotation for a
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []

    for line in lines:
        values = list(map(int, line.split(',')))
        annotations.append(values)

    return annotations
