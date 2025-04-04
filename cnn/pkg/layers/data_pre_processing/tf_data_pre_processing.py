import os

import numpy as np
from metadata import visDrone
import tensorflow as tf
from keras.api.ops import convert_to_numpy


class DataPreProcessor:
    def __init__(self, images_folder: str, annotations_folder: str, data_idx: int = 0):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.data_idx = data_idx

    def load_images_for_training(self, target_images_shape: tuple):
        """
        Load data for training
        Returns:
            tuple: original image, downscaled image
        """
        # TODO: currently only for single image
        image_files = os.listdir(self.images_folder)
        image_file = image_files[0]
        image_path = os.path.join(self.images_folder, image_file)

        image, downscaled_image = self.prepare_single_image(image_path=image_path, target_size=target_images_shape)

        self.factor_H = image.shape[0] / target_images_shape[0]
        self.factor_W = image.shape[1] / target_images_shape[1]
        self.downscale_shape = target_images_shape

        return image, downscaled_image

    def load_annotations_for_training(self, S: int, B: int, C: int, anchors: tf.Tensor, annotation_file_name: str):
        annotation_path = os.path.join(self.annotations_folder, annotation_file_name)
        annotations = _extract_visDrone_annotations(annotation_path)
        downscaled_annotations = _downscale_annotations(annotations, self.factor_H, self.factor_W)
        downscaled_annotations = _cook_annotations(np.array(downscaled_annotations), self.downscale_shape, S, B, C, anchors)
        
        return downscaled_annotations   

    def find_out_anchors(self, factor_H: float, factor_W: float, grid_cell_size: tuple, B: int):
        annotation_files = os.listdir(self.annotations_folder)
        boxes = []

        for annotation_file in annotation_files:
            annotations = _extract_visDrone_annotations(os.path.join(self.annotations_folder, annotation_file))
            if len(annotations) == 0:
                continue

            downscaled_annotations = _downscale_annotations(annotations, factor_H, factor_W)

            for annotation in downscaled_annotations:
                boxes.append([annotation[visDrone.width_idx] / grid_cell_size[1],
                              annotation[visDrone.height_idx] / grid_cell_size[0]])

        default_anchors = _kmeans(np.array(boxes), k=B, dist=np.median, max_iter=300)

        return default_anchors

    def load_single_image(self, image_path: str):
        return tf.io.decode_image(tf.io.read_file(image_path), channels=3)

    def prepare_single_image(self, image_path: str, target_size: tuple):
        orig_image = self.load_single_image(image_path=image_path)
        downscaled_image = tf.image.resize(orig_image, target_size, method='nearest', antialias=True)   
        downscaled_image = tf.cast(downscaled_image, tf.float32) / 255.0
        downscaled_image = tf.image.per_image_standardization(downscaled_image)
        downscaled_image = tf.convert_to_tensor(downscaled_image, dtype=tf.float32)

        return tf.convert_to_tensor(orig_image), tf.convert_to_tensor(downscaled_image)


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
        values = list(map(int, ''.join(c for c in line.strip().rstrip(',') if c.isdigit() or c == ',').split(',')))
        if values[visDrone.object_existence_idx] == 0 or values[visDrone.category_idx] > 10:
            continue

        annotations.append(values)

    return annotations


def _downscale_annotations(annotations: list, factor_H: float, factor_W: float):
    """
    Downscale the annotations to the input image size

    Args:
        annotations (list): list of annotations
        factor_H (float): height factor
        factor_W (float): width factor

    Returns:
        list: list of downscaled annotations of shape (8,)
    """
    downscaled_annotations = []

    for annotation in annotations:
        annotation[visDrone.top_left_x_idx] = int(annotation[visDrone.top_left_x_idx] / factor_W)
        annotation[visDrone.top_left_y_idx] = int(annotation[visDrone.top_left_y_idx] / factor_H)
        annotation[visDrone.width_idx] = int(annotation[visDrone.width_idx] / factor_W)
        annotation[visDrone.height_idx] = int(annotation[visDrone.height_idx] / factor_H)

        downscaled_annotations.append(annotation)

    return downscaled_annotations


def _kmeans(boxes, k, dist=np.median, max_iter=300):
    """
    Runs k-means clustering with an IoU-based distance metric.

    boxes: numpy array of shape (n, 2) where n is the number of boxes and each box is (width, height)
    k: number of clusters (anchors)
    dist: function to calculate the cluster center, here median is used
    max_iter: maximum iterations to run
    """
    num_boxes = boxes.shape[0]

    # initialize clusters by randomly choosing k boxes from the dataset.
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    last_clusters = np.zeros((num_boxes,))

    for _ in range(max_iter):
        distances = np.zeros((num_boxes, k))

        for i in range(num_boxes):
            distances[i] = 1 - _kmeans_iou(boxes[i], clusters)

        current_clusters = np.argmin(distances, axis=1)
        if (last_clusters == current_clusters).all():
            break

        for cluster in range(k):
            if np.sum(current_clusters == cluster) == 0:
                continue

            clusters[cluster] = dist(boxes[current_clusters == cluster], axis=0)

        last_clusters = current_clusters

    return clusters


def _kmeans_iou(box, clusters):
    """
    Calculate the Intersection over Union (IoU) between a box and k clusters.
    box: tuple or array, (width, height)
    clusters: numpy array of shape (k, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y

    box_area = box[0] * box[1]
    clusters_area = clusters[:, 0] * clusters[:, 1]

    res = intersection / (box_area + clusters_area - intersection)
    return res


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


def _cook_annotations(annotations: np.ndarray, model_input_img_shape: tuple, S: int, B: int, C: int, anchors: tf.Tensor):
    """
    convert list of annotations to the model output shape

    Args:
        annotations (np.ndarray): true values - matrix of shape (m, 8), represents a batch of m annotations
        original_img_shape (tuple): shape of the original image (width, height)
        model_input_img_shape (tuple): shape of the image that the model accepts (width, height)
        anchors (np.ndarray): normalized ([0, 1]) anchors for the model - matrix of shape (B, 2), represents the anchors for the model

    Returns:
        Y_target (np.ndarray): true values - matrix of shape (S, S, B*5+C), represents a batch of m images
    """
    m = annotations.shape[0]

    Y_norm = annotations.copy()

    # assign annotations to a grid cells
    X_center = (Y_norm[:, visDrone.top_left_x_idx] + Y_norm[:, visDrone.width_idx] / 2)
    Y_center = (Y_norm[:, visDrone.top_left_y_idx] + Y_norm[:, visDrone.height_idx] / 2)
    cell_height = model_input_img_shape[0] / S
    cell_width = model_input_img_shape[1] / S

    # assign bounding box to a cell
    cells_x = X_center // cell_width
    cells_y = Y_center // cell_height

    # calc offset of the x and y inside the cell where the center of the box is. Relative values [0, 1]
    X_cells_offset = (X_center - cells_x * cell_width) / cell_width  # shape (m,)
    Y_cells_offset = (Y_center - cells_y * cell_height) / cell_height  # shape (m,)

    # create one-hot encoding vector for the class
    class_targets = np.zeros((m, C))
    categories = Y_norm[:, visDrone.category_idx].astype(np.int8)
    class_targets[np.arange(m), categories] = 1
    # smooth one-hot encoding vector
    class_targets = (1 - 1e-6) * class_targets + 1e-6 / C

    # construct output matrix - aka reshape annotations to the model output shape
    Y_target = np.zeros((S, S, B * 5 + C)).astype(np.float32)

    for i in range(m):
        cell_x = int(cells_x[i])
        cell_y = int(cells_y[i])

        # bounding box target
        bbox_target = np.array([
            X_cells_offset[i],
            Y_cells_offset[i],
            Y_norm[i, visDrone.width_idx] / cell_width,
            Y_norm[i, visDrone.height_idx] / cell_height,
            1,
        ])

        true_center_offset = np.array([X_cells_offset[i], Y_cells_offset[i]])
        
        distances = np.linalg.norm(convert_to_numpy(anchors) - true_center_offset, axis=1)
        best_bbox_slot = np.argmin(distances)

        start_index = best_bbox_slot * 5
        Y_target[cell_y, cell_x, start_index:start_index+5] = bbox_target
        Y_target[cell_y, cell_x, B * 5:] = class_targets[i]

    return Y_target
