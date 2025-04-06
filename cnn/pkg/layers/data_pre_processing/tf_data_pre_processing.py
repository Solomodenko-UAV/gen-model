import os
import re

import numpy as np
from metadata import visDrone
import tensorflow as tf
from keras.api.ops import convert_to_numpy


class DataPreProcessor:
    def __init__(self, images_folder: str, annotations_folder: str, data_idx: int = 0):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder

        self.orig_image_shape = {}
        self.factors_H = {}
        self.factors_W = {}

    def create_dataset(self, target_images_shape: tuple, S: int, B: int, C: int, anchors: tf.Tensor, batch_size: int = 32):
        """
        Create a dataset for training

        Args:
            target_images_shape (tuple): shape of the image that the model accepts (height, width)
            S (int): number of grid cells in the image
            B (int): number of bounding boxes per cell
            C (int): number of classes
            anchors (tf.Tensor): normalized ([0, 1]) anchors for the model - matrix of shape (B, 2), represents the anchors for the model
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            tf.data.Dataset: dataset for training
        """
        def process_image_and_annotation(image_path: str, annotation_path: str):
            """
            Process image and annotation

            Args:
                image_path (str): path to the image
                annotation_path (str): path to the annotation file
                target_size (tuple): target size of the image

            Returns:
                tuple: processed image and annotation
            """
            img, downscaled_image = self.prepare_single_image(image_path=image_path, target_size=target_images_shape)

            factor_H = img.shape[0] / target_images_shape[0]
            factor_W = img.shape[1] / target_images_shape[1]

            annotations_list = extract_visDrone_annotations_from_file(annotation_path)

            cooked_annotations = []
            for annotations in annotations_list:
                downscaled_annotations = _downscale_annotations(annotations, factor_H, factor_W)
                downscaled_annotations = _cook_annotations(np.array(downscaled_annotations), self.downscale_shape, S, B, C, anchors)
                cooked_annotations.append(downscaled_annotations)

            return downscaled_image, np.array(cooked_annotations)

        image_files = tf.data.Dataset.list_files(os.path.join(self.images_folder, '*.jpg'))

        dataset = image_files.map(
            lambda x: tf.py_function(
                func=process_image_and_annotation,
                inp=[x, tf.strings.regex_replace(x, '.jpg', '.txt')],
                Tout=[tf.float32, tf.float32]),

            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        
        return dataset

    def load_images_batch_for_training(self, target_images_shape: tuple, start_idx: int = 0, end_index: int = -1):
        """
        Load data for training
        Returns:
            list of images (tf.Tensor): batch of images - matrix of shape (m, height, width, num_channels), represents a batch of m images
        """

        downscaled_images = self.prepare_multiple_images(self.images_folder, target_images_shape, start_idx, end_index)
        self.downscale_shape = target_images_shape

        return tf.convert_to_tensor(downscaled_images, dtype=tf.float32)

    def load_annotations_for_training(self, S: int, B: int, C: int, anchors: tf.Tensor):
        """
        Load annotations for training

        Raises:
            ValueError: _description_

        Returns:
            cooked_annotations (np.ndarray): annotations for the model - matrix of shape (m, S, S, B*5+C), represents a batch of m images
        """
        annotations_list = self.extract_visDrone_annotations()

        if len(annotations_list) != len(self.factors_H):
            raise ValueError("Number of annotations does not match the number of images")

        cooked_annotations = []
        for idx, annotations in enumerate(annotations_list):
            downscaled_annotations = _downscale_annotations(annotations, self.factors_H[idx], self.factors_W[idx])
            downscaled_annotations = _cook_annotations(np.array(downscaled_annotations), self.downscale_shape, S, B, C, anchors)
            cooked_annotations.append(downscaled_annotations)

        return np.array(cooked_annotations)

    def find_out_anchors(self, grid_cell_size: tuple, B: int):
        tf.print("Finding out anchors...")

        annotations_list = self.extract_visDrone_annotations()

        if len(annotations_list) != len(self.factors_H):
            raise ValueError("Number of annotations does not match the number of images")

        boxes = []
        for idx, annotations in enumerate(annotations_list):
            downscaled_annotations = _downscale_annotations(annotations, self.factors_H[idx], self.factors_W[idx])
            for annotation in downscaled_annotations:
                boxes.append([annotation[visDrone.width_idx] / grid_cell_size[1],
                              annotation[visDrone.height_idx] / grid_cell_size[0]])

        default_anchors = _kmeans(np.array(boxes), k=B, dist=np.median, max_iter=300)

        return default_anchors

    def read_multiple_images(self, images_dirs: list, start_idx: int = 0, end_idx: int = -1):
        images = []

        counter = 0
        for image_dir in images_dirs:
            for image_file in os.listdir(image_dir):
                if counter < start_idx:
                    counter += 1
                    continue

                if end_idx != -1 and counter >= end_idx:
                    break

                image_full_path = os.path.join(image_dir, image_file)

                if os.path.isfile(image_full_path):
                    img = tf.io.decode_image(tf.io.read_file(image_full_path), channels=3)
                    images.append(img)

                counter += 1

        return images

    def prepare_multiple_images(self, images_dir: str, target_size: tuple, start_idx: int = 0, end_idx: int = -1):
        images = self.read_multiple_images(images_dirs=[images_dir], start_idx=start_idx, end_idx=end_idx)

        downscaled_images = []
        for idx, image in enumerate(images):
            downscaled_image = tf.image.resize(image, target_size, method='nearest')
            downscaled_image = tf.image.convert_image_dtype(downscaled_image, dtype=tf.float32)
            downscaled_image = tf.image.per_image_standardization(downscaled_image)
            downscaled_images.append(tf.convert_to_tensor(downscaled_image, dtype=tf.float32))

            factor_H = image.shape[0] / target_size[0]
            factor_W = image.shape[1] / target_size[1]

            self.factors_H[idx] = factor_H
            self.factors_W[idx] = factor_W
            self.orig_image_shape[idx] = image.shape

        return downscaled_images

    def load_single_image(self, image_path: str):
        return tf.io.decode_image(tf.io.read_file(image_path), channels=3)

    def prepare_single_image(self, image_path: str, target_size: tuple):
        orig_image = self.load_single_image(image_path=image_path)
        downscaled_image = tf.image.resize(orig_image, target_size, method='nearest')
        downscaled_image = tf.image.convert_image_dtype(downscaled_image, dtype=tf.float32)
        downscaled_image = tf.image.per_image_standardization(downscaled_image)
        downscaled_image = tf.convert_to_tensor(downscaled_image, dtype=tf.float32)

        return tf.convert_to_tensor(orig_image, dtype=tf.uint8), tf.convert_to_tensor(downscaled_image, dtype=tf.float32)

    def extract_visDrone_annotations(self):
        """
        Extract annotations from the predefined folder

        Returns:
            list: list of dictionaries, each dictionary contains the annotation for a
        """
        annotation_files = os.listdir(self.annotations_folder)
        annotations_list = []

        for annotation_file in annotation_files:
            annotations = extract_visDrone_annotations_from_file(os.path.join(self.annotations_folder, annotation_file))
            annotations_list.append(annotations)

        return annotations_list


def extract_visDrone_annotations_from_file(file_path: str):
    """
    Extract annotations from the file

    Returns:
        list: list of dictionaries, each dictionary contains the annotation for a
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []

    for line in lines:
        values = list(map(float, re.findall(r'[-+]?\d*\.?\d+', line.strip())))
        if values[visDrone.object_existence_idx] == 0 or values[visDrone.category_idx] > 10:
            continue

        values[visDrone.category_idx] -= 1  # convert to 0-based index
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
        new_ann = annotation.copy()

        new_ann[visDrone.top_left_x_idx] = annotation[visDrone.top_left_x_idx] / factor_W
        new_ann[visDrone.top_left_y_idx] = annotation[visDrone.top_left_y_idx] / factor_H
        new_ann[visDrone.width_idx] = annotation[visDrone.width_idx] / factor_W
        new_ann[visDrone.height_idx] = annotation[visDrone.height_idx] / factor_H

        downscaled_annotations.append(new_ann)

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
    # class_targets = (1 - 1e-6) * class_targets + 1e-6 / C

    # construct output matrix - aka reshape annotations to the model output shape
    Y_target = np.zeros((S, S, B * 5 + C)).astype(np.float32)
    used_anchors = np.zeros((S, S, B), dtype=bool)

    for i in range(m):
        cell_x = int(cells_x[i])
        cell_y = int(cells_y[i])

        gt_w = Y_norm[i, visDrone.width_idx] / cell_width
        gt_h = Y_norm[i, visDrone.height_idx] / cell_height

        iou_scores = _kmeans_iou(np.array([gt_w, gt_h]), convert_to_numpy(anchors))

        sorted_anchors = np.argsort(iou_scores)[::-1]
        for anchor_idx in sorted_anchors:
            if not used_anchors[cell_y, cell_x, anchor_idx]:
                used_anchors[cell_y, cell_x, anchor_idx] = True
                start_index = anchor_idx * 5

                gt_w_transformed = np.log(gt_w / anchors[anchor_idx][0] + 1e-10)
                gt_h_transformed = np.log(gt_h / anchors[anchor_idx][1] + 1e-10)

                # bounding box target
                bbox_target = np.array([
                    X_cells_offset[i],
                    Y_cells_offset[i],
                    gt_w_transformed,
                    gt_h_transformed,
                    1,
                ])

                Y_target[cell_y, cell_x, start_index:start_index+5] = bbox_target
                Y_target[cell_y, cell_x, B*5:] = class_targets[i]
                break
        # if no anchors left, the object is skipped

    return Y_target
