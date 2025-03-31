import os

import numpy as np
from metadata import visDrone
import tensorflow as tf


class DataPreProcessor:
    def __init__(self, images_folder: str, annotations_folder: str, data_idx: int = 0):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.data_idx = data_idx

    def load_data_for_training(self, target_images_shape: tuple):
        """
        Load data for training
        Returns:
            tuple: original image, downscaled images and downscaled annotations
        """
        # TODO: currently only for single image
        image_files = os.listdir(self.images_folder)
        image_file = image_files[0]
        image_path = os.path.join(self.images_folder, image_file)
        annotation_path = os.path.join(self.annotations_folder, image_file) + ".txt"

        image, downscaled_image = self.prepare_single_image(image_path=image_path, target_size=target_images_shape)
        factor_H = image.shape[0] / target_images_shape[0]
        factor_W = image.shape[1] / target_images_shape[1]

        annotations = _extract_visDrone_annotations(annotation_path)
        downscaled_annotations = _downscale_annotations(annotations, factor_H, factor_W)
        downscaled_annotations = np.array(downscaled_annotations, dtype=np.float32)

        return image, downscaled_image, downscaled_annotations

    def find_out_anchors(self, factor_H: float, factor_W: float, grid_cell_size: tuple, B: int):
        annotation_files = os.listdir(self.annotations_folder)
        boxes = [[] for _ in annotation_files]

        for annotation_file in annotation_files:
            annotations = _extract_visDrone_annotations(os.path.join(self.annotation_folder, annotation_file) + ".txt")
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

        downscaled_image = _downscale_img(orig_image.numpy(), target_size)

        mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
        std = np.array([0.229, 0.224, 0.225])   # ImageNet std
        downscaled_image = (np.array(downscaled_image) / 255.0 - mean) / std

        downscaled_image = tf.Tensor(downscaled_image, dtype=tf.float32)

        return orig_image, downscaled_image


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
        list: list of downscaled annotations
    """
    downscaled_annotations = []

    for annotation in annotations:
        x1 = int(annotation[visDrone.x1_idx] / factor_W)
        y1 = int(annotation[visDrone.y1_idx] / factor_H)
        x2 = int(annotation[visDrone.x2_idx] / factor_W)
        y2 = int(annotation[visDrone.y2_idx] / factor_H)

        downscaled_annotations.append([x1, y1, x2, y2])

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
