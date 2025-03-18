import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from metadata import visDrone


class Model(ABC):
    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, input_data):
        pass


class YOLOActorPhoto():
    def __init__(self,
                 data_folder: str,
                 annotation_folder: str,
                 img_res: tuple,
                 tinyYOLO_model: Model,
                 mini_batch_size: int
                 ):

        self.data_folder = data_folder
        self.annotation_folder = annotation_folder
        self.img_res = img_res
        self.tinyYOLO_model = tinyYOLO_model
        self.mini_batch_size = mini_batch_size

    def load_data(self):
        entries = os.listdir(self.data_folder)
        img_files = {}
        annotation_files = {}
        for entry in entries:
            file_name = entry.split('.')[0]
            img_files[file_name] = os.path.join(self.data_folder, entry)
            annotation_files[file_name] = os.path.join(self.annotation_folder, file_name + '.txt')

        for file_name, img_path in img_files.items():
            img, img_resized = self._prepare_single_image(img_path)
            img_resized = img_resized.transpose(1, 2, 0)  # reshape (num_channels, height, width) -> (height, width, num_channels)
            annotations = _extract_visDrone_annotations(annotation_files[file_name])
            annotations_resized = _downscale_annotation(annotations, img.shape[0] // img_resized.shape[0], img.shape[1] // img_resized.shape[1])

            fig, ax = plt.subplots(1)
            ax.imshow(img_resized)
            for i in range(len(annotations)):
                x, y, w, h, _, c, _, _ = annotations_resized[i]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                label = visDrone.categories.get(c)
                ax.text(x, y - 20, label, color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            plt.axis('off')
            plt.show()

    def _prepare_single_image(self, img_path: str):
        img = Image.open(img_path)

        img_array = np.array(img)
        img_resized = _downscale_img(img_array, self.img_res)
        return (img_array, img_resized)


def _downscale_img(image: np.ndarray, new_shape: tuple):
    """
    Downscale the image to the new shape

    Args:
        image (np.ndarray): matrix of shape (height, width, num_channels), represents an image
        new_shape (np.ndarray): tuple (new_height, new_width), represents the new shape of the image

    Returns:
        np.ndarray: downsampled image of shape(num_channels, new_height, new_width)
    """

    (new_height, new_width) = new_shape

    factor_H = image.shape[0] // new_height
    factor_W = image.shape[1] // new_width

    image_cropped = image[:new_height * factor_H, :new_width * factor_W, :]
    image_cropped = image_cropped.transpose(2, 0, 1)  # reshape (height, width, num_channels) -> (num_channels, height, width)
    reshaped = image_cropped.reshape(image.shape[2], new_height, factor_H, new_width, factor_W)

    return reshaped.mean(axis=(2, 4)).astype(np.uint8)


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
