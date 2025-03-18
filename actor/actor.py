import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import os
import matplotlib.pyplot as plt

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

        files = [os.path.join(self.data_folder, entry) for entry in entries]
        
        for file in files:
            img, img_resized = self._prepare_single_image(file)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(np.transpose(img_resized, (1, 2, 0)))
            axes[0].set_title("Resized Image")
            axes[1].imshow(img)
            axes[1].set_title("Original Image")
            plt.tight_layout()
            plt.show()
    
    def _prepare_single_image(self, img_path: str):
        img = Image.open(img_path)
        
        img_array = np.array(img)
        print(img_array.shape)
        img_resized = _downscale_img(img_array, self.img_res)
        print(img_resized.shape)
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
    image_cropped = image_cropped.transpose(2, 0, 1) # reshape (height, width, num_channels) -> (num_channels, height, width)
    reshaped = image_cropped.reshape(image.shape[2], new_height, factor_H, new_width, factor_W)
    
    return reshaped.mean(axis=(2, 4)).astype(np.uint8)
    