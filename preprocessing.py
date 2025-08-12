# -*- coding: utf-8 -*-

"""
Image preprocessing utilities
"""

import logging
import os
import sys
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
)

# Constants and configuration
CONFIG = {
    "image_size": (224, 224),
    "data_dir": "data",
    "log_dir": "logs",
    "debug": False,
}

class ImagePreprocessor:
    """
    Image preprocessing utilities
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image preprocessor

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _validate_image(self, image: np.ndarray) -> bool:
        """
        Validate an image

        Args:
            image (np.ndarray): Image array

        Returns:
            bool: Whether the image is valid
        """
        if image is None:
            self.logger.error("Image is None")
            return False
        if image.ndim != 3:
            self.logger.error("Image must have 3 dimensions")
            return False
        if image.shape[2] != 3:
            self.logger.error("Image must have 3 color channels")
            return False
        return True

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize an image

        Args:
            image (np.ndarray): Image array

        Returns:
            np.ndarray: Resized image array
        """
        if not self._validate_image(image):
            return None
        image = Image.fromarray(image)
        image = image.resize(self.config["image_size"])
        return np.array(image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image

        Args:
            image (np.ndarray): Image array

        Returns:
            np.ndarray: Normalized image array
        """
        if not self._validate_image(image):
            return None
        image = image / 255.0
        return image

    def _convert_image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert an image to a tensor

        Args:
            image (np.ndarray): Image array

        Returns:
            torch.Tensor: Image tensor
        """
        if not self._validate_image(image):
            return None
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess an image

        Args:
            image_path (str): Path to the image

        Returns:
            Optional[torch.Tensor]: Preprocessed image tensor
        """
        start_time = time.time()
        image = np.load(image_path)
        image = self._resize_image(image)
        image = self._normalize_image(image)
        image = self._convert_image_to_tensor(image)
        end_time = time.time()
        self.logger.info(f"Preprocessed image in {end_time - start_time:.2f} seconds")
        return image

class ImageDataset(Dataset):
    """
    Image dataset
    """

    def __init__(self, image_paths: List[str], config: Dict[str, Any]):
        """
        Initialize the image dataset

        Args:
            image_paths (List[str]): List of image paths
            config (Dict[str, Any]): Configuration dictionary
        """
        self.image_paths = image_paths
        self.config = config
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get an item from the dataset

        Args:
            index (int): Index of the item

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image_path = self.image_paths[index]
        image = self.config["preprocessor"].preprocess_image(image_path)
        return image

class Config:
    """
    Configuration
    """

    def __init__(self):
        """
        Initialize the configuration
        """
        self.image_size = (224, 224)
        self.data_dir = "data"
        self.log_dir = "logs"
        self.debug = False

def main():
    """
    Main function
    """
    config = Config()
    preprocessor = ImagePreprocessor(config.__dict__)
    dataset = ImageDataset(
        image_paths=["image1.npy", "image2.npy"],
        config=config.__dict__,
    )
    for image in dataset:
        print(image.shape)

if __name__ == "__main__":
    main()