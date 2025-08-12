import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extraction layer for computer vision tasks.
    """
    def __init__(self, config: Dict):
        """
        Initializes the FeatureExtractor with configuration settings.

        Parameters:
            config (Dict): Dictionary containing feature extractor settings.
                        Example:
                            {
                                "input_size": (224, 224),
                                "output_size": (1024,),
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                                "random_seed": 42
                            }
        """
        self.config = config
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.mean = torch.as_tensor(config["mean"], dtype=torch.float32)
        self.std = torch.as_tensor(config["std"], dtype=torch.float32)
        self.random_seed = config["random_seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """
        Builds the feature extractor model architecture.
        """
        try:
            # Example model architecture
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                torch.nn.Flatten(),
                torch.nn.Linear(4096, self.output_size[0])
            )
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise e

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from a batch of images.

        Parameters:
            images (torch.Tensor): Batch of images of shape (N, C, H, W)

        Returns:
            torch.Tensor: Extracted features of shape (N, output_size)
        """
        try:
            # Normalize images
            images = self._normalize(images)

            # Move data to device
            images = images.to(self.device)

            # Extract features using the model
            features = self.model(images)

            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise e

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a batch of images.

        Parameters:
            images (torch.Tensor): Batch of images of shape (N, C, H, W)

        Returns:
            torch.Tensor: Normalized images
        """
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return (images - mean) / std

    def load_pretrained(self, model_path: str):
        """
        Loads pretrained weights into the feature extractor model.

        Parameters:
            model_path (str): Path to the pretrained model weights.
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Pretrained weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            raise e

    def save_model(self, model_path: str):
        """
        Saves the feature extractor model to a file.

        Parameters:
            model_path (str): Path to save the model weights.
        """
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e

class FeatureExtractionException(Exception):
    """
    Custom exception class for feature extraction errors.
    """
    pass

# Example usage
if __name__ == "__main__":
    config = {
        "input_size": (224, 224),
        "output_size": (1024,),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "random_seed": 42
    }

    extractor = FeatureExtractor(config)

    # Load pretrained weights (optional)
    # extractor.load_pretrained("path_to_pretrained_model.pth")

    # Generate random input data
    input_data = torch.rand(1, 3, 224, 224)

    # Extract features
    features = extractor.extract_features(input_data)
    print(features.shape)

    # Save model (optional)
    # extractor.save_model("path_to_save_model.pth")