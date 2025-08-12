import logging
import random
from typing import List, Tuple, Dict
import imgaug.augmenters as iaa
import imutils
import cv2
import os
import numpy as np
from imgaug import parameters as iaa_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAugmentation:
    """
    Class for applying data augmentation techniques to images.

    ...

    Attributes
    ----------
    aug_pipeline : imgaug.augmenters.Sequential
        Augmentation pipeline consisting of multiple augmentation techniques.

    Methods
    -------
    apply_augmentations(image: np.array) -> np.array:
        Apply the augmentation pipeline to the input image.

    add_augmentation(augmentation: imgaug.augmenters.Augmenter) -> None:
        Add a new augmentation technique to the pipeline.

    load_config(config_file: str) -> None:
        Load augmentation techniques and probabilities from a configuration file.

    save_config(config_file: str) -> None:
        Save the current augmentation pipeline and probabilities to a file.
    """

    def __init__(self):
        self.aug_pipeline = iaa.Sequential([])

    def apply_augmentations(self, image: np.array) -> np.array:
        """
        Apply the augmentation pipeline to the input image.

        Parameters
        ----------
        image : np.array
            Input image to be augmented.

        Returns
        -------
        np.array
            Augmented image.
        """
        augmented_image = self.aug_pipeline(image=image)
        return augmented_image

    def add_augmentation(self, augmentation: iaa.Augmenter) -> None:
        """
        Add a new augmentation technique to the pipeline.

        Parameters
        ----------
        augmentation : imgaug.augmenters.Augmenter
            Augmentation technique to be added.

        Returns
        -------
        None
        """
        self.aug_pipeline.append(augmentation)

    def load_config(self, config_file: str) -> None:
        """
        Load augmentation techniques and probabilities from a configuration file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.

        Returns
        -------
        None
        """
        if not os.path.exists(config_file):
            logger.error("Configuration file not found.")
            return

        # Load augmentation techniques and probabilities from the file
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
                augmentations = config['augmentations']
                probabilities = config['probabilities']

                # Add augmentations to the pipeline with specified probabilities
                for aug, prob in zip(augmentations, probabilities):
                    aug_obj = self._create_augmentation(aug, prob)
                    self.aug_pipeline.append(aug_obj)
        except FileNotFoundError:
            logger.error("Configuration file not found.")
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON: %s", str(e))
        except KeyError as e:
            logger.error("Missing key in configuration file: %s", str(e))
        except Exception as e:
            logger.error("Error loading configuration: %s", str(e))

    def save_config(self, config_file: str) -> None:
        """
        Save the current augmentation pipeline and probabilities to a file.

        Parameters
        ----------
        config_file : str
            Path to the file where the configuration will be saved.

        Returns
        -------
        None
        """
        config = {
            "augmentations": [],
            "probabilities": []
        }

        # Iterate through the augmentation pipeline and store each augmentation
        # technique and its probability in the configuration dictionary
        for aug in self.aug_pipeline.augmenters:
            aug_type = type(aug.augmenters[0]).__name__
            aug_prob = aug.probability
            config["augmentations"].append(aug_type)
            config["probabilities"].append(aug_prob)

        # Save the configuration to a file
        try:
            with open(config_file, 'w') as file:
                json.dump(config, file, indent=4)
        except Exception as e:
            logger.error("Error saving configuration: %s", str(e))

    def _create_augmentation(self, aug_type: str, prob: float) -> iaa.Augmenter:
        """
        Create an augmentation object based on the provided type and probability.

        Parameters
        ----------
        aug_type : str
            Type of augmentation (e.g., "Fliplr", "Affine").
        prob : float
            Probability of applying the augmentation.

        Returns
        -------
        imgaug.augmenters.Augmenter
            Augmentation object.
        """
        if aug_type == "Fliplr":
            return iaa.Fliplr(prob)
        elif aug_type == "Affine":
            return iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2, 0.2), scale=(0.8, 1.2), shear=(-8, 8), mode='constant', cval=0, fit_output=True)
        else:
            logger.warning("Unsupported augmentation type. Skipping...")
            return None

class EyeTrackingAugmentation:
    """
    Class for applying data augmentation techniques specific to eye-tracking data.

    ...

    Attributes
    ----------
    image_aug : ImageAugmentation
        Object for applying general image augmentation techniques.
    gaze_data : List[Tuple[float, float]]
        List of gaze coordinates (x, y) to be augmented.
    head_pose : List[float]
        List of head pose angles (pitch, yaw, roll) to be augmented.

    Methods
    -------
    apply_eye_tracking_augmentations() -> None:
        Apply augmentation techniques specific to eye-tracking data.

    save_augmented_data(output_dir: str) -> None:
        Save the augmented eye-tracking data and corresponding images.
    """

    def __init__(self, image_path: str, gaze_data: List[Tuple[float, float]], head_pose: List[float]):
        self.image_aug = ImageAugmentation()
        self.gaze_data = gaze_data
        self.head_pose = head_pose
        self.image = cv2.imread(image_path)

    def apply_eye_tracking_augmentations(self) -> None:
        """
        Apply augmentation techniques specific to eye-tracking data.

        This includes random adjustments to gaze coordinates and head pose angles.

        Returns
        -------
        None
        """
        # Augment gaze coordinates
        for i in range(len(self.gaze_data)):
            x, y = self.gaze_data[i]
            random_angle = random.uniform(-5, 5)
            x_new = x + np.sin(random_angle)
            y_new = y + np.cos(random_angle)
            self.gaze_data[i] = (x_new, y_new)

        # Augment head pose angles
        for i in range(len(self.head_pose)):
            pitch, yaw, roll = self.head_pose[i]
            random_pitch = random.uniform(-2, 2)
            random_yaw = random.uniform(-2, 2)
            random_roll = random.uniform(-1, 1)
            self.head_pose[i] = (pitch + random_pitch, yaw + random_yaw, roll + random_roll)

        # Apply image augmentations
        self.image_aug.add_augmentation(iaa.Fliplr(0.5))  # Randomly flip image horizontally
        self.image_aug.add_augmentation(iaa.Affine(rotate=(-10, 10), mode='constant', cval=0))  # Random rotation
        self.augmented_image = self.image_aug.apply_augmentations(self.image)

    def save_augmented_data(self, output_dir: str) -> None:
        """
        Save the augmented eye-tracking data and corresponding images.

        Parameters
        ----------
        output_dir : str
            Directory where the augmented data will be saved.

        Returns
        -------
        None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save augmented gaze data and head pose data
        aug_gaze_data_file = os.path.join(output_dir, "augmented_gaze_data.csv")
        with open(aug_gaze_data_file, 'w') as file:
            for x, y in self.gaze_data:
                file.write(f"{x},{y}\n")

        aug_head_pose_file = os.path.join(output_dir, "augmented_head_pose.csv")
        with open(aug_head_pose_file, 'w') as file:
            for pitch, yaw, roll in self.head_pose:
                file.write(f"{pitch},{yaw},{roll}\n")

        # Save augmented image
        aug_image_file = os.path.join(output_dir, "augmented_image.png")
        cv2.imwrite(aug_image_file, self.augmented_image)

def main():
    # Example usage of the augmentation classes
    image_path = "example.jpg"
    gaze_data = [(0.5, 0.5), (0.3, 0.7), (0.2, 0.8)]
    head_pose = [(10, 20, 5), (15, 30, -2), (20, 10, 3)]

    aug = EyeTrackingAugmentation(image_path, gaze_data, head_pose)
    aug.apply_eye_tracking_augmentations()
    aug.save_augmented_data("augmented_data/")

if __name__ == "__main__":
    main()

# Example configuration file (augmentations.json)
# {
#     "augmentations": ["Fliplr", "Affine"],
#     "probabilities": [0.5, 0.8]
# }