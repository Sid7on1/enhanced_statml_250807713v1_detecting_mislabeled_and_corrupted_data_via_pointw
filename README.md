"""
Project: enhanced_stat.ML_2508.07713v1_Detecting_Mislabeled_and_Corrupted_Data_via_Pointw
Type: computer_vision
Description: Enhanced AI project based on stat.ML_2508.07713v1_Detecting-Mislabeled-and-Corrupted-Data-via-Pointw with content analysis.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataProcessor:
    """
    DataProcessor class to handle data loading, preprocessing, and splitting.
    """

    def __init__(self, data_path, noise_type, noise_level):
        """
        Initialize the DataProcessor.

        Args:
            data_path (str): Path to the dataset.
            noise_type (str): Type of noise to add to the data (e.g., "label", "input").
            noise_level (float): Level of noise to add to the data (e.g., 0.1 for 10% noise).
        """
        self.data_path = data_path
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.X = None
        self.y = None

    def load_data(self):
        """
        Load the dataset from the specified path.
        """
        try:
            self.X = pd.read_csv(os.path.join(self.data_path, "features.csv"))
            self.y = pd.read_csv(os.path.join(self.data_path, "labels.csv"))
        except FileNotFoundError:
            logging.error("Dataset not found.")
            sys.exit(1)

    def add_noise(self):
        """
        Add noise to the data based on the specified type and level.
        """
        if self.noise_type == "label":
            self.y += np.random.normal(0, self.noise_level, size=self.y.shape[0])
        elif self.noise_type == "input":
            self.X += np.random.normal(0, self.noise_level, size=self.X.shape)
        else:
            logging.error("Invalid noise type.")
            sys.exit(1)

    def preprocess_data(self):
        """
        Preprocess the data by scaling and splitting it into training and testing sets.
        """
        try:
            self.X = StandardScaler().fit_transform(self.X)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        except NotFittedError:
            logging.error("Data not preprocessed.")
            sys.exit(1)

    def get_data(self):
        """
        Get the preprocessed data.

        Returns:
            tuple: Preprocessed training and testing data.
        """
        return self.X_train, self.X_test, self.y_train, self.y_test


class MutualInformationCalculator:
    """
    MutualInformationCalculator class to calculate mutual information between inputs and labels.
    """

    def __init__(self, X, y):
        """
        Initialize the MutualInformationCalculator.

        Args:
            X (numpy array): Input data.
            y (numpy array): Label data.
        """
        self.X = X
        self.y = y

    def calculate_mutual_information(self):
        """
        Calculate the mutual information between inputs and labels.

        Returns:
            float: Mutual information score.
        """
        try:
            return mutual_info_score(self.X, self.y)
        except ValueError:
            logging.error("Mutual information calculation failed.")
            sys.exit(1)


class DataSelector:
    """
    DataSelector class to select high-quality data based on mutual information.
    """

    def __init__(self, X, y, threshold):
        """
        Initialize the DataSelector.

        Args:
            X (numpy array): Input data.
            y (numpy array): Label data.
            threshold (float): Threshold for selecting high-quality data.
        """
        self.X = X
        self.y = y
        self.threshold = threshold

    def select_data(self):
        """
        Select high-quality data based on mutual information.

        Returns:
            tuple: Selected input and label data.
        """
        try:
            mutual_info = MutualInformationCalculator(self.X, self.y).calculate_mutual_information()
            selected_X = self.X[self.X[:, 0] > self.threshold]
            selected_y = self.y[selected_X[:, 0] > self.threshold]
            return selected_X, selected_y
        except ValueError:
            logging.error("Data selection failed.")
            sys.exit(1)


class Project:
    """
    Project class to manage the entire project.
    """

    def __init__(self, data_path, noise_type, noise_level, threshold):
        """
        Initialize the Project.

        Args:
            data_path (str): Path to the dataset.
            noise_type (str): Type of noise to add to the data (e.g., "label", "input").
            noise_level (float): Level of noise to add to the data (e.g., 0.1 for 10% noise).
            threshold (float): Threshold for selecting high-quality data.
        """
        self.data_path = data_path
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.threshold = threshold
        self.data_processor = DataProcessor(data_path, noise_type, noise_level)
        self.data_selector = DataSelector(None, None, threshold)

    def run_project(self):
        """
        Run the project by loading data, adding noise, preprocessing data, and selecting high-quality data.
        """
        try:
            self.data_processor.load_data()
            self.data_processor.add_noise()
            self.data_processor.preprocess_data()
            self.data_selector.X, self.data_selector.y = self.data_processor.get_data()
            selected_X, selected_y = self.data_selector.select_data()
            logging.info("Project completed successfully.")
        except Exception as e:
            logging.error("Project failed with error: {}".format(str(e)))
            sys.exit(1)


if __name__ == "__main__":
    project = Project("data", "label", 0.1, 0.5)
    project.run_project()