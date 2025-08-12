import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Input must be numpy arrays")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Input must be 2D for X and 1D for y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    def _validate_config(self, config: Dict) -> None:
        required_keys = ["velocity_threshold", "flow_threshold"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

    def _calculate_velocity(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate velocity between consecutive frames.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)

        Returns:
        np.ndarray: Velocity between consecutive frames (n_samples, n_features)
        """
        velocity = np.zeros((X.shape[0], X.shape[1]))
        for i in range(1, X.shape[0]):
            velocity[i] = np.linalg.norm(X[i] - X[i-1])
        return velocity

    def _calculate_flow(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate flow between consecutive frames.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)

        Returns:
        np.ndarray: Flow between consecutive frames (n_samples, n_features)
        """
        flow = np.zeros((X.shape[0], X.shape[1]))
        for i in range(1, X.shape[0]):
            flow[i] = distance.euclidean(X[i], X[i-1])
        return flow

    def _calculate_mutual_info(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mutual information between input data and labels.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)

        Returns:
        float: Mutual information between input data and labels
        """
        return mutual_info_score(X.ravel(), y)

    def _filter_noisy_samples(self, X: np.ndarray, y: np.ndarray, velocity_threshold: float, flow_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter noisy samples based on velocity and flow thresholds.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)
        velocity_threshold (float): Velocity threshold
        flow_threshold (float): Flow threshold

        Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered input data and labels
        """
        velocity = self._calculate_velocity(X)
        flow = self._calculate_flow(X)
        mask = (velocity < velocity_threshold) & (flow < flow_threshold)
        return X[mask], y[mask]

    def _calculate_pointwise_contribution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate pointwise contribution of each sample to mutual information.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)

        Returns:
        np.ndarray: Pointwise contribution of each sample to mutual information (n_samples,)
        """
        mutual_info = self._calculate_mutual_info(X, y)
        pointwise_contribution = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            X_i = X[i]
            y_i = y[i]
            pointwise_contribution[i] = mutual_info_score(X_i, y_i)
        return pointwise_contribution

    def detect_noisy_samples(self, X: np.ndarray, y: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect noisy samples using velocity and flow thresholds.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)
        config (Dict): Configuration dictionary

        Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered input data and labels
        """
        self._validate_input(X, y)
        self._validate_config(config)
        velocity_threshold = config["velocity_threshold"]
        flow_threshold = config["flow_threshold"]
        X_filtered, y_filtered = self._filter_noisy_samples(X, y, velocity_threshold, flow_threshold)
        return X_filtered, y_filtered

    def detect_mislabeled_samples(self, X: np.ndarray, y: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect mislabeled samples using pointwise contribution.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)
        config (Dict): Configuration dictionary

        Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered input data and labels
        """
        self._validate_input(X, y)
        self._validate_config(config)
        pointwise_contribution = self._calculate_pointwise_contribution(X, y)
        threshold = np.mean(pointwise_contribution)
        mask = pointwise_contribution < threshold
        return X[mask], y[mask]

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, model: torch.nn.Module) -> float:
        """
        Evaluate model performance using mutual information.

        Args:
        X (np.ndarray): Input data (n_samples, n_features)
        y (np.ndarray): Labels (n_samples,)
        model (torch.nn.Module): Model to evaluate

        Returns:
        float: Mutual information between input data and model predictions
        """
        self._validate_input(X, y)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_tensor = torch.from_numpy(y).long().to(self.device)
            predictions = model(X_tensor)
            mutual_info = self._calculate_mutual_info(X_tensor.cpu().numpy(), predictions.cpu().numpy())
        return mutual_info

def main():
    # Example usage
    config = {
        "velocity_threshold": 0.5,
        "flow_threshold": 0.5
    }
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    utils = Utils(config)
    X_filtered, y_filtered = utils.detect_noisy_samples(X, y, config)
    print(X_filtered.shape, y_filtered.shape)

if __name__ == "__main__":
    main()