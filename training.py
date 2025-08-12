import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
MODEL_FILE = 'model.pth'
RESULTS_FILE = 'results.csv'

# Define custom exception classes
class InvalidDataError(Exception):
    """Raised when invalid data is encountered."""
    pass

class ModelNotTrainedError(Exception):
    """Raised when the model is not trained."""
    pass

# Define data structures and models
class PointwiseMutualInformation:
    """Calculates pointwise mutual information between inputs and labels."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initializes the PointwiseMutualInformation class.

        Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data.
        """
        self.X = X
        self.y = y

    def calculate_mutual_information(self) -> np.ndarray:
        """
        Calculates pointwise mutual information between inputs and labels.

        Returns:
        np.ndarray: Pointwise mutual information values.
        """
        # Calculate mutual information using the formula from the paper
        mutual_info = np.zeros(len(self.X))
        for i in range(len(self.X)):
            p_xy = np.mean(self.X[i] * self.y[i])
            p_x = np.mean(self.X[i])
            p_y = np.mean(self.y[i])
            mutual_info[i] = p_xy * np.log2(p_xy / (p_x * p_y))
        return mutual_info

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesses data by scaling and encoding categorical variables."""
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the scaler to the data.

        Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data (optional).
        """
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data by scaling.

        Args:
        X (np.ndarray): Input data.

        Returns:
        np.ndarray: Scaled data.
        """
        return self.scaler.transform(X)

class NeuralNetwork(nn.Module):
    """Defines a neural network model."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the NeuralNetwork class.

        Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TrainingPipeline:
    """Defines a training pipeline for the model."""
    def __init__(self, model: NeuralNetwork, device: str, batch_size: int, epochs: int, learning_rate: float):
        """
        Initializes the TrainingPipeline class.

        Args:
        model (NeuralNetwork): Neural network model.
        device (str): Device to use (e.g., 'cpu', 'cuda').
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_loader: DataLoader) -> None:
        """
        Trains the model.

        Args:
        train_loader (DataLoader): Training data loader.
        """
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluates the model.

        Args:
        test_loader (DataLoader): Testing data loader.

        Returns:
        Tuple[float, float, float]: Accuracy, precision, recall.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()

        accuracy = correct / len(test_loader.dataset)
        precision = accuracy
        recall = accuracy
        return accuracy, precision, recall

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data from a CSV file.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Input data and label data.
    """
    data = pd.read_csv(file_path)
    X = data.drop('label', axis=1).values
    y = data['label'].values
    return X, y

def main() -> None:
    """
    Main function.
    """
    # Load data
    X_train, y_train = load_data(os.path.join(DATA_DIR, TRAIN_FILE))
    X_test, y_test = load_data(os.path.join(DATA_DIR, TEST_FILE))

    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Split data into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.from_numpy(X_train_split).float(), torch.from_numpy(y_train_split).long()), batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.from_numpy(X_val_split).float(), torch.from_numpy(y_val_split).long()), batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.from_numpy(X_test_scaled).float(), torch.from_numpy(y_test).long()), batch_size=32, shuffle=False)

    # Create model and training pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(input_dim=X_train.shape[1], hidden_dim=128, output_dim=10)
    pipeline = TrainingPipeline(model, device, batch_size=32, epochs=10, learning_rate=0.001)

    # Train model
    pipeline.train(train_loader)

    # Evaluate model
    accuracy, precision, recall = pipeline.evaluate(test_loader)
    logger.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_FILE))

if __name__ == '__main__':
    main()