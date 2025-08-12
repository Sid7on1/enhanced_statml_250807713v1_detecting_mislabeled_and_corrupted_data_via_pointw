# loss_functions.py

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLossFunctions(Module):
    """
    Custom loss functions for the computer vision project.
    """

    def __init__(self):
        super(CustomLossFunctions, self).__init__()

    def _calculate_pointwise_mutual_information(
        self, x: Tensor, y: Tensor, epsilon: float = 1e-8
    ) -> Tensor:
        """
        Calculate the pointwise mutual information between two tensors.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The pointwise mutual information between x and y.
        """
        # Calculate the joint probability distribution
        joint_prob = torch.histc(torch.stack([x, y], dim=1), bins=100, min=0, max=1)

        # Calculate the marginal probability distributions
        x_prob = torch.histc(x, bins=100, min=0, max=1)
        y_prob = torch.histc(y, bins=100, min=0, max=1)

        # Calculate the pointwise mutual information
        pmi = torch.log(joint_prob / (x_prob * y_prob) + epsilon)

        return pmi

    def _calculate_velocity_threshold(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Calculate the velocity threshold between two tensors.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The velocity threshold between x and y.
        """
        # Calculate the pointwise mutual information
        pmi = self._calculate_pointwise_mutual_information(x, y, epsilon)

        # Calculate the velocity threshold
        velocity_threshold = torch.mean(pmi) - torch.std(pmi)

        return velocity_threshold

    def _calculate_flow_theory(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Calculate the flow theory between two tensors.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The flow theory between x and y.
        """
        # Calculate the pointwise mutual information
        pmi = self._calculate_pointwise_mutual_information(x, y, epsilon)

        # Calculate the flow theory
        flow_theory = torch.mean(pmi) + torch.std(pmi)

        return flow_theory

    def _calculate_mislabeling_rate(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Calculate the mislabeling rate between two tensors.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The mislabeling rate between x and y.
        """
        # Calculate the pointwise mutual information
        pmi = self._calculate_pointwise_mutual_information(x, y, epsilon)

        # Calculate the mislabeling rate
        mislabeling_rate = torch.mean(pmi) - torch.std(pmi)

        return mislabeling_rate

    def _calculate_corrupted_data_rate(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Calculate the corrupted data rate between two tensors.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The corrupted data rate between x and y.
        """
        # Calculate the pointwise mutual information
        pmi = self._calculate_pointwise_mutual_information(x, y, epsilon)

        # Calculate the corrupted data rate
        corrupted_data_rate = torch.mean(pmi) + torch.std(pmi)

        return corrupted_data_rate

    def custom_loss_function(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Custom loss function that combines the velocity threshold and flow theory.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The custom loss function value.
        """
        # Calculate the velocity threshold
        velocity_threshold = self._calculate_velocity_threshold(x, y, epsilon)

        # Calculate the flow theory
        flow_theory = self._calculate_flow_theory(x, y, epsilon)

        # Calculate the custom loss function
        custom_loss = velocity_threshold + flow_theory

        return custom_loss

    def mislabeling_loss_function(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Mislabeling loss function that calculates the mislabeling rate.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The mislabeling loss function value.
        """
        # Calculate the mislabeling rate
        mislabeling_rate = self._calculate_mislabeling_rate(x, y, epsilon)

        # Calculate the mislabeling loss function
        mislabeling_loss = mislabeling_rate

        return mislabeling_loss

    def corrupted_data_loss_function(self, x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Corrupted data loss function that calculates the corrupted data rate.

        Args:
        x (Tensor): The input tensor.
        y (Tensor): The label tensor.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
        Tensor: The corrupted data loss function value.
        """
        # Calculate the corrupted data rate
        corrupted_data_rate = self._calculate_corrupted_data_rate(x, y, epsilon)

        # Calculate the corrupted data loss function
        corrupted_data_loss = corrupted_data_rate

        return corrupted_data_loss

def custom_loss_function(x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Custom loss function that combines the velocity threshold and flow theory.

    Args:
    x (Tensor): The input tensor.
    y (Tensor): The label tensor.
    epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
    Tensor: The custom loss function value.
    """
    custom_loss_functions = CustomLossFunctions()
    custom_loss = custom_loss_functions.custom_loss_function(x, y, epsilon)

    return custom_loss

def mislabeling_loss_function(x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Mislabeling loss function that calculates the mislabeling rate.

    Args:
    x (Tensor): The input tensor.
    y (Tensor): The label tensor.
    epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
    Tensor: The mislabeling loss function value.
    """
    custom_loss_functions = CustomLossFunctions()
    mislabeling_loss = custom_loss_functions.mislabeling_loss_function(x, y, epsilon)

    return mislabeling_loss

def corrupted_data_loss_function(x: Tensor, y: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    Corrupted data loss function that calculates the corrupted data rate.

    Args:
    x (Tensor): The input tensor.
    y (Tensor): The label tensor.
    epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
    Tensor: The corrupted data loss function value.
    """
    custom_loss_functions = CustomLossFunctions()
    corrupted_data_loss = custom_loss_functions.corrupted_data_loss_function(x, y, epsilon)

    return corrupted_data_loss