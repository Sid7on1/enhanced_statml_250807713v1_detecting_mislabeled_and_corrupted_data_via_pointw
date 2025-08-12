import logging
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple
from evaluation.metrics import calculate_mutual_information, calculate_pointwise_contribution
from evaluation.config import Config
from evaluation.exceptions import EvaluationError
from evaluation.models import ModelEvaluator
from evaluation.utils import load_data, load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluation:
    def __init__(self, config: Config):
        self.config = config
        self.model_evaluator = ModelEvaluator(config)

    def evaluate(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            model: The model to evaluate.
            data: The data to evaluate on.

        Returns:
            A dictionary containing the evaluation metrics.
        """
        try:
            # Load the model and data
            model.load_state_dict(torch.load(self.config.model_path))
            data = load_data(self.config.data_path)

            # Evaluate the model
            predictions = self.model_evaluator.evaluate(model, data)

            # Calculate the evaluation metrics
            accuracy = accuracy_score(data['labels'], predictions)
            report = classification_report(data['labels'], predictions)
            matrix = confusion_matrix(data['labels'], predictions)

            # Calculate the mutual information and pointwise contribution
            mutual_info = calculate_mutual_information(data['inputs'], data['labels'])
            pointwise_contribution = calculate_pointwise_contribution(data['inputs'], data['labels'])

            # Log the results
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{report}")
            logger.info(f"Confusion Matrix:\n{matrix}")
            logger.info(f"Mutual Information: {mutual_info:.4f}")
            logger.info(f"Pointwise Contribution: {pointwise_contribution:.4f}")

            # Return the evaluation metrics
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': matrix,
                'mutual_info': mutual_info,
                'pointwise_contribution': pointwise_contribution
            }
        except Exception as e:
            # Log the error
            logger.error(f"Error evaluating model: {e}")
            raise EvaluationError(f"Error evaluating model: {e}")

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, model: nn.Module, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the model on the given data.

        Args:
            model: The model to evaluate.
            data: The data to evaluate on.

        Returns:
            The predictions made by the model.
        """
        try:
            # Load the model and data
            model.load_state_dict(torch.load(self.config.model_path))
            data = load_data(self.config.data_path)

            # Evaluate the model
            predictions = model(data['inputs'])

            # Return the predictions
            return predictions
        except Exception as e:
            # Log the error
            logger.error(f"Error evaluating model: {e}")
            raise EvaluationError(f"Error evaluating model: {e}")

def load_data(path: str) -> Dict[str, torch.Tensor]:
    """
    Load the data from the given path.

    Args:
        path: The path to the data.

    Returns:
        A dictionary containing the data.
    """
    try:
        # Load the data
        data = torch.load(path)

        # Return the data
        return data
    except Exception as e:
        # Log the error
        logger.error(f"Error loading data: {e}")
        raise EvaluationError(f"Error loading data: {e}")

def load_model(path: str) -> nn.Module:
    """
    Load the model from the given path.

    Args:
        path: The path to the model.

    Returns:
        The loaded model.
    """
    try:
        # Load the model
        model = torch.load(path)

        # Return the model
        return model
    except Exception as e:
        # Log the error
        logger.error(f"Error loading model: {e}")
        raise EvaluationError(f"Error loading model: {e}")

if __name__ == "__main__":
    # Load the configuration
    config = Config()

    # Create an evaluation instance
    evaluation = Evaluation(config)

    # Load the model and data
    model = load_model(config.model_path)
    data = load_data(config.data_path)

    # Evaluate the model
    results = evaluation.evaluate(model, data)

    # Print the results
    print(results)