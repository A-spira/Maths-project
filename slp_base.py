"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the SLP estimator.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        """
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases.
        """
        # TODO: Initialize weights using small random values (e.g., normal distribution with 0 mean and 0.01 standard deviation)
        pass

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
