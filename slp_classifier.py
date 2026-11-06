"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.
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
        Initialize the SLP classifier.

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
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, random_state
        )

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        # TODO: Implement forward propagation
        pass

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

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        # TODO: Implement backpropagation
        # Compute output layer error
        # Compute hidden layer error
        # Compute gradients
        pass

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # TODO: Implement cross-entropy loss
        # Clip predictions to avoid log(0)
        pass

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # TODO: Implement training loop
        # 1. Set random seed if provided
        # 2. (Bonus for multi-class) Identify unique classes and encode y
        # 3. Initialize weights
        # 4. For each iteration:
        #    - Forward propagation
        #    - Compute loss
        #    - Backward propagation
        #    - Update weights
        #    - Store loss in loss_curve_
        pass

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        # TODO: Implement prediction
        # Use forward propagation and return softmax output
        pass

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # TODO: Implement prediction
        # Get probabilities and return class with highest probability
        pass

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        # TODO: Implement accuracy computation
        pass
