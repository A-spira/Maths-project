"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import logistic, softmax
from slp_base import BaseSLPEstimator

# Probabilities are kept in [EPSILON, 1 - EPSILON] so that log() stays finite.
PROBABILITY_EPSILON: float = 1e-15


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Binary problems use one output neuron with a sigmoid.
    Problems with three classes or more use one neuron per class with a softmax.

    Compatible interface with sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        solver: str = "adam",
        batch_size: Optional[int] = 32,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        alpha: float = 0.0,
    ) -> None:
        """
        Initialize the SLP classifier.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'), default='logistic'
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations (epochs)
        random_state : int or None
            Random seed for reproducibility
        solver : str
            Optimizer: 'sgd' (plain gradient descent) or 'adam'
        batch_size : int or None
            Number of samples per update; None uses all samples (full batch)
        early_stopping : bool
            Whether to stop when the validation loss stops improving
        validation_fraction : float
            Share of the training data held out for validation, between 0 and 1
        n_iter_no_change : int
            Number of epochs without improvement allowed before stopping
        tol : float
            Minimum decrease of the validation loss that counts as an improvement
        alpha : float
            L2 regularization strength; 0 disables regularization
        """
        super().__init__(
            hidden_layer_size,
            activation,
            learning_rate,
            max_iter,
            random_state,
            solver,
            batch_size,
            early_stopping,
            validation_fraction,
            n_iter_no_change,
            tol,
            alpha,
        )

        # Classifier-specific attributes
        self.classes_: Optional[NDArray] = None  # Unique class labels

    def _validate_labels(self, y: Any, n_samples: int) -> NDArray:
        """
        Convert the labels to a 1D array and check them.

        Parameters:
        -----------
        y : array-like, shape (n_samples,) or (n_samples, 1)
            Class labels, which may be numbers or strings
        n_samples : int
            Number of labels that y must contain

        Returns:
        --------
        y : array, shape (n_samples,)
            Validated labels
        """
        y_array = np.asarray(y)
        # A single column holds the same information as a 1D array.
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.ravel()
        if y_array.ndim != 1:
            raise ValueError(f"y must be a 1D array of labels, got shape {y_array.shape}.")
        if y_array.shape[0] != n_samples:
            raise ValueError(f"X has {n_samples} samples, but y has {y_array.shape[0]}.")
        # NaN can only appear in floating-point labels.
        if np.issubdtype(y_array.dtype, np.floating) and np.any(np.isnan(y_array)):
            raise ValueError("y contains NaN values.")
        return y_array

    def _encode_labels(self, y: NDArray) -> NDArray[np.floating]:
        """
        Store the classes and encode the labels for the output layer.

        Parameters:
        -----------
        y : array, shape (n_samples,)
            Validated labels

        Returns:
        --------
        y_encoded : array, shape (n_samples, n_outputs)
            One column for one or two classes, one-hot columns otherwise
        """
        # class_indices gives the position of each label in the sorted classes.
        self.classes_, class_indices = np.unique(y, return_inverse=True)
        n_classes = self.classes_.shape[0]
        if n_classes == 1:
            # Only one class: the single output neuron learns a probability of 1.
            return np.ones((y.shape[0], 1))  # shape: (n, 1)
        if n_classes == 2:
            # Two classes: the output neuron gives the probability of the second one.
            return class_indices.reshape(-1, 1).astype(float)  # shape: (n, 1)
        # Three classes or more: row k of the identity matrix is the one-hot code of k.
        return np.eye(n_classes)[class_indices]  # shape: (n, K)

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
        activation, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_  # (n, d) @ (d, h) + (h,) -> (n, h)
        a1 = activation(z1)  # shape: (n, h)
        z2 = a1 @ self.W2_ + self.b2_  # (n, h) @ (h, K) + (K,) -> (n, K)
        # One neuron: sigmoid. Several neurons: softmax over the classes.
        y_pred = logistic(z2) if self.n_outputs_ == 1 else softmax(z2)  # shape: (n, K)
        return z1, a1, z2, y_pred

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
            Encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        n_samples = X.shape[0]
        _, activation_derivative = self._get_activation_function()

        # Output layer error: for cross-entropy with a sigmoid or a softmax output,
        # the derivative of the output cancels out and dL/dz2 = y_pred - y.
        dz2 = y_pred - y  # shape: (n, K)

        # Since z2 = a1 @ W2 + b2, the gradients of W2 and b2 follow directly.
        dW2 = a1.T @ dz2 / n_samples  # (h, n) @ (n, K) -> (h, K)
        db2 = np.mean(dz2, axis=0)  # shape: (K,)

        # Hidden layer error: send the output error back through W2.
        da1 = dz2 @ self.W2_.T  # (n, K) @ (K, h) -> (n, h)
        # Then through the activation, element by element: da1/dz1 = activation'(z1).
        dz1 = da1 * activation_derivative(z1)  # shape: (n, h)

        # Since z1 = X @ W1 + b1, the gradients of W1 and b1 follow directly.
        dW1 = X.T @ dz1 / n_samples  # (d, n) @ (n, h) -> (d, h)
        db1 = np.mean(dz1, axis=0)  # shape: (h,)

        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss, averaged over the samples.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_outputs)
            Encoded true labels
        y_pred : array-like, shape (n_samples, n_outputs)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # The sigmoid can return exactly 1, so clip to avoid log(0).
        p = np.clip(y_pred, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)  # shape: (n, K)
        if y_true.shape[1] == 1:
            # Binary: L = -mean(y * log(p) + (1 - y) * log(1 - p)).
            return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
        # Multi-class: L = -mean(sum_k y_k * log(p_k)).
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    def fit(self, X: NDArray[np.floating], y: NDArray) -> "SimpleSLPClassifier":
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
        self._validate_params()
        X_array = self._validate_X(X, reset=True)  # shape: (n, d)
        y_array = self._validate_labels(y, X_array.shape[0])  # shape: (n,)
        y_encoded = self._encode_labels(y_array)  # shape: (n, K)
        return self._train(X_array, y_encoded)

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
        self._check_is_fitted()
        X_array = self._validate_X(X)  # shape: (n, d)
        y_pred = self._forward_propagation(X_array)[3]  # shape: (n, K)
        n_classes = self.classes_.shape[0]
        if n_classes == 1:
            # Only one class is known, so its probability is 1.
            return np.ones((X_array.shape[0], 1))  # shape: (n, 1)
        if n_classes == 2:
            # The neuron gives P(second class); the first class gets the remainder.
            return np.hstack([1.0 - y_pred, y_pred])  # shape: (n, 2)
        return y_pred  # Softmax output, shape: (n, K)

    def predict(self, X: NDArray[np.floating]) -> NDArray:
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
        # Position of the most probable class in each row.
        class_indices = np.argmax(self.predict_proba(X), axis=1)  # shape: (n,)
        return self.classes_[class_indices]  # shape: (n,)

    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
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
        y_pred = self.predict(X)  # shape: (n,)
        y_true = self._validate_labels(y, y_pred.shape[0])  # shape: (n,)
        return float(np.mean(y_pred == y_true))
