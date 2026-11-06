"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.
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
    ) -> None:
        """
        Initialize the SLP regressor.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('logistic', 'tanh', 'relu'), default='logistic'
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
        )

    def _validate_y(self, y: Any, n_samples: int) -> NDArray[np.floating]:
        """
        Convert the targets to a 2D float array and check them.

        Parameters:
        -----------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values
        n_samples : int
            Number of rows that y must have

        Returns:
        --------
        y : array, shape (n_samples, n_outputs)
            Validated targets, always with two dimensions
        """
        y_array = np.asarray(y, dtype=float)
        # A 1D target is one output per sample: (n,) -> (n, 1).
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        if y_array.ndim != 2:
            raise ValueError(f"y must be a 1D or 2D array, got {y_array.ndim} dimensions.")
        if y_array.shape[0] != n_samples:
            raise ValueError(f"X has {n_samples} samples, but y has {y_array.shape[0]}.")
        if not np.all(np.isfinite(y_array)):
            raise ValueError("y contains NaN or infinite values.")
        return y_array

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
        y_pred = z2  # Identity output: the prediction can take any real value
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

        Each gradient comes from the chain rule, starting from the output layer.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            Target values
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        n_samples = X.shape[0]
        _, activation_derivative = self._get_activation_function()

        # Output layer error: with L = 1/(2n) * sum ||y_pred - y||^2, dL/dz2 = y_pred - y.
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
        Compute mean squared error loss, with a factor 1/2 that simplifies its gradient.

        L = 1/(2n) * sum_i ||y_pred_i - y_true_i||^2

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_outputs)
            True values
        y_pred : array-like, shape (n_samples, n_outputs)
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        # Sum the squared errors over the outputs, then average over the samples.
        squared_errors = np.sum((y_pred - y_true) ** 2, axis=1)  # shape: (n,)
        return float(0.5 * np.mean(squared_errors))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self._validate_params()
        X_array = self._validate_X(X, reset=True)  # shape: (n, d)
        y_array = self._validate_y(y, X_array.shape[0])  # shape: (n, K)
        return self._train(X_array, y_array)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        self._check_is_fitted()
        X_array = self._validate_X(X)  # shape: (n, d)
        y_pred = self._forward_propagation(X_array)[3]  # shape: (n, K)
        # With one output, return (n,) like scikit-learn.
        return y_pred.ravel() if self.n_outputs_ == 1 else y_pred

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        R² = 1 - SS_res / SS_tot. With several outputs, the scores are averaged.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y_pred = np.asarray(self.predict(X)).reshape(-1, self.n_outputs_)  # shape: (n, K)
        y_true = self._validate_y(y, y_pred.shape[0])  # shape: (n, K)
        # Squared error of the model, for each output.
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)  # shape: (K,)
        # Squared error of a model that always predicts the mean, for each output.
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)  # shape: (K,)
        return float(np.mean(1.0 - ss_res / ss_tot))
