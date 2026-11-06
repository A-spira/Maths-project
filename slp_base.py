"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any, Callable, Dict, Optional, Tuple
import warnings
import numpy as np
from numpy.typing import NDArray
from activations import (
    identity,
    identity_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)

# Adam hyperparameters: the usual values, which are also scikit-learn's defaults.
ADAM_BETA1: float = 0.9  # Decay rate of the running mean of gradients
ADAM_BETA2: float = 0.999  # Decay rate of the running mean of squared gradients
ADAM_EPSILON: float = 1e-8  # Avoids a division by zero in the Adam step

VALID_ACTIVATIONS: Tuple[str, ...] = ("identity", "logistic", "tanh", "relu")
VALID_SOLVERS: Tuple[str, ...] = ("sgd", "adam")


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators: parameter and
    data validation, weight initialization, the optimizers and the training loop.
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
        Initialize the SLP estimator.

        Following scikit-learn conventions, parameters are only stored here and
        are validated when fit() is called.

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
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.solver: str = solver
        self.batch_size: Optional[int] = batch_size
        self.early_stopping: bool = early_stopping
        self.validation_fraction: float = validation_fraction
        self.n_iter_no_change: int = n_iter_no_change
        self.tol: float = tol
        self.alpha: float = alpha

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training
        self.validation_loss_curve_: list[float] = []  # Filled with early stopping
        self.best_validation_loss_: Optional[float] = None  # Lowest validation loss
        self.n_iter_: int = 0  # Number of completed epochs
        self.diverged_: bool = False  # True if training stopped on a non-finite value
        self.n_features_in_: Optional[int] = None  # Number of input features
        self.n_outputs_: Optional[int] = None  # Number of output neurons

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _check_positive_int(name: str, value: Any) -> None:
        """Raise ValueError unless value is an integer greater than or equal to 1."""
        # bool is a subclass of int in Python, so it must be rejected explicitly.
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _check_real(name: str, value: Any) -> None:
        """Raise ValueError unless value is a finite real number."""
        # Same bool exclusion as above; np.isfinite rejects NaN and infinity.
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}.")

    def _validate_params(self) -> None:
        """Check every hyperparameter and raise ValueError on the first invalid one."""
        self._check_positive_int("hidden_layer_size", self.hidden_layer_size)
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Unknown activation: {self.activation}")
        self._check_real("learning_rate", self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}.")
        self._check_positive_int("max_iter", self.max_iter)
        # numpy's default_rng only accepts non-negative seeds.
        if self.random_state is not None:
            if (
                isinstance(self.random_state, bool)
                or not isinstance(self.random_state, Integral)
                or self.random_state < 0
            ):
                raise ValueError(
                    f"random_state must be None or an integer >= 0, got {self.random_state!r}."
                )
        if self.solver not in VALID_SOLVERS:
            raise ValueError(f"solver must be one of {VALID_SOLVERS}, got {self.solver!r}.")
        if self.batch_size is not None:
            self._check_positive_int("batch_size", self.batch_size)
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError(f"early_stopping must be a boolean, got {self.early_stopping!r}.")
        self._check_real("validation_fraction", self.validation_fraction)
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be strictly between 0 and 1, "
                f"got {self.validation_fraction!r}."
            )
        self._check_positive_int("n_iter_no_change", self.n_iter_no_change)
        self._check_real("tol", self.tol)
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol!r}.")
        self._check_real("alpha", self.alpha)
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha!r}.")

    def _validate_X(self, X: Any, reset: bool = False) -> NDArray[np.floating]:
        """
        Convert X to a 2D float array and check it.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        reset : bool
            True in fit() to store the number of features; False in predict()
            to compare X with the number of features seen during fit()

        Returns:
        --------
        X : array, shape (n_samples, n_features)
            Validated data
        """
        X_array = np.asarray(X, dtype=float)  # shape: (n, d)
        if X_array.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X_array.ndim} dimension(s).")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError(f"X must not be empty, got shape {X_array.shape}.")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X contains NaN or infinite values.")
        if reset:
            self.n_features_in_ = X_array.shape[1]
        elif X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_array.shape[1]} features, expected {self.n_features_in_}."
            )
        return X_array

    def _check_is_fitted(self) -> None:
        """Raise ValueError if the model has not been trained yet."""
        if self.W1_ is None:
            raise ValueError("This model is not fitted yet. Call fit() first.")

    # ------------------------------------------------------------------
    # Activation and weight initialization
    # ------------------------------------------------------------------

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "identity":
            return identity, identity_derivative
        elif self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(
        self, n_features: int, n_outputs: int, rng: np.random.Generator
    ) -> None:
        """
        Initialize weights and biases with Glorot uniform initialization.

        Each value is drawn from U(-bound, bound), with
        bound = sqrt(6 / (fan_in + fan_out)), so the scale adapts to the layer size.

        Different random values make hidden neurons compute different functions.
        If they all started equal, they would receive equal gradients and stay equal.

        Parameters:
        -----------
        n_features : int
            Number of input features (d)
        n_outputs : int
            Number of output neurons (K)
        rng : numpy Generator
            Source of randomness
        """
        hidden = self.hidden_layer_size  # h
        bound1 = np.sqrt(6.0 / (n_features + hidden))  # Input -> hidden layer
        bound2 = np.sqrt(6.0 / (hidden + n_outputs))  # Hidden -> output layer
        self.W1_ = rng.uniform(-bound1, bound1, size=(n_features, hidden))  # shape: (d, h)
        self.b1_ = rng.uniform(-bound1, bound1, size=hidden)  # shape: (h,)
        self.W2_ = rng.uniform(-bound2, bound2, size=(hidden, n_outputs))  # shape: (h, K)
        self.b2_ = rng.uniform(-bound2, bound2, size=n_outputs)  # shape: (K,)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def _initialize_optimizer(self) -> None:
        """Reset the Adam state: step counter and both running means set to zero."""
        self._adam_t: int = 0  # Number of updates done so far
        # First moments: running means of the gradients, one array per parameter.
        self._adam_m: Dict[str, NDArray[np.floating]] = {
            "W1": np.zeros_like(self.W1_),
            "b1": np.zeros_like(self.b1_),
            "W2": np.zeros_like(self.W2_),
            "b2": np.zeros_like(self.b2_),
        }
        # Second moments: running means of the squared gradients.
        self._adam_v: Dict[str, NDArray[np.floating]] = {
            "W1": np.zeros_like(self.W1_),
            "b1": np.zeros_like(self.b1_),
            "W2": np.zeros_like(self.W2_),
            "b2": np.zeros_like(self.b2_),
        }

    def _adam_step(
        self, name: str, gradient: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute the Adam step of one parameter array.

        Parameters:
        -----------
        name : str
            Parameter name: 'W1', 'b1', 'W2' or 'b2'
        gradient : array
            Gradient of the loss with respect to this parameter

        Returns:
        --------
        step : array, same shape as gradient
            Quantity to subtract from the parameter
        """
        # m = beta1 * m + (1 - beta1) * g: smoothed gradient direction.
        self._adam_m[name] = ADAM_BETA1 * self._adam_m[name] + (1 - ADAM_BETA1) * gradient
        # v = beta2 * v + (1 - beta2) * g^2: smoothed gradient magnitude.
        self._adam_v[name] = (
            ADAM_BETA2 * self._adam_v[name] + (1 - ADAM_BETA2) * gradient**2
        )
        # Both means start at 0, so they are too small at first: correct this bias.
        m_hat = self._adam_m[name] / (1 - ADAM_BETA1**self._adam_t)
        v_hat = self._adam_v[name] / (1 - ADAM_BETA2**self._adam_t)
        # Dividing by sqrt(v_hat) gives every weight its own effective learning rate.
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)

    def _update_parameters(
        self,
        dW1: NDArray[np.floating],
        db1: NDArray[np.floating],
        dW2: NDArray[np.floating],
        db2: NDArray[np.floating],
    ) -> None:
        """Update all weights and biases in place using the selected optimizer."""
        if self.solver == "sgd":
            # Plain gradient descent: theta = theta - learning_rate * gradient.
            self.W1_ -= self.learning_rate * dW1
            self.b1_ -= self.learning_rate * db1
            self.W2_ -= self.learning_rate * dW2
            self.b2_ -= self.learning_rate * db2
        else:
            # Adam: count this update, then apply one step per parameter array.
            self._adam_t += 1
            self.W1_ -= self._adam_step("W1", dW1)
            self.b1_ -= self._adam_step("b1", db1)
            self.W2_ -= self._adam_step("W2", dW2)
            self.b2_ -= self._adam_step("b2", db2)

    # ------------------------------------------------------------------
    # Helpers used by the training loop
    # ------------------------------------------------------------------

    def _copy_parameters(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Return independent copies of the four parameter arrays."""
        return self.W1_.copy(), self.b1_.copy(), self.W2_.copy(), self.b2_.copy()

    def _restore_parameters(
        self, parameters: Tuple[NDArray, NDArray, NDArray, NDArray]
    ) -> None:
        """Put back parameter arrays saved by _copy_parameters."""
        self.W1_, self.b1_, self.W2_, self.b2_ = parameters

    def _parameters_are_finite(self) -> bool:
        """Return True if no weight or bias contains NaN or infinity."""
        return bool(
            np.all(np.isfinite(self.W1_))
            and np.all(np.isfinite(self.b1_))
            and np.all(np.isfinite(self.W2_))
            and np.all(np.isfinite(self.b2_))
        )

    def _split_validation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Randomly split the data into a training part and a validation part.

        Returns:
        --------
        X_train, y_train, X_val, y_val : tuple of arrays
            Rows of X and y; the validation part holds validation_fraction of them
        """
        n_samples = X.shape[0]
        # Round up so that the validation part contains at least one sample.
        n_val = int(np.ceil(self.validation_fraction * n_samples))
        if n_val >= n_samples:
            raise ValueError(
                f"Not enough samples ({n_samples}) for early stopping "
                f"with validation_fraction={self.validation_fraction}."
            )
        indices = rng.permutation(n_samples)  # shape: (n,)
        val_indices = indices[:n_val]  # shape: (n_val,)
        train_indices = indices[n_val:]  # shape: (n - n_val,)
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def _objective_and_gradients(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> Tuple[float, Tuple[NDArray, NDArray, NDArray, NDArray]]:
        """
        Compute the training objective and its gradients on one batch.

        Objective = loss + alpha / (2m) * (||W1||^2 + ||W2||^2), with m samples.
        Biases are not penalized: they only shift the output and do not make the
        model more sensitive to its inputs.

        Parameters:
        -----------
        X : array, shape (m, n_features)
            Batch of input data
        y : array, shape (m, n_outputs)
            Batch of encoded targets

        Returns:
        --------
        objective : float
            Loss plus L2 penalty
        gradients : tuple of arrays
            dW1, db1, dW2, db2, including the penalty
        """
        n_samples = X.shape[0]
        z1, a1, z2, y_pred = self._forward_propagation(X)
        # Squared Frobenius norm of the weights: the sum of all squared weights.
        squared_norm = np.sum(self.W1_**2) + np.sum(self.W2_**2)
        objective = self._compute_loss(y, y_pred) + self.alpha / (2 * n_samples) * squared_norm
        dW1, db1, dW2, db2 = self._backward_propagation(X, y, z1, a1, z2, y_pred)
        # The derivative of alpha / (2m) * W^2 with respect to W is alpha / m * W.
        dW1 = dW1 + self.alpha / n_samples * self.W1_  # shape: (d, h)
        dW2 = dW2 + self.alpha / n_samples * self.W2_  # shape: (h, K)
        return objective, (dW1, db1, dW2, db2)

    def _run_epoch(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Run one pass over the data: shuffle it, then update once per batch.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Training data
        y : array, shape (n_samples, n_outputs)
            Encoded targets
        batch_size : int
            Number of samples per update
        rng : numpy Generator
            Source of randomness for the shuffling

        Returns:
        --------
        loss : float
            Mean of the batch objectives, weighted by the size of each batch
        """
        n_samples = X.shape[0]
        # A new random order at each epoch, so that batches differ between epochs.
        order = rng.permutation(n_samples)  # shape: (n,)
        total_loss = 0.0
        # Each batch starts batch_size samples after the previous one.
        for start in range(0, n_samples, batch_size):
            batch = order[start : start + batch_size]  # shape: (m,), the last may be smaller
            X_batch, y_batch = X[batch], y[batch]  # shapes: (m, d) and (m, K)
            objective, gradients = self._objective_and_gradients(X_batch, y_batch)
            # Multiply by m because the last batch may be smaller than the others.
            total_loss += objective * batch.shape[0]
            self._update_parameters(*gradients)
        return total_loss / n_samples

    def _train(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "BaseSLPEstimator":
        """
        Training loop shared by the classifier and the regressor.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Validated training data
        y : array, shape (n_samples, n_outputs)
            Encoded targets

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # A local generator gives reproducible runs and leaves np.random untouched.
        rng = np.random.default_rng(self.random_state)

        # Hold out validation data only when early stopping is enabled.
        if self.early_stopping:
            X_train, y_train, X_val, y_val = self._split_validation(X, y, rng)
        else:
            X_train, y_train, X_val, y_val = X, y, None, None

        # Start from a clean state, so that calling fit() twice does not continue training.
        self.n_outputs_ = y.shape[1]
        self._initialize_weights(X.shape[1], self.n_outputs_, rng)
        self._initialize_optimizer()
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        self.best_validation_loss_ = None
        self.n_iter_ = 0
        self.diverged_ = False

        # None means full batch; a batch cannot be larger than the training set.
        n_train = X_train.shape[0]
        batch_size = n_train if self.batch_size is None else min(self.batch_size, n_train)

        best_parameters = None  # Weights at the lowest validation loss
        best_val_loss = np.inf  # Any finite loss is an improvement over this
        epochs_without_improvement = 0

        # Numerical errors are handled by the guard below, which warns only once.
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(self.max_iter):
                # Saved so that a diverging epoch can be undone.
                parameters_before = self._copy_parameters()
                epoch_loss = self._run_epoch(X_train, y_train, batch_size, rng)

                # Guard: stop if the loss or the weights are no longer finite.
                if not np.isfinite(epoch_loss) or not self._parameters_are_finite():
                    self._restore_parameters(parameters_before)
                    self.diverged_ = True
                    warnings.warn(
                        f"Training stopped at epoch {epoch + 1}: non-finite loss or weights. "
                        "Try a smaller learning_rate or standardize the data.",
                        RuntimeWarning,
                    )
                    break

                self.loss_curve_.append(float(epoch_loss))
                self.n_iter_ = epoch + 1

                if self.early_stopping:
                    val_prediction = self._forward_propagation(X_val)[3]  # shape: (n_val, K)
                    # The penalty is left out: this measures prediction quality only.
                    val_loss = self._compute_loss(y_val, val_prediction)
                    self.validation_loss_curve_.append(float(val_loss))
                    # Only a decrease larger than tol counts as an improvement.
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        best_parameters = self._copy_parameters()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    if epochs_without_improvement >= self.n_iter_no_change:
                        break

        # Keep the weights of the best epoch rather than those of the last one.
        if best_parameters is not None:
            self._restore_parameters(best_parameters)
            self.best_validation_loss_ = float(best_val_loss)
        return self

    # ------------------------------------------------------------------
    # Methods implemented by the subclasses
    # ------------------------------------------------------------------

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
