"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # TODO: Implement ReLU
    pass


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # TODO: Implement ReLU derivative
    pass


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # TODO: Implement tanh (can use np.tanh)
    pass


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # TODO: Implement tanh derivative
    pass


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # TODO: Implement sigmoid
    # Hint: Use np.clip to avoid overflow
    pass


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # TODO: Implement sigmoid derivative
    pass


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # TODO: Implement softmax
    pass
