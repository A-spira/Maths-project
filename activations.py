"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray

# Largest |z| passed to exp() in the logistic function.
# exp() overflows a 64-bit float just above 709, so 500 stays safely below that limit.
LOGISTIC_CLIP: float = 500.0


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function: relu(z) = max(0, z).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values, same shape as z
    """
    # Element-wise max(0, z): negative values become 0, others are unchanged.
    return np.maximum(0.0, z)  # shape: same as z


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation: 1 if z > 0, otherwise 0.

    ReLU is not differentiable at z = 0. By convention, 0 is used at that point.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values, same shape as z
    """
    # The slope is 1 on the positive side and 0 elsewhere, including z = 0.
    return np.where(z > 0, 1.0, 0.0)  # shape: same as z


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function: tanh(z) = sinh(z) / cosh(z).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values in [-1, 1], same shape as z
    """
    # NumPy's tanh stays finite for large |z|, so no clipping is needed.
    return np.tanh(z)  # shape: same as z


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation: tanh'(z) = 1 - tanh(z)^2.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values, same shape as z
    """
    # Quotient rule on sinh/cosh gives (cosh^2 - sinh^2) / cosh^2 = 1 - tanh^2.
    return 1.0 - np.tanh(z) ** 2  # shape: same as z


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function: sigma(z) = 1 / (1 + exp(-z)).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values in [0, 1], same shape as z
    """
    # Bound z so that exp(-z) cannot overflow when z is very negative.
    z_clipped = np.clip(z, -LOGISTIC_CLIP, LOGISTIC_CLIP)  # shape: same as z
    # Apply the sigmoid formula element by element.
    return 1.0 / (1.0 + np.exp(-z_clipped))  # shape: same as z


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation: sigma'(z) = sigma(z) * (1 - sigma(z)).

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values, same shape as z
    """
    # Reuse logistic() so that the derivative gets the same overflow protection.
    sigma = logistic(z)  # shape: same as z
    # Largest value is 0.25 at z = 0; it tends to 0 when |z| is large (saturation).
    return sigma * (1.0 - sigma)  # shape: same as z


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    softmax(z)_k = exp(z_k) / sum_j exp(z_j), computed separately for each sample.

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # Subtracting a constant from one row leaves its softmax unchanged.
    # With the row maximum, the largest exponent is 0, so exp() cannot overflow.
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # (n, K) - (n, 1) -> (n, K)
    # Numerators of the softmax formula.
    exp_z = np.exp(z_shifted)  # shape: (n, K)
    # Divide each row by its sum; keepdims lets (n, 1) broadcast over (n, K).
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # shape: (n, K)
