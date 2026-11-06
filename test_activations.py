"""
Activation function tests - Correctness and numerical properties.

Tests every activation function and its derivative against the mathematical
definition, using hand-verifiable values.

Run with: pytest tests/test_activations.py -v
"""

import numpy as np
import pytest
from activations import (
    logistic,
    logistic_derivative,
    relu,
    relu_derivative,
    softmax,
    tanh,
    tanh_derivative,
)


class TestReLU:

    def test_negative_becomes_zero(self):
        np.testing.assert_array_equal(relu(np.array([-2.0, -1.0, 0.0])), [0.0, 0.0, 0.0])

    def test_positive_unchanged(self):
        np.testing.assert_array_equal(relu(np.array([1.0, 2.0, 5.0])), [1.0, 2.0, 5.0])

    def test_full_array(self):
        np.testing.assert_array_almost_equal(
            relu(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
            [0.0, 0.0, 0.0, 1.0, 2.0],
        )

    def test_derivative_negative(self):
        np.testing.assert_array_equal(
            relu_derivative(np.array([-2.0, -1.0, 0.0])), [0, 0, 0]
        )

    def test_derivative_positive(self):
        np.testing.assert_array_equal(
            relu_derivative(np.array([1.0, 2.0, 5.0])), [1, 1, 1]
        )

    def test_derivative_full_array(self):
        np.testing.assert_array_almost_equal(
            relu_derivative(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
            [0, 0, 0, 1, 1],
        )


class TestLogistic:

    def test_at_zero(self):
        np.testing.assert_almost_equal(logistic(np.array([0.0]))[0], 0.5)

    def test_formula(self):
        z = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(logistic(z), 1 / (1 + np.exp(-z)))

    def test_output_range(self):
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = logistic(z)
        assert np.all(result > 0) and np.all(result < 1)

    def test_derivative_at_zero(self):
        np.testing.assert_almost_equal(logistic_derivative(np.array([0.0]))[0], 0.25)

    def test_derivative_formula(self):
        z = np.array([0.0, 1.0, -1.0])
        sig = 1 / (1 + np.exp(-z))
        np.testing.assert_array_almost_equal(logistic_derivative(z), sig * (1 - sig))


class TestTanh:

    def test_at_zero(self):
        np.testing.assert_almost_equal(tanh(np.array([0.0]))[0], 0.0)

    def test_formula(self):
        z = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(tanh(z), np.tanh(z))

    def test_derivative_at_zero(self):
        np.testing.assert_almost_equal(tanh_derivative(np.array([0.0]))[0], 1.0)

    def test_derivative_formula(self):
        z = np.array([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(tanh_derivative(z), 1 - np.tanh(z) ** 2)


class TestSoftmax:

    def test_sums_to_one(self):
        np.testing.assert_almost_equal(
            np.sum(softmax(np.array([[1.0, 2.0, 3.0]]))), 1.0
        )

    def test_uniform_input(self):
        np.testing.assert_array_almost_equal(
            softmax(np.array([[1.0, 1.0, 1.0]])), [[1 / 3, 1 / 3, 1 / 3]]
        )

    def test_ordering(self):
        result = softmax(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
        assert np.all(result[:, 2] > result[:, 1])
        assert np.all(result[:, 1] > result[:, 0])

    def test_all_positive(self):
        assert np.all(softmax(np.array([[1.0, 2.0, 3.0]])) > 0)

    def test_batch_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        np.testing.assert_array_almost_equal(np.sum(softmax(z), axis=1), [1.0, 1.0])

    def test_numerical_stability(self):
        """Large inputs should not produce NaN or inf."""
        result = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_almost_equal(np.sum(result), 1.0)
