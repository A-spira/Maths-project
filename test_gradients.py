"""
Additional tests (bonus) - Gradient checks and behaviour of the bonus features.

The backpropagation gradients are compared with centred finite differences of the
objective, for every parameter, activation, output type and L2 strength. The file
also covers L2 regularization, the identity activation and R² on a constant target.

Run with: pytest test_gradients.py -v
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from activations import identity, identity_derivative
from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor

ACTIVATIONS = ["identity", "logistic", "tanh", "relu"]
ALPHAS = [0.0, 0.5]
STEP = 1e-6  # Finite-difference step
TOLERANCE = 1e-6  # Largest accepted gap between the two gradients


def numerical_gradient(model, parameter, X, y):
    """Centred finite differences of the objective, one parameter value at a time."""
    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + STEP
        objective_plus, _ = model._objective_and_gradients(X, y)
        parameter[index] = original - STEP
        objective_minus, _ = model._objective_and_gradients(X, y)
        parameter[index] = original  # Put the value back before the next one
        gradient[index] = (objective_plus - objective_minus) / (2 * STEP)
    return gradient


def assert_gradients_match(model, X, y):
    """Check the four analytic gradients against finite differences."""
    _, analytic = model._objective_and_gradients(X, y)
    parameters = (model.W1_, model.b1_, model.W2_, model.b2_)
    for name, parameter, gradient in zip(("W1", "b1", "W2", "b2"), parameters, analytic):
        numeric = numerical_gradient(model, parameter, X, y)
        gap = np.max(np.abs(numeric - gradient))
        assert gap < TOLERANCE, f"Gradient of {name} is off by {gap:.2e}"


def initialized_model(model, n_features, n_outputs):
    """Prepare a model's weights without training it."""
    model._validate_params()
    model.n_outputs_ = n_outputs
    model._initialize_weights(n_features, n_outputs, np.random.default_rng(1))
    return model


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(20, 4))


class TestRegressorGradients:

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    @pytest.mark.parametrize("n_outputs", [1, 3])
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_matches_finite_differences(self, X, activation, n_outputs, alpha):
        y = np.random.default_rng(2).normal(size=(20, n_outputs))
        model = SimpleSLPRegressor(hidden_layer_size=6, activation=activation, alpha=alpha)
        assert_gradients_match(initialized_model(model, 4, n_outputs), X, y)


class TestClassifierGradients:

    @pytest.mark.parametrize("activation", ACTIVATIONS)
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_matches_finite_differences(self, X, activation, n_classes, alpha):
        labels = np.arange(20) % n_classes  # Every class is present
        model = SimpleSLPClassifier(hidden_layer_size=6, activation=activation, alpha=alpha)
        y = model._encode_labels(labels)
        assert_gradients_match(initialized_model(model, 4, y.shape[1]), X, y)


class TestL2Regularization:

    def test_objective_adds_expected_penalty(self, X):
        y = np.ones((20, 1))
        model = initialized_model(SimpleSLPRegressor(hidden_layer_size=3, alpha=2.0), 4, 1)
        objective, _ = model._objective_and_gradients(X, y)
        loss = model._compute_loss(y, model._forward_propagation(X)[3])
        penalty = 2.0 / (2 * 20) * (np.sum(model.W1_**2) + np.sum(model.W2_**2))
        assert objective == pytest.approx(loss + penalty)

    def test_biases_are_not_penalized(self, X):
        y = np.ones((20, 1))
        plain = initialized_model(SimpleSLPRegressor(hidden_layer_size=3, alpha=0.0), 4, 1)
        strong = initialized_model(SimpleSLPRegressor(hidden_layer_size=3, alpha=9.0), 4, 1)
        _, plain_gradients = plain._objective_and_gradients(X, y)
        _, strong_gradients = strong._objective_and_gradients(X, y)
        np.testing.assert_array_equal(plain_gradients[1], strong_gradients[1])  # db1
        np.testing.assert_array_equal(plain_gradients[3], strong_gradients[3])  # db2

    def test_alpha_shrinks_weights(self, X):
        y = X[:, 0] - 2 * X[:, 1]
        settings = dict(hidden_layer_size=10, learning_rate=0.01, max_iter=100, random_state=0)
        plain = SimpleSLPRegressor(alpha=0.0, **settings).fit(X, y)
        strong = SimpleSLPRegressor(alpha=10.0, **settings).fit(X, y)
        assert np.sum(strong.W1_**2) < np.sum(plain.W1_**2)

    def test_negative_alpha_is_rejected(self, X):
        with pytest.raises(ValueError):
            SimpleSLPRegressor(alpha=-1.0).fit(X, np.ones(20))


class TestIdentityActivation:

    def test_values_and_derivative(self):
        z = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_array_equal(identity(z), z)
        np.testing.assert_array_equal(identity_derivative(z), [1.0, 1.0, 1.0])

    def test_learns_a_linear_function(self):
        X = np.random.default_rng(0).normal(size=(100, 3))
        y = 2 * X[:, 0] - X[:, 1] + 3 * X[:, 2]
        model = SimpleSLPRegressor(
            hidden_layer_size=10, activation="identity", learning_rate=0.01, random_state=0
        ).fit(X, y)
        assert model.score(X, y) > 0.99

    def test_cannot_solve_xor(self):
        """An affine decision boundary classifies at most 3 of the 4 XOR points."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        model = SimpleSLPClassifier(
            hidden_layer_size=10, activation="identity",
            learning_rate=0.1, max_iter=500, random_state=42,
        ).fit(X, y)
        assert model.score(X, y) <= 0.75


class TestR2OnConstantTarget:

    @staticmethod
    def fitted_regressor(constant, n_outputs=1):
        """A regressor with zero weights, which always predicts `constant` exactly."""
        model = SimpleSLPRegressor(hidden_layer_size=2)
        model.W1_, model.b1_ = np.zeros((1, 2)), np.zeros(2)
        model.W2_, model.b2_ = np.zeros((2, n_outputs)), np.full(n_outputs, constant)
        model.n_features_in_, model.n_outputs_ = 1, n_outputs
        return model

    # 0.1 is the case where rounding leaves SS_tot slightly above 0.
    @pytest.mark.parametrize("value", [5.0, 0.1])
    def test_exact_prediction_gives_one(self, value):
        model = self.fitted_regressor(value)
        assert model.score(np.zeros((30, 1)), np.full(30, value)) == 1.0

    @pytest.mark.parametrize("value", [5.0, 0.1])
    def test_wrong_prediction_gives_zero(self, value):
        model = self.fitted_regressor(value + 1.0)
        assert model.score(np.zeros((30, 1)), np.full(30, value)) == 0.0

    def test_one_constant_output_among_two(self):
        model = self.fitted_regressor(1.0, n_outputs=2)
        y = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
        expected = (1.0 + r2_score(y[:, 1], np.ones(4))) / 2
        assert model.score(np.zeros((4, 1)), y) == pytest.approx(expected)

    def test_matches_sklearn_when_y_varies(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(50, 2)), rng.normal(size=50)
        model = SimpleSLPRegressor(hidden_layer_size=4, max_iter=5, random_state=0).fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))
