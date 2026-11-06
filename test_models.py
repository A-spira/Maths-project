"""
Model tests - Correctness, learning behaviour, and integration.

Covers classifier and regressor: parameter handling, weight shapes, output
shapes, probability validity, learning on realistic datasets, specific
learning problems (XOR, linear, constant), edge cases, reproducibility,
and a sanity comparison against sklearn.

Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data():
    """Scaled binary classification dataset (200 samples, 10 features)."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8, n_redundant=2,
        n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


@pytest.fixture
def multiclass_data():
    """Scaled 3-class classification dataset (300 samples, 15 features)."""
    X, y = make_classification(
        n_samples=300, n_features=15, n_informative=10, n_redundant=5,
        n_classes=3, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


@pytest.fixture
def regression_data():
    """Scaled regression dataset (200 samples, 10 features)."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=8, noise=10.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:

    def test_initialization(self):
        clf = SimpleSLPClassifier(
            hidden_layer_size=50, activation="relu",
            learning_rate=0.01, max_iter=100, random_state=42,
        )
        assert clf.hidden_layer_size == 50
        assert clf.activation == "relu"
        assert clf.learning_rate == 0.01
        assert clf.max_iter == 100
        assert clf.random_state == 42

    def test_weight_shapes(self, binary_data):
        X_train, _, y_train, _ = binary_data
        clf = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf.fit(X_train, y_train)
        assert clf.W1_.shape == (X_train.shape[1], 20)
        assert clf.b1_.shape == (20,)
        assert clf.W2_.shape[0] == 20
        assert clf.b2_.shape[0] == clf.n_outputs_

    def test_predict_shape_and_labels(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        clf = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        assert predictions.shape == (X_test.shape[0],)
        assert set(predictions).issubset(set(y_train))

    def test_predict_proba(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        clf = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], len(np.unique(y_train)))
        assert np.all(proba >= 0) and np.all(proba <= 1)
        np.testing.assert_array_almost_equal(
            np.sum(proba, axis=1), np.ones(X_test.shape[0])
        )

    def test_score(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        clf = SimpleSLPClassifier(hidden_layer_size=30, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.6

    def test_loss_decreases(self, binary_data):
        X_train, _, y_train, _ = binary_data
        clf = SimpleSLPClassifier(
            hidden_layer_size=30, learning_rate=0.01, max_iter=200, random_state=42
        )
        clf.fit(X_train, y_train)
        assert clf.loss_curve_[-1] < clf.loss_curve_[0]

    def test_multiclass(self, multiclass_data):
        X_train, X_test, y_train, y_test = multiclass_data
        clf = SimpleSLPClassifier(hidden_layer_size=40, max_iter=300, random_state=42)
        clf.fit(X_train, y_train)
        assert len(clf.classes_) == 3
        assert clf.score(X_test, y_test) > 0.5
        assert clf.predict_proba(X_test).shape[1] == 3

    def test_activations(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        for activation in ["relu", "tanh", "logistic"]:
            clf = SimpleSLPClassifier(
                hidden_layer_size=20, activation=activation,
                max_iter=100, random_state=42,
            )
            clf.fit(X_train, y_train)
            assert clf.score(X_test, y_test) > 0.5, f"Poor performance with {activation}"

    def test_reproducibility(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        clf1 = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf1.fit(X_train, y_train)
        clf2 = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf2.fit(X_train, y_train)
        np.testing.assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))

    def test_different_seeds_different_weights(self, binary_data):
        X_train, _, y_train, _ = binary_data
        clf1 = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=42)
        clf1.fit(X_train, y_train)
        clf2 = SimpleSLPClassifier(hidden_layer_size=20, max_iter=100, random_state=123)
        clf2.fit(X_train, y_train)
        assert not np.allclose(clf1.W1_, clf2.W1_)


# ---------------------------------------------------------------------------
# Classifier: specific learning problems
# ---------------------------------------------------------------------------


class TestClassifierProblems:

    def test_linearly_separable(self):
        """Two classes split cleanly by the first feature."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = SimpleSLPClassifier(
            hidden_layer_size=5, activation="relu",
            learning_rate=0.1, max_iter=100, random_state=42,
        )
        clf.fit(X, y)
        assert np.mean(clf.predict(X) == y) >= 0.75

    def test_all_same_class(self):
        """When all labels are the same, should predict that class."""
        X = np.random.randn(20, 3)
        y = np.zeros(20, dtype=int)
        clf = SimpleSLPClassifier(hidden_layer_size=5, max_iter=50, random_state=42)
        clf.fit(X, y)
        assert np.all(clf.predict(X) == 0)

    def test_xor(self):
        """XOR requires non-linear decision boundary."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        clf = SimpleSLPClassifier(
            hidden_layer_size=10, activation="relu",
            learning_rate=0.1, max_iter=500, random_state=42,
        )
        clf.fit(X, y)
        assert np.mean(clf.predict(X) == y) >= 0.75


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class TestRegressor:

    def test_initialization(self):
        reg = SimpleSLPRegressor(
            hidden_layer_size=50, activation="relu",
            learning_rate=0.01, max_iter=100, random_state=42,
        )
        assert reg.hidden_layer_size == 50
        assert reg.activation == "relu"
        assert reg.learning_rate == 0.01
        assert reg.max_iter == 100
        assert reg.random_state == 42

    def test_predict_shape_and_finite(self, regression_data):
        X_train, X_test, y_train, _ = regression_data
        reg = SimpleSLPRegressor(hidden_layer_size=20, max_iter=100, random_state=42)
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        assert predictions.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(predictions))

    def test_score(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        reg = SimpleSLPRegressor(
            hidden_layer_size=30, learning_rate=0.01, max_iter=300, random_state=42
        )
        reg.fit(X_train, y_train)
        assert reg.score(X_test, y_test) > 0.5

    def test_loss_decreases(self, regression_data):
        X_train, _, y_train, _ = regression_data
        reg = SimpleSLPRegressor(
            hidden_layer_size=30, learning_rate=0.01, max_iter=200, random_state=42
        )
        reg.fit(X_train, y_train)
        assert reg.loss_curve_[-1] < reg.loss_curve_[0]

    def test_constant_output(self):
        """Should converge to the constant target mean."""
        X = np.random.randn(30, 3)
        y = np.ones(30) * 5.0
        reg = SimpleSLPRegressor(
            hidden_layer_size=5, learning_rate=0.01, max_iter=100, random_state=42
        )
        reg.fit(X, y)
        assert 4.0 < np.mean(reg.predict(X)) < 6.0

    def test_linear_function(self):
        """Should achieve high R² on a clean linear relationship."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(200) * 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        reg = SimpleSLPRegressor(
            hidden_layer_size=20, learning_rate=0.01, max_iter=300, random_state=42
        )
        reg.fit(X_train, y_train)
        assert reg.score(X_test, y_test) > 0.8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_single_feature(self):
        X, y = np.random.randn(20, 1), np.random.randint(0, 2, 20)
        clf = SimpleSLPClassifier(hidden_layer_size=3, max_iter=50, random_state=42)
        clf.fit(X, y)
        assert clf.predict(X).shape == (20,)

    def test_many_features(self):
        X, y = np.random.randn(30, 50), np.random.randint(0, 2, 30)
        clf = SimpleSLPClassifier(hidden_layer_size=5, max_iter=20, random_state=42)
        clf.fit(X, y)
        assert clf.predict(X).shape == (30,)

    def test_single_sample_prediction(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = SimpleSLPClassifier(hidden_layer_size=10, max_iter=100, random_state=42)
        clf.fit(X, y)
        assert clf.predict(X[0:1]).shape == (1,)

    def test_small_hidden_layer(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = SimpleSLPClassifier(hidden_layer_size=2, max_iter=100, random_state=42)
        clf.fit(X, y)
        assert 0 <= clf.score(X, y) <= 1

    def test_large_hidden_layer(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = SimpleSLPClassifier(hidden_layer_size=200, max_iter=50, random_state=42)
        clf.fit(X, y)
        assert 0 <= clf.score(X, y) <= 1

    def test_regressor_minimal_data(self):
        X, y = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])
        reg = SimpleSLPRegressor(hidden_layer_size=3, max_iter=100, random_state=42)
        reg.fit(X, y)
        assert reg.predict(X).shape == (2,)


# ---------------------------------------------------------------------------
# Integration: sanity comparison with sklearn
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_classifier_reasonable_accuracy(self):
        """Our classifier should reach >60 % accuracy on a simple dataset."""
        from sklearn.neural_network import MLPClassifier

        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=8, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SimpleSLPClassifier(
            hidden_layer_size=30, activation="relu",
            learning_rate=0.01, max_iter=200, random_state=42,
        )
        clf.fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.6

    def test_regressor_reasonable_r2(self):
        """Our regressor should reach R² > 0.6 on a simple dataset."""
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=8, noise=10.0, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        reg = SimpleSLPRegressor(
            hidden_layer_size=30, activation="relu",
            learning_rate=0.01, max_iter=300, random_state=42,
        )
        reg.fit(X_train, y_train)
        assert reg.score(X_test, y_test) > 0.6
