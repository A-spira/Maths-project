"""
Smoke tests - Verify the pipeline runs end-to-end without errors.

These tests check that the code can be imported and that the basic
fit/predict cycle completes without crashing. Correctness is tested
in test_activations.py and test_models.py.

Run with: pytest tests/test_smoke.py -v
"""

import numpy as np
from activations import logistic, relu, softmax, tanh
from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


class TestActivationSmoke:
    """Verify activation functions are callable and return arrays."""

    def test_relu(self):
        assert relu(np.array([-1.0, 1.0])).shape == (2,)

    def test_logistic(self):
        assert logistic(np.array([0.0])).shape == (1,)

    def test_tanh(self):
        assert tanh(np.array([0.0])).shape == (1,)

    def test_softmax(self):
        assert softmax(np.array([[1.0, 2.0, 3.0]])).shape == (1, 3)


class TestClassifierSmoke:
    """Verify the classifier pipeline runs without errors."""

    def test_fit_predict(self):
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        clf = SimpleSLPClassifier(hidden_layer_size=5, max_iter=10, random_state=42)
        clf.fit(X, y)
        assert clf.predict(X).shape == (20,)
        assert clf.W1_ is not None
        assert len(clf.loss_curve_) > 0


class TestRegressorSmoke:
    """Verify the regressor pipeline runs without errors."""

    def test_fit_predict(self):
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        reg = SimpleSLPRegressor(hidden_layer_size=5, max_iter=10, random_state=42)
        reg.fit(X, y)
        predictions = reg.predict(X)
        assert predictions.shape == (20,)
        assert np.all(np.isfinite(predictions))
        assert reg.W1_ is not None
        assert len(reg.loss_curve_) > 0
