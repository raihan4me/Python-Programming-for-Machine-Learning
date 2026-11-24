import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from mlkit.numops import log_loss, sigmoid, standardize


def test_sigmoid_zero():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_extremes_no_warnings():
    z = np.array([-1000.0, 0.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any RuntimeWarning fails the test
        out = sigmoid(z)
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_sigmoid_preserves_shape():
    z = np.zeros((4, 3))
    assert sigmoid(z).shape == (4, 3)


def test_standardize_mean_std():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=[0, 50, -10], scale=[1, 5, 20], size=(1000, 3))
    Z, mean, std = standardize(X)
    assert Z.shape == X.shape
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(mean, X.mean(axis=0))


def test_standardize_constant_column():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Z, mean, std = standardize(X)
    assert np.all(np.isfinite(Z))
    assert np.all(Z[:, 1] == 0.0)
    assert std[1] == 1.0


def test_log_loss_matches_sklearn():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.6, 0.3, 0.99])
    assert log_loss(y, p) == pytest.approx(sk_log_loss(y, p))
    assert np.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))
