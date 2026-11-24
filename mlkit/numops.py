"""Vectorised numerical helpers used across the toolkit."""
import numpy as np


def standardize(X):
    """Column-wise z-score standardisation. Returns (Z, mean, safe_std)."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)                       # shape (n_features,)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)          # avoid division by zero
    return (X - mean) / std, mean, std          # (n,d) - (d,) broadcasts row-wise


def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def log_loss(y_true, p, eps=1e-15):
    """Mean binary cross-entropy with clipping."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
