"""Math exercises for the app.

Implement these yourself before comparing against NumPy / scikit-learn.
"""
from __future__ import annotations
import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TODO: Implement MSE from the mathematical definition."""
    raise NotImplementedError("Implement mean_squared_error yourself.")

def sigmoid(z: np.ndarray) -> np.ndarray:
    """TODO: Implement a numerically stable sigmoid."""
    raise NotImplementedError("Implement sigmoid yourself.")

def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """TODO: Implement BCE. Think carefully about probabilities near 0 and 1."""
    raise NotImplementedError("Implement binary_cross_entropy yourself.")

def numerical_gradient(fn, theta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """OPTIONAL challenge: finite-difference gradient checker."""
    raise NotImplementedError("Optional: implement numerical_gradient.")
