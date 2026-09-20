"""Ordinary least-squares exercises."""
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean of the squared residuals.

    Validate that the arrays are compatible. Do not call a library metric.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.shape != y_true.shape:
        raise ValueError("Shapes of y_pred and y_true must be equal.")
    return np.mean((y_pred - y_true)**2)


def fit_simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = intercept + slope*x and return ``(intercept, slope)``.

    Use the closed-form covariance/variance relationship. Consider what should
    happen when every x value is identical.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x and y must have the same dimensions.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if np.isclose(denominator, 0):
        raise ValueError("Cannot estimate a slope when x has no variation.")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return float(intercept), float(slope)


def predict(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Generate predictions from a fitted simple linear model."""
    return intercept + slope * np.asarray(x, dtype=float)


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return observed values minus predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")
    return y_true - y_pred


def matrix_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stretch: estimate coefficients for multiple predictors with matrix algebra.

    Add an intercept column deliberately and prefer a numerically stable linear
    solve or pseudoinverse over explicitly inverting a matrix.
    """
    X = np.asarray(X, dtype = "float")
    y = np.asarray(y, dtype = "float")

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if y.ndim != 1:
        raise ValueError("Y must be one-dimensional.")
    if X.shape[0] == 0 or X.shape[0] != len(y):
        raise ValueError("X and y must have the same nonzero number of observations.")

    intercept_col = np.ones((X.shape[0], 1))
    design_matrix = np.column_stack((intercept_col, X))
    # Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
    return np.linalg.pinv(design_matrix) @ y
