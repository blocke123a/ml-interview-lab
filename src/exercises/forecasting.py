"""Time-series forecasting exercises."""
import numpy as np

def naive_forecast(series: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Forecast every future step using the most recent observation."""
    series = np.asarray(series, float)
    if series.ndim != 1 or len(series) == 0 or horizon <= 0:
        raise ValueError("Use a nonempty 1D series and positive horizon.")
    return np.repeat(series[-1], horizon)

def seasonal_naive_forecast(series: np.ndarray, season_length: int, horizon: int) -> np.ndarray:
    """Repeat values from the most recently observed seasonal cycle."""
    series = np.asarray(series, float)
    if series.ndim != 1 or season_length <= 0 or horizon <= 0 or len(series) < season_length:
        raise ValueError("Invalid series, season length, or horizon.")
    return np.resize(series[-season_length:], horizon)

def moving_average_forecast(series: np.ndarray, window: int, horizon: int = 1) -> np.ndarray:
    """Forecast recursively from the mean of the latest window."""
    history = list(np.asarray(series, float))
    if window <= 0 or horizon <= 0 or len(history) < window:
        raise ValueError("Invalid window or horizon.")
    forecasts = []
    for _ in range(horizon):
        value = float(np.mean(history[-window:]))
        forecasts.append(value)
        history.append(value)
    return np.asarray(forecasts)

def simple_exponential_smoothing(series: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Return fitted one-step levels and the next forecast."""
    series = np.asarray(series, float)
    if series.ndim != 1 or len(series) == 0 or not 0 < alpha <= 1:
        raise ValueError("Use a nonempty series and alpha in (0, 1].")
    levels = np.empty_like(series)
    levels[0] = series[0]
    for index in range(1, len(series)):
        levels[index] = alpha * series[index] + (1 - alpha) * levels[index - 1]
    return levels, float(levels[-1])

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the average absolute forecast error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Inputs must have equal nonzero shapes.")
    return float(np.mean(np.abs(y_true - y_pred)))

def rolling_origin_splits(n_observations: int, initial_window: int, horizon: int, step: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window train/test index pairs without future leakage."""
    if min(n_observations, initial_window, horizon, step) <= 0 or initial_window + horizon > n_observations:
        raise ValueError("Invalid rolling-origin configuration.")
    splits = []
    train_end = initial_window
    while train_end + horizon <= n_observations:
        splits.append((np.arange(train_end), np.arange(train_end, train_end + horizon)))
        train_end += step
    return splits
