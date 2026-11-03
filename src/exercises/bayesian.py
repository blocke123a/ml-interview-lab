"""Conjugate Bayesian-updating exercises."""
import math
import numpy as np

def beta_binomial_update(prior_alpha: float, prior_beta: float, successes: int, failures: int) -> tuple[float, float]:
    """Return posterior Beta parameters; use conjugacy and count both outcomes."""
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive.")
    if successes < 0 or failures < 0:
        raise ValueError("Observation counts cannot be negative.")
    return prior_alpha + successes, prior_beta + failures

def posterior_mean(alpha: float, beta: float) -> float:
    """Return the expected probability under a Beta(alpha, beta) distribution."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")
    return alpha / (alpha + beta)


def beta_pdf(x, alpha: float, beta: float) -> np.ndarray:
    """Evaluate the Beta density with NumPy, without scipy.stats."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")
    x = np.asarray(x, dtype=float)
    coefficient = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
    return coefficient * np.power(x, alpha - 1) * np.power(1 - x, beta - 1)


def _beta_cdf(x: float, alpha: float, beta: float, steps: int = 20000) -> float:
    """Numerically integrate the Beta density for credible intervals."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    grid = np.linspace(0, x, steps)
    density = beta_pdf(grid, alpha, beta)
    return float(np.trapezoid(density, grid)) if hasattr(np, "trapezoid") else float(np.trapz(density, grid))


def _beta_ppf(probability: float, alpha: float, beta: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(50):
        midpoint = (low + high) / 2
        if _beta_cdf(midpoint, alpha, beta) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def credible_interval(alpha: float, beta: float, level: float = 0.95) -> tuple[float, float]:
    """Return an equal-tailed interval by numerically computing Beta quantiles."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1.")
    tail = (1 - level) / 2
    lower = _beta_ppf(tail, alpha, beta)
    upper = _beta_ppf(1 - tail, alpha, beta)
    return float(lower), float(upper)