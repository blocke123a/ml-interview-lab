"""Conjugate Bayesian-updating exercises."""
from scipy.stats import beta as beta_distribution

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

def credible_interval(alpha: float, beta: float, level: float = 0.95) -> tuple[float, float]:
    """Stretch: return an equal-tailed interval by converting level to Beta quantiles."""
    if alpha <= 0 or beta <= 0:
                raise ValueError("Beta parameters must be positive.")
    if not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1.")
    tail = (1 - level) / 2
    lower, upper = beta_distribution.ppf([tail, 1 - tail], alpha, beta)
    return float(lower), float(upper)