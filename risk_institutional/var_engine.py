"""
risk_institutional/var_engine.py — Value-at-Risk Calculator.

Implements three VaR estimation methods:

1. **Historical VaR**: empirical percentile of the loss distribution
2. **Parametric VaR**: assumes normally distributed returns (Gaussian)
3. **Monte Carlo VaR**: simulates N paths from estimated distribution

All methods output VaR at specified confidence levels (95%, 99%).
A negative VaR value indicates a loss at that confidence level.

Integration (Loop #13): VaR/CVaR → Kill Switch
  Results feed into kill_switch.py to determine if portfolio risk
  exceeds institutional limits and whether to trigger circuit breakers.
"""
import logging
from typing import Dict, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger("VaREngine")

_DEFAULT_CONFIDENCES = (0.95, 0.99)
_MIN_SAMPLES = 20
_MC_SIMS = 10_000


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Historical VaR at given confidence level.

    VaR = -percentile(returns, 1-confidence)
    A positive value = max expected loss at confidence level.

    Parameters
    ----------
    returns    : 1-D array of portfolio returns (fractions)
    confidence : confidence level, e.g. 0.95 for 95%

    Returns
    -------
    float — VaR as a positive loss value (0.05 = 5% loss)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < _MIN_SAMPLES:
        return 0.0
    percentile = 100.0 * (1.0 - confidence)
    var = float(-np.percentile(r, percentile))
    return max(var, 0.0)


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Parametric VaR assuming normally distributed returns.

    VaR = -(mu + z * sigma)  where z is the confidence z-score.

    Parameters
    ----------
    returns    : 1-D array of portfolio returns
    confidence : confidence level

    Returns
    -------
    float — VaR as positive loss value
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < _MIN_SAMPLES:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r))
    z = float(stats.norm.ppf(1.0 - confidence))
    var = -(mu + z * sigma)
    return max(var, 0.0)


def monte_carlo_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_sims: int = _MC_SIMS,
) -> float:
    """
    Monte Carlo VaR by simulating from estimated return distribution.

    Uses a Student-t distribution to capture fat tails.

    Parameters
    ----------
    returns    : 1-D array of historical returns
    confidence : confidence level
    n_sims     : number of Monte Carlo simulations

    Returns
    -------
    float — VaR as positive loss value
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < _MIN_SAMPLES:
        return 0.0

    mu = float(np.mean(r))
    sigma = float(np.std(r))

    # Fit Student-t degrees of freedom from excess kurtosis
    excess_kurt = float(stats.kurtosis(r))
    if excess_kurt > 0.5:
        nu = max(4.5, 6.0 / excess_kurt + 4.0)
    else:
        nu = 30.0

    # Generate MC samples
    t_samples = stats.t.rvs(df=nu, loc=mu, scale=sigma, size=n_sims)
    percentile = 100.0 * (1.0 - confidence)
    var = float(-np.percentile(t_samples, percentile))
    return max(var, 0.0)


def compute_all(
    returns: np.ndarray,
    confidences: tuple = _DEFAULT_CONFIDENCES,
) -> Dict:
    """
    Compute all VaR estimates for multiple confidence levels.

    Parameters
    ----------
    returns     : 1-D array of portfolio returns
    confidences : tuple of confidence levels to compute

    Returns
    -------
    dict with keys like 'historical_95', 'parametric_99', etc.
    """
    result: Dict = {}
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < _MIN_SAMPLES:
        logger.warning(f"VaREngine: insufficient data ({len(r)} < {_MIN_SAMPLES})")
        for c in confidences:
            pct = int(c * 100)
            result[f"historical_{pct}"] = 0.0
            result[f"parametric_{pct}"] = 0.0
            result[f"monte_carlo_{pct}"] = 0.0
        result["n_samples"] = len(r)
        return result

    for c in confidences:
        pct = int(c * 100)
        result[f"historical_{pct}"] = historical_var(r, c)
        result[f"parametric_{pct}"] = parametric_var(r, c)
        result[f"monte_carlo_{pct}"] = monte_carlo_var(r, c)

    result["n_samples"] = len(r)
    result["mean_return"] = float(np.mean(r))
    result["volatility"] = float(np.std(r))

    return result
