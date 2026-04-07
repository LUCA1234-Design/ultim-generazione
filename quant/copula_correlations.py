"""
quant/copula_correlations.py — Advanced Cross-Asset Correlation Analysis.

Implements:

1. **Tail Dependency** (empirical copula)
   - Lower tail dependency: P(X < q | Y < q) as q → 0
   - Upper tail dependency: P(X > q | Y > q) as q → 1
   - Measures co-movement in extreme events

2. **Regime-Switching Correlations**
   - Correlation computed separately for each market regime
   - Identifies which regimes drive correlation spikes

3. **Rolling Correlation with Adaptive Window**
   - Window shrinks when recent volatility is high (more responsive)
   - Window expands during calm periods (less noise)

Integration: used by CorrelationAgent and concentration_risk module.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("CopulaCorrelations")

_TAIL_Q = 0.1         # quantile threshold for tail dependency
_MIN_SAMPLES = 20
_WINDOW_BASE = 30
_WINDOW_MIN = 10
_WINDOW_MAX = 60


def _to_uniform(x: np.ndarray) -> np.ndarray:
    """Transform observations to uniform marginals (empirical CDF)."""
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1  # 1-based ranks
    return ranks / (n + 1)


def tail_dependency(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    q: float = _TAIL_Q
) -> Dict[str, float]:
    """
    Estimate empirical tail dependence coefficients between two return series.

    Lower tail lambda_L: how often both assets crash together.
    Upper tail lambda_U: how often both assets spike together.

    Parameters
    ----------
    returns_a, returns_b : 1-D arrays of equal length
    q                    : tail quantile (default 0.10 = bottom/top 10%)

    Returns
    -------
    dict with 'lambda_lower', 'lambda_upper', 'n_samples'
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)

    # Align and remove NaN
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) < _MIN_SAMPLES:
        return {"lambda_lower": 0.0, "lambda_upper": 0.0, "n_samples": len(a)}

    u = _to_uniform(a)
    v = _to_uniform(b)

    # Lower tail: P(U < q, V < q) / q
    lower_joint = np.mean((u < q) & (v < q))
    lambda_lower = float(lower_joint / q) if q > 0 else 0.0

    # Upper tail: P(U > 1-q, V > 1-q) / q
    upper_joint = np.mean((u > 1 - q) & (v > 1 - q))
    lambda_upper = float(upper_joint / q) if q > 0 else 0.0

    return {
        "lambda_lower": min(lambda_lower, 1.0),
        "lambda_upper": min(lambda_upper, 1.0),
        "n_samples": len(a),
    }


def regime_correlation(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    regimes: np.ndarray,
    regime_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Pearson correlation between returns_a and returns_b
    separately for each market regime.

    Parameters
    ----------
    returns_a   : 1-D array of returns for asset A
    returns_b   : 1-D array of returns for asset B
    regimes     : 1-D integer array of regime labels (same length)
    regime_names: optional list mapping regime indices to names

    Returns
    -------
    dict mapping regime name/index to correlation coefficient.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    r = np.asarray(regimes, dtype=int)

    # Align lengths
    n = min(len(a), len(b), len(r))
    a, b, r = a[:n], b[:n], r[:n]

    result: Dict[str, float] = {}
    for reg_id in np.unique(r):
        mask = r == reg_id
        if mask.sum() < 5:
            continue
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        name = (
            regime_names[reg_id] if regime_names and reg_id < len(regime_names)
            else f"regime_{reg_id}"
        )
        result[name] = corr

    return result


def rolling_correlation_adaptive(
    series_a: pd.Series,
    series_b: pd.Series,
    base_window: int = _WINDOW_BASE,
    min_window: int = _WINDOW_MIN,
    max_window: int = _WINDOW_MAX,
) -> Optional[pd.Series]:
    """
    Rolling Pearson correlation with volatility-adaptive window.

    The window shrinks when recent volatility of series_a is high
    (need more responsiveness) and expands during low-vol periods
    (can afford more averaging for stability).

    Parameters
    ----------
    series_a, series_b : pd.Series (must share index)
    base_window : nominal window size
    min_window  : minimum window size during high volatility
    max_window  : maximum window size during low volatility

    Returns
    -------
    pd.Series of rolling correlations aligned to series_a.index, or None.
    """
    try:
        a = series_a.values.astype(float)
        b = series_b.values.astype(float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        if n < min_window:
            return None

        # Compute rolling volatility of a to adapt window
        roll_std = pd.Series(a).rolling(base_window).std().values + 1e-8
        # Normalise to [0,1]
        std_min = np.nanmin(roll_std)
        std_max = np.nanmax(roll_std)
        if std_max - std_min < 1e-10:
            normalised = np.full(n, 0.5)
        else:
            normalised = (roll_std - std_min) / (std_max - std_min)
        # High vol → small window; low vol → large window
        adaptive_windows = (
            max_window - (max_window - min_window) * normalised
        ).astype(int)
        adaptive_windows = np.clip(adaptive_windows, min_window, max_window)

        correlations = np.full(n, np.nan)
        for i in range(n):
            w = int(adaptive_windows[i])
            if i < w - 1:
                continue
            sa = a[i - w + 1: i + 1]
            sb = b[i - w + 1: i + 1]
            if np.std(sa) < 1e-10 or np.std(sb) < 1e-10:
                correlations[i] = 0.0
                continue
            c = float(np.corrcoef(sa, sb)[0, 1])
            correlations[i] = c if np.isfinite(c) else 0.0

        return pd.Series(correlations, index=series_a.index[:n])
    except Exception as exc:
        logger.debug(f"rolling_correlation_adaptive error: {exc}")
        return None


def correlation_matrix(returns_dict: Dict[str, np.ndarray]) -> Optional[pd.DataFrame]:
    """
    Compute a correlation matrix from a dict of return series.

    Parameters
    ----------
    returns_dict : {symbol: np.ndarray of returns}

    Returns
    -------
    pd.DataFrame correlation matrix, or None.
    """
    try:
        symbols = list(returns_dict.keys())
        if len(symbols) < 2:
            return None

        # Find common length
        n = min(len(v) for v in returns_dict.values())
        data = {s: returns_dict[s][:n] for s in symbols}
        df = pd.DataFrame(data)
        return df.corr(method="pearson")
    except Exception as exc:
        logger.debug(f"correlation_matrix error: {exc}")
        return None
