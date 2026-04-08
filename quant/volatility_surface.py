"""
quant/volatility_surface.py — Advanced Volatility Models.

Implements:

1. **GARCH(1,1)** — Generalised AutoRegressive Conditional Heteroscedasticity
   - Estimates time-varying conditional volatility from returns
   - Parameters: omega, alpha (ARCH), beta (GARCH)
   - Fitted via MLE with quasi-Newton optimisation (scipy)

2. **Volatility Cones**
   - Percentiles of realised volatility across multiple horizons (5, 10, 20, 60)
   - Shows whether current vol is historically high/low on each horizon

3. **Term Structure**
   - Compares short-term (5-day) vs long-term (20-day) realised vol
   - Contango: short < long (normal), Backwardation: short > long (stress)

Integration: used by the Volatility Surface monitor and feeds into
VaR/CVaR calculations and regime-aware position sizing.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("VolatilitySurface")

_GARCH_WINDOWS = [5, 10, 20, 60]
_ANNUALISATION = 365  # crypto trades 24/7


def _realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling realised volatility (annualised)."""
    series = pd.Series(returns)
    rv = series.rolling(window).std().values * np.sqrt(_ANNUALISATION)
    return rv


class GARCHModel:
    """
    GARCH(1,1) model fitted via maximum likelihood.

    Model:
        r_t = sigma_t * z_t,   z_t ~ N(0,1)
        sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}

    Constraints:
        omega > 0,  alpha >= 0,  beta >= 0,  alpha + beta < 1
    """

    def __init__(self):
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self._is_fitted = False
        self._last_sigma2: Optional[float] = None
        self._last_r2: Optional[float] = None
        self._returns: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> bool:
        """
        Fit GARCH(1,1) parameters via MLE.

        Parameters
        ----------
        returns : 1-D array of log returns

        Returns
        -------
        bool: True if converged, False otherwise.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 30:
            return False

        # Initial parameter guess
        var0 = float(np.var(r))
        x0 = np.array([var0 * 0.05, 0.1, 0.85])

        def neg_loglik(params: np.ndarray) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.empty(len(r))
            sigma2[0] = var0
            for t in range(1, len(r)):
                sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
                if sigma2[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)
            return -ll

        bounds = [(1e-9, None), (0, 1), (0, 1)]
        try:
            res = minimize(
                neg_loglik, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200},
            )
            if res.success or res.fun < 1e9:
                self.omega, self.alpha, self.beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
                # Clamp alpha+beta < 1
                if self.alpha + self.beta >= 1.0:
                    self.alpha, self.beta = 0.05, 0.90
                self._returns = r.copy()
                # Compute final sigma²
                s2 = float(np.var(r))
                for t in range(1, len(r)):
                    s2 = self.omega + self.alpha * r[t - 1] ** 2 + self.beta * s2
                self._last_sigma2 = s2
                self._last_r2 = float(r[-1] ** 2)
                self._is_fitted = True
                return True
        except Exception as exc:
            logger.debug(f"GARCHModel.fit optimisation error: {exc}")
        return False

    def forecast_volatility(self, steps: int = 1) -> Optional[np.ndarray]:
        """
        Forecast conditional volatility for the next `steps` periods.

        Returns annualised daily volatility (sqrt of conditional variance).
        """
        if not self._is_fitted:
            return None
        s2 = self._last_sigma2
        r2 = self._last_r2
        forecasts = []
        long_run_var = self.omega / max(1 - self.alpha - self.beta, 1e-8)

        for h in range(1, steps + 1):
            if h == 1:
                s2_next = self.omega + self.alpha * r2 + self.beta * s2
            else:
                # Multi-step: converge to long-run variance
                s2_next = self.omega + (self.alpha + self.beta) ** (h - 1) * (
                    s2 - long_run_var
                ) + long_run_var
            forecasts.append(float(np.sqrt(max(s2_next, 0)) * np.sqrt(_ANNUALISATION)))

        return np.array(forecasts)

    def get_params(self) -> Dict:
        if not self._is_fitted:
            return {}
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.alpha + self.beta,
            "long_run_vol_annualised": float(
                np.sqrt(self.omega / max(1 - self.alpha - self.beta, 1e-8))
                * np.sqrt(_ANNUALISATION)
            ),
        }


# ---- Module-level functions -------------------------------------------------

def fit_garch(returns: np.ndarray) -> GARCHModel:
    """Fit and return a GARCHModel on the given return series."""
    model = GARCHModel()
    model.fit(returns)
    return model


def forecast_volatility(model: GARCHModel, steps: int = 5) -> Optional[np.ndarray]:
    """
    Return volatility forecast from a fitted GARCHModel.

    Parameters
    ----------
    model : fitted GARCHModel
    steps : forecast horizon in periods

    Returns
    -------
    np.ndarray of annualised volatility forecasts, or None.
    """
    return model.forecast_volatility(steps)


def volatility_cone(df: pd.DataFrame,
                    windows: List[int] = None) -> Optional[Dict]:
    """
    Compute volatility cones: for each window, calculate the
    [5th, 25th, 50th, 75th, 95th] percentiles of realised volatility
    over the full history, plus the current level.

    Parameters
    ----------
    df      : OHLCV DataFrame
    windows : list of rolling windows (default [5, 10, 20, 60])

    Returns
    -------
    dict keyed by window size, each containing percentile stats and current value.
    """
    if windows is None:
        windows = _GARCH_WINDOWS
    if df is None or len(df) < max(windows) + 10:
        return None

    try:
        close = df["close"].values.astype(float)
        log_ret = np.diff(np.log(np.maximum(close, 1e-8)))
        result = {}

        for w in windows:
            rv = _realized_vol(log_ret, window=w)
            rv_clean = rv[~np.isnan(rv)]
            if len(rv_clean) < 10:
                continue
            current_rv = float(rv_clean[-1]) if not np.isnan(rv_clean[-1]) else 0.0
            result[w] = {
                "current": current_rv,
                "p05": float(np.percentile(rv_clean, 5)),
                "p25": float(np.percentile(rv_clean, 25)),
                "p50": float(np.percentile(rv_clean, 50)),
                "p75": float(np.percentile(rv_clean, 75)),
                "p95": float(np.percentile(rv_clean, 95)),
                "pct_rank": float(
                    np.mean(rv_clean <= current_rv) * 100
                ),
            }

        return result if result else None
    except Exception as exc:
        logger.debug(f"volatility_cone error: {exc}")
        return None


def term_structure(df: pd.DataFrame) -> Optional[Dict]:
    """
    Compute volatility term structure: compare short vs long realised vol.

    Returns a dict with:
        'short_vol'       : float — 5-period annualised vol
        'long_vol'        : float — 20-period annualised vol
        'slope'           : float — (long - short) / long  (+ = contango, - = backwardation)
        'regime'          : str   — 'contango' | 'backwardation' | 'flat'
        'vol_ratio'       : float — short / long
    """
    if df is None or len(df) < 25:
        return None
    try:
        close = df["close"].values.astype(float)
        log_ret = np.diff(np.log(np.maximum(close, 1e-8)))

        short_vols = _realized_vol(log_ret, window=5)
        long_vols = _realized_vol(log_ret, window=20)

        # Use last non-NaN values
        sv = float(pd.Series(short_vols).dropna().iloc[-1]) if len(pd.Series(short_vols).dropna()) > 0 else 0.0
        lv = float(pd.Series(long_vols).dropna().iloc[-1]) if len(pd.Series(long_vols).dropna()) > 0 else 0.0

        if lv < 1e-8:
            return None

        slope = (lv - sv) / lv
        ratio = sv / lv

        if slope > 0.1:
            regime = "contango"
        elif slope < -0.1:
            regime = "backwardation"
        else:
            regime = "flat"

        return {
            "short_vol": sv,
            "long_vol": lv,
            "slope": slope,
            "vol_ratio": ratio,
            "regime": regime,
        }
    except Exception as exc:
        logger.debug(f"term_structure error: {exc}")
        return None
