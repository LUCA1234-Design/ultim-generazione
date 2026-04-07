"""
risk_institutional/cvar_stress.py — Conditional VaR and Stress Testing.

Implements:

1. **CVaR / Expected Shortfall (ES)**
   - Mean of the worst (1-confidence) fraction of losses
   - More risk-sensitive than VaR: captures tail severity, not just threshold

2. **Stress Testing**
   - Predefined scenarios: crypto crash 2022, flash crash, correlation spike
   - Custom scenario generator from parameter overrides
   - Applies scenario shocks to portfolio and computes impact

Integration (Loop #13): VaR/CVaR → Kill Switch
  CVaR beyond thresholds triggers Level-3 or Level-5 kill switch.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("CVaRStress")

_MIN_SAMPLES = 20

# ---- Predefined stress scenarios -----------------------------------------------

_PREDEFINED_SCENARIOS: Dict[str, Dict] = {
    "crypto_crash_2022": {
        "description": "Crypto market collapse (May 2022): -80% peak-to-trough",
        "returns_shock": -0.30,        # -30% single-period shock
        "vol_multiplier": 5.0,
        "correlation_spike": 0.95,
        "n_periods": 5,
    },
    "flash_crash": {
        "description": "Flash crash: sudden -20% in single session",
        "returns_shock": -0.20,
        "vol_multiplier": 10.0,
        "correlation_spike": 0.99,
        "n_periods": 1,
    },
    "correlation_spike": {
        "description": "All assets move together (correlation → 0.90)",
        "returns_shock": -0.10,
        "vol_multiplier": 3.0,
        "correlation_spike": 0.90,
        "n_periods": 3,
    },
    "liquidity_crisis": {
        "description": "Liquidity crunch: wide bid-ask, forced liquidations",
        "returns_shock": -0.15,
        "vol_multiplier": 4.0,
        "correlation_spike": 0.85,
        "n_periods": 2,
        "slippage_mult": 10.0,
    },
    "regulatory_shock": {
        "description": "Regulatory ban/restriction announcement",
        "returns_shock": -0.25,
        "vol_multiplier": 6.0,
        "correlation_spike": 0.80,
        "n_periods": 2,
    },
}


def compute_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Compute CVaR (Expected Shortfall) at the given confidence level.

    CVaR = E[loss | loss > VaR(confidence)]
         = mean of worst (1-confidence) fraction of losses

    Parameters
    ----------
    returns    : 1-D array of portfolio returns (negative = loss)
    confidence : confidence level

    Returns
    -------
    float — CVaR as a positive expected loss value
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < _MIN_SAMPLES:
        return 0.0

    sorted_r = np.sort(r)
    n = len(sorted_r)
    cutoff_idx = max(1, int(np.floor(n * (1 - confidence))))
    tail = sorted_r[:cutoff_idx]  # worst returns
    cvar = float(-np.mean(tail)) if len(tail) > 0 else 0.0
    return max(cvar, 0.0)


def stress_test(
    portfolio: Dict,
    scenario: Dict,
) -> Dict:
    """
    Apply a stress scenario to a portfolio and compute impact.

    Parameters
    ----------
    portfolio : dict with:
        'positions'  : list of {symbol, size, direction, entry_price, current_price}
        'balance'    : total portfolio value
        'returns'    : recent return history (np.ndarray)
    scenario  : stress scenario dict (from predefined or custom)

    Returns
    -------
    dict with stress impact metrics
    """
    balance = float(portfolio.get("balance", 10000.0))
    positions = portfolio.get("positions", [])
    shock = float(scenario.get("returns_shock", -0.10))
    vol_mult = float(scenario.get("vol_multiplier", 3.0))
    corr_spike = float(scenario.get("correlation_spike", 0.9))
    n_periods = int(scenario.get("n_periods", 1))

    # Compound shock over n_periods
    compound_shock = (1 + shock) ** n_periods - 1

    # Gross portfolio impact (simplified: each position takes the shock)
    total_notional = sum(
        abs(float(p.get("size", 0))) * float(p.get("current_price", 0))
        for p in positions
    )

    # With correlation spike, diversification benefit disappears
    # Simple model: loss = sum of individual position losses * (1 + corr_spike)
    individual_loss = total_notional * abs(compound_shock)
    correlated_loss = individual_loss * (1 + corr_spike * 0.5)

    # Slippage multiplier for illiquid scenarios
    slippage_mult = float(scenario.get("slippage_mult", 1.0))
    total_loss = correlated_loss * slippage_mult

    # Volatility impact on remaining returns distribution
    base_returns = portfolio.get("returns", np.zeros(30))
    base_vol = float(np.std(base_returns)) if len(base_returns) > 1 else 0.01
    stressed_vol = base_vol * vol_mult

    pct_loss = total_loss / max(balance, 1e-8)

    return {
        "scenario_name": scenario.get("description", "Custom"),
        "compound_shock": compound_shock,
        "total_loss": total_loss,
        "pct_loss": pct_loss,
        "balance_after": balance - total_loss,
        "stressed_volatility": stressed_vol,
        "n_periods": n_periods,
        "survivable": pct_loss < 0.50,  # can survive if less than 50% loss
    }


def generate_scenario(params: Dict) -> Dict:
    """
    Create a custom stress scenario from parameters.

    Parameters
    ----------
    params : dict with optional keys:
        'returns_shock'    : float (negative = loss)
        'vol_multiplier'   : float
        'correlation_spike': float [0, 1]
        'n_periods'        : int
        'description'      : str

    Returns
    -------
    scenario dict suitable for stress_test()
    """
    return {
        "description": params.get("description", "Custom Scenario"),
        "returns_shock": float(params.get("returns_shock", -0.10)),
        "vol_multiplier": float(params.get("vol_multiplier", 2.0)),
        "correlation_spike": float(np.clip(params.get("correlation_spike", 0.7), 0, 1)),
        "n_periods": int(params.get("n_periods", 1)),
        "slippage_mult": float(params.get("slippage_mult", 1.0)),
    }


def get_predefined_scenarios() -> Dict[str, Dict]:
    """Return all predefined stress scenarios."""
    return dict(_PREDEFINED_SCENARIOS)


def run_all_scenarios(portfolio: Dict) -> Dict[str, Dict]:
    """
    Run all predefined stress scenarios against a portfolio.

    Returns
    -------
    dict mapping scenario_name → stress_test result
    """
    results = {}
    for name, scenario in _PREDEFINED_SCENARIOS.items():
        try:
            results[name] = stress_test(portfolio, scenario)
        except Exception as exc:
            logger.debug(f"stress_test error for {name}: {exc}")
            results[name] = {"error": str(exc)}
    return results
