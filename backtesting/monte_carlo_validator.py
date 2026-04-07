"""
backtesting/monte_carlo_validator.py — Monte Carlo Strategy Validation.

Validates strategy robustness by:
1. Bootstrapping N permutations of the trade result sequence
2. Computing confidence intervals for Sharpe, win rate, max drawdown
3. Determining if the strategy is robust at a given confidence level

A strategy is considered robust if:
  - Lower CI bound of Sharpe > 0.5
  - Lower CI bound of win rate > 0.45
  - Upper CI bound of max drawdown < 0.20

Integration (Loop #15): results feed into StrategyEvolver to gate
which strategies can be promoted to live trading.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MonteCarloValidator")

_DEFAULT_N_SIMS = 1000
_DEFAULT_CONFIDENCE = 0.95
_ROBUST_MIN_SHARPE = 0.5
_ROBUST_MIN_WIN_RATE = 0.45
_ROBUST_MAX_DRAWDOWN = 0.20


def validate_strategy(
    trade_results: List[Dict],
    n_sims: int = _DEFAULT_N_SIMS,
) -> Dict:
    """
    Validate strategy robustness via Monte Carlo simulation.

    Shuffles the trade sequence N times and computes performance
    metrics for each permutation. Returns confidence intervals.

    Parameters
    ----------
    trade_results : list of trade dicts with 'pnl' and 'win' keys
    n_sims        : number of Monte Carlo simulations

    Returns
    -------
    dict with confidence intervals and robustness flags
    """
    if not trade_results or len(trade_results) < 10:
        return {
            "error": "Insufficient trades for validation",
            "n_trades": len(trade_results),
        }

    pnls = np.array([float(t.get("pnl", 0)) for t in trade_results])
    wins = np.array([float(t.get("win", (t.get("pnl", 0) > 0))) for t in trade_results])

    sim_sharpes = []
    sim_win_rates = []
    sim_max_drawdowns = []

    for _ in range(n_sims):
        # Bootstrap resample (with replacement)
        idx = np.random.choice(len(pnls), size=len(pnls), replace=True)
        sim_pnl = pnls[idx]
        sim_win = wins[idx]

        # Sharpe
        std = float(np.std(sim_pnl))
        sharpe = float(np.mean(sim_pnl) / std * np.sqrt(252)) if std > 1e-8 else 0.0
        sim_sharpes.append(sharpe)

        # Win rate
        sim_win_rates.append(float(np.mean(sim_win)))

        # Max drawdown from equity curve
        equity = np.cumsum(sim_pnl)
        equity = equity - equity.min() + abs(equity.min()) + 1  # shift to positive
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.max(1 - equity / np.maximum(peak, 1e-8)))
        sim_max_drawdowns.append(max_dd)

    sharpes_arr = np.array(sim_sharpes)
    wr_arr = np.array(sim_win_rates)
    dd_arr = np.array(sim_max_drawdowns)

    return {
        "n_trades": len(pnls),
        "n_simulations": n_sims,
        "sharpe": {
            "mean": float(np.mean(sharpes_arr)),
            "std": float(np.std(sharpes_arr)),
            "ci_lower_95": float(np.percentile(sharpes_arr, 2.5)),
            "ci_upper_95": float(np.percentile(sharpes_arr, 97.5)),
        },
        "win_rate": {
            "mean": float(np.mean(wr_arr)),
            "std": float(np.std(wr_arr)),
            "ci_lower_95": float(np.percentile(wr_arr, 2.5)),
            "ci_upper_95": float(np.percentile(wr_arr, 97.5)),
        },
        "max_drawdown": {
            "mean": float(np.mean(dd_arr)),
            "std": float(np.std(dd_arr)),
            "ci_lower_95": float(np.percentile(dd_arr, 2.5)),
            "ci_upper_95": float(np.percentile(dd_arr, 97.5)),
        },
    }


def get_confidence_intervals(validation_result: Dict) -> Dict[str, Tuple[float, float]]:
    """
    Extract confidence intervals from a validation result.

    Returns
    -------
    dict mapping metric_name → (lower_ci, upper_ci)
    """
    intervals = {}
    for metric in ["sharpe", "win_rate", "max_drawdown"]:
        m = validation_result.get(metric, {})
        intervals[metric] = (
            float(m.get("ci_lower_95", 0)),
            float(m.get("ci_upper_95", 1)),
        )
    return intervals


def is_strategy_robust(
    validation_result: Dict,
    min_sharpe: float = _ROBUST_MIN_SHARPE,
    min_win_rate: float = _ROBUST_MIN_WIN_RATE,
    max_drawdown: float = _ROBUST_MAX_DRAWDOWN,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> bool:
    """
    Determine if a strategy is robust enough for live trading.

    A strategy is robust if its lower confidence bound exceeds
    minimum thresholds.

    Parameters
    ----------
    validation_result : output from validate_strategy()
    min_sharpe        : minimum acceptable Sharpe ratio (lower CI bound)
    min_win_rate      : minimum acceptable win rate (lower CI bound)
    max_drawdown      : maximum acceptable drawdown (upper CI bound)

    Returns
    -------
    bool — True if strategy passes all robustness checks
    """
    if "error" in validation_result:
        return False

    intervals = get_confidence_intervals(validation_result)

    sharpe_lo, _ = intervals.get("sharpe", (0, 0))
    wr_lo, _ = intervals.get("win_rate", (0, 0))
    _, dd_hi = intervals.get("max_drawdown", (0, 1))

    is_robust = (
        sharpe_lo >= min_sharpe and
        wr_lo >= min_win_rate and
        dd_hi <= max_drawdown
    )

    if not is_robust:
        logger.debug(
            f"Strategy NOT robust: sharpe_lo={sharpe_lo:.3f} (min={min_sharpe}), "
            f"wr_lo={wr_lo:.3f} (min={min_win_rate}), "
            f"dd_hi={dd_hi:.3f} (max={max_drawdown})"
        )

    return is_robust
