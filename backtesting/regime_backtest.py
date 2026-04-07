"""
backtesting/regime_backtest.py — Regime-Segmented Backtest.

Separates backtest results by market regime to identify:
  - Which regimes the strategy excels in
  - Which regimes the strategy should avoid
  - Regime-specific performance metrics

Integration: Used by the EvolutionEngine to decide which regime-specific
strategies to promote, and by MAMLAdapter to load regime-specific priors.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("RegimeBacktest")

_REGIME_NAMES = ["trending", "ranging", "volatile"]


def backtest_by_regime(
    data: pd.DataFrame,
    regimes: np.ndarray,
    trade_results: Optional[List[Dict]] = None,
) -> Dict:
    """
    Segment backtest results by market regime.

    Parameters
    ----------
    data          : full OHLCV DataFrame
    regimes       : 1-D integer or string array of regime labels, same length as data
    trade_results : optional pre-computed trade list (each with 'entry_i', 'exit_i', 'pnl', 'win')
                    If None, uses a simple buy-and-hold per regime.

    Returns
    -------
    dict mapping regime_name → performance metrics
    """
    if regimes is None or len(regimes) == 0:
        return {}

    # Map integer regimes to names if needed
    regime_arr = np.asarray(regimes)
    if regime_arr.dtype.kind in ("i", "u"):
        label_map = {0: "trending", 1: "ranging", 2: "volatile"}
        regime_labels = np.array([label_map.get(int(r), "ranging") for r in regime_arr])
    else:
        regime_labels = np.array([str(r) for r in regime_arr])

    result = {}

    for regime_name in _REGIME_NAMES:
        mask = regime_labels == regime_name
        if not mask.any():
            result[regime_name] = {"n_periods": 0, "n_trades": 0}
            continue

        n_periods = int(mask.sum())

        if trade_results:
            # Filter trades that occurred mostly during this regime
            regime_trades = []
            for trade in trade_results:
                entry_i = trade.get("entry_i", 0)
                exit_i = trade.get("exit_i", 0)
                if entry_i < len(regime_labels) and regime_labels[entry_i] == regime_name:
                    regime_trades.append(trade)

            if regime_trades:
                metrics = _compute_regime_metrics(regime_trades)
            else:
                metrics = {"n_trades": 0}
        else:
            # Simple buy-and-hold return during regime periods
            closes = data["close"].values.astype(float)
            regime_indices = np.where(mask)[0]
            if len(regime_indices) > 1:
                regime_close = closes[regime_indices]
                log_returns = np.diff(np.log(np.maximum(regime_close, 1e-8)))
                metrics = {
                    "n_trades": 0,
                    "avg_return": float(np.mean(log_returns)) if len(log_returns) > 0 else 0.0,
                    "volatility": float(np.std(log_returns)) if len(log_returns) > 0 else 0.0,
                    "buy_hold_return": float(closes[regime_indices[-1]] / closes[regime_indices[0]] - 1),
                }
            else:
                metrics = {"n_trades": 0}

        result[regime_name] = {
            "n_periods": n_periods,
            "pct_of_history": n_periods / max(len(regime_labels), 1),
            **metrics,
        }

    return result


def get_regime_performance(backtest_by_regime_result: Dict) -> Dict:
    """
    Extract and rank regime performance from backtest_by_regime() output.

    Returns
    -------
    dict with 'ranking', 'best_regime', 'worst_regime', 'per_regime_summary'
    """
    performance_scores = {}
    for regime, metrics in backtest_by_regime_result.items():
        n_trades = int(metrics.get("n_trades", 0))
        if n_trades == 0:
            # Use buy-hold return if available
            score = float(metrics.get("buy_hold_return", 0.0))
        else:
            # Composite score: win rate + Sharpe
            wr = float(metrics.get("win_rate", 0.5))
            sharpe = float(metrics.get("sharpe_ratio", 0.0))
            score = 0.6 * wr + 0.4 * max(sharpe, 0) / 3.0
        performance_scores[regime] = score

    if not performance_scores:
        return {}

    sorted_regimes = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "ranking": [{"regime": r, "score": s} for r, s in sorted_regimes],
        "best_regime": sorted_regimes[0][0] if sorted_regimes else None,
        "worst_regime": sorted_regimes[-1][0] if sorted_regimes else None,
        "per_regime_summary": backtest_by_regime_result,
    }


def best_regime(performance_result: Dict) -> Optional[str]:
    """Return the regime with the best performance."""
    return performance_result.get("best_regime")


def worst_regime(performance_result: Dict) -> Optional[str]:
    """Return the regime with the worst performance."""
    return performance_result.get("worst_regime")


# ---- Private helpers --------------------------------------------------------

def _compute_regime_metrics(trades: List[Dict]) -> Dict:
    """Compute performance metrics for a list of trades."""
    pnls = np.array([float(t.get("pnl", 0)) for t in trades])
    wins = [t for t in trades if t.get("win", False)]

    win_rate = len(wins) / max(len(trades), 1)
    total_return = float(np.sum(pnls))

    std = float(np.std(pnls))
    sharpe = float(np.mean(pnls) / std * np.sqrt(252)) if std > 1e-8 else 0.0

    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if not t.get("win", False)))
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "avg_pnl": float(np.mean(pnls)),
    }
