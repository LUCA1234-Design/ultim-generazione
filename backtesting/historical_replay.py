"""
backtesting/historical_replay.py — Walk-Forward Backtest Engine.

Replays historical OHLCV data through a simplified version of the
agent pipeline and measures performance metrics.

Supports:
  - Single-pass backtest over full history
  - Walk-forward validation (train/test splits)
  - Configurable slippage and commission

Integration (Loop #15): Backtest Validator → Strategy Evolver
  Results feed back into StrategyEvolver to promote only strategies
  that show robust walk-forward validation.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("HistoricalReplay")

_DEFAULT_SLIPPAGE = 0.0005   # 0.05% slippage per trade
_DEFAULT_COMMISSION = 0.0006  # 0.06% commission per side
_WALK_FORWARD_FOLDS = 5
_TRAIN_PCT = 0.70


def run_backtest(
    data: pd.DataFrame,
    config: Dict,
    signal_fn: Optional[Callable] = None,
) -> Dict:
    """
    Run a single backtest pass over historical data.

    Parameters
    ----------
    data      : OHLCV DataFrame
    config    : dict with:
        'initial_balance' : float
        'slippage'        : float (fraction)
        'commission'      : float (fraction per side)
        'max_positions'   : int
        'sl_pct'          : float stop-loss %
        'tp_pct'          : float take-profit %
    signal_fn : optional callable(df, i) → (direction, size)
                If None, uses a simple RSI-based signal

    Returns
    -------
    dict with backtest performance metrics
    """
    initial_balance = float(config.get("initial_balance", 10000.0))
    slippage = float(config.get("slippage", _DEFAULT_SLIPPAGE))
    commission = float(config.get("commission", _DEFAULT_COMMISSION))
    sl_pct = float(config.get("sl_pct", 0.02))     # 2% stop loss
    tp_pct = float(config.get("tp_pct", 0.04))     # 4% take profit

    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    equity_curve = [initial_balance]

    position = None  # {'side': 'long'/'short', 'entry': price, 'size': fraction}

    close_prices = data["close"].values.astype(float)
    # Use intracandle high/low for SL/TP evaluation to avoid look-ahead bias.
    # A stop-loss at, say, 2 % below entry may be hit by the candle's low even
    # if the candle closes above the SL level.  Using only the close price
    # systematically underestimates stop-outs.
    high_prices = data["high"].values.astype(float) if "high" in data.columns else close_prices
    low_prices = data["low"].values.astype(float) if "low" in data.columns else close_prices
    n = len(close_prices)

    for i in range(20, n):
        price = close_prices[i]
        bar_high = high_prices[i]
        bar_low = low_prices[i]

        # Check stop/take profit for open position
        if position is not None:
            entry = position["entry"]
            side = position["side"]
            size = position["size"]
            sl_price = position["sl_price"]
            tp_price = position["tp_price"]

            if side == "long":
                # SL hit if the candle low touched the stop-loss level
                if bar_low <= sl_price:
                    exit_price = sl_price * (1 + slippage)
                    realised_pnl_pct = (exit_price - entry) / entry
                    pnl = balance * size * realised_pnl_pct - balance * size * commission
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    trades.append({"pnl": pnl, "pnl_pct": realised_pnl_pct, "win": pnl > 0,
                                   "entry_i": position.get("entry_i", i), "exit_i": i})
                    position = None
                elif bar_high >= tp_price:
                    exit_price = tp_price * (1 - slippage)
                    realised_pnl_pct = (exit_price - entry) / entry
                    pnl = balance * size * realised_pnl_pct - balance * size * commission
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    trades.append({"pnl": pnl, "pnl_pct": realised_pnl_pct, "win": pnl > 0,
                                   "entry_i": position.get("entry_i", i), "exit_i": i})
                    position = None
            else:  # short
                if bar_high >= sl_price:
                    exit_price = sl_price * (1 - slippage)
                    realised_pnl_pct = (entry - exit_price) / entry
                    pnl = balance * size * realised_pnl_pct - balance * size * commission
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    trades.append({"pnl": pnl, "pnl_pct": realised_pnl_pct, "win": pnl > 0,
                                   "entry_i": position.get("entry_i", i), "exit_i": i})
                    position = None
                elif bar_low <= tp_price:
                    exit_price = tp_price * (1 + slippage)
                    realised_pnl_pct = (entry - exit_price) / entry
                    pnl = balance * size * realised_pnl_pct - balance * size * commission
                    balance += pnl
                    peak_balance = max(peak_balance, balance)
                    trades.append({"pnl": pnl, "pnl_pct": realised_pnl_pct, "win": pnl > 0,
                                   "entry_i": position.get("entry_i", i), "exit_i": i})
                    position = None

        equity_curve.append(balance)

        # Generate new signal if no position open
        if position is None:
            if signal_fn is not None:
                direction, size = signal_fn(data, i)
            else:
                direction, size = _default_signal(close_prices, i)

            if direction in ("long", "short") and size > 0:
                # Enter with slippage and commission
                entry_price = price * (1 + slippage if direction == "long" else 1 - slippage)
                balance -= balance * size * commission
                sl_price = entry_price * (1 - sl_pct) if direction == "long" else entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 + tp_pct) if direction == "long" else entry_price * (1 - tp_pct)
                position = {
                    "side": direction,
                    "entry": entry_price,
                    "size": size,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "entry_i": i,
                }

    # Force close at end
    if position is not None:
        price = close_prices[-1]
        side = position["side"]
        entry = position["entry"]
        size = position["size"]
        if side == "long":
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry
        pnl = balance * size * pnl_pct
        balance += pnl
        trades.append({"pnl": pnl, "pnl_pct": pnl_pct, "win": pnl > 0,
                        "entry_i": position.get("entry_i", n-1), "exit_i": n-1})

    return _compute_metrics(trades, equity_curve, initial_balance)


def walk_forward(
    data: pd.DataFrame,
    config: Dict,
    train_pct: float = _TRAIN_PCT,
    n_folds: int = _WALK_FORWARD_FOLDS,
    signal_fn: Optional[Callable] = None,
) -> Dict:
    """
    Walk-forward validation over multiple train/test folds.

    Parameters
    ----------
    data      : full historical OHLCV DataFrame
    config    : backtest configuration dict
    train_pct : fraction of each window used for training
    n_folds   : number of walk-forward folds
    signal_fn : signal generation function

    Returns
    -------
    dict with per-fold and aggregate metrics
    """
    n = len(data)
    fold_size = n // n_folds
    fold_results = []

    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = fold_start + fold_size

        train_end = fold_start + int(fold_size * train_pct)
        test_start = train_end
        test_end = fold_end

        if test_end > n:
            break

        test_data = data.iloc[test_start:test_end].reset_index(drop=True)
        if len(test_data) < 30:
            continue

        fold_result = run_backtest(test_data, config, signal_fn)
        fold_result["fold"] = fold
        fold_result["train_size"] = train_end - fold_start
        fold_result["test_size"] = test_end - test_start
        fold_results.append(fold_result)

    if not fold_results:
        return {"error": "Insufficient data for walk-forward"}

    # Aggregate metrics
    sharpes = [r.get("sharpe_ratio", 0) for r in fold_results]
    win_rates = [r.get("win_rate", 0) for r in fold_results]
    max_dds = [r.get("max_drawdown", 1) for r in fold_results]

    return {
        "folds": fold_results,
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_win_rate": float(np.mean(win_rates)),
        "avg_max_drawdown": float(np.mean(max_dds)),
        "sharpe_std": float(np.std(sharpes)),
        "n_folds": len(fold_results),
        "is_robust": float(np.mean(sharpes)) > 0.5 and float(np.mean(win_rates)) > 0.45,
    }


def get_results(trades: List[Dict]) -> Dict:
    """Compute performance metrics from a list of trade dicts."""
    return _compute_metrics(trades, [], 10000.0)


# ---- Private helpers --------------------------------------------------------

def _default_signal(prices: np.ndarray, i: int, period: int = 14) -> Tuple[str, float]:
    """Simple RSI-based signal for testing."""
    if i < period + 1:
        return "neutral", 0.0
    diffs = np.diff(prices[i - period:i + 1])
    gains = diffs[diffs > 0]
    losses = -diffs[diffs < 0]
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-8
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)

    if rsi < 30:
        return "long", 0.5
    elif rsi > 70:
        return "short", 0.5
    return "neutral", 0.0


def _compute_metrics(trades: List[Dict], equity_curve: List[float], initial: float) -> Dict:
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            "total_return": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0,
            "max_drawdown": 0.0, "n_trades": 0, "profit_factor": 0.0,
        }

    pnls = [t.get("pnl", 0) for t in trades]
    wins = [t for t in trades if t.get("win", False)]
    losses = [t for t in trades if not t.get("win", True)]

    total_pnl = sum(pnls)
    total_return = total_pnl / max(initial, 1e-8)
    win_rate = len(wins) / max(len(trades), 1)

    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losses))
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    # Sharpe (using daily P&L approximation)
    pnl_arr = np.array(pnls)
    sharpe = (
        float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252))
        if np.std(pnl_arr) > 1e-8 else 0.0
    )

    # Max drawdown from equity curve
    if equity_curve and len(equity_curve) > 1:
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(1 - eq / np.maximum(peak, 1e-8)))
    else:
        max_dd = 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "n_trades": len(trades),
        "profit_factor": profit_factor,
        "avg_pnl": float(np.mean(pnls)),
        "final_balance": initial + total_pnl,
    }
