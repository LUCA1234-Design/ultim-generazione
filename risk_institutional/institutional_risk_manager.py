import logging
from typing import Any, Dict, Iterable, Optional

import numpy as np

from risk_institutional.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class InstitutionalRiskManager:
    """Advanced institutional risk controls for kill switch, ATR sizing and trailing SL."""

    def __init__(
        self,
        kill_switch: Optional[KillSwitch] = None,
        atr_risk_fraction: float = 0.01,
        atr_stop_mult: float = 1.5,
        trailing_window: int = 20,
        trailing_std_mult: float = 2.0,
        flash_crash_pct: float = 0.08,
        flash_crash_lookback: int = 3,
    ):
        self.kill_switch = kill_switch or KillSwitch()
        self.atr_risk_fraction = float(max(0.0001, atr_risk_fraction))
        self.atr_stop_mult = float(max(0.1, atr_stop_mult))
        self.trailing_window = int(max(5, trailing_window))
        self.trailing_std_mult = float(max(0.5, trailing_std_mult))
        self.flash_crash_pct = float(max(0.01, flash_crash_pct))
        self.flash_crash_lookback = int(max(2, flash_crash_lookback))

    def compute_market_state(self, closes: Iterable[float]) -> Dict[str, Any]:
        prices = np.asarray(list(closes), dtype=float)
        prices = prices[np.isfinite(prices)]
        if prices.size < max(10, self.flash_crash_lookback + 1):
            return {"market_vol": 0.0, "baseline_vol": 0.01, "flash_crash": False}

        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)
        returns = returns[np.isfinite(returns)]
        if returns.size == 0:
            return {"market_vol": 0.0, "baseline_vol": 0.01, "flash_crash": False}

        short_window = min(5, len(returns))
        market_vol = float(np.std(returns[-short_window:]))
        baseline_window = min(50, len(returns))
        baseline_vol = float(np.std(returns[-baseline_window:]))
        baseline_vol = max(baseline_vol, 1e-6)

        lookback_start_idx = max(0, len(prices) - self.flash_crash_lookback - 1)
        ref_price = float(prices[lookback_start_idx])
        latest = float(prices[-1])
        drop = (latest - ref_price) / max(ref_price, 1e-8)
        flash_crash = drop <= -self.flash_crash_pct

        return {
            "market_vol": market_vol,
            "baseline_vol": baseline_vol,
            "flash_crash": flash_crash,
            "flash_crash_drop_pct": float(drop),
        }

    def should_kill_globally(self, kill_result: Dict) -> bool:
        triggered = set(kill_result.get("triggered_levels", []))
        return (2 in triggered) or (5 in triggered)

    def apply_atr_position_sizing(
        self,
        current_size: float,
        balance: float,
        entry_price: float,
        atr_value: float,
    ) -> float:
        current_size = float(max(0.0, current_size))
        atr_value = float(max(0.0, atr_value))
        entry_price = float(max(0.0, entry_price))
        balance = float(max(0.0, balance))
        if atr_value <= 0 or entry_price <= 0 or balance <= 0:
            return current_size

        risk_per_unit = atr_value * self.atr_stop_mult
        if risk_per_unit <= 1e-10:
            return current_size

        risk_amount = balance * self.atr_risk_fraction
        atr_size = risk_amount / risk_per_unit
        if atr_size <= 0:
            return current_size

        return float(min(current_size, atr_size))

    def trailing_stop_from_std(
        self,
        direction: str,
        current_price: float,
        existing_sl: float,
        closes: Iterable[float],
    ) -> float:
        prices = np.asarray(list(closes), dtype=float)
        prices = prices[np.isfinite(prices)]
        if prices.size < self.trailing_window:
            return float(existing_sl)

        window = prices[-self.trailing_window:]
        std = float(np.std(window))
        if std <= 0:
            return float(existing_sl)

        distance = self.trailing_std_mult * std
        if direction == "long":
            candidate = float(current_price - distance)
            return float(max(existing_sl, candidate))
        candidate = float(current_price + distance)
        return float(min(existing_sl, candidate))
