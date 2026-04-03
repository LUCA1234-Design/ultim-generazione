"""
Risk Agent for V17.
Kelly Criterion with real win rates from experience DB, adaptive position sizing.
Computes entry, SL, TP1, TP2 levels and position size.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import atr, adx
from config.settings import ACCOUNT_BALANCE, LEVERAGE

logger = logging.getLogger("RiskAgent")

DEFAULT_WIN_RATE = 0.55


class RiskAgent(BaseAgent):
    """Adaptive risk agent with Kelly sizing and ATR-based levels."""

    def __init__(self):
        super().__init__("risk", initial_weight=0.15)
        # Win rates per pattern/symbol, populated by memory module
        self._win_rates: Dict[str, float] = {}
        self._balance = ACCOUNT_BALANCE

    # ------------------------------------------------------------------
    # Win rate management
    # ------------------------------------------------------------------

    def set_win_rate(self, key: str, win_rate: float) -> None:
        """Set win rate for a key (e.g. symbol, pattern, interval)."""
        self._win_rates[key] = float(np.clip(win_rate, 0.01, 0.99))

    def get_win_rate(self, symbol: str, interval: str, pattern: str = "") -> float:
        """Look up win rate with fallback hierarchy."""
        keys = [
            f"{symbol}_{interval}_{pattern}",
            f"{symbol}_{interval}",
            f"{interval}_{pattern}",
            interval,
            "global",
        ]
        for key in keys:
            if key in self._win_rates:
                return self._win_rates[key]
        return DEFAULT_WIN_RATE

    def update_balance(self, new_balance: float) -> None:
        self._balance = max(new_balance, 0.0)

    # ------------------------------------------------------------------
    # Kelly Criterion
    # ------------------------------------------------------------------

    @staticmethod
    def kelly_fraction(win_rate: float, rr: float) -> float:
        """Half-Kelly fraction capped at 5%."""
        if rr <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01
        q = 1.0 - win_rate
        k = (win_rate * rr - q) / rr
        return float(np.clip(k * 0.5, 0.005, 0.05))

    # ------------------------------------------------------------------
    # Level calculation
    # ------------------------------------------------------------------

    def calc_levels(self, df: pd.DataFrame, direction: str,
                    atr_sl_mult: float = 1.5,       # was 2.0 — tighter SL
                    atr_tp1_mult: float = 2.5,      # was 2.0 — wider TP1, gives R/R = 2.5/1.5 = 1.67
                    atr_tp2_mult: float = 5.0) -> Tuple[float, float, float, float]:
        """Compute SL, TP1, TP2 from ATR, and return (sl, tp1, tp2, rr)."""
        _atr = atr(df, 14).iloc[-1]
        close = df["close"].iloc[-1]
        if direction == "long":
            sl = close - atr_sl_mult * _atr
            tp1 = close + atr_tp1_mult * _atr
            tp2 = close + atr_tp2_mult * _atr
        else:
            sl = close + atr_sl_mult * _atr
            tp1 = close - atr_tp1_mult * _atr
            tp2 = close - atr_tp2_mult * _atr
        rr = abs(tp1 - close) / max(abs(close - sl), 1e-10)
        return float(sl), float(tp1), float(tp2), float(rr)

    def calc_position_size(self, entry: float, sl: float,
                            win_rate: float = DEFAULT_WIN_RATE,
                            rr: float = 2.0) -> float:
        """Return position size in base currency units."""
        risk_per_unit = abs(entry - sl)
        if risk_per_unit < 1e-10:
            return 0.0
        k = self.kelly_fraction(win_rate, rr)
        risk_amount = self._balance * k
        size = risk_amount / risk_per_unit
        return round(float(size), 4)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df,
                direction: str = "long") -> Optional[AgentResult]:
        if df is None or len(df) < 20:
            return None

        win_rate = self.get_win_rate(symbol, interval)
        sl, tp1, tp2, rr = self.calc_levels(df, direction)
        entry = float(df["close"].iloc[-1])
        kelly = self.kelly_fraction(win_rate, rr)
        size = self.calc_position_size(entry, sl, win_rate, rr)

        # Risk score: higher R/R and win rate → higher score
        rr_score = float(np.clip(rr / 3.0, 0.0, 1.0))                           # was (rr-1.0)/3.0; R/R=2 now gives 0.67 instead of 0.33
        wr_score = float(np.clip((win_rate - 0.3) / 0.4, 0.0, 1.0))             # was (win_rate-0.4)/0.4; WR=0.55 now gives 0.625 instead of 0.375
        score = 0.5 * rr_score + 0.5 * wr_score                                 # was 0.6/0.4, balanced 50/50

        details = [
            f"entry={entry:.4f}",
            f"sl={sl:.4f}",
            f"tp1={tp1:.4f}",
            f"tp2={tp2:.4f}",
            f"rr={rr:.2f}",
            f"kelly={kelly*100:.1f}%",
            f"size={size}",
            f"win_rate={win_rate:.2%}",
        ]

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=score,
            direction=direction,
            confidence=win_rate,
            details=details,
            metadata={
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr": rr,
                "kelly": kelly,
                "size": size,
                "win_rate": win_rate,
                "balance": self._balance,
            },
        )
