"""
risk_institutional/kill_switch.py — Multi-Level Circuit Breaker.

Implements a 5-level kill switch system that automatically halts
or restricts trading when risk thresholds are breached:

  Level 1 (Position):   single position loses > X% → close it
  Level 2 (Daily):      daily P&L loss > threshold → stop new trades today
  Level 3 (Portfolio):  total drawdown > threshold → safe mode (close only)
  Level 4 (Correlation): avg correlation > 0.85 → block new positions
  Level 5 (Volatility): extreme market volatility → full kill

Each level has its own cooldown and recovery process.

Integration (Loop #13): VaR/CVaR → Kill Switch
  Called continuously by EvolutionEngine._check_drawdown() and
  by the new risk monitoring loops.
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("KillSwitch")

# Default thresholds
_L1_POSITION_LOSS_PCT = 0.05       # 5% single-position loss
_L2_DAILY_LOSS_PCT = 0.03          # 3% daily loss
_L3_DRAWDOWN_PCT = 0.12            # 12% total drawdown
_L4_CORRELATION = 0.85             # average correlation threshold
_L5_VOL_MULTIPLIER = 4.0           # 4x normal volatility

# Cooldown times (seconds)
_L1_COOLDOWN = 300       # 5 min
_L2_COOLDOWN = 86400     # 24 hours (end of day)
_L3_COOLDOWN = 3600      # 1 hour
_L4_COOLDOWN = 1800      # 30 min
_L5_COOLDOWN = 7200      # 2 hours


class CircuitBreakerLevel:
    """A single circuit breaker level with cooldown and recovery."""

    def __init__(self, level_id: int, cooldown_sec: float, threshold: float):
        self.level_id = level_id
        self.cooldown_sec = cooldown_sec
        self.threshold = threshold
        self._active = False
        self._tripped_at: Optional[float] = None
        self._trip_count = 0
        self._reason = ""

    def trip(self, reason: str = "") -> None:
        if not self._active:
            self._active = True
            self._tripped_at = time.time()
            self._trip_count += 1
            self._reason = reason
            logger.warning(f"🔴 Kill Switch Level {self.level_id} TRIPPED: {reason}")

    def attempt_recovery(self) -> bool:
        """Try to recover (cooldown expired). Returns True if recovered."""
        if not self._active:
            return True
        if self._tripped_at and (time.time() - self._tripped_at) >= self.cooldown_sec:
            self._active = False
            self._tripped_at = None
            logger.info(f"🟢 Kill Switch Level {self.level_id} RECOVERED")
            return True
        return False

    def is_active(self) -> bool:
        return self._active

    def time_remaining(self) -> float:
        if not self._active or self._tripped_at is None:
            return 0.0
        elapsed = time.time() - self._tripped_at
        return max(0.0, self.cooldown_sec - elapsed)

    def get_status(self) -> Dict:
        return {
            "level": self.level_id,
            "active": self._active,
            "trip_count": self._trip_count,
            "reason": self._reason,
            "cooldown_remaining": self.time_remaining(),
        }


class KillSwitch:
    """
    Multi-level circuit breaker for institutional-grade risk management.

    Usage:
        kill = KillSwitch()
        state = {
            'balance': 10000,
            'initial_balance': 10000,
            'daily_pnl': -200,
            'positions': [...],
            'market_vol': 0.05,
            'baseline_vol': 0.02,
            'flash_crash': False,
        }
        result = kill.check_all_levels(state)
        if kill.is_killed():
            ...
    """

    def __init__(self,
                 l1_loss_pct: float = _L1_POSITION_LOSS_PCT,
                 l2_daily_loss_pct: float = _L2_DAILY_LOSS_PCT,
                 l3_drawdown_pct: float = _L3_DRAWDOWN_PCT,
                 l4_correlation: float = _L4_CORRELATION,
                 l5_vol_multiplier: float = _L5_VOL_MULTIPLIER):
        self._levels = {
            1: CircuitBreakerLevel(1, _L1_COOLDOWN, l1_loss_pct),
            2: CircuitBreakerLevel(2, _L2_COOLDOWN, l2_daily_loss_pct),
            3: CircuitBreakerLevel(3, _L3_COOLDOWN, l3_drawdown_pct),
            4: CircuitBreakerLevel(4, _L4_COOLDOWN, l4_correlation),
            5: CircuitBreakerLevel(5, _L5_COOLDOWN, l5_vol_multiplier),
        }
        self._active_levels: Set[int] = set()
        self._lock = threading.RLock()  # RLock allows re-entrant acquisition

    # ------------------------------------------------------------------

    def check_all_levels(self, portfolio_state: Dict) -> Dict:
        """
        Check all 5 circuit breaker levels against current portfolio state.

        Parameters
        ----------
        portfolio_state : dict with:
            'balance'         : current balance
            'initial_balance' : starting balance
            'peak_balance'    : historical peak
            'daily_pnl'       : today's P&L
            'positions'       : list of position dicts
            'market_vol'      : current market volatility
            'baseline_vol'    : average market volatility
            'flash_crash'     : abnormal multi-minute crash flag
            'avg_correlation' : average correlation between open positions

        Returns
        -------
        dict with triggered levels and recommended actions
        """
        with self._lock:
            triggered = []
            actions = []

            balance = float(portfolio_state.get("balance", 10000))
            initial_bal = float(portfolio_state.get("initial_balance", balance))
            peak_bal = float(portfolio_state.get("peak_balance", balance))
            daily_pnl = float(portfolio_state.get("daily_pnl", 0))
            positions = portfolio_state.get("positions", [])
            market_vol = float(portfolio_state.get("market_vol", 0))
            baseline_vol = float(portfolio_state.get("baseline_vol", market_vol + 1e-8))
            flash_crash = bool(portfolio_state.get("flash_crash", False))
            avg_corr = float(portfolio_state.get("avg_correlation", 0))

            # Level 1: Individual position losses
            for pos in positions:
                pnl_pct = float(pos.get("pnl_pct", 0))
                if pnl_pct < -self._levels[1].threshold:
                    self._levels[1].trip(
                        f"Position {pos.get('symbol', '?')} loss {pnl_pct:.1%}"
                    )
                    actions.append(f"CLOSE_POSITION:{pos.get('symbol', '?')}")

            if self._levels[1].is_active():
                triggered.append(1)

            # Level 2: Daily loss
            daily_pnl_pct = daily_pnl / max(initial_bal, 1e-8)
            if daily_pnl_pct < -self._levels[2].threshold:
                self._levels[2].trip(f"Daily loss {daily_pnl_pct:.1%}")
                triggered.append(2)
                actions.append("STOP_NEW_TRADES_TODAY")

            # Level 3: Portfolio drawdown
            drawdown = (peak_bal - balance) / max(peak_bal, 1e-8)
            if drawdown > self._levels[3].threshold:
                self._levels[3].trip(f"Drawdown {drawdown:.1%}")
                triggered.append(3)
                actions.append("SAFE_MODE:CLOSE_ONLY")

            # Level 4: Correlation risk
            if avg_corr > self._levels[4].threshold and len(positions) > 1:
                self._levels[4].trip(f"Avg correlation {avg_corr:.2f}")
                triggered.append(4)
                actions.append("BLOCK_NEW_POSITIONS")

            # Level 5: Volatility explosion
            if baseline_vol > 1e-8:
                vol_ratio = market_vol / baseline_vol
                if vol_ratio > self._levels[5].threshold or flash_crash:
                    if flash_crash:
                        reason = "Flash crash detected"
                    else:
                        reason = f"Volatility {vol_ratio:.1f}x baseline"
                    self._levels[5].trip(reason)
                    triggered.append(5)
                    actions.append("KILL_ALL")

            self._active_levels = set(
                lvl for lvl, cb in self._levels.items() if cb.is_active()
            )

            return {
                "triggered_levels": triggered,
                "active_levels": list(self._active_levels),
                "recommended_actions": list(set(actions)),
                "is_killed": self.is_killed(),
                "allow_new_trades": self._allow_new_trades(),
            }

    def is_killed(self) -> bool:
        """True if Level 5 (total kill) is active."""
        with self._lock:
            return self._levels[5].is_active()

    def is_safe_mode(self) -> bool:
        """True if Level 3+ is active (close-only mode)."""
        with self._lock:
            return (
                self._levels[3].is_active() or
                self._levels[5].is_active()
            )

    def get_active_breakers(self) -> List[Dict]:
        """Return status of all active circuit breakers."""
        with self._lock:
            return [
                self._levels[lvl].get_status()
                for lvl in sorted(self._active_levels)
            ]

    def attempt_recovery(self) -> List[int]:
        """
        Attempt recovery for all active breakers where cooldown has expired.

        Returns list of levels that successfully recovered.
        """
        recovered = []
        with self._lock:
            for lvl, cb in self._levels.items():
                if cb.is_active() and cb.attempt_recovery():
                    recovered.append(lvl)
                    self._active_levels.discard(lvl)
        return recovered

    def force_reset(self, levels: Optional[List[int]] = None) -> None:
        """Force reset specific levels (for emergency manual override)."""
        with self._lock:
            targets = levels or list(self._levels.keys())
            for lvl in targets:
                if lvl in self._levels:
                    self._levels[lvl]._active = False
                    self._active_levels.discard(lvl)
            logger.warning(f"KillSwitch force_reset for levels: {targets}")

    def get_full_status(self) -> Dict:
        """Return complete status of all circuit breaker levels."""
        with self._lock:
            return {
                "is_killed": self.is_killed(),
                "is_safe_mode": self.is_safe_mode(),
                "allow_new_trades": self._allow_new_trades(),
                "levels": {lvl: cb.get_status() for lvl, cb in self._levels.items()},
            }

    def _allow_new_trades(self) -> bool:
        """True if new trades are allowed (no blocking levels active)."""
        blocking = {2, 3, 4, 5}
        return len(self._active_levels & blocking) == 0
