"""
Performance Tracker for V17.
Tracks real win rates, P&L, and feeds back into agents and the risk model.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from engine.execution import Position
from memory import experience_db

logger = logging.getLogger("PerformanceTracker")


class PerformanceTracker:
    """Monitors closed positions and updates agent win rates."""

    def __init__(self):
        self._snapshots: List[Dict[str, Any]] = []
        self._symbol_stats: Dict[str, Dict[str, float]] = {}
        self._interval_stats: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_position(self, pos: Position) -> None:
        """Record a closed position and update statistics."""
        if pos.pnl is None:
            return

        snapshot = pos.to_dict()
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 1000:
            self._snapshots.pop(0)

        # Update symbol stats
        sym_stats = self._symbol_stats.setdefault(pos.symbol, {
            "trades": 0, "wins": 0, "total_pnl": 0.0,
        })
        sym_stats["trades"] += 1
        if pos.pnl > 0:
            sym_stats["wins"] += 1
        sym_stats["total_pnl"] += pos.pnl

        # Update interval stats
        ivl_stats = self._interval_stats.setdefault(pos.interval, {
            "trades": 0, "wins": 0, "total_pnl": 0.0,
        })
        ivl_stats["trades"] += 1
        if pos.pnl > 0:
            ivl_stats["wins"] += 1
        ivl_stats["total_pnl"] += pos.pnl

        # Persist to DB
        experience_db.save_trade_outcome(
            position_id=pos.position_id,
            ts_open=pos.open_time,
            ts_close=pos.close_time or time.time(),
            symbol=pos.symbol,
            interval=pos.interval,
            direction=pos.direction,
            entry_price=pos.entry_price,
            close_price=pos.close_price or pos.entry_price,
            size=pos.size,
            pnl=pos.pnl,
            status=pos.status,
            strategy=pos.strategy,
            decision_id=pos.decision_id,
            paper=pos.paper,
        )

        emoji = "✅" if pos.pnl > 0 else "❌"
        logger.info(
            f"{emoji} PerformanceTracker: {pos.symbol}/{pos.interval} {pos.direction} "
            f"PnL={pos.pnl:+.4f} status={pos.status}"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def global_win_rate(self) -> float:
        """Overall win rate across all recorded positions."""
        if not self._snapshots:
            return 0.55  # Default before any data
        wins = sum(1 for s in self._snapshots if (s.get("pnl") or 0) > 0)
        return wins / len(self._snapshots)

    def symbol_win_rate(self, symbol: str) -> float:
        stats = self._symbol_stats.get(symbol)
        if stats is None or stats["trades"] == 0:
            # Try DB
            db_wr = experience_db.get_win_rate_by_symbol(symbol)
            return db_wr if db_wr is not None else 0.55
        return stats["wins"] / stats["trades"]

    def interval_win_rate(self, interval: str) -> float:
        stats = self._interval_stats.get(interval)
        if stats is None or stats["trades"] == 0:
            db_wr = experience_db.get_win_rate_by_interval(interval)
            return db_wr if db_wr is not None else 0.55
        return stats["wins"] / stats["trades"]

    def total_pnl(self) -> float:
        return sum((s.get("pnl") or 0) for s in self._snapshots)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Approximate Sharpe from per-trade P&L."""
        pnls = [(s.get("pnl") or 0) for s in self._snapshots]
        if len(pnls) < 5:
            return 0.0
        mean = np.mean(pnls) - risk_free_rate
        std = np.std(pnls)
        return float(mean / std) if std > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        total = len(self._snapshots)
        wins = sum(1 for s in self._snapshots if (s.get("pnl") or 0) > 0)
        losses = total - wins
        pnl_list = [(s.get("pnl") or 0) for s in self._snapshots]
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_pnl": sum(pnl_list),
            "avg_pnl": float(np.mean(pnl_list)) if pnl_list else 0.0,
            "best_trade": float(max(pnl_list)) if pnl_list else 0.0,
            "worst_trade": float(min(pnl_list)) if pnl_list else 0.0,
            "sharpe": self.sharpe_ratio(),
            "symbol_stats": {
                sym: {
                    "wr": s["wins"] / s["trades"] if s["trades"] > 0 else 0,
                    "pnl": s["total_pnl"],
                    "trades": s["trades"],
                }
                for sym, s in self._symbol_stats.items()
            },
        }

    def update_risk_agent_win_rates(self, risk_agent, current_balance: float = None) -> None:
        """Push real win rates from tracker into RiskAgent."""
        risk_agent.set_win_rate("global", self.global_win_rate())
        if current_balance is not None:
            risk_agent.update_balance(current_balance)
        for symbol, stats in self._symbol_stats.items():
            if stats["trades"] >= 5:
                wr = stats["wins"] / stats["trades"]
                risk_agent.set_win_rate(symbol, wr)
        for interval, stats in self._interval_stats.items():
            if stats["trades"] >= 5:
                wr = stats["wins"] / stats["trades"]
                risk_agent.set_win_rate(interval, wr)
