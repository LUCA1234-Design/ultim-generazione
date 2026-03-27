"""
Execution Engine for V17.
Paper trading (default, PAPER_TRADING=True) simulates orders and tracks P&L.
Real mode uses Binance Futures futures_create_order() via binance_client.
"""
import logging
import time
import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from config.settings import (
    PAPER_TRADING,
    ACCOUNT_BALANCE,
    LEVERAGE,
    MAX_DAILY_LOSS_USDT,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
)
from data.binance_client import place_futures_order

logger = logging.getLogger("Execution")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents an open (or closed) position."""
    position_id: str
    symbol: str
    interval: str
    direction: str          # "long" | "short"
    entry_price: float
    size: float             # base currency units
    sl: float
    tp1: float
    tp2: float
    strategy: str = ""
    open_time: float = field(default_factory=time.time)
    close_time: Optional[float] = None
    close_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"   # "open" | "closed" | "sl_hit" | "tp1_hit" | "tp2_hit"
    tp1_hit: bool = False
    decision_id: str = ""
    paper: bool = True

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "size": self.size,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "close_price": self.close_price,
            "pnl": self.pnl,
            "status": self.status,
            "paper": self.paper,
        }


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Handles order placement in paper or live mode."""

    def __init__(self, paper_trading: bool = PAPER_TRADING,
                 initial_balance: float = ACCOUNT_BALANCE):
        self.paper_trading = paper_trading
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._open_positions: Dict[str, Position] = {}   # position_id → Position
        self._closed_positions: List[Position] = []
        self._total_pnl = 0.0
        self._trade_count = 0
        self._win_count = 0
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._current_day = datetime.datetime.now(datetime.timezone.utc).date()
        logger.info(
            f"ExecutionEngine: {'PAPER' if paper_trading else 'LIVE'} trading | "
            f"balance={initial_balance}"
        )
    def _roll_day_if_needed(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self._current_day:
            self._current_day = today
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            logger.info("🔄 Daily risk counters reset")

    def is_risk_blocked(self) -> tuple[bool, str]:
        self._roll_day_if_needed()

        daily_loss_usdt = max(0.0, -self._daily_pnl)
        daily_loss_pct = (
            (daily_loss_usdt / self._initial_balance) * 100
            if self._initial_balance > 0 else 0.0
        )

        if daily_loss_usdt >= MAX_DAILY_LOSS_USDT:
            return True, "max_daily_loss_usdt"

        if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            return True, "max_daily_loss_pct"

        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return True, "max_consecutive_losses"

        return False, ""

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, symbol: str, interval: str, direction: str,
                      entry_price: float, size: float, sl: float,
                      tp1: float, tp2: float, strategy: str = "",
                      decision_id: str = "") -> Optional[Position]:
        """Open a new position (paper or live)."""
        pos_id = str(uuid.uuid4())[:8]
        pos = Position(
            position_id=pos_id,
            symbol=symbol,
            interval=interval,
            direction=direction,
            entry_price=entry_price,
            size=size,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            strategy=strategy,
            decision_id=decision_id,
            paper=self.paper_trading,
        )

        if self.paper_trading:
            self._open_positions[pos_id] = pos
            logger.info(
                f"📄 PAPER OPEN [{pos_id}] {symbol} {direction.upper()} "
                f"@ {entry_price:.4f} size={size} sl={sl:.4f} tp1={tp1:.4f}"
            )
        else:
            # Real execution
            side = "BUY" if direction == "long" else "SELL"
            order = place_futures_order(symbol, side, "MARKET", size)
            if order is None:
                logger.error(f"Failed to open live position for {symbol}")
                return None
            self._open_positions[pos_id] = pos
            logger.info(f"✅ LIVE OPEN [{pos_id}] {symbol} {direction.upper()} {order}")

        return pos

    def close_position(self, position_id: str, close_price: float,
                        reason: str = "manual") -> Optional[Position]:
        """Close an open position."""
        pos = self._open_positions.pop(position_id, None)
        if pos is None:
            return None

        pos.close_price = close_price
        pos.close_time = time.time()
        pos.pnl = pos.unrealised_pnl(close_price)
        pos.status = reason
        self._roll_day_if_needed()
        self._balance += pos.pnl
        self._total_pnl += pos.pnl
        self._daily_pnl += pos.pnl
        self._trade_count += 1

        if pos.pnl > 0:
            self._win_count += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._closed_positions.append(pos)
        emoji = "✅" if pos.pnl > 0 else "❌"
        logger.info(
            f"{emoji} {'PAPER ' if pos.paper else ''}CLOSE [{position_id}] "
            f"{pos.symbol} {pos.direction.upper()} @ {close_price:.4f} "
            f"PnL={pos.pnl:+.4f} ({reason})"
        )

        if not self.paper_trading:
            side = "SELL" if pos.direction == "long" else "BUY"
            place_futures_order(pos.symbol, side, "MARKET", pos.size, reduce_only=True)

        return pos

    def check_position_levels(self, symbol: str, current_price: float) -> List[Position]:
        """Check all open positions for SL/TP hits and return closed positions."""
        to_close: List[Tuple[str, float, str]] = []
        closed_positions: List[Position] = []

        for pos_id, pos in list(self._open_positions.items()):
            if pos.symbol != symbol:
                continue

            if pos.direction == "long":
                if current_price <= pos.sl:
                    to_close.append((pos_id, current_price, "sl_hit"))
                elif not pos.tp1_hit and current_price >= pos.tp1:
                    pos.tp1_hit = True
                    # Move SL to entry (breakeven)
                    pos.sl = pos.entry_price
                    logger.info(f"🎯 TP1 hit [{pos_id}] {pos.symbol} — SL moved to entry")
                elif pos.tp1_hit and current_price >= pos.tp2:
                    to_close.append((pos_id, current_price, "tp2_hit"))
            else:
                if current_price >= pos.sl:
                    to_close.append((pos_id, current_price, "sl_hit"))
                elif not pos.tp1_hit and current_price <= pos.tp1:
                    pos.tp1_hit = True
                    pos.sl = pos.entry_price
                    logger.info(f"🎯 TP1 hit [{pos_id}] {pos.symbol} — SL moved to entry")
                elif pos.tp1_hit and current_price <= pos.tp2:
                    to_close.append((pos_id, current_price, "tp2_hit"))

        for pos_id, price, reason in to_close:
            closed = self.close_position(pos_id, price, reason)
            if closed is not None:
                closed_positions.append(closed)

        return closed_positions

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        self._roll_day_if_needed()
        risk_blocked, risk_reason = self.is_risk_blocked()

        return {
            _skip_reasons
            "balance": self._balance,
            "initial_balance": self._initial_balance,
            "total_pnl": self._total_pnl,
            "pnl_pct": self._total_pnl / self._initial_balance * 100,
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "win_rate": self._win_count / max(self._trade_count, 1),
            "open_positions": len(self._open_positions),
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "risk_blocked": risk_blocked,
            "risk_block_reason": risk_reason,
        }

    def get_open_positions(self) -> List[Position]:
        return list(self._open_positions.values())

    def get_closed_positions(self, limit: int = 50) -> List[Position]:
        return self._closed_positions[-limit:]
