"""
Execution Engine for V17.
Paper trading (default, PAPER_TRADING=True) simulates orders and tracks P&L.
Real mode uses Binance Futures futures_create_order() via binance_client.
"""
import logging
import threading
import time
import datetime
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from config.settings import (
    PAPER_TRADING,
    ACCOUNT_BALANCE,
    TRAINING_MODE,
    LEVERAGE,
    MAX_DAILY_LOSS_USDT,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_TOTAL_DRAWDOWN_PCT,
    MAX_WEEKLY_LOSS_PCT,
    ORDER_ROUTING_ENABLED,
)
from data import data_store
from data.binance_client import place_futures_order

try:
    from risk_institutional import InstitutionalRiskManager
except Exception:
    InstitutionalRiskManager = None

logger = logging.getLogger("Execution")

# Maximum age per interval before a position is force-closed (in seconds)
_MAX_POSITION_AGE = {"15m": 86400, "1h": 172800, "4h": 259200}  # 1d, 2d, 3d
# Trailing stop ratio: fraction of TP1 distance to trail after TP1 is hit
_TRAIL_STOP_RATIO = 0.5


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
    tp2_hit: bool = False
    decision_id: str = ""
    paper: bool = True
    tp1_partial_pnl: Optional[float] = None   # PnL parziale registrato alla chiusura 50% a TP1

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
        # Lock per thread-safety tra position monitor e WebSocket thread
        self._lock = threading.Lock()
        # Drawdown protection: peak balance e P&L settimanale (non-resettanti)
        self._peak_balance = initial_balance
        self._weekly_pnl = 0.0
        self._week_start = datetime.datetime.now(datetime.timezone.utc).isocalendar()[1]
        self._standby = False
        self._standby_reason = ""
        self._institutional_risk_manager = None
        logger.info(
            f"ExecutionEngine: {'PAPER' if paper_trading else 'LIVE'} trading | "
            f"balance={initial_balance}"
        )
    def _roll_day_if_needed(self) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).date()
        if today != self._current_day:
            self._current_day = today
            self._daily_pnl = 0.0
            # consecutive_losses NON viene azzerato al cambio giorno —
            # si azzera SOLO dopo un trade profittevole (in close_position)
            logger.info("🔄 Daily risk counters reset")
        # Reset settimanale P&L se la settimana ISO è cambiata
        current_week = datetime.datetime.now(datetime.timezone.utc).isocalendar()[1]
        if current_week != self._week_start:
            self._week_start = current_week
            self._weekly_pnl = 0.0
            logger.info("🔄 Weekly PnL reset")

    def is_risk_blocked(self) -> tuple[bool, str, dict]:
        self._roll_day_if_needed()

        daily_loss_usdt = max(0.0, -self._daily_pnl)
        daily_loss_pct = (
            (daily_loss_usdt / self._initial_balance) * 100
            if self._initial_balance > 0 else 0.0
        )

        # Drawdown totale dal peak balance
        total_drawdown_pct = (
            (self._peak_balance - self._balance) / self._peak_balance * 100
            if self._peak_balance > 0 else 0.0
        )
        # Perdita settimanale (solo perdite, non guadagni)
        weekly_loss_pct = (
            abs(min(0.0, self._weekly_pnl)) / self._initial_balance * 100
            if self._initial_balance > 0 else 0.0
        )

        details = {
            "daily_loss_usdt": daily_loss_usdt,
            "daily_loss_pct": daily_loss_pct,
            "daily_loss_pct_max": MAX_DAILY_LOSS_PCT,
            "daily_loss_usdt_max": MAX_DAILY_LOSS_USDT,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_losses_max": MAX_CONSECUTIVE_LOSSES,
            "total_drawdown_pct": total_drawdown_pct,
            "total_drawdown_pct_max": MAX_TOTAL_DRAWDOWN_PCT,
            "weekly_loss_pct": weekly_loss_pct,
            "weekly_loss_pct_max": MAX_WEEKLY_LOSS_PCT,
        }

        if TRAINING_MODE:
            if (
                daily_loss_usdt >= MAX_DAILY_LOSS_USDT
                or daily_loss_pct >= MAX_DAILY_LOSS_PCT
                or self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES
                or total_drawdown_pct >= MAX_TOTAL_DRAWDOWN_PCT
                or weekly_loss_pct >= MAX_WEEKLY_LOSS_PCT
            ):
                logger.debug(f"[TRAINING] Risk limit reached but not blocking: {details}")
            return False, "", details

        if daily_loss_usdt >= MAX_DAILY_LOSS_USDT:
            return True, "max_daily_loss_usdt", details

        if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            return True, "max_daily_loss_pct", details

        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return True, "max_consecutive_losses", details

        if total_drawdown_pct >= MAX_TOTAL_DRAWDOWN_PCT:
            return True, "max_total_drawdown", details

        if weekly_loss_pct >= MAX_WEEKLY_LOSS_PCT:
            return True, "max_weekly_loss", details

        return False, "", details

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, symbol: str, interval: str, direction: str,
                      entry_price: float, size: float, sl: float,
                      tp1: float, tp2: float, strategy: str = "",
                      decision_id: str = "") -> Optional[Position]:
        """Open a new position (paper or live)."""
        if self._standby:
            logger.warning(f"🛑 Standby active: blocking new position for {symbol} ({self._standby_reason})")
            return None
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

        # ---- Institutional risk checks ----
        try:
            from risk_institutional.regulatory_limits import check_position_limits
            notional = entry_price * size
            current_regime = "ranging"  # fallback; ideally pass regime from caller
            lev = float(LEVERAGE)
            pos_dict = {"notional": notional, "leverage": lev, "symbol": symbol}
            reg_check = check_position_limits(pos_dict, self._balance, regime=current_regime)
            if not reg_check["allowed"]:
                logger.warning(
                    f"⚖️ [RegulatoryLimits] Position blocked for {symbol}: {reg_check['violations']}"
                )
                return None
        except Exception as _reg_err:
            logger.debug(f"⚖️ RegulatoryLimits check skipped: {_reg_err}")

        try:
            from risk_institutional.margin_monitor import MarginMonitor
            lev = float(LEVERAGE)
            _mm = MarginMonitor(leverage=lev)
            with self._lock:
                _open_positions_snapshot = list(self._open_positions.values())
            open_pos_list = [
                {"notional": abs(p.entry_price * p.size), "margin": abs(p.entry_price * p.size) / max(lev, 1e-8)}
                for p in _open_positions_snapshot
            ]
            margin_status = _mm.compute_margin_usage(open_pos_list, self._balance)
            if not margin_status.get("can_open_new", True):
                logger.warning(
                    f"⚖️ [MarginMonitor] Margin utilisation {margin_status.get('utilisation_pct', 0):.1%} "
                    f"— blocking new position for {symbol} (status={margin_status.get('status')})"
                )
                return None
        except Exception as _mm_err:
            logger.debug(f"⚖️ MarginMonitor check skipped: {_mm_err}")

        with self._lock:
            if self.paper_trading:
                self._open_positions[pos_id] = pos
                logger.info(
                    f"📄 PAPER OPEN [{pos_id}] {symbol} {direction.upper()} "
                    f"@ {entry_price:.4f} size={size} sl={sl:.4f} tp1={tp1:.4f}"
                )
            else:
                # Esecuzione reale: leggo il fill price effettivo da Binance
                side = "BUY" if direction == "long" else "SELL"
                order = None
                if ORDER_ROUTING_ENABLED:
                    try:
                        from engine.order_slicer import SmartOrderRouter
                        router = SmartOrderRouter()
                        result = router.route_order(symbol, side, size * entry_price, entry_price)
                        if result.success:
                            real_entry = float(result.avg_fill_price)
                            if real_entry > 0:
                                pos.entry_price = real_entry
                                slippage_pct = abs(real_entry - entry_price) / max(entry_price, 1e-8) * 100.0
                                logger.info(
                                    f"🧠 Smart routing [{symbol}] strategy={result.strategy_used} "
                                    f"slippage={slippage_pct:.3f}% orders={result.n_orders_placed}"
                                )
                            self._open_positions[pos_id] = pos
                            logger.info(f"✅ LIVE OPEN [{pos_id}] {symbol} {direction.upper()} smart-routed")
                            return pos
                    except Exception as e:
                        logger.warning(f"SmartOrderRouter error [{symbol}]: {e}")

                order = place_futures_order(symbol, side, "MARKET", size)
                if order is None:
                    logger.error(f"Failed to open live position for {symbol}")
                    return None
                # Aggiorna l'entry price con il fill price reale
                real_entry = float(order.get("avgPrice", entry_price))
                if real_entry > 0:
                    slippage_pct = abs(real_entry - entry_price) / entry_price * 100
                    logger.info(
                        f"💹 Slippage: requested={entry_price:.4f} "
                        f"filled={real_entry:.4f} diff={slippage_pct:.3f}%"
                    )
                    # Ricalcola SL/TP se lo slippage è significativo (>0.1%)
                    if slippage_pct > 0.1:
                        sl_dist = abs(sl - entry_price)
                        tp1_dist = abs(tp1 - entry_price)
                        tp2_dist = abs(tp2 - entry_price)
                        if direction == "long":
                            pos.sl = real_entry - sl_dist
                            pos.tp1 = real_entry + tp1_dist
                            pos.tp2 = real_entry + tp2_dist
                        else:
                            pos.sl = real_entry + sl_dist
                            pos.tp1 = real_entry - tp1_dist
                            pos.tp2 = real_entry - tp2_dist
                    pos.entry_price = real_entry
                self._open_positions[pos_id] = pos
                logger.info(f"✅ LIVE OPEN [{pos_id}] {symbol} {direction.upper()} {order}")

        return pos

    def close_position(self, position_id: str, close_price: float,
                        reason: str = "manual") -> Optional[Position]:
        """Close an open position."""
        with self._lock:
            pos = self._open_positions.pop(position_id, None)
        if pos is None:
            return None

        if not self.paper_trading:
            # Live: piazza l'ordine di chiusura e leggi il fill price reale
            side = "SELL" if pos.direction == "long" else "BUY"
            close_order = place_futures_order(pos.symbol, side, "MARKET", pos.size, reduce_only=True)
            if close_order is not None:
                real_close = float(close_order.get("avgPrice", close_price))
                if real_close > 0:
                    close_price = real_close

        pos.close_price = close_price
        pos.close_time = time.time()
        pos.pnl = pos.unrealised_pnl(close_price)
        pos.status = reason
        self._roll_day_if_needed()

        with self._lock:
            self._balance += pos.pnl
            self._total_pnl += pos.pnl
            self._daily_pnl += pos.pnl
            self._weekly_pnl += pos.pnl
            self._trade_count += 1
            # Aggiorna il peak balance per il calcolo del drawdown
            self._peak_balance = max(self._peak_balance, self._balance)

            if pos.pnl > 0:
                self._win_count += 1
                self._consecutive_losses = 0  # Reset SOLO dopo un trade profittevole
            else:
                self._consecutive_losses += 1

            self._closed_positions.append(pos)
            if len(self._closed_positions) > 1000:
                del self._closed_positions[:100]

        emoji = "✅" if pos.pnl > 0 else "❌"
        logger.info(
            f"{emoji} {'PAPER ' if pos.paper else ''}CLOSE [{position_id}] "
            f"{pos.symbol} {pos.direction.upper()} @ {close_price:.4f} "
            f"PnL={pos.pnl:+.4f} ({reason})"
        )

        return pos

    def check_position_levels(self, symbol: str, current_price: float) -> List[Position]:
        """Check all open positions for SL/TP hits and return closed positions."""
        to_close: List[Tuple[str, float, str]] = []
        closed_positions: List[Position] = []

        with self._lock:
            snapshot = list(self._open_positions.items())

        for pos_id, pos in snapshot:
            if pos.symbol != symbol:
                continue

            # Check position timeout before SL/TP
            max_age = _MAX_POSITION_AGE.get(pos.interval, 172800)
            if time.time() - pos.open_time > max_age:
                logger.info(
                    f"⏰ TIMEOUT [{pos_id}] {pos.symbol}/{pos.interval} — "
                    f"open for >{max_age}s, closing at {current_price:.4f}"
                )
                to_close.append((pos_id, current_price, "timeout"))
                continue

            if pos.direction == "long":
                if current_price <= pos.sl:
                    to_close.append((pos_id, current_price, "sl_hit"))
                elif not pos.tp1_hit and current_price >= pos.tp1:
                    pos.tp1_hit = True
                    # Partial take profit: chiude 50% della posizione
                    partial_size = pos.size / 2.0
                    partial_pnl = (current_price - pos.entry_price) * partial_size
                    pos.tp1_partial_pnl = partial_pnl
                    pos.size = partial_size
                    # Sposta SL a breakeven
                    pos.sl = pos.entry_price
                    logger.info(
                        f"🎯 TP1 hit [{pos_id}] {pos.symbol} — "
                        f"closed 50%, trailing remaining 50% | partial_pnl={partial_pnl:+.4f}"
                    )
                    # In live trading, piazza ordine MARKET per la metà della size
                    if not self.paper_trading:
                        try:
                            place_futures_order(pos.symbol, "SELL", "MARKET", partial_size, reduce_only=True)
                        except Exception as _e:
                            logger.error(f"Partial TP1 order error [{pos_id}]: {_e}")
                    with self._lock:
                        self._balance += partial_pnl
                        self._total_pnl += partial_pnl
                        self._daily_pnl += partial_pnl
                        self._weekly_pnl += partial_pnl
                        self._peak_balance = max(self._peak_balance, self._balance)
                elif pos.tp1_hit and not pos.tp2_hit and current_price >= pos.tp2:
                    pos.tp2_hit = True
                    to_close.append((pos_id, current_price, "tp2_hit"))
                elif pos.tp1_hit and not pos.tp2_hit:
                    # Dynamic trailing stop from rolling standard deviation
                    new_sl = self._compute_dynamic_trailing_sl(pos, current_price)
                    if new_sl > pos.sl:
                        pos.sl = new_sl
                        logger.debug(f"📈 Trail SL [{pos_id}] {pos.symbol} → {new_sl:.4f}")
            else:
                if current_price >= pos.sl:
                    to_close.append((pos_id, current_price, "sl_hit"))
                elif not pos.tp1_hit and current_price <= pos.tp1:
                    pos.tp1_hit = True
                    # Partial take profit: chiude 50% della posizione
                    partial_size = pos.size / 2.0
                    partial_pnl = (pos.entry_price - current_price) * partial_size
                    pos.tp1_partial_pnl = partial_pnl
                    pos.size = partial_size
                    pos.sl = pos.entry_price
                    logger.info(
                        f"🎯 TP1 hit [{pos_id}] {pos.symbol} — "
                        f"closed 50%, trailing remaining 50% | partial_pnl={partial_pnl:+.4f}"
                    )
                    # In live trading, piazza ordine MARKET per la metà della size
                    if not self.paper_trading:
                        try:
                            place_futures_order(pos.symbol, "BUY", "MARKET", partial_size, reduce_only=True)
                        except Exception as _e:
                            logger.error(f"Partial TP1 order error [{pos_id}]: {_e}")
                    with self._lock:
                        self._balance += partial_pnl
                        self._total_pnl += partial_pnl
                        self._daily_pnl += partial_pnl
                        self._weekly_pnl += partial_pnl
                        self._peak_balance = max(self._peak_balance, self._balance)
                elif pos.tp1_hit and not pos.tp2_hit and current_price <= pos.tp2:
                    pos.tp2_hit = True
                    to_close.append((pos_id, current_price, "tp2_hit"))
                elif pos.tp1_hit and not pos.tp2_hit:
                    # Dynamic trailing stop from rolling standard deviation
                    new_sl = self._compute_dynamic_trailing_sl(pos, current_price)
                    if new_sl < pos.sl:
                        pos.sl = new_sl
                        logger.debug(f"📉 Trail SL [{pos_id}] {pos.symbol} → {new_sl:.4f}")

        for pos_id, price, reason in to_close:
            closed = self.close_position(pos_id, price, reason)
            if closed is not None:
                closed_positions.append(closed)

        return closed_positions

    def _compute_dynamic_trailing_sl(self, pos: Position, current_price: float) -> float:
        # Fallback trail distance is used for graceful degradation if
        # institutional risk utilities or market data are unavailable.
        fallback_distance = abs(pos.tp1 - pos.entry_price) * _TRAIL_STOP_RATIO
        fallback_sl = (
            current_price - fallback_distance
            if pos.direction == "long"
            else current_price + fallback_distance
        )
        try:
            df = data_store.get_df(pos.symbol, pos.interval)
            closes = [] if df is None else list(df["close"].tail(50).astype(float))
            if InstitutionalRiskManager is None:
                return float(fallback_sl)
            if self._institutional_risk_manager is None:
                self._institutional_risk_manager = InstitutionalRiskManager()
            return self._institutional_risk_manager.trailing_stop_from_std(
                direction=pos.direction,
                current_price=float(current_price),
                existing_sl=float(pos.sl),
                closes=closes,
            )
        except Exception:
            return float(fallback_sl)

    def set_standby(self, enabled: bool, reason: str = "") -> None:
        with self._lock:
            self._standby = bool(enabled)
            self._standby_reason = reason if enabled else ""
        status = "ON" if enabled else "OFF"
        logger.warning(f"🛑 Execution standby {status} {('- ' + reason) if reason else ''}")

    def is_standby(self) -> bool:
        with self._lock:
            return bool(self._standby)

    def cancel_pending_orders(self) -> int:
        # No pending order book is tracked internally in this engine version.
        return 0

    def close_all_positions(self, reason: str = "kill_switch") -> List[Position]:
        closed_positions: List[Position] = []
        with self._lock:
            snapshot = list(self._open_positions.items())
        for pos_id, pos in snapshot:
            close_price = pos.entry_price
            try:
                df = data_store.get_df(pos.symbol, pos.interval)
                if df is not None and len(df) > 0:
                    close_price = float(df["close"].iloc[-1])
            except Exception:
                pass
            closed = self.close_position(pos_id, close_price, reason=reason)
            if closed is not None:
                closed_positions.append(closed)
        return closed_positions

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        self._roll_day_if_needed()
        risk_blocked, risk_reason, _risk_details = self.is_risk_blocked()

        with self._lock:
            return {
                "paper_trading": self.paper_trading,
                "balance": self._balance,
                "initial_balance": self._initial_balance,
                "total_pnl": self._total_pnl,
                "pnl_pct": self._total_pnl / self._initial_balance * 100,
                "trade_count": self._trade_count,
                "win_count": self._win_count,
                "win_rate": self._win_count / max(self._trade_count, 1),
                "open_positions": len(self._open_positions),
                "daily_pnl": self._daily_pnl,
                "weekly_pnl": self._weekly_pnl,
                "consecutive_losses": self._consecutive_losses,
                "risk_blocked": risk_blocked,
                "risk_block_reason": risk_reason,
                "peak_balance": self._peak_balance,
                "total_drawdown_pct": (self._peak_balance - self._balance) / self._peak_balance * 100
                    if self._peak_balance > 0 else 0.0,
                "standby": self._standby,
                "standby_reason": self._standby_reason,
            }

    def get_open_positions(self) -> List[Position]:
        with self._lock:
            return list(self._open_positions.values())

    def get_closed_positions(self, limit: int = 50) -> List[Position]:
        with self._lock:
            return self._closed_positions[-limit:]
