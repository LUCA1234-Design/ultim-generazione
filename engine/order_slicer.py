"""
engine/order_slicer.py — Advanced Order Execution Algorithms.
"""
import logging
import time
from dataclasses import dataclass
from typing import Tuple

from config.settings import (
    ORDER_ADV_LOOKBACK_CANDLES,
    ORDER_ICEBERG_N_SLICES,
    ORDER_ICEBERG_VISIBLE_PCT,
    ORDER_MARKET_THRESHOLD_PCT,
    ORDER_TWAP_INTERVAL_SEC,
    ORDER_TWAP_N_SLICES,
    ORDER_TWAP_TIMEOUT_SEC,
)
from data import data_store
from data.binance_client import (
    cancel_order,
    get_best_bid_ask,
    get_order_status,
    place_futures_order,
    place_limit_order,
)

logger = logging.getLogger("OrderSlicer")


@dataclass
class OrderResult:
    success: bool
    avg_fill_price: float
    total_filled_qty: float
    slippage_pct: float
    strategy_used: str
    n_orders_placed: int
    execution_time_sec: float


class TWAPSlicer:
    def __init__(self, n_slices: int = ORDER_TWAP_N_SLICES, interval_sec: int = ORDER_TWAP_INTERVAL_SEC,
                 timeout_sec: int = ORDER_TWAP_TIMEOUT_SEC):
        self.n_slices = max(1, int(n_slices))
        self.interval_sec = max(1, int(interval_sec))
        self.timeout_sec = max(5, int(timeout_sec))

    def execute(self, symbol: str, side: str, total_qty: float, entry_price: float) -> OrderResult:
        start = time.time()
        slice_qty = total_qty / self.n_slices
        filled_qty = 0.0
        cost = 0.0
        n_orders = 0
        n_market_fallbacks = 0

        for i in range(self.n_slices):
            bid, ask = get_best_bid_ask(symbol)
            px = bid if side == "BUY" else ask
            if px <= 0:
                px = entry_price

            order = place_limit_order(symbol, side, slice_qty, px)
            n_orders += 1
            if not order:
                market = place_futures_order(symbol, side, "MARKET", slice_qty)
                n_orders += 1
                n_market_fallbacks += 1
                if market:
                    fq = float(market.get("executedQty", slice_qty) or slice_qty)
                    fp = float(market.get("avgPrice", px) or px)
                    filled_qty += fq
                    cost += fq * fp
                continue

            order_id = order.get("orderId")
            st = time.time()
            done = False
            last_status = None
            while time.time() - st < self.timeout_sec:
                last_status = get_order_status(symbol, order_id)
                if last_status and last_status.get("status") in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                    done = True
                    break
                time.sleep(1)

            if not done:
                cancel_order(symbol, order_id)
                n_market_fallbacks += 1
                market = place_futures_order(symbol, side, "MARKET", slice_qty)
                n_orders += 1
                if market:
                    fq = float(market.get("executedQty", slice_qty) or slice_qty)
                    fp = float(market.get("avgPrice", px) or px)
                    filled_qty += fq
                    cost += fq * fp
            else:
                if last_status and last_status.get("status") == "FILLED":
                    fq = float(last_status.get("executedQty", slice_qty) or slice_qty)
                    fp = float(last_status.get("avgPrice", px) or px)
                    filled_qty += fq
                    cost += fq * fp
                else:
                    executed = float((last_status or {}).get("executedQty", 0.0) or 0.0)
                    if executed > 0:
                        fp = float((last_status or {}).get("avgPrice", px) or px)
                        filled_qty += executed
                        cost += executed * fp

            if i < self.n_slices - 1:
                time.sleep(self.interval_sec)

        avg_fill = (cost / filled_qty) if filled_qty > 1e-12 else 0.0
        slippage = abs(avg_fill - entry_price) / entry_price * 100.0 if entry_price > 0 and avg_fill > 0 else 0.0
        logger.info(
            "🧊 TWAP %s %s: filled=%.6f/%0.6f avg=%.4f fallbacks=%s",
            symbol, side, filled_qty, total_qty, avg_fill, n_market_fallbacks,
        )
        return OrderResult(
            success=filled_qty > 0,
            avg_fill_price=avg_fill if avg_fill > 0 else entry_price,
            total_filled_qty=filled_qty,
            slippage_pct=slippage,
            strategy_used="twap",
            n_orders_placed=n_orders,
            execution_time_sec=time.time() - start,
        )


class IcebergOrder:
    def __init__(self, n_slices: int = ORDER_ICEBERG_N_SLICES, visible_pct: float = ORDER_ICEBERG_VISIBLE_PCT,
                 timeout_sec: int = 30, price_drift_pct: float = 0.1):
        self.n_slices = max(1, int(n_slices))
        self.visible_pct = max(0.05, min(float(visible_pct), 1.0))
        self.timeout_sec = max(5, int(timeout_sec))
        self.price_drift_pct = max(0.01, float(price_drift_pct))

    def execute(self, symbol: str, side: str, total_qty: float, entry_price: float) -> OrderResult:
        start = time.time()
        per_slice_qty = total_qty / self.n_slices
        visible_qty = max(per_slice_qty * self.visible_pct, total_qty * 0.01)
        remaining = total_qty
        filled_qty = 0.0
        cost = 0.0
        n_orders = 0

        while remaining > 1e-8:
            bid, ask = get_best_bid_ask(symbol)
            px = bid if side == "BUY" else ask
            if px <= 0:
                px = entry_price

            drift = abs(px - entry_price) / max(entry_price, 1e-8) * 100.0
            if drift > self.price_drift_pct:
                logger.info("🧊 Iceberg drift pause %s: %.3f%%", symbol, drift)
                time.sleep(1)
                continue

            qty = min(visible_qty, remaining)
            order = place_limit_order(symbol, side, qty, px)
            n_orders += 1
            if not order:
                market = place_futures_order(symbol, side, "MARKET", qty)
                n_orders += 1
                if market:
                    fq = float(market.get("executedQty", qty) or qty)
                    fp = float(market.get("avgPrice", px) or px)
                    filled_qty += fq
                    remaining -= fq
                    cost += fq * fp
                continue

            oid = order.get("orderId")
            st = time.time()
            status = None
            while time.time() - st < self.timeout_sec:
                status = get_order_status(symbol, oid)
                if status and status.get("status") in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                    break
                time.sleep(1)

            status = status or get_order_status(symbol, oid) or {}
            executed = float(status.get("executedQty", 0.0) or 0.0)
            avg_price = float(status.get("avgPrice", px) or px)
            if status.get("status") != "FILLED":
                cancel_order(symbol, oid)
            if executed <= 0.0:
                market = place_futures_order(symbol, side, "MARKET", qty)
                n_orders += 1
                if market:
                    executed = float(market.get("executedQty", qty) or qty)
                    avg_price = float(market.get("avgPrice", px) or px)

            if executed > 0.0:
                filled_qty += executed
                remaining -= executed
                cost += executed * avg_price

        avg_fill = (cost / filled_qty) if filled_qty > 1e-12 else 0.0
        slippage = abs(avg_fill - entry_price) / entry_price * 100.0 if entry_price > 0 and avg_fill > 0 else 0.0
        return OrderResult(
            success=filled_qty > 0,
            avg_fill_price=avg_fill if avg_fill > 0 else entry_price,
            total_filled_qty=filled_qty,
            slippage_pct=slippage,
            strategy_used="iceberg",
            n_orders_placed=n_orders,
            execution_time_sec=time.time() - start,
        )


class SmartOrderRouter:
    def _compute_adv_usdt(self, symbol: str) -> float:
        df = data_store.get_df(symbol, "15m")
        if df is None or df.empty:
            return 0.0
        lookback = min(len(df), ORDER_ADV_LOOKBACK_CANDLES)
        recent = df.iloc[-lookback:]
        if "quote_volume" in recent.columns:
            return float(recent["quote_volume"].astype(float).mean() * 96.0)
        return float((recent["close"].astype(float) * recent["volume"].astype(float)).mean() * 96.0)

    def route_order(self, symbol: str, side: str, total_size_usdt: float, entry_price: float) -> OrderResult:
        start = time.time()
        total_qty = total_size_usdt / max(entry_price, 1e-8)
        adv_usdt = self._compute_adv_usdt(symbol)

        ratio = (total_size_usdt / adv_usdt) if adv_usdt > 1e-8 else 0.0

        if ratio < ORDER_MARKET_THRESHOLD_PCT:
            order = place_futures_order(symbol, side, "MARKET", total_qty)
            if not order:
                return OrderResult(False, entry_price, 0.0, 0.0, "market", 1, time.time() - start)
            avg_fill = float(order.get("avgPrice", entry_price) or entry_price)
            filled = float(order.get("executedQty", total_qty) or total_qty)
            slip = abs(avg_fill - entry_price) / max(entry_price, 1e-8) * 100.0
            return OrderResult(True, avg_fill, filled, slip, "market", 1, time.time() - start)

        if ratio < 0.01:
            twap = TWAPSlicer(n_slices=ORDER_TWAP_N_SLICES, interval_sec=ORDER_TWAP_INTERVAL_SEC, timeout_sec=ORDER_TWAP_TIMEOUT_SEC)
            return twap.execute(symbol, side, total_qty, entry_price)

        iceberg = IcebergOrder(n_slices=ORDER_ICEBERG_N_SLICES, visible_pct=ORDER_ICEBERG_VISIBLE_PCT)
        return iceberg.execute(symbol, side, total_qty, entry_price)
