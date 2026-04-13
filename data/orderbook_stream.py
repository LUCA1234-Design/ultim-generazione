"""
data/orderbook_stream.py — Real-time L2 order book and trade flow streams.
"""
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import websocket

from config.settings import (
    ORDERBOOK_MAX_SYMBOLS,
    ORDERBOOK_TRADE_WINDOW_SEC,
    ORDERBOOK_STALE_SEC,
    ORDERBOOK_UPDATE_MS,
    ORDERBOOK_DEPTH_LEVELS,
    ORDERBOOK_STREAM_ENABLED,
)

logger = logging.getLogger("OrderBookStream")

_orderbook_cache: Dict[str, dict] = {}
_threads: Dict[str, threading.Thread] = {}
_cache_lock = threading.Lock()

_RECONNECT_BASE_SEC = 2
_RECONNECT_MAX_SEC = 60


def _stream_url(symbol: str) -> str:
    sym = symbol.lower()
    depth_stream = f"{sym}@depth{ORDERBOOK_DEPTH_LEVELS}@{ORDERBOOK_UPDATE_MS}ms"
    return f"wss://fstream.binance.com/stream?streams={depth_stream}/{sym}@aggTrade"


def _init_symbol(symbol: str) -> None:
    with _cache_lock:
        _orderbook_cache.setdefault(symbol, {
            "symbol": symbol,
            "bids": [],
            "asks": [],
            "bid_ask_imbalance": 0.5,
            "last_update_ts": 0.0,
            "cumulative_buy_volume": 0.0,
            "cumulative_sell_volume": 0.0,
            "trade_window_start_ts": time.time(),
        })


def _apply_trade_window_decay(snapshot: dict) -> None:
    now = time.time()
    start_ts = float(snapshot.get("trade_window_start_ts", now))
    if (now - start_ts) >= ORDERBOOK_TRADE_WINDOW_SEC:
        snapshot["cumulative_buy_volume"] = 0.0
        snapshot["cumulative_sell_volume"] = 0.0
        snapshot["trade_window_start_ts"] = now


def _on_depth(symbol: str, data: dict) -> None:
    bids_raw = data.get("b", [])[:ORDERBOOK_DEPTH_LEVELS]
    asks_raw = data.get("a", [])[:ORDERBOOK_DEPTH_LEVELS]

    bids: List[Tuple[float, float]] = [(float(p), float(q)) for p, q in bids_raw]
    asks: List[Tuple[float, float]] = [(float(p), float(q)) for p, q in asks_raw]

    bid_qty = sum(q for _, q in bids)
    ask_qty = sum(q for _, q in asks)
    denom = bid_qty + ask_qty
    imbalance = (bid_qty / denom) if denom > 1e-12 else 0.5

    with _cache_lock:
        snapshot = _orderbook_cache.get(symbol)
        if snapshot is None:
            return
        _apply_trade_window_decay(snapshot)
        snapshot["bids"] = bids
        snapshot["asks"] = asks
        snapshot["bid_ask_imbalance"] = float(imbalance)
        snapshot["last_update_ts"] = time.time()


def _on_agg_trade(symbol: str, data: dict) -> None:
    qty = float(data.get("q", 0.0) or 0.0)
    is_buyer_maker = bool(data.get("m", False))

    with _cache_lock:
        snapshot = _orderbook_cache.get(symbol)
        if snapshot is None:
            return
        _apply_trade_window_decay(snapshot)
        if is_buyer_maker:
            snapshot["cumulative_sell_volume"] = float(snapshot.get("cumulative_sell_volume", 0.0) + qty)
        else:
            snapshot["cumulative_buy_volume"] = float(snapshot.get("cumulative_buy_volume", 0.0) + qty)
        snapshot["last_update_ts"] = time.time()


def _run_symbol_stream(symbol: str) -> None:
    _init_symbol(symbol)
    url = _stream_url(symbol)
    retries = 0

    while True:
        try:
            def _on_message(_ws, raw_message: str):
                try:
                    msg = json.loads(raw_message)
                    data = msg.get("data", msg)
                    event_type = data.get("e", "")
                    if event_type == "depthUpdate":
                        _on_depth(symbol, data)
                    elif event_type == "aggTrade":
                        _on_agg_trade(symbol, data)
                except Exception as exc:
                    logger.debug(f"OrderBookStream[{symbol}] message parse error: {exc}")

            def _on_open(_ws):
                logger.info(f"📡 L2 stream connected: {symbol}")

            def _on_error(_ws, err):
                logger.warning(f"OrderBookStream[{symbol}] error: {err}")

            def _on_close(_ws, code, msg):
                logger.info(f"OrderBookStream[{symbol}] closed ({code})")

            ws_app = websocket.WebSocketApp(
                url,
                on_message=_on_message,
                on_open=_on_open,
                on_error=_on_error,
                on_close=_on_close,
            )
            retries = 0
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            logger.warning(f"OrderBookStream[{symbol}] run exception: {exc}")

        retries += 1
        wait = min(_RECONNECT_BASE_SEC * (2 ** min(retries, 6)), _RECONNECT_MAX_SEC)
        logger.info(f"OrderBookStream[{symbol}] reconnecting in {wait}s...")
        time.sleep(wait)


def start_orderbook_streams(symbols: List[str]) -> None:
    """Start L2 streams for up to ORDERBOOK_MAX_SYMBOLS symbols."""
    if not ORDERBOOK_STREAM_ENABLED:
        logger.info("OrderBook stream disabled by settings")
        return

    selected = [s.upper() for s in (symbols or [])][:ORDERBOOK_MAX_SYMBOLS]
    for symbol in selected:
        if symbol in _threads:
            continue
        t = threading.Thread(
            target=_run_symbol_stream,
            args=(symbol,),
            daemon=True,
            name=f"OB-{symbol}",
        )
        _threads[symbol] = t
        t.start()
        time.sleep(0.1)
    logger.info(f"📡 L2 stream threads active: {len(_threads)}")


def get_orderbook_snapshot(symbol: str) -> Optional[dict]:
    sym = symbol.upper()
    with _cache_lock:
        snap = _orderbook_cache.get(sym)
        if not snap:
            return None
        _apply_trade_window_decay(snap)
        return {
            "symbol": snap.get("symbol"),
            "bids": list(snap.get("bids", [])),
            "asks": list(snap.get("asks", [])),
            "bid_ask_imbalance": float(snap.get("bid_ask_imbalance", 0.5)),
            "last_update_ts": float(snap.get("last_update_ts", 0.0)),
            "cumulative_buy_volume": float(snap.get("cumulative_buy_volume", 0.0)),
            "cumulative_sell_volume": float(snap.get("cumulative_sell_volume", 0.0)),
        }


def get_real_imbalance(symbol: str) -> Optional[float]:
    snap = get_orderbook_snapshot(symbol)
    if not snap:
        return None
    age = time.time() - float(snap.get("last_update_ts", 0.0))
    if age > ORDERBOOK_STALE_SEC:
        return None
    return float(snap.get("bid_ask_imbalance", 0.5))


def get_real_trade_flow(symbol: str) -> Optional[dict]:
    snap = get_orderbook_snapshot(symbol)
    if not snap:
        return None
    age = time.time() - float(snap.get("last_update_ts", 0.0))
    if age > ORDERBOOK_STALE_SEC:
        return None
    return {
        "cumulative_buy_volume": float(snap.get("cumulative_buy_volume", 0.0)),
        "cumulative_sell_volume": float(snap.get("cumulative_sell_volume", 0.0)),
    }
