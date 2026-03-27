"""
WebSocket manager for V17.
Multi-stream WebSocket connections to Binance Futures with:
- Exponential backoff reconnection
- REST fallback for missed candles
- Health monitoring
"""
import json
import logging
import ssl
import threading
import time
from typing import Callable, Dict, List, Optional

import websocket

from config.settings import (
    WS_GROUP_SIZE,
    WS_RECONNECT_DELAY_BASE,
    WS_MAX_RECONNECT_DELAY,
    POLL_CLOSED_ENABLE,
    POLL_CLOSED_INTERVAL,
    WS_STALE_TIMEOUT,
    WS_HEALTH_LOG_INTERVAL,
    WS_MAX_FAIL_COUNT_ALERT,
)

logger = logging.getLogger("WebSocketManager")

# ---------------------------------------------------------------------------
# Global WS health registry
# ---------------------------------------------------------------------------

WS_HEALTH: Dict[str, dict] = {}
_LAST_KLINE_TIME: Dict[str, int] = {}
_LAST_MESSAGE_TIME: Dict[str, float] = {}
_WS_FAILCOUNT: Dict[str, int] = {}
_WS_URLS: Dict[str, str] = {}

# Callbacks registered by the event processor
_on_kline_closed: Optional[Callable] = None
_on_kline_update: Optional[Callable] = None


def register_callbacks(on_closed: Callable, on_update: Callable) -> None:
    """Register callbacks for kline events."""
    global _on_kline_closed, _on_kline_update
    _on_kline_closed = on_closed
    _on_kline_update = on_update

def _init_ws_state(ws_name: str) -> None:
    WS_HEALTH[ws_name] = {
        "alive": False,
        "last_msg": 0.0,
        "last_error": None,
        "fail_count": _WS_FAILCOUNT.get(ws_name, 0),
        "restarts": 0,
    }
    _LAST_MESSAGE_TIME.setdefault(ws_name, 0.0)
    _WS_FAILCOUNT.setdefault(ws_name, 0)


# ---------------------------------------------------------------------------
# Stream URL helpers
# ---------------------------------------------------------------------------

def _build_stream_url(symbols_group: List[str], tf: str) -> str:
    streams = [f"{s.lower()}@kline_{tf}" for s in symbols_group]
    return "wss://fstream.binance.com/stream?streams=" + "/".join(streams)


def _split_into_groups(symbols: List[str], group_size: int = WS_GROUP_SIZE) -> List[List[str]]:
    groups = []
    current: List[str] = []
    for sym in symbols:
        current.append(sym)
        if len(current) >= group_size:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------

def _handle_message(ws_name: str, raw_message: str) -> None:
    try:
        data = json.loads(raw_message)
        stream_data = data.get("data", data)
        kline = stream_data.get("k")
        if not kline:
            return
        symbol: str = kline["s"]
        interval: str = kline["i"]
        is_closed: bool = kline.get("x", False)
        key = f"{symbol}_{interval}"

        WS_HEALTH[ws_name] = {"alive": True, "last_msg": time.time()}

        if _on_kline_update:
            _on_kline_update(symbol, interval, kline)

        if is_closed:
            open_time = int(kline["t"])
            if _LAST_KLINE_TIME.get(key) != open_time:
                _LAST_KLINE_TIME[key] = open_time
                if _on_kline_closed:
                    threading.Thread(
                        target=_on_kline_closed,
                        args=(symbol, interval, kline),
                        daemon=True,
                        name=f"kline-{symbol}-{interval}",
                    ).start()
    except Exception as e:
        logger.debug(f"WS message error [{ws_name}]: {e}")


# ---------------------------------------------------------------------------
# Single WebSocket runner (with exponential backoff)
# ---------------------------------------------------------------------------

def _run_ws(ws_name: str, url: str) -> None:
    retries = 0
    while True:
        try:
            ws_app = websocket.WebSocketApp(
                url,
                on_message=lambda ws, msg: _handle_message(ws_name, msg),
                on_error=lambda ws, err: logger.warning(f"WS [{ws_name}] error: {err}"),
                on_close=lambda ws, code, msg: logger.info(f"WS [{ws_name}] closed ({code})"),
                on_open=lambda ws: logger.debug(f"WS [{ws_name}] connected"),
            )
            WS_HEALTH[ws_name] = {"alive": True, "last_msg": time.time()}
            retries = 0  # Reset backoff after successful connection
            ws_app.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=0,  # Disable client-side ping; Binance server handles keepalive with its own pings
            )
        except Exception as e:
            logger.error(f"WS [{ws_name}] exception: {e}")
        WS_HEALTH[ws_name] = {"alive": False, "last_msg": time.time()}
        retries += 1
        wait = min(WS_RECONNECT_DELAY_BASE * (2 ** retries), WS_MAX_RECONNECT_DELAY)
        logger.info(f"WS [{ws_name}] reconnecting in {wait}s (attempt {retries})...")
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_websockets(symbols: List[str], timeframes: List[str] = None) -> int:
    """Start all WebSocket threads for the given symbols and timeframes.

    Returns the number of threads started.
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]
    count = 0
    for tf in timeframes:
        groups = _split_into_groups(symbols, WS_GROUP_SIZE)
        for i, group in enumerate(groups):
            ws_name = f"WS_{tf}_{i}"
            url = _build_stream_url(group, tf)
            t = threading.Thread(
                target=_run_ws,
                args=(ws_name, url),
                daemon=True,
                name=ws_name,
            )
            t.start()
            count += 1
            time.sleep(0.3)
    logger.info(f"🌐 {count} WebSocket threads started")
    return count


def startup_health_check(timeout: int = 25) -> bool:
    """Wait for all registered WS to become healthy.

    Returns True if all healthy within timeout, False otherwise.
    """
    start = time.time()
    logger.info("🔍 WebSocket health check started...")
    while time.time() - start < timeout:
        if not WS_HEALTH:
            time.sleep(1)
            continue
        now = time.time()
        all_ok = all(
            d.get("alive", False) and (now - d.get("last_msg", 0) < 15)
            for d in WS_HEALTH.values()
        )
        if all_ok:
            logger.info("✅ All WebSockets healthy")
            return True
        time.sleep(1)
    logger.warning(f"⚠️ Health check timeout after {timeout}s")
    return False


def get_health_summary() -> dict:
    """Return a summary of WS health for monitoring."""
    now = time.time()
    total = len(WS_HEALTH)
    alive = sum(1 for d in WS_HEALTH.values() if d.get("alive", False))
    stale = sum(
        1 for d in WS_HEALTH.values()
        if d.get("alive", False) and (now - d.get("last_msg", 0)) > 30
    )
    return {"total": total, "alive": alive, "stale": stale, "dead": total - alive}


# ---------------------------------------------------------------------------
# REST fallback poller
# ---------------------------------------------------------------------------

def start_rest_fallback(symbols: List[str], on_closed_candle: Callable) -> None:
    """Start the REST fallback polling thread."""
    if not POLL_CLOSED_ENABLE:
        return
    t = threading.Thread(
        target=_poll_loop,
        args=(symbols, on_closed_candle),
        daemon=True,
        name="POLL-CLOSED",
    )
    t.start()
    logger.info("🔄 REST fallback poller started")


def _get_tf_seconds(interval_str: str) -> int:
    mapping = {"15m": 900, "1h": 3600, "4h": 14400}
    return mapping.get(interval_str, 3600)


def _poll_loop(symbols: List[str], on_closed_candle: Callable) -> None:
    time.sleep(30)  # Let WS settle first
    from data import data_store
    from data.binance_client import fetch_futures_klines
    while True:
        try:
            for interval in ["15m", "1h", "4h"]:
                tf_seconds = _get_tf_seconds(interval)
                current_time = time.time()
                for symbol in symbols:
                    try:
                        df = data_store.get_df(symbol, interval)
                        if df is not None and not df.empty:
                            last_candle_time = df.index[-1].timestamp()
                            if (current_time - last_candle_time) < (tf_seconds + 60):
                                continue
                        kl = fetch_futures_klines(symbol, interval, limit=2)
                        if not kl or len(kl) < 2:
                            continue
                        last = kl[-1]
                        close_time_ms = int(last[6])
                        if close_time_ms > int(current_time * 1000):
                            continue
                        open_time = int(last[0])
                        key = f"{symbol}_{interval}"
                        if _LAST_KLINE_TIME.get(key) == open_time:
                            continue
                        k = {
                            "t": open_time, "o": last[1], "h": last[2], "l": last[3],
                            "c": last[4], "v": last[5], "V": last[10], "x": True,
                            "s": symbol, "i": interval,
                        }
                        _LAST_KLINE_TIME[key] = open_time
                        data_store.update_realtime(symbol, interval, k)
                        if interval in ("1h", "15m", "4h"):
                            on_closed_candle(symbol, interval, k)
                        time.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"[POLL-{interval}] {symbol}: {e}")
        except Exception as e:
            logger.error(f"[POLL-MASTER] {e}")
        time.sleep(POLL_CLOSED_INTERVAL)
