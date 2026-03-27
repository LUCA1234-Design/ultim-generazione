"""
DataFrame store for V17.
Thread-safe in-memory store for historical and realtime OHLCV data.
"""
import logging
import pandas as pd
import numpy as np
from threading import Lock
from typing import Optional, Dict

logger = logging.getLogger("DataStore")

_lock = Lock()
_historical: Dict[str, Dict[str, pd.DataFrame]] = {}   # {symbol: {interval: df}}
_realtime: Dict[str, Dict[str, pd.DataFrame]] = {}     # {symbol: {interval: df}}

HISTORICAL_LIMIT = 500


def _parse_klines(klines: list) -> Optional[pd.DataFrame]:
    """Convert raw Binance klines list to a OHLCV DataFrame."""
    if not klines:
        return None
    rows = []
    for k in klines:
        rows.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": int(k[6]),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_vol": float(k[10]),
        })
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_vol"]]
    return df


def store_historical(symbol: str, interval: str, klines: list) -> None:
    """Store fetched historical klines for a symbol/interval."""
    df = _parse_klines(klines)
    if df is None or df.empty:
        return
    with _lock:
        if symbol not in _historical:
            _historical[symbol] = {}
        if symbol not in _realtime:
            _realtime[symbol] = {}
        _historical[symbol][interval] = df
        _realtime[symbol][interval] = df.copy()


def update_realtime(symbol: str, interval: str, kline: dict) -> None:
    """Update realtime DataFrame with a single kline from WebSocket."""
    try:
        open_time_ms = int(kline["t"])
        row = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline.get("q", 0)),
            "trades": int(kline.get("n", 0)),
            "taker_buy_vol": float(kline.get("V", 0)),
        }
        idx = pd.to_datetime(open_time_ms, unit="ms", utc=True)
        with _lock:
            if symbol not in _realtime:
                _realtime[symbol] = {}
            df = _realtime[symbol].get(interval)
            if df is None or df.empty:
                df = _historical.get(symbol, {}).get(interval)
                if df is None:
                    return
                df = df.copy()
            new_row = pd.DataFrame([row], index=[idx])
            if idx in df.index:
                df.loc[idx] = row
            else:
                df = pd.concat([df, new_row])
            if len(df) > HISTORICAL_LIMIT:
                df = df.iloc[-HISTORICAL_LIMIT:]
            _realtime[symbol][interval] = df
    except Exception as e:
        logger.debug(f"update_realtime {symbol} {interval}: {e}")


def get_df(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Return the most up-to-date DataFrame for a symbol/interval."""
    with _lock:
        rt = _realtime.get(symbol, {}).get(interval)
        if rt is not None and not rt.empty:
            return rt.copy()
        hist = _historical.get(symbol, {}).get(interval)
        if hist is not None and not hist.empty:
            return hist.copy()
    return None


def get_all_symbols() -> list:
    """Return list of all symbols currently stored."""
    with _lock:
        return list(_historical.keys())


def has_data(symbol: str, interval: str, min_rows: int = 50) -> bool:
    """Check if sufficient data exists for a symbol/interval."""
    df = get_df(symbol, interval)
    return df is not None and len(df) >= min_rows


def get_latest_close(symbol: str, interval: str) -> Optional[float]:
    """Return the latest close price."""
    df = get_df(symbol, interval)
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    return None


def get_latest_volume(symbol: str, interval: str) -> Optional[float]:
    """Return the latest candle volume."""
    df = get_df(symbol, interval)
    if df is not None and not df.empty:
        return float(df["volume"].iloc[-1])
    return None


def get_avg_volume(symbol: str, interval: str, lookback: int = 20) -> Optional[float]:
    """Return the average volume over the last `lookback` closed candles."""
    df = get_df(symbol, interval)
    if df is not None and len(df) > lookback:
        return float(df["volume"].iloc[-lookback - 1:-1].mean())
    return None


def clear_symbol(symbol: str) -> None:
    """Remove all stored data for a symbol."""
    with _lock:
        _historical.pop(symbol, None)
        _realtime.pop(symbol, None)
