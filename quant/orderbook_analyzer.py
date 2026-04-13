"""
quant/orderbook_analyzer.py — Order Book Proxy Analysis.

Since live L2 order book data is not available in the current WebSocket
setup, this module constructs order book *proxies* from OHLCV data using
well-established heuristics:

1. **Bid-Ask Imbalance Proxy**
   - Uses the within-candle price range and close position to infer
     the relative weight of buy vs sell pressure.
   - Close near high → bids dominate; close near low → asks dominate.

2. **Iceberg Detection Proxy**
   - Detects candles with anomalously high volume but small price
     movement — a signature of hidden liquidity absorbing flow.

3. **Absorption Score**
   - Combines imbalance + iceberg signals into a single absorption
     score measuring how much supply/demand is being absorbed at
     current price levels.

Integration: used by OrderFlowAgent to complement VPIN/Kyle signals.
"""
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("OrderBookAnalyzer")

try:
    from config.settings import ORDERBOOK_STALE_SEC
except Exception:
    ORDERBOOK_STALE_SEC = 5

try:
    from data.orderbook_stream import get_orderbook_snapshot
    _REAL_ORDERBOOK_AVAILABLE = True
except Exception:
    _REAL_ORDERBOOK_AVAILABLE = False

# Tuning constants
_ICEBERG_VOL_ZSCORE = 2.0    # volume z-score threshold for anomaly
_ICEBERG_RANGE_PCT = 0.002   # max price range (as fraction of close) for iceberg
_WINDOW = 20


def compute_imbalance_proxy(df: pd.DataFrame, window: int = _WINDOW) -> Optional[pd.Series]:
    """
    Bid-ask imbalance proxy derived from candle close position within range.

    Formula:
        imbalance = (close - low) / (high - low)  → [0, 1]
        0 = close at low (ask side dominates)
        1 = close at high (bid side dominates)
        0.5 = balanced

    Rolling mean over `window` bars smooths the signal.

    Returns
    -------
    pd.Series in [0, 1], or None on error.
    """
    if df is None or len(df) < window:
        return None
    try:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        hl_range = high - low
        hl_range = np.where(hl_range < 1e-10, 1e-10, hl_range)
        raw_imbalance = (close - low) / hl_range

        series = pd.Series(raw_imbalance, index=df.index)
        smoothed = series.rolling(window).mean().fillna(0.5)
        return smoothed
    except Exception as exc:
        logger.debug(f"compute_imbalance_proxy error: {exc}")
        return None


def detect_iceberg_proxy(df: pd.DataFrame,
                         vol_zscore_thresh: float = _ICEBERG_VOL_ZSCORE,
                         range_pct_thresh: float = _ICEBERG_RANGE_PCT,
                         window: int = _WINDOW) -> Optional[pd.Series]:
    """
    Detect potential iceberg orders: high volume with small price range.

    Returns a boolean Series (True = potential iceberg candle).

    An iceberg candle satisfies:
      - volume > mean + vol_zscore_thresh * std  (anomalously high volume)
      - (high - low) / close < range_pct_thresh  (small price movement)

    Parameters
    ----------
    df              : OHLCV DataFrame
    vol_zscore_thresh : standard deviations above mean volume to flag
    range_pct_thresh  : max range as fraction of close price
    window          : lookback window for volume statistics

    Returns
    -------
    pd.Series[bool] or None.
    """
    if df is None or len(df) < window:
        return None
    try:
        vol = df["volume"].values.astype(float)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        roll_mean = pd.Series(vol).rolling(window).mean().values
        roll_std = pd.Series(vol).rolling(window).std().values + 1e-8

        high_vol_mask = vol > (roll_mean + vol_zscore_thresh * roll_std)
        small_range_mask = (high - low) / np.maximum(close, 1e-8) < range_pct_thresh

        iceberg = high_vol_mask & small_range_mask
        return pd.Series(iceberg, index=df.index)
    except Exception as exc:
        logger.debug(f"detect_iceberg_proxy error: {exc}")
        return None


def absorption_score(df: pd.DataFrame, window: int = _WINDOW) -> Optional[float]:
    """
    Aggregate absorption score combining imbalance + iceberg detection.

    Score interpretation:
        > 0  : buying absorption (bids absorbing selling pressure)
        < 0  : selling absorption (asks absorbing buying pressure)
        ≈ 0  : balanced

    Returns
    -------
    float in [-1, +1] or None on error.
    """
    if df is None or len(df) < window:
        return None
    try:
        imb = compute_imbalance_proxy(df, window=window)
        ice = detect_iceberg_proxy(df, window=window)

        score = 0.0
        components = 0

        if imb is not None and not imb.dropna().empty:
            last_imb = float(imb.dropna().iloc[-1])
            # map [0,1] → [-1, +1]: 0.5=neutral → 0
            score += 2.0 * (last_imb - 0.5)
            components += 1

        if ice is not None and not ice.dropna().empty:
            # Recent iceberg detections — look at last 5 candles
            recent_icebergs = ice.dropna().iloc[-5:]
            if recent_icebergs.any():
                # Determine direction of absorption by checking recent imbalance
                if imb is not None and not imb.dropna().empty:
                    imb_at_iceberg = float(imb.dropna().iloc[-1])
                    # Iceberg absorbing sells (high imbalance) = buying absorption
                    ice_signal = 1.0 if imb_at_iceberg > 0.5 else -1.0
                    score += ice_signal * 0.5
                    components += 1

        if components == 0:
            return 0.0
        return float(np.clip(score / components, -1.0, 1.0))
    except Exception as exc:
        logger.debug(f"absorption_score error: {exc}")
        return 0.0


def get_orderbook_signal(df: pd.DataFrame) -> dict:
    """
    Compute all order book proxy signals and return a summary dict.

    Returns
    -------
    dict with keys:
        'imbalance'      : float [0,1] — latest imbalance proxy
        'iceberg_recent' : bool — any iceberg in last 5 candles
        'absorption'     : float [-1,1] — combined score
        'direction'      : str — 'long' | 'short' | 'neutral'
    """
    result = {
        "imbalance": 0.5,
        "iceberg_recent": False,
        "absorption": 0.0,
        "direction": "neutral",
    }

    try:
        imb = compute_imbalance_proxy(df)
        ice = detect_iceberg_proxy(df)
        ab = absorption_score(df)

        if imb is not None and not imb.dropna().empty:
            result["imbalance"] = float(imb.dropna().iloc[-1])

        if ice is not None and not ice.dropna().empty:
            result["iceberg_recent"] = bool(ice.dropna().iloc[-5:].any())

        if ab is not None:
            result["absorption"] = float(ab)
            if ab > 0.3:
                result["direction"] = "long"
            elif ab < -0.3:
                result["direction"] = "short"
    except Exception as exc:
        logger.debug(f"get_orderbook_signal error: {exc}")

    return result


def get_realtime_orderbook_signal(symbol: str, df: Optional[pd.DataFrame] = None) -> dict:
    """Use real L2 snapshot if fresh, otherwise fallback to OHLCV proxies."""
    proxy = get_orderbook_signal(df) if df is not None else {
        "imbalance": 0.5,
        "iceberg_recent": False,
        "absorption": 0.0,
        "direction": "neutral",
    }
    result = {
        "real_imbalance": float(proxy.get("imbalance", 0.5)),
        "order_flow_imbalance": 0.0,
        "absorption": float(proxy.get("absorption", 0.0)),
        "using_real_data": False,
        "direction": proxy.get("direction", "neutral"),
        "iceberg_recent": bool(proxy.get("iceberg_recent", False)),
    }
    if not _REAL_ORDERBOOK_AVAILABLE:
        return result

    try:
        snap = get_orderbook_snapshot(symbol)
        if not snap:
            return result
        age = time.time() - float(snap.get("last_update_ts", 0.0))
        if age > ORDERBOOK_STALE_SEC:
            return result

        buy_vol = float(snap.get("cumulative_buy_volume", 0.0))
        sell_vol = float(snap.get("cumulative_sell_volume", 0.0))
        flow_denom = buy_vol + sell_vol
        flow_imb = ((buy_vol - sell_vol) / flow_denom) if flow_denom > 1e-12 else 0.0
        real_imb = float(snap.get("bid_ask_imbalance", 0.5))

        absorption = 0.0
        if real_imb > 0.65 and flow_imb > 0.1:
            absorption = 1.0
        elif real_imb < 0.35 and flow_imb < -0.1:
            absorption = -1.0

        result.update({
            "real_imbalance": real_imb,
            "order_flow_imbalance": float(np.clip(flow_imb, -1.0, 1.0)),
            "absorption": float(absorption),
            "using_real_data": True,
            "direction": "long" if absorption > 0 else ("short" if absorption < 0 else "neutral"),
            "iceberg_recent": False,
        })
    except Exception as exc:
        logger.debug(f"get_realtime_orderbook_signal error: {exc}")
    return result
