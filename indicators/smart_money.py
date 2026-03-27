"""
Smart Money indicators for V17.
Includes: CVD, Volume Delta, Liquidity Sweep detection.
"""
import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# CVD — Cumulative Volume Delta
# ---------------------------------------------------------------------------

def cumulative_volume_delta(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (CVD, delta_per_bar).

    Delta = buy_volume - sell_volume, estimated from candle body/range ratio.
    """
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    total_range = (high - low).replace(0, np.nan)
    buy_pct = ((close - low) / total_range).clip(0, 1)
    buy_vol = buy_pct * volume
    sell_vol = (1 - buy_pct) * volume
    delta = buy_vol - sell_vol
    cvd = delta.cumsum()
    return cvd, delta


def volume_delta(df: pd.DataFrame) -> pd.Series:
    """Per-bar volume delta (buy vol - sell vol)."""
    _, delta = cumulative_volume_delta(df)
    return delta


# ---------------------------------------------------------------------------
# Taker-based delta (more accurate if taker_buy_vol available)
# ---------------------------------------------------------------------------

def taker_delta(df: pd.DataFrame) -> pd.Series:
    """Volume delta using taker buy volume when available."""
    if "taker_buy_vol" not in df.columns:
        return volume_delta(df)
    buy_vol = df["taker_buy_vol"]
    sell_vol = df["volume"] - buy_vol
    return buy_vol - sell_vol


def cumulative_taker_delta(df: pd.DataFrame) -> pd.Series:
    """Cumulative taker delta."""
    return taker_delta(df).cumsum()


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------

def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Detect liquidity sweeps: price wicks beyond recent high/low then closes back.

    Returns a Series:
        +1 = bullish sweep (price swept lows, closed above)
        -1 = bearish sweep (price swept highs, closed below)
         0 = no sweep
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    sweeps = pd.Series(0, index=df.index, dtype=int)

    for i in range(lookback, len(df)):
        recent_high = high.iloc[i - lookback:i].max()
        recent_low = low.iloc[i - lookback:i].min()
        c = close.iloc[i]
        h = high.iloc[i]
        lo = low.iloc[i]
        if h > recent_high and c < recent_high:
            sweeps.iloc[i] = -1   # swept highs → bearish
        elif lo < recent_low and c > recent_low:
            sweeps.iloc[i] = 1    # swept lows → bullish
    return sweeps


# ---------------------------------------------------------------------------
# Order Block detection
# ---------------------------------------------------------------------------

def detect_order_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Simple order block detection based on large-body candles followed by strong moves.

    Returns Series with 1 (bullish OB), -1 (bearish OB), 0 (none).
    """
    close = df["close"]
    open_ = df["open"]
    volume = df["volume"]
    body = (close - open_).abs()
    avg_body = body.rolling(lookback).mean()
    avg_vol = volume.rolling(lookback).mean()

    ob = pd.Series(0, index=df.index, dtype=int)
    for i in range(lookback, len(df) - 1):
        if body.iloc[i] > 1.5 * avg_body.iloc[i] and volume.iloc[i] > 1.5 * avg_vol.iloc[i]:
            # Strong bullish candle
            if close.iloc[i] > open_.iloc[i]:
                ob.iloc[i] = 1
            else:
                ob.iloc[i] = -1
    return ob


# ---------------------------------------------------------------------------
# Volume Profile (simple approximation)
# ---------------------------------------------------------------------------

def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Compute a simple volume profile (price levels vs accumulated volume).

    Returns a DataFrame with columns ['price_level', 'volume'].
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_min >= price_max:
        return pd.DataFrame(columns=["price_level", "volume"])
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volume = np.zeros(bins)
    for _, row in df.iterrows():
        # distribute volume proportionally across bins covered by the candle
        candle_low = row["low"]
        candle_high = row["high"]
        candle_vol = row["volume"]
        for b in range(bins):
            lo_bin = bin_edges[b]
            hi_bin = bin_edges[b + 1]
            overlap_lo = max(candle_low, lo_bin)
            overlap_hi = min(candle_high, hi_bin)
            if overlap_hi > overlap_lo:
                candle_range = candle_high - candle_low if candle_high > candle_low else 1e-10
                fraction = (overlap_hi - overlap_lo) / candle_range
                bin_volume[b] += candle_vol * fraction
    price_levels = (bin_edges[:-1] + bin_edges[1:]) / 2
    return pd.DataFrame({"price_level": price_levels, "volume": bin_volume})


def poc(df: pd.DataFrame, bins: int = 20) -> float:
    """Point of Control: price level with highest accumulated volume."""
    vp = volume_profile(df, bins)
    if vp.empty:
        return float(df["close"].iloc[-1])
    return float(vp.loc[vp["volume"].idxmax(), "price_level"])
