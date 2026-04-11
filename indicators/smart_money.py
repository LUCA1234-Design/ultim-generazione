"""
Smart Money indicators for V17/V18.
Includes: CVD, Volume Delta, Liquidity Sweep, Break of Structure (BOS),
          Change of Character (CHoCH), Fair Value Gaps (FVG), CVD Divergence.
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

    Vectorised implementation — no Python loop.

    Returns a Series:
        +1 = bullish sweep (price swept lows, closed above)
        -1 = bearish sweep (price swept highs, closed below)
         0 = no sweep
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Rolling max/min of the previous `lookback` bars (shift excludes current bar)
    prev_high = high.shift(1).rolling(lookback).max()
    prev_low = low.shift(1).rolling(lookback).min()

    bear_sweep = (high > prev_high) & (close < prev_high)
    bull_sweep = (low < prev_low) & (close > prev_low)

    result = pd.Series(0, index=df.index, dtype=int)
    result[bull_sweep] = 1
    result[bear_sweep] = -1
    return result


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


# ---------------------------------------------------------------------------
# Break of Structure (BOS)
# ---------------------------------------------------------------------------

def detect_bos(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Break of Structure: close breaks through the previous swing high or low.

    The signal fires only on the first candle of the breakout (transition bar),
    so it does not repeat while price stays outside the prior range.

    Returns: +1 bullish BOS, -1 bearish BOS, 0 no event.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Swing levels computed from the `lookback` bars prior to each bar
    swing_high = high.shift(1).rolling(lookback).max()
    swing_low = low.shift(1).rolling(lookback).min()

    above = close > swing_high
    below = close < swing_low

    # Only the first bar that crosses (transition detection)
    bos_up = above & ~above.shift(1).fillna(False)
    bos_down = below & ~below.shift(1).fillna(False)

    result = pd.Series(0, index=df.index, dtype=int)
    result[bos_up] = 1
    result[bos_down] = -1
    return result


# ---------------------------------------------------------------------------
# Change of Character (CHoCH)
# ---------------------------------------------------------------------------

def detect_choch(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Change of Character: first BOS against the prevailing short-term trend.

    Trend context is determined by EMA(lookback):
    - Bullish CHoCH: price in downtrend (below EMA) then breaks above swing high
    - Bearish CHoCH: price in uptrend (above EMA) then breaks below swing low

    Returns: +1 bullish CHoCH, -1 bearish CHoCH, 0 none.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_line = close.ewm(span=lookback, min_periods=lookback).mean()
    in_uptrend = close > ema_line

    half = max(lookback // 2, 5)
    swing_high = high.shift(1).rolling(half).max()
    swing_low = low.shift(1).rolling(half).min()

    above = close > swing_high
    below = close < swing_low

    bos_up = above & ~above.shift(1).fillna(False)
    bos_down = below & ~below.shift(1).fillna(False)

    # CHoCH: BOS that opposes the prevailing trend (one bar ago)
    prev_downtrend = ~in_uptrend.shift(1).fillna(True)
    prev_uptrend = in_uptrend.shift(1).fillna(False)

    bull_choch = bos_up & prev_downtrend
    bear_choch = bos_down & prev_uptrend

    result = pd.Series(0, index=df.index, dtype=int)
    result[bull_choch] = 1
    result[bear_choch] = -1
    return result


# ---------------------------------------------------------------------------
# Fair Value Gap (FVG)
# ---------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame) -> pd.Series:
    """Fair Value Gap: 3-candle imbalance zone.

    At bar[i], evaluates the pattern formed by bars [i-2, i-1, i]:
    - Bullish FVG: high[i-2] < low[i]  → upward gap, imbalance zone below price
    - Bearish FVG: low[i-2]  > high[i] → downward gap, imbalance zone above price

    Returns: +1 bullish FVG, -1 bearish FVG, 0 none.
    """
    high = df["high"]
    low = df["low"]

    bull_fvg = low > high.shift(2)
    bear_fvg = high < low.shift(2)

    result = pd.Series(0, index=df.index, dtype=int)
    result[bull_fvg] = 1
    result[bear_fvg] = -1
    return result


# ---------------------------------------------------------------------------
# CVD Divergence
# ---------------------------------------------------------------------------

def detect_cvd_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Divergence between price and Cumulative Volume Delta (CVD).

    Uses net change over `lookback` bars as a slope proxy (fully vectorised):
    - price_slope > 0 but CVD_slope < 0 → bearish divergence (-1)
    - price_slope < 0 but CVD_slope > 0 → bullish divergence (+1)

    Returns: +1 bullish divergence, -1 bearish divergence, 0 none.
    """
    cvd, _ = cumulative_volume_delta(df)
    close = df["close"]

    price_slope = close.diff(lookback)
    cvd_slope = cvd.diff(lookback)

    bull_div = (price_slope < 0) & (cvd_slope > 0)
    bear_div = (price_slope > 0) & (cvd_slope < 0)

    result = pd.Series(0, index=df.index, dtype=int)
    result[bull_div] = 1
    result[bear_div] = -1
    return result
