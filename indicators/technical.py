"""
Technical indicators for V17/V18.
All indicators accept pandas Series / DataFrame and return pandas objects.
Includes: RSI, ATR, MACD, OBV, Bollinger Bands, Keltner Channels, ADX, Z-Score,
          VWAP, Anchored VWAP, Supertrend.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using exponential moving averages."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    h = df["high"]
    lo = df["low"]
    c = df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(series: pd.Series, period: int = 20,
                    num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle (SMA), lower band."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + num_std * std, ma, ma - num_std * std


# ---------------------------------------------------------------------------
# Keltner Channels
# ---------------------------------------------------------------------------

def keltner_channels(df: pd.DataFrame, period: int = 20,
                     atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper channel, middle (EMA), lower channel."""
    ema = df["close"].ewm(span=period, min_periods=period).mean()
    _atr = atr(df, period)
    return ema + atr_mult * _atr, ema, ema - atr_mult * _atr


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ADX, +DI, -DI."""
    h = df["high"]
    lo = df["low"]
    c = df["close"]
    prev_h = h.shift(1)
    prev_lo = lo.shift(1)
    prev_c = c.shift(1)

    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    dm_plus_raw = np.where((h - prev_h) > (prev_lo - lo), np.maximum(h - prev_h, 0), 0)
    dm_minus_raw = np.where((prev_lo - lo) > (h - prev_h), np.maximum(prev_lo - lo, 0), 0)

    dm_plus = pd.Series(dm_plus_raw, index=df.index)
    dm_minus = pd.Series(dm_minus_raw, index=df.index)

    _atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / period, min_periods=period).mean() / _atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, min_periods=period).mean() / _atr.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx_line, di_plus, di_minus


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-score of a price series."""
    rolling_mean = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# EMA slope (for regime detection)
# ---------------------------------------------------------------------------

def ema_slope(series: pd.Series, ema_period: int = 20, slope_lookback: int = 5) -> pd.Series:
    """Normalised slope of EMA over `slope_lookback` bars."""
    _ema = ema(series, ema_period)
    slope = _ema.diff(slope_lookback) / _ema.shift(slope_lookback).replace(0, np.nan)
    return slope


# ---------------------------------------------------------------------------
# BB / Keltner squeeze indicator
# ---------------------------------------------------------------------------

def squeeze_intensity(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                      kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Returns 1 where BB is inside KC (squeeze), 0 otherwise."""
    bb_upper, _, bb_lower = bollinger_bands(df["close"], bb_period, bb_std)
    kc_upper, _, kc_lower = keltner_channels(df, kc_period, kc_mult)
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)
    return squeeze


# ---------------------------------------------------------------------------
# Volume ratio
# ---------------------------------------------------------------------------

def volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Current volume divided by rolling average volume."""
    avg = df["volume"].rolling(lookback).mean()
    return df["volume"] / avg.replace(0, np.nan)


# ---------------------------------------------------------------------------
# VWAP — Volume Weighted Average Price
# ---------------------------------------------------------------------------

def vwap(df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
    """Volume Weighted Average Price.

    Parameters
    ----------
    df     : OHLCV DataFrame
    period : rolling window in bars.  If None, computes cumulative VWAP
             from the first bar of the DataFrame (full-history mode).

    Returns
    -------
    pd.Series of VWAP values.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"]
    if period is None:
        return tp_vol.cumsum() / df["volume"].cumsum().replace(0, np.nan)
    return (
        tp_vol.rolling(period).sum()
        / df["volume"].rolling(period).sum().replace(0, np.nan)
    )


def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """VWAP anchored to a specific bar index.

    Parameters
    ----------
    df         : OHLCV DataFrame
    anchor_idx : bar index (0-based) from which accumulation starts.
                 Negative values count from the end.

    Returns
    -------
    pd.Series — NaN before *anchor_idx*, VWAP values from *anchor_idx* onward.
    """
    n = len(df)
    if anchor_idx < 0:
        anchor_idx = max(0, n + anchor_idx)
    anchor_idx = max(0, min(anchor_idx, n - 1))

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"]
    avwap = tp_vol.iloc[anchor_idx:].cumsum() / (
        df["volume"].iloc[anchor_idx:].cumsum().replace(0, np.nan)
    )
    result = pd.Series(np.nan, index=df.index)
    result.iloc[anchor_idx:] = avwap.values
    return result


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Uses a recursive trailing-stop algorithm on ATR-based upper/lower bands.

    Parameters
    ----------
    df         : OHLCV DataFrame
    period     : ATR period
    multiplier : ATR band multiplier

    Returns
    -------
    st_line   : pd.Series — trailing stop value
    direction : pd.Series[int] — +1 uptrend (price above band), -1 downtrend
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    _atr = atr(df, period)
    basic_upper = (hl2 + multiplier * _atr).values
    basic_lower = (hl2 - multiplier * _atr).values
    close = df["close"].values
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    valid = np.where(~np.isnan(basic_upper))[0]
    if len(valid) == 0:
        return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)

    i0 = int(valid[0])
    final_upper[i0] = basic_upper[i0]
    final_lower[i0] = basic_lower[i0]
    st[i0] = basic_upper[i0]
    direction[i0] = -1

    for i in range(i0 + 1, n):
        if np.isnan(basic_upper[i]):
            st[i] = st[i - 1] if not np.isnan(st[i - 1]) else np.nan
            direction[i] = direction[i - 1]
            continue

        prev_fu = final_upper[i - 1] if not np.isnan(final_upper[i - 1]) else basic_upper[i]
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < prev_fu or close[i - 1] > prev_fu
            else prev_fu
        )

        prev_fl = final_lower[i - 1] if not np.isnan(final_lower[i - 1]) else basic_lower[i]
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > prev_fl or close[i - 1] < prev_fl
            else prev_fl
        )

        prev_dir = direction[i - 1]
        if prev_dir == 1:  # was uptrend
            if close[i] < final_lower[i]:
                st[i] = final_upper[i]
                direction[i] = -1
            else:
                st[i] = final_lower[i]
                direction[i] = 1
        else:  # was downtrend
            if close[i] > final_upper[i]:
                st[i] = final_lower[i]
                direction[i] = 1
            else:
                st[i] = final_upper[i]
                direction[i] = -1

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)
