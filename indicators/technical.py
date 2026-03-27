"""
Technical indicators for V17.
All indicators accept pandas Series / DataFrame and return pandas objects.
Includes: RSI, ATR, MACD, OBV, Bollinger Bands, Keltner Channels, ADX, Z-Score.
"""
import numpy as np
import pandas as pd
from typing import Tuple


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
