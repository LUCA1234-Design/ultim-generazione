"""
Technical indicators for V17/V19.
All indicators accept pandas Series / DataFrame and return pandas objects.
Includes: RSI, ATR, MACD, OBV, Bollinger Bands, Keltner Channels, ADX, Z-Score,
          VWAP, Supertrend, Ehlers Fisher Transform, KAMA, HV Percentile, CMO.
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

# ===========================================================================
# V19 — NEW INDICATORS
# ===========================================================================

# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------
def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP calcolato sull'intero DataFrame."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def vwap_bands(df: pd.DataFrame, n_std: float = 1.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP + upper/lower bands a N deviazioni standard.
    Returns (vwap_series, upper_band, lower_band).
    """
    vwap_s = vwap(df)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    deviation = (typical - vwap_s) ** 2
    variance = deviation.rolling(20).mean()
    std = np.sqrt(variance)
    return vwap_s, vwap_s + n_std * std, vwap_s - n_std * std


def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """Anchored VWAP da un indice specifico."""
    if anchor_idx < 0 or anchor_idx >= len(df):
        anchor_idx = 0
    df_slice = df.iloc[anchor_idx:]
    typical = (df_slice["high"] + df_slice["low"] + df_slice["close"]) / 3.0
    cum_tp_vol = (typical * df_slice["volume"]).cumsum()
    cum_vol = df_slice["volume"].cumsum().replace(0, np.nan)
    avwap = cum_tp_vol / cum_vol
    result = pd.Series(np.nan, index=df.index)
    result.iloc[anchor_idx:] = avwap.values
    return result


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------
def supertrend(df: pd.DataFrame, period: int = 10,
               multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Supertrend indicator.
    Returns (supertrend_line, direction_series) where direction +1=bullish, -1=bearish.
    """
    _atr_s = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_band = (hl2 + multiplier * _atr_s).copy()
    lower_band = (hl2 - multiplier * _atr_s).copy()

    supertrend_s = pd.Series(np.nan, index=df.index)
    direction_s = pd.Series(1, index=df.index, dtype=int)

    for i in range(1, len(df)):
        curr_upper = float(upper_band.iloc[i])
        curr_lower = float(lower_band.iloc[i])
        prev_upper = float(upper_band.iloc[i - 1])
        prev_lower = float(lower_band.iloc[i - 1])
        prev_close = float(df["close"].iloc[i - 1])
        curr_close = float(df["close"].iloc[i])

        # Adjust upper band
        if curr_upper < prev_upper or prev_close > prev_upper:
            upper_band.iloc[i] = curr_upper
        else:
            upper_band.iloc[i] = prev_upper

        # Adjust lower band
        if curr_lower > prev_lower or prev_close < prev_lower:
            lower_band.iloc[i] = curr_lower
        else:
            lower_band.iloc[i] = prev_lower

        prev_st = supertrend_s.iloc[i - 1]
        if np.isnan(prev_st):
            direction_s.iloc[i] = 1
            supertrend_s.iloc[i] = float(lower_band.iloc[i])
        elif prev_st == float(upper_band.iloc[i - 1]):
            if curr_close <= float(upper_band.iloc[i]):
                direction_s.iloc[i] = -1
                supertrend_s.iloc[i] = float(upper_band.iloc[i])
            else:
                direction_s.iloc[i] = 1
                supertrend_s.iloc[i] = float(lower_band.iloc[i])
        else:
            if curr_close >= float(lower_band.iloc[i]):
                direction_s.iloc[i] = 1
                supertrend_s.iloc[i] = float(lower_band.iloc[i])
            else:
                direction_s.iloc[i] = -1
                supertrend_s.iloc[i] = float(upper_band.iloc[i])

    return supertrend_s, direction_s


# ---------------------------------------------------------------------------
# Ehlers Fisher Transform
# ---------------------------------------------------------------------------
def ehlers_fisher(series: pd.Series, period: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Ehlers Fisher Transform.
    Trasforma i prezzi in distribuzione normale. Estremi +/-2.5 = inversioni ad alta affidabilita.
    Returns (fisher_series, signal_series).
    """
    highest = series.rolling(period).max()
    lowest = series.rolling(period).min()
    price_range = (highest - lowest).replace(0, np.nan)
    value = (2.0 * ((series - lowest) / price_range) - 1.0).clip(-0.999, 0.999)
    fisher = (0.5 * np.log((1 + value) / (1 - value))).ewm(span=3, min_periods=1).mean()
    signal = fisher.shift(1)
    return fisher, signal


# ---------------------------------------------------------------------------
# KAMA — Kaufman Adaptive Moving Average
# ---------------------------------------------------------------------------
def kama(series: pd.Series, period: int = 10,
         fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average.
    Veloce in trend forte, lento in mercato laterale — riduce whipsaw.
    """
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    kama_values = np.full(len(series), np.nan)
    sv = series.values
    if len(sv) <= period:
        return pd.Series(kama_values, index=series.index)
    kama_values[period] = sv[period]
    for i in range(period + 1, len(sv)):
        direction = abs(sv[i] - sv[i - period])
        volatility = float(np.sum(np.abs(np.diff(sv[i - period:i + 1]))))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_values[i] = kama_values[i - 1] + sc * (sv[i] - kama_values[i - 1])
    return pd.Series(kama_values, index=series.index)


# ---------------------------------------------------------------------------
# HV Percentile — Historical Volatility Percentile
# ---------------------------------------------------------------------------
def hv_percentile(df: pd.DataFrame, hv_period: int = 20,
                  lookback: int = 252) -> pd.Series:
    """Historical Volatility Percentile.
    0.0 = volatilita al minimo storico, 1.0 = al massimo storico.
    Utile per il dimensionamento dinamico delle posizioni.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv = log_ret.rolling(hv_period).std() * np.sqrt(252)

    def pct_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        return float(np.sum(x[:-1] <= x[-1])) / (len(x) - 1)

    return hv.rolling(lookback, min_periods=30).apply(pct_rank, raw=True)


# ---------------------------------------------------------------------------
# CMO — Chande Momentum Oscillator
# ---------------------------------------------------------------------------
def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator. Range -100 a +100.
    Misura il momentum puro senza smoothing, piu reattivo del RSI.
    """
    diff = series.diff()
    up = diff.clip(lower=0).rolling(period).sum()
    down = (-diff.clip(upper=0)).rolling(period).sum()
    total = (up + down).replace(0, np.nan)
    return 100 * (up - down) / total