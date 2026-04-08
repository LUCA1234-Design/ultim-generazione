"""
quant/microstructure.py — Market Microstructure Analysis.

Implements order-flow metrics computed from OHLCV data:

1. **VPIN** (Volume-synchronized Probability of Informed Trading)
   - Classifies volume as buy-initiated vs sell-initiated using
     the bulk-classification rule (price change sign)
   - Computes VPIN = |buy_volume - sell_volume| / total_volume
     over fixed volume buckets

2. **Kyle's Lambda** (price impact / market depth proxy)
   - OLS regression: Δprice ~ λ × signed_volume
   - High λ → thin market, informed trading; Low λ → deep market

3. **Trade Imbalance**
   - Rolling buy_volume / total_volume ratio
   - Proxy for order-flow direction

Integration: used by OrderFlowAgent to generate microstructure signals.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("Microstructure")


def _classify_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bulk-classify each candle's volume as buy or sell using the
    tick rule: if close > open → buy volume; else → sell volume.
    Neutral (close == open) volume is split 50/50.

    Returns df with added 'buy_vol' and 'sell_vol' columns.
    """
    out = df.copy()
    direction = np.sign(out["close"].values - out["open"].values)
    # +1 = buy candle, -1 = sell candle, 0 = neutral
    buy_frac = np.where(direction > 0, 1.0, np.where(direction < 0, 0.0, 0.5))
    out["buy_vol"] = out["volume"] * buy_frac
    out["sell_vol"] = out["volume"] * (1.0 - buy_frac)
    return out


def compute_vpin(df: pd.DataFrame, n_buckets: int = 50) -> Optional[pd.Series]:
    """
    Compute VPIN on fixed-volume buckets.

    VPIN = |V_buy - V_sell| / V_bucket  averaged over the last n_buckets buckets.

    Parameters
    ----------
    df       : OHLCV DataFrame
    n_buckets: number of volume buckets (default 50)

    Returns
    -------
    pd.Series of VPIN values aligned to df index, or None if insufficient data.
    """
    if df is None or len(df) < n_buckets:
        return None
    try:
        df2 = _classify_volume(df)
        total_vol = df2["volume"].sum()
        if total_vol <= 0:
            return None

        bucket_size = total_vol / n_buckets

        vpin_values = []
        cum_buy = 0.0
        cum_sell = 0.0
        cum_vol = 0.0
        bucket_vpin: list = []

        for _, row in df2.iterrows():
            remaining_buy = float(row["buy_vol"])
            remaining_sell = float(row["sell_vol"])
            remaining_vol = float(row["volume"])

            while remaining_vol > 0:
                fill = min(remaining_vol, bucket_size - cum_vol)
                ratio = fill / max(remaining_vol, 1e-12)
                cum_buy += remaining_buy * ratio
                cum_sell += remaining_sell * ratio
                cum_vol += fill
                remaining_buy -= remaining_buy * ratio
                remaining_sell -= remaining_sell * ratio
                remaining_vol -= fill

                if cum_vol >= bucket_size * 0.999:
                    imbalance = abs(cum_buy - cum_sell) / max(bucket_size, 1e-12)
                    bucket_vpin.append(imbalance)
                    cum_buy = 0.0
                    cum_sell = 0.0
                    cum_vol = 0.0

            vpin_values.append(
                float(np.mean(bucket_vpin[-n_buckets:])) if bucket_vpin else 0.0
            )

        result = pd.Series(vpin_values, index=df2.index)
        return result
    except Exception as exc:
        logger.debug(f"compute_vpin error: {exc}")
        return None


def compute_kyle_lambda(df: pd.DataFrame, window: int = 20) -> Optional[pd.Series]:
    """
    Estimate Kyle's Lambda (price impact coefficient) via rolling OLS.

    Model: Δprice_t = λ × signed_volume_t + ε_t
    where signed_volume = buy_vol - sell_vol

    A higher λ indicates a less liquid market where orders move price more.

    Parameters
    ----------
    df     : OHLCV DataFrame
    window : rolling window size

    Returns
    -------
    pd.Series of lambda estimates, or None if insufficient data.
    """
    if df is None or len(df) < window + 1:
        return None
    try:
        df2 = _classify_volume(df)
        delta_price = df2["close"].diff().fillna(0).values.astype(float)
        signed_vol = (df2["buy_vol"] - df2["sell_vol"]).values.astype(float)

        lambdas = np.full(len(df2), np.nan)

        for i in range(window, len(df2)):
            y = delta_price[i - window:i]
            x = signed_vol[i - window:i]
            if np.std(x) < 1e-10:
                lambdas[i] = 0.0
                continue
            slope, _, _, _, _ = stats.linregress(x, y)
            lambdas[i] = float(slope)

        result = pd.Series(lambdas, index=df2.index).fillna(method="bfill").fillna(0.0)
        return result
    except Exception as exc:
        logger.debug(f"compute_kyle_lambda error: {exc}")
        return None


def compute_trade_imbalance(df: pd.DataFrame, window: int = 20) -> Optional[pd.Series]:
    """
    Compute rolling trade imbalance = buy_volume / total_volume.

    Values > 0.6 → strong buy pressure
    Values < 0.4 → strong sell pressure
    Values ≈ 0.5 → balanced

    Parameters
    ----------
    df     : OHLCV DataFrame
    window : rolling window for smoothing

    Returns
    -------
    pd.Series in [0, 1], or None if insufficient data.
    """
    if df is None or len(df) < window:
        return None
    try:
        df2 = _classify_volume(df)
        roll_buy = df2["buy_vol"].rolling(window).sum()
        roll_total = df2["volume"].rolling(window).sum()
        imbalance = (roll_buy / roll_total.clip(lower=1e-12)).fillna(0.5)
        return imbalance
    except Exception as exc:
        logger.debug(f"compute_trade_imbalance error: {exc}")
        return None


def get_microstructure_score(df: pd.DataFrame) -> Optional[float]:
    """
    Aggregate microstructure score in [-1, +1]:
      - VPIN contribution: high VPIN → uncertainty (negative)
      - Trade imbalance: buy pressure → positive, sell pressure → negative
      - Kyle lambda: normalised price impact

    Returns
    -------
    float in [-1, +1] or None if data insufficient.
    Positive = buy pressure / informed buying; Negative = sell pressure.
    """
    if df is None or len(df) < 50:
        return None
    try:
        vpin = compute_vpin(df, n_buckets=min(50, len(df) // 2))
        imbalance = compute_trade_imbalance(df, window=20)
        kyle = compute_kyle_lambda(df, window=20)

        scores = []

        if vpin is not None and not vpin.dropna().empty:
            v = float(vpin.dropna().iloc[-1])
            # VPIN in [0,1]: high → uncertain, penalise signal confidence
            scores.append(-0.5 * v + 0.25)  # centred around 0

        if imbalance is not None and not imbalance.dropna().empty:
            i = float(imbalance.dropna().iloc[-1])
            # imbalance in [0,1], 0.5 = neutral → map to [-1, 1]
            scores.append(2.0 * (i - 0.5))

        if kyle is not None and not kyle.dropna().empty:
            lam = float(kyle.dropna().iloc[-1])
            # Normalise lambda: positive lambda with buy pressure is positive signal
            # We just use sign since magnitude is instrument-dependent
            scores.append(float(np.sign(lam)) * 0.5)

        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores), -1.0, 1.0))
    except Exception as exc:
        logger.debug(f"get_microstructure_score error: {exc}")
        return 0.0
