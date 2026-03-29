"""
Multi-Timeframe (MTF) Correlation module for V17.

Provides MTFCorrelator to compare trend direction and volume across 15m/1h/4h
timeframes and compute an alignment score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MTFCorrelator")

# Supported timeframe order from shortest to longest
_TF_WEIGHTS: Dict[str, float] = {"15m": 0.25, "1h": 0.40, "4h": 0.35}
_DEFAULT_TFS = list(_TF_WEIGHTS.keys())


@dataclass
class MTFResult:
    """Output of MTFCorrelator.correlate()."""

    alignment_score: float          # 0.0 → fully misaligned, 1.0 → perfect alignment
    dominant_trend: str             # "bullish" | "bearish" | "neutral"
    tf_signals: Dict[str, str]      # {"15m": "bullish", "1h": "neutral", ...}
    volume_confirmed: bool          # True if volume supports the dominant trend
    conflicts: List[str]            # Human-readable list of conflicting TFs

    def __repr__(self) -> str:
        return (
            f"MTFResult(alignment={self.alignment_score:.2f}, "
            f"trend={self.dominant_trend}, vol={self.volume_confirmed}, "
            f"conflicts={self.conflicts})"
        )


class MTFCorrelator:
    """
    Compares trend direction and volume across multiple timeframes.

    Usage
    -----
    correlator = MTFCorrelator()
    result = correlator.correlate("BTCUSDT", {"15m": df_15m, "1h": df_1h, "4h": df_4h})
    """

    def __init__(self, timeframes: Optional[List[str]] = None):
        self._tfs = timeframes or _DEFAULT_TFS
        self._weights = {tf: _TF_WEIGHTS.get(tf, 0.33) for tf in self._tfs}
        # Normalise weights so they sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    # ------------------------------------------------------------------
    # Per-TF signal extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_signal(df: pd.DataFrame) -> str:
        """
        Determine simple trend direction for a single DataFrame.
        Returns "bullish", "bearish", or "neutral".
        """
        if df is None or len(df) < 26:
            return "neutral"
        try:
            close = df["close"]
            # EMA-based trend
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            last_close = float(close.iloc[-1])
            last_ema20 = float(ema20.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])

            # MACD histogram sign
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
            hist_sign = float(macd_hist.iloc[-1])

            bullish_signals = 0
            bearish_signals = 0

            # EMA stack
            if last_close > last_ema20 > last_ema50:
                bullish_signals += 2
            elif last_close < last_ema20 < last_ema50:
                bearish_signals += 2
            elif last_close > last_ema20:
                bullish_signals += 1
            elif last_close < last_ema20:
                bearish_signals += 1

            # MACD
            if hist_sign > 0:
                bullish_signals += 1
            elif hist_sign < 0:
                bearish_signals += 1

            if bullish_signals > bearish_signals + 1:
                return "bullish"
            if bearish_signals > bullish_signals + 1:
                return "bearish"
            return "neutral"
        except Exception as exc:
            logger.debug(f"Trend signal error: {exc}")
            return "neutral"

    @staticmethod
    def _volume_above_avg(df: pd.DataFrame, lookback: int = 20) -> bool:
        """True if last bar's volume is above the rolling average."""
        if df is None or len(df) < lookback + 1:
            return False
        try:
            vol = df["volume"]
            avg = float(vol.iloc[-lookback - 1:-1].mean())
            last = float(vol.iloc[-1])
            return last > avg if avg > 0 else False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Main correlate method
    # ------------------------------------------------------------------

    def correlate(self, symbol: str, tf_data: Dict[str, pd.DataFrame]) -> MTFResult:
        """
        Compute multi-timeframe alignment.

        Parameters
        ----------
        symbol  : Trading pair (for logging)
        tf_data : Dict mapping interval strings to OHLCV DataFrames
                  e.g. {"15m": df_15m, "1h": df_1h, "4h": df_4h}

        Returns
        -------
        MTFResult
        """
        tf_signals: Dict[str, str] = {}
        available_tfs: List[str] = []

        for tf in self._tfs:
            df = tf_data.get(tf)
            if df is not None and len(df) >= 26:
                signal = self._trend_signal(df)
                tf_signals[tf] = signal
                available_tfs.append(tf)
            else:
                tf_signals[tf] = "neutral"

        if not available_tfs:
            return MTFResult(
                alignment_score=0.0,
                dominant_trend="neutral",
                tf_signals=tf_signals,
                volume_confirmed=False,
                conflicts=["No timeframe data available"],
            )

        # ------------------------------------------------------------------
        # Determine dominant trend via weighted voting
        # ------------------------------------------------------------------
        votes: Dict[str, float] = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        for tf in available_tfs:
            w = self._weights.get(tf, 0.33)
            votes[tf_signals[tf]] += w

        dominant_trend = max(votes, key=votes.get)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # Alignment score: fraction of weighted votes for the dominant trend
        # ------------------------------------------------------------------
        total_weight = sum(self._weights.get(tf, 0.33) for tf in available_tfs)
        dominant_weight = votes.get(dominant_trend, 0.0)
        alignment_score = float(dominant_weight / total_weight) if total_weight > 0 else 0.0
        alignment_score = float(np.clip(alignment_score, 0.0, 1.0))

        # ------------------------------------------------------------------
        # Identify conflicting timeframes
        # ------------------------------------------------------------------
        conflicts: List[str] = []
        if dominant_trend != "neutral":
            opposite = "bearish" if dominant_trend == "bullish" else "bullish"
            for tf in available_tfs:
                if tf_signals[tf] == opposite:
                    conflicts.append(f"{tf}:{tf_signals[tf]} (vs dominant {dominant_trend})")

        # ------------------------------------------------------------------
        # Volume confirmation: majority of available TFs show above-avg volume
        # ------------------------------------------------------------------
        vol_confirmations = 0
        for tf in available_tfs:
            df = tf_data.get(tf)
            if df is not None and self._volume_above_avg(df):
                vol_confirmations += 1
        volume_confirmed = vol_confirmations > len(available_tfs) / 2

        logger.debug(
            f"MTFCorrelator [{symbol}]: alignment={alignment_score:.2f} "
            f"trend={dominant_trend} vol={volume_confirmed} "
            f"signals={tf_signals} conflicts={conflicts}"
        )

        return MTFResult(
            alignment_score=alignment_score,
            dominant_trend=dominant_trend,
            tf_signals=tf_signals,
            volume_confirmed=volume_confirmed,
            conflicts=conflicts,
        )
