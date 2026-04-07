"""
agents/sentiment_agent.py — Market Sentiment Analysis Agent.

Derives sentiment signals from:
  1. Funding rate proxy (from price momentum and volume)
  2. Open Interest changes proxy (from volume analysis)
  3. Long/Short ratio proxy (from volume and price action)

Since real funding rate / OI data requires exchange API calls
that aren't always available, this agent uses OHLCV proxies
that approximate these signals with reasonable accuracy.

Score > 0.6 → bullish sentiment → long signal
Score < 0.4 → bearish sentiment → short signal
Extreme sentiment (> 0.75 or < 0.25) → contrarian signal
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger("SentimentAgent")

_MIN_BARS = 30
_FUNDING_WINDOW = 20      # periods for funding rate proxy
_OI_WINDOW = 10           # periods for OI change proxy
_LS_WINDOW = 20           # periods for long/short ratio proxy
_EXTREME_SENTIMENT = 0.75  # above this → consider contrarian signal


class SentimentAgent(BaseAgent):
    """
    Derives sentiment signals from OHLCV data proxies.

    Sentiment proxies:
    - Funding rate proxy: rolling momentum / mean deviation from price
    - OI change proxy: unusual volume divergence from price movement
    - L/S ratio proxy: rolling buy-side volume fraction
    """

    def __init__(self):
        super().__init__("sentiment", initial_weight=0.15)

    def analyse(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Optional[AgentResult]:
        if df is None or len(df) < _MIN_BARS:
            return None

        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)
            open_p = df["open"].values.astype(float)

            details = []
            metadata = {}

            # ---- 1. Funding Rate Proxy ----
            # Positive funding = longs paying shorts → overheated longs
            # Proxy: price deviation from rolling mean
            ma20 = np.convolve(close, np.ones(_FUNDING_WINDOW) / _FUNDING_WINDOW, mode="valid")
            last_close = close[-1]
            last_ma = float(ma20[-1]) if len(ma20) > 0 else last_close
            funding_proxy = (last_close - last_ma) / max(abs(last_ma), 1e-8)
            funding_proxy = float(np.clip(funding_proxy * 10, -1, 1))  # scale

            # High positive funding proxy → longs crowded → bearish contrarian
            # High negative funding proxy → shorts crowded → bullish contrarian

            # ---- 2. Open Interest Change Proxy ----
            # OI expansion = new positions entering (high volume on breakout)
            # OI contraction = position unwinding (high volume on reversal)
            n = min(_OI_WINDOW, len(df) - 1)
            price_change = abs(close[-1] - close[-n - 1]) / max(close[-n - 1], 1e-8)
            vol_change = volume[-n:].mean() / max(volume[:-n].mean() if len(volume[:-n]) > 0 else volume.mean(), 1e-8)

            # OI expansion proxy: high volume AND significant price move
            oi_expanding = price_change > 0.01 and vol_change > 1.2
            oi_signal = 1.0 if (oi_expanding and close[-1] > close[-n - 1]) else (
                -1.0 if (oi_expanding and close[-1] < close[-n - 1]) else 0.0
            )

            # ---- 3. Long/Short Ratio Proxy ----
            # Buy volume fraction as proxy for L/S ratio
            buy_vol = np.where(close[-_LS_WINDOW:] > open_p[-_LS_WINDOW:],
                               volume[-_LS_WINDOW:], 0).sum()
            total_vol = volume[-_LS_WINDOW:].sum()
            ls_ratio = float(buy_vol / max(total_vol, 1e-8))  # [0,1], 0.5 = balanced

            # ---- Composite Sentiment Score ----
            # Funding proxy: positive = longs crowded → neutral/bearish
            # Map funding to score component (contrarian: overcrowded longs → sell)
            funding_component = 0.5 - funding_proxy * 0.3  # positive funding → lower score

            # OI: expanding up = bullish, expanding down = bearish
            oi_component = 0.5 + oi_signal * 0.2

            # L/S ratio component
            ls_component = float(ls_ratio)

            raw_score = (
                0.30 * funding_component +
                0.30 * oi_component +
                0.40 * ls_component
            )
            score = float(np.clip(raw_score, 0.0, 1.0))

            # Extreme sentiment → contrarian flag
            contrarian = score > _EXTREME_SENTIMENT or score < (1 - _EXTREME_SENTIMENT)
            if contrarian:
                # Flip for contrarian interpretation
                contrarian_score = 1.0 - score
                details.append(
                    f"⚠️ Extreme sentiment={score:.2f} → contrarian signal={contrarian_score:.2f}"
                )
                score = contrarian_score

            # Direction
            if score > 0.58:
                direction = "long"
            elif score < 0.42:
                direction = "short"
            else:
                direction = "neutral"

            details.append(
                f"Funding proxy={funding_proxy:.3f}, OI signal={oi_signal:.1f}, "
                f"L/S ratio={ls_ratio:.2f}"
            )

            metadata.update({
                "funding_proxy": funding_proxy,
                "oi_signal": oi_signal,
                "oi_expanding": oi_expanding,
                "ls_ratio": ls_ratio,
                "contrarian_flag": contrarian,
            })

            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=score,
                direction=direction,
                confidence=0.5 * (1 - abs(score - 0.5) * 2),  # highest confidence near extremes
                details=details,
                metadata=metadata,
            )

        except Exception as exc:
            logger.warning(f"SentimentAgent.analyse error [{symbol}/{interval}]: {exc}")
            return None
