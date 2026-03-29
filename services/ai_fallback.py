"""
AI Fallback service for V17.

Provides rule-based analysis when the LM Studio AI is unavailable.
Uses indicator values (RSI, MACD, BB) to generate a simple text summary.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("AIFallback")


def _rsi_commentary(rsi_val: float) -> str:
    if rsi_val < 30:
        return f"RSI={rsi_val:.1f} → deeply oversold, potential mean-reversion long setup"
    if rsi_val < 45:
        return f"RSI={rsi_val:.1f} → mildly oversold, bias long"
    if rsi_val > 70:
        return f"RSI={rsi_val:.1f} → overbought, potential reversal short setup"
    if rsi_val > 55:
        return f"RSI={rsi_val:.1f} → mildly overbought, bias short"
    return f"RSI={rsi_val:.1f} → neutral territory"


def _macd_commentary(macd_val: float, macd_signal: float) -> str:
    diff = macd_val - macd_signal
    if diff > 0:
        return f"MACD={macd_val:.4f} above signal={macd_signal:.4f} → bullish momentum"
    return f"MACD={macd_val:.4f} below signal={macd_signal:.4f} → bearish momentum"


def _bb_commentary(close: float, bb_upper: float, bb_lower: float) -> str:
    band_width = bb_upper - bb_lower
    if band_width == 0:
        return "BB bands degenerate"
    pct_b = (close - bb_lower) / band_width
    if pct_b < 0.1:
        return f"Price near BB lower (pct_b={pct_b:.2f}) → oversold / support"
    if pct_b > 0.9:
        return f"Price near BB upper (pct_b={pct_b:.2f}) → overbought / resistance"
    return f"Price mid-band (pct_b={pct_b:.2f}) → no extreme"


def generate_fallback_analysis(
    symbol: str,
    interval: str,
    df: Optional[pd.DataFrame] = None,
    indicators: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a rule-based analysis when AI is unavailable.

    Parameters
    ----------
    symbol      : Trading pair, e.g. "BTCUSDT"
    interval    : Candle interval, e.g. "1h"
    df          : Optional OHLCV DataFrame; indicators extracted if provided
    indicators  : Optional pre-computed dict with keys rsi, macd, macd_signal,
                  bb_upper, bb_lower, close

    Returns
    -------
    dict with keys: analysis (str), degraded (bool), source (str)
    """
    lines: list[str] = [
        f"[FALLBACK ANALYSIS] {symbol} {interval} — AI unavailable, using rule-based heuristics."
    ]

    try:
        # Extract indicators from DataFrame if not provided directly
        if indicators is None and df is not None and len(df) >= 26:
            close_s = df["close"]
            rsi_val = _compute_rsi(close_s, 14)
            macd_val, macd_sig = _compute_macd(close_s)
            sma20 = close_s.rolling(20).mean().iloc[-1]
            std20 = close_s.rolling(20).std().iloc[-1]
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            indicators = {
                "rsi": rsi_val,
                "macd": macd_val,
                "macd_signal": macd_sig,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "close": float(close_s.iloc[-1]),
            }

        if indicators:
            rsi_val = indicators.get("rsi", 50.0)
            macd_val = indicators.get("macd", 0.0)
            macd_sig = indicators.get("macd_signal", 0.0)
            bb_upper = indicators.get("bb_upper", 0.0)
            bb_lower = indicators.get("bb_lower", 0.0)
            close_val = indicators.get("close", 0.0)

            lines.append(_rsi_commentary(rsi_val))
            lines.append(_macd_commentary(macd_val, macd_sig))
            if bb_upper > 0 and bb_lower > 0 and close_val > 0:
                lines.append(_bb_commentary(close_val, bb_upper, bb_lower))
        else:
            lines.append("No indicator data available; analysis not possible.")

    except Exception as exc:
        logger.warning(f"Fallback analysis computation error: {exc}")
        lines.append("Indicator computation failed; degraded mode active.")

    analysis = " | ".join(lines)
    return {
        "analysis": analysis,
        "degraded": True,
        "source": "fallback",
    }


# ---------------------------------------------------------------------------
# Private helpers (avoid importing from indicators/ to keep this self-contained)
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - 100 / (1 + rs)
    val = rsi_series.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _compute_macd(close: pd.Series) -> tuple[float, float]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal.iloc[-1])
