"""
Multi-Timeframe Agent for V17.

Uses MTFCorrelator to assess cross-timeframe trend alignment and return
a directional score that boosts signal quality.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentResult
from indicators.mtf_correlation import MTFCorrelator, MTFResult

logger = logging.getLogger("MTFAgent")

# Alignment thresholds
_STRONG_ALIGNMENT = 0.70    # above this → follow dominant trend
_WEAK_ALIGNMENT = 0.50      # below this → neutral, too much conflict


class MTFAgent(BaseAgent):
    """
    Analyses trend alignment across 15m / 1h / 4h timeframes.

    Parameters
    ----------
    timeframes : list of interval strings to consider (default: ["15m", "1h", "4h"])
    """

    def __init__(self, timeframes=None):
        super().__init__("mtf", initial_weight=0.20)
        self._correlator = MTFCorrelator(timeframes=timeframes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(
        self,
        symbol: str,
        interval: str,
        df: Optional[pd.DataFrame],
        tf_data: Optional[Dict[str, pd.DataFrame]] = None,
        *args,
        **kwargs,
    ) -> Optional[AgentResult]:
        """
        Compute MTF alignment score.

        Parameters
        ----------
        symbol   : Trading pair, e.g. "BTCUSDT"
        interval : Primary interval for context (not used for analysis itself)
        df       : Primary OHLCV DataFrame (used as fallback for the primary TF)
        tf_data  : Dict mapping interval → DataFrame for each timeframe.
                   If None or empty, the method returns None gracefully.

        Returns
        -------
        AgentResult or None when insufficient cross-TF data is available.
        """
        # Build tf_data: if only primary df provided, use it for the primary TF
        if not tf_data:
            if df is not None and len(df) >= 26:
                tf_data = {interval: df}
            else:
                logger.debug(
                    f"MTFAgent [{symbol}]: no multi-TF data provided, returning None"
                )
                return None

        mtf: MTFResult = self._correlator.correlate(symbol, tf_data)

        # Map alignment + trend to score and direction
        score, direction = self._score_from_mtf(mtf)

        details = [
            f"alignment={mtf.alignment_score:.2f}",
            f"dominant={mtf.dominant_trend}",
            f"vol_confirmed={mtf.volume_confirmed}",
        ]
        if mtf.conflicts:
            details.append(f"conflicts={','.join(mtf.conflicts)}")
        details += [f"{tf}:{sig}" for tf, sig in mtf.tf_signals.items()]

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(score, 0.0, 1.0)),
            direction=direction,
            confidence=float(mtf.alignment_score),
            details=details,
            metadata={
                "alignment_score": mtf.alignment_score,
                "dominant_trend": mtf.dominant_trend,
                "tf_signals": mtf.tf_signals,
                "volume_confirmed": mtf.volume_confirmed,
                "conflicts": mtf.conflicts,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_from_mtf(mtf: MTFResult):
        """
        Convert MTFResult to (score, direction).

        Rules
        -----
        - alignment > 0.70 → high-conviction directional score
        - alignment 0.50..0.70 → moderate score, follow dominant
        - alignment < 0.50 → neutral (too much conflict)
        - volume confirmation adds a 0.05 bonus
        """
        alignment = mtf.alignment_score
        trend = mtf.dominant_trend

        if trend == "neutral" or alignment < _WEAK_ALIGNMENT:
            return 0.50, "neutral"

        # Base score: scales from 0.50 (at weak threshold) to 0.90 (perfect alignment)
        base_score = 0.50 + 0.40 * ((alignment - _WEAK_ALIGNMENT) / (1.0 - _WEAK_ALIGNMENT))

        # Volume bonus
        if mtf.volume_confirmed:
            base_score += 0.05

        direction = "long" if trend == "bullish" else "short"
        return base_score, direction
