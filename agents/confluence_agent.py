"""
Confluence Agent for V17.
Replaces the binary "Muro di Berlino" with probabilistic multi-timeframe confluence.
Scores alignment across 15m / 1h / 4h timeframes.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import rsi, macd, adx, bollinger_bands, obv, zscore
from indicators.smart_money import cumulative_volume_delta, liquidity_sweep
from data import data_store

logger = logging.getLogger("ConfluenceAgent")

TF_WEIGHTS = {"15m": 0.25, "1h": 0.40, "4h": 0.35}
TF_ORDER = ["15m", "1h", "4h"]


class ConfluenceAgent(BaseAgent):
    """Probabilistic multi-timeframe confluence scoring."""

    def __init__(self):
        super().__init__("confluence", initial_weight=0.25)

    # ------------------------------------------------------------------
    # Per-timeframe directional bias score
    # ------------------------------------------------------------------

    def _tf_bias(self, df: pd.DataFrame, direction: str) -> float:
        """Return 0.0 – 1.0 bias score for a given direction on one TF."""
        if df is None or len(df) < 50:
            return 0.0
        try:
            close = df["close"]
            rsi_val = rsi(close, 14).iloc[-1]
            adx_s, di_p, di_m = adx(df, 14)
            last_adx = adx_s.iloc[-1]
            last_di_p = di_p.iloc[-1]
            last_di_m = di_m.iloc[-1]
            macd_l, macd_sig, macd_hist = macd(close)
            last_hist = macd_hist.iloc[-1]
            prev_hist = macd_hist.iloc[-2] if len(macd_hist) > 2 else last_hist
            bb_up, bb_mid, bb_lo = bollinger_bands(close, 20, 2.0)
            last_close = close.iloc[-1]
            obv_s = obv(df)
            obv_slope = float(obv_s.iloc[-1] - obv_s.iloc[-5]) if len(obv_s) >= 5 else 0
            _, delta = cumulative_volume_delta(df)
            last_delta = float(delta.iloc[-1])
            sweep = liquidity_sweep(df, 20).iloc[-1]

            score = 0.0
            if direction == "long":
                if rsi_val < 50: score += 0.15
                if rsi_val < 35: score += 0.10
                if last_di_p > last_di_m: score += 0.10
                if last_hist > 0 and prev_hist < 0: score += 0.15
                elif last_hist > 0: score += 0.05
                if last_close < bb_lo.iloc[-1] * 1.005: score += 0.15
                if obv_slope > 0: score += 0.10
                if last_delta > 0: score += 0.10
                if sweep == 1: score += 0.20  # bullish liquidity sweep
            else:
                if rsi_val > 50: score += 0.15
                if rsi_val > 65: score += 0.10
                if last_di_m > last_di_p: score += 0.10
                if last_hist < 0 and prev_hist > 0: score += 0.15
                elif last_hist < 0: score += 0.05
                if last_close > bb_up.iloc[-1] * 0.995: score += 0.15
                if obv_slope < 0: score += 0.10
                if last_delta < 0: score += 0.10
                if sweep == -1: score += 0.20  # bearish liquidity sweep

            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"_tf_bias error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Confluence computation
    # ------------------------------------------------------------------

    def compute_confluence(self, symbol: str, primary_interval: str,
                            direction: str) -> Dict[str, float]:
        """Return per-TF bias scores and weighted confluence score."""
        tf_scores: Dict[str, float] = {}
        for tf in TF_ORDER:
            df = data_store.get_df(symbol, tf)
            bias = self._tf_bias(df, direction)
            tf_scores[tf] = bias

        # Weighted average, with primary TF having extra weight
        total_weight = 0.0
        weighted_sum = 0.0
        for tf, w in TF_WEIGHTS.items():
            if tf == primary_interval:
                w_adj = w * 1.5
            else:
                w_adj = w
            weighted_sum += tf_scores[tf] * w_adj
            total_weight += w_adj

        confluence_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return {
            "tf_scores": tf_scores,
            "confluence": float(np.clip(confluence_score, 0.0, 1.0)),
        }

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df,
                direction: str = "long") -> Optional[AgentResult]:
        if df is None or len(df) < 50:
            return None

        result = self.compute_confluence(symbol, interval, direction)
        confluence = result["confluence"]
        tf_scores = result["tf_scores"]

        details = [f"{tf}={v:.2f}" for tf, v in tf_scores.items()]
        details.append(f"confluence={confluence:.2f}")

        # Determine how many TFs agree
        agreeing = sum(1 for v in tf_scores.values() if v > 0.40)
        total_tfs = len(tf_scores)

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=confluence,
            direction=direction,
            confidence=agreeing / max(total_tfs, 1),
            details=details,
            metadata={"tf_scores": tf_scores, "agreeing_tfs": agreeing},
        )
