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
from indicators.technical import rsi, macd, adx, bollinger_bands, obv, zscore, vwap, supertrend, ema
from indicators.smart_money import cumulative_volume_delta, liquidity_sweep
from data import data_store

logger = logging.getLogger("ConfluenceAgent")

TF_WEIGHTS = {"15m": 0.25, "1h": 0.40, "4h": 0.35}
TF_ORDER = ["15m", "1h", "4h"]

# Bidirectional evaluation constants
_OPPOSITE_DOMINANCE_MARGIN = 0.15   # opposite direction must exceed primary by this to flip
_OPPOSITE_SCORE_PENALTY = 0.85      # score multiplier applied when flipping direction
_TF_AGREEMENT_THRESHOLD = 0.50      # minimum TF score to be considered "agreeing"
_PRIMARY_TF_WEAK_THRESHOLD = 0.45   # primary TF score below this triggers penalty
_PRIMARY_TF_WEAK_PENALTY = 0.80     # multiplier applied when primary TF is weak


class ConfluenceAgent(BaseAgent):
    """Probabilistic multi-timeframe confluence scoring."""

    def __init__(self):
        super().__init__("confluence", initial_weight=0.25)
        # Instance-level TF weights so the evolution engine can adapt them at runtime
        self._tf_weights: Dict[str, float] = dict(TF_WEIGHTS)

    def update_tf_weights(self, new_weights: dict) -> None:
        """Update per-timeframe weights used in confluence scoring.

        Values are clipped to [0.05, 0.70] and normalised to sum to 1.0 so the
        weighted average in ``compute_confluence()`` remains well-behaved.
        """
        _default_weight = 1.0 / len(TF_ORDER)
        cleaned = {}
        for tf in TF_ORDER:
            raw = new_weights.get(tf, self._tf_weights.get(tf, _default_weight))
            cleaned[tf] = float(np.clip(float(raw), 0.05, 0.70))
        total = sum(cleaned.values())
        if total > 0:
            self._tf_weights = {tf: cleaned[tf] / total for tf in TF_ORDER}
            logger.info(f"🌊 ConfluenceAgent: TF weights updated → {self._tf_weights}")
        else:
            logger.warning("ConfluenceAgent.update_tf_weights: all-zero weights ignored")

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

            # VWAP position (rolling 20-bar)
            _vwap_val = float(vwap(df, period=20).iloc[-1])
            above_vwap = float(last_close) > _vwap_val

            # Supertrend direction
            try:
                _, _st_dir_s = supertrend(df, period=10, multiplier=3.0)
                _st_long = int(_st_dir_s.iloc[-1]) == 1
            except Exception:
                _st_long = None

            # EMA 200 position
            _ema200_long = None  # type: Optional[bool]
            if len(df) >= 200:
                try:
                    _ema200_val = float(ema(close, 200).iloc[-1])
                    _ema200_long = float(last_close) > _ema200_val
                except Exception:
                    pass

            score = 0.0
            if direction == "long":
                if rsi_val < 55: score += 0.15       # was < 50
                if rsi_val < 40: score += 0.10       # was < 35
                if last_di_p > last_di_m: score += 0.10
                if last_hist > 0 and prev_hist < 0: score += 0.15
                elif last_hist > 0: score += 0.08    # was 0.05
                if last_close < bb_lo.iloc[-1] * 1.01: score += 0.15   # was * 1.005
                if obv_slope > 0: score += 0.10
                if last_delta > 0: score += 0.10
                if sweep == 1: score += 0.20  # bullish liquidity sweep
                # V18 additions
                if above_vwap: score += 0.10
                if _st_long is True: score += 0.10
                if _ema200_long is True: score += 0.08
            else:
                if rsi_val > 45: score += 0.15       # was > 50
                if rsi_val > 60: score += 0.10       # was > 65
                if last_di_m > last_di_p: score += 0.10
                if last_hist < 0 and prev_hist > 0: score += 0.15
                elif last_hist < 0: score += 0.08    # was 0.05
                if last_close > bb_up.iloc[-1] * 0.99: score += 0.15   # was * 0.995
                if obv_slope < 0: score += 0.10
                if last_delta < 0: score += 0.10
                if sweep == -1: score += 0.20  # bearish liquidity sweep
                # V18 additions
                if not above_vwap: score += 0.10
                if _st_long is False: score += 0.10
                if _ema200_long is False: score += 0.08

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
        tf_directions: list = []
        for tf in TF_ORDER:
            df = data_store.get_df(symbol, tf)
            bias = self._tf_bias(df, direction)
            tf_scores[tf] = bias
            # Determine per-TF direction based on bias vs opposite
            opposite = "short" if direction == "long" else "long"
            opp_bias = self._tf_bias(df, opposite)
            tf_directions.append(direction if bias >= opp_bias else opposite)

        # Weighted average, with primary TF having extra weight
        total_weight = 0.0
        weighted_sum = 0.0
        for tf, w in self._tf_weights.items():
            if tf == primary_interval:
                w_adj = w * 1.5
            else:
                w_adj = w
            weighted_sum += tf_scores[tf] * w_adj
            total_weight += w_adj

        confluence_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Direction agreement across timeframes
        n_long = sum(1 for d in tf_directions if d == "long")
        n_short = sum(1 for d in tf_directions if d == "short")
        direction_agreement = max(n_long, n_short) / max(len(tf_directions), 1)

        if direction_agreement >= 0.80:
            agreement_mult = 1.30
        elif direction_agreement <= 0.40:
            agreement_mult = 0.60
        else:
            agreement_mult = 1.0

        confluence_score = float(np.clip(confluence_score * agreement_mult, 0.0, 1.0))

        return {
            "tf_scores": tf_scores,
            "confluence": confluence_score,
            "direction_agreement": direction_agreement,
            "agreement_mult": agreement_mult,
        }

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df,
                direction: str = "long") -> Optional[AgentResult]:
        if df is None or len(df) < 50:
            return None

        # Evaluate BOTH directions
        result_primary = self.compute_confluence(symbol, interval, direction)
        opposite = "short" if direction == "long" else "long"
        result_opposite = self.compute_confluence(symbol, interval, opposite)

        primary_score = result_primary["confluence"]
        opposite_score = result_opposite["confluence"]

        # If opposite direction is significantly stronger, penalize or flip
        if opposite_score > primary_score + _OPPOSITE_DOMINANCE_MARGIN:
            # Use opposite direction
            final_direction = opposite
            final_score = opposite_score * _OPPOSITE_SCORE_PENALTY  # slight penalty for disagreeing with pattern
            tf_scores = result_opposite["tf_scores"]
        else:
            final_direction = direction
            final_score = primary_score
            tf_scores = result_primary["tf_scores"]

        details = [f"{tf}={v:.2f}" for tf, v in tf_scores.items()]
        details.append(f"confluence={final_score:.2f}")
        details.append(f"primary={primary_score:.2f}")
        details.append(f"opposite={opposite_score:.2f}")

        # === SOGLIA ALZATA A 0.50 ===
        agreeing = sum(1 for v in tf_scores.values() if v > _TF_AGREEMENT_THRESHOLD)
        total_tfs = len(tf_scores)

        # Penalità se il TF primario è debole
        primary_tf_score = tf_scores.get(interval, 0.0)
        if primary_tf_score < _PRIMARY_TF_WEAK_THRESHOLD:
            final_score *= _PRIMARY_TF_WEAK_PENALTY
            details.append(f"primary_tf_weak({primary_tf_score:.2f})x{_PRIMARY_TF_WEAK_PENALTY}")

        # confidence = frazione TF sopra 0.50 (non 0.40)
        confidence = agreeing / max(total_tfs, 1)

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=final_score,
            direction=final_direction,
            confidence=confidence,
            details=details,
            metadata={
                "tf_scores": tf_scores,
                "agreeing_tfs": agreeing,
                "primary_score": primary_score,
                "opposite_score": opposite_score,
                "direction_agreement": result_primary.get("direction_agreement", 1.0)
                    if final_direction == direction else result_opposite.get("direction_agreement", 1.0),
                "agreement_mult": result_primary.get("agreement_mult", 1.0)
                    if final_direction == direction else result_opposite.get("agreement_mult", 1.0),
            },
        )
