"""
agents/contrarian_agent.py — Devil's Advocate Agent.

The contrarian agent actively searches for reasons NOT to trade.
It receives the emerging consensus direction from other agents
and seeks counter-evidence that the trade is a mistake.

Key checks:
  1. Oversaturation: is everyone in the same direction? (RSI extreme)
  2. Divergences: price vs oscillator divergence against consensus
  3. Volume exhaustion: high volume with diminishing price moves
  4. Consensus fragility: score distribution is overly uniform
  5. Regime mismatch: strategy conflicts with current volatility regime

Score interpretation (INVERTED):
    High score (> 0.7) → strong reason to NOT trade → triggers veto in consensus
    Low score (< 0.3)  → no counter-evidence found → trade can proceed

Used in ConsensusProtocol Round 2 as devil's advocate veto mechanism.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger("ContrarianAgent")

_MIN_BARS = 30
_RSI_EXTREME_LOW = 25
_RSI_EXTREME_HIGH = 75
_VOLUME_EXHAUSTION_THRESHOLD = 2.0   # volume × this but small price move
_DIVERGENCE_WINDOW = 14


class ContrarianAgent(BaseAgent):
    """
    Devil's advocate: assigns high scores to reasons NOT to trade.

    Unlike other agents that score how GOOD the trade looks,
    the ContrarianAgent scores how BAD it is — how many red flags exist.

    Integration: ConsensusProtocol.debate() uses this score to
    penalise all other agents if contrarian score > threshold.
    """

    def __init__(self):
        super().__init__("contrarian", initial_weight=0.25)

    def analyse(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        consensus_direction: Optional[str] = None,
        agent_results: Optional[Dict] = None,
        *args,
        **kwargs,
    ) -> Optional[AgentResult]:
        """
        Analyse counter-evidence against the consensus trade.

        Parameters
        ----------
        symbol              : trading pair
        interval            : timeframe
        df                  : OHLCV DataFrame
        consensus_direction : emerging consensus direction ('long'/'short')
        agent_results       : dict of other agent results for consensus analysis

        Returns
        -------
        AgentResult where score = strength of counter-evidence (0=none, 1=strong)
        """
        if df is None or len(df) < _MIN_BARS:
            return None

        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)
            open_p = df["open"].values.astype(float)

            details = []
            red_flags = []
            metadata = {}

            # ---- 1. RSI Extreme Check ----
            rsi = self._compute_rsi(close, period=14)
            rsi_flag = False

            if consensus_direction == "long" and rsi > _RSI_EXTREME_HIGH:
                red_flags.append(f"RSI={rsi:.1f} extremely overbought for LONG trade")
                rsi_flag = True
            elif consensus_direction == "short" and rsi < _RSI_EXTREME_LOW:
                red_flags.append(f"RSI={rsi:.1f} extremely oversold for SHORT trade")
                rsi_flag = True
            elif rsi > 75 or rsi < 25:
                red_flags.append(f"RSI={rsi:.1f} extreme value (contrarian warning)")
                rsi_flag = True

            # ---- 2. RSI Divergence vs Price ----
            divergence_flag = False
            if len(close) >= _DIVERGENCE_WINDOW + 2:
                divergence = self._detect_divergence(close, high, low, rsi_val=rsi)
                if divergence != 0:
                    divergence_type = "bearish" if divergence < 0 else "bullish"
                    if (consensus_direction == "long" and divergence < 0) or \
                       (consensus_direction == "short" and divergence > 0):
                        red_flags.append(f"{divergence_type} RSI divergence against consensus")
                        divergence_flag = True

            # ---- 3. Volume Exhaustion ----
            exhaustion_flag = False
            if len(close) >= 5 and len(volume) >= 5:
                recent_range = abs(close[-1] - close[-5]) / max(close[-5], 1e-8)
                avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
                recent_vol_ratio = float(volume[-1] / max(avg_vol, 1e-8))

                # High volume but small price range = exhaustion
                if recent_vol_ratio > _VOLUME_EXHAUSTION_THRESHOLD and recent_range < 0.005:
                    red_flags.append(
                        f"Volume exhaustion: vol_ratio={recent_vol_ratio:.1f}, range={recent_range:.3f}"
                    )
                    exhaustion_flag = True

            # ---- 4. Consensus Fragility Check ----
            fragility_flag = False
            if agent_results:
                scores = []
                for name, result in agent_results.items():
                    if name == "contrarian":
                        continue
                    if hasattr(result, "score"):
                        scores.append(float(result.score))
                    elif isinstance(result, dict):
                        scores.append(float(result.get("score", 0.5)))

                if len(scores) >= 3:
                    score_std = float(np.std(scores))
                    score_mean = float(np.mean(scores))
                    # Very uniform scores (low std) near neutral → consensus fragile
                    if score_std < 0.05 and abs(score_mean - 0.5) < 0.08:
                        red_flags.append(
                            f"Fragile consensus: mean={score_mean:.2f}, std={score_std:.3f}"
                        )
                        fragility_flag = True

            # ---- 5. Support/Resistance Clash ----
            sr_flag = False
            if len(high) >= 20 and len(low) >= 20:
                recent_high = float(np.max(high[-20:]))
                recent_low = float(np.min(low[-20:]))
                current_price = float(close[-1])

                # Distance from key levels
                dist_to_resistance = (recent_high - current_price) / max(current_price, 1e-8)
                dist_to_support = (current_price - recent_low) / max(current_price, 1e-8)

                if consensus_direction == "long" and dist_to_resistance < 0.005:
                    red_flags.append(
                        f"Price near resistance ({dist_to_resistance:.3f}) → risky LONG"
                    )
                    sr_flag = True
                elif consensus_direction == "short" and dist_to_support < 0.005:
                    red_flags.append(
                        f"Price near support ({dist_to_support:.3f}) → risky SHORT"
                    )
                    sr_flag = True

            # ---- Compute Contrarian Score ----
            n_flags = sum([rsi_flag, divergence_flag, exhaustion_flag, fragility_flag, sr_flag])

            # Weight flags (RSI and divergence are most important)
            weighted_flags = (
                0.30 * float(rsi_flag) +
                0.25 * float(divergence_flag) +
                0.20 * float(exhaustion_flag) +
                0.15 * float(fragility_flag) +
                0.10 * float(sr_flag)
            )

            # Contrarian score: 0 = no red flags, 1 = many red flags
            contrarian_score = float(np.clip(weighted_flags, 0.0, 1.0))

            # When contrarian score is high, direction is the OPPOSITE of consensus
            if contrarian_score > 0.5 and consensus_direction:
                opp = {"long": "short", "short": "long"}.get(consensus_direction, "neutral")
                direction = opp
                details.append(f"Contrarian: opposing consensus '{consensus_direction}'")
            else:
                direction = "neutral"

            details.extend(red_flags)
            if not red_flags:
                details.append("✅ No significant red flags found")

            metadata.update({
                "n_red_flags": n_flags,
                "rsi": rsi,
                "rsi_extreme": rsi_flag,
                "divergence_detected": divergence_flag,
                "volume_exhaustion": exhaustion_flag,
                "consensus_fragile": fragility_flag,
                "sr_clash": sr_flag,
                "consensus_direction": consensus_direction,
            })

            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=contrarian_score,  # HIGH score = reasons to NOT trade
                direction=direction,
                confidence=min(contrarian_score * 1.2, 1.0),
                details=details,
                metadata=metadata,
            )

        except Exception as exc:
            logger.warning(f"ContrarianAgent.analyse error [{symbol}/{interval}]: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI for the last price."""
        if len(prices) < period + 1:
            return 50.0
        diffs = np.diff(prices[-period - 1:])
        gains = diffs[diffs > 0]
        losses = -diffs[diffs < 0]
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 1e-8
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-8
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _detect_divergence(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        rsi_val: float,
        period: int = 14,
    ) -> int:
        """
        Simple divergence detection.

        Returns:
            +1 = bullish divergence (price makes lower low, RSI makes higher low)
            -1 = bearish divergence (price makes higher high, RSI makes lower high)
             0 = no divergence
        """
        if len(close) < period + 2:
            return 0

        # Compare last 2 highs/lows with recent RSI trend
        price_change = float(close[-1] - close[-period])
        rsi_trend = rsi_val - 50.0  # positive = RSI trending up

        # Bearish divergence: price higher, RSI lower (making lower highs in RSI)
        if price_change > 0.01 * close[-period] and rsi_trend < -5:
            return -1

        # Bullish divergence: price lower, RSI higher
        if price_change < -0.01 * close[-period] and rsi_trend > 5:
            return +1

        return 0
