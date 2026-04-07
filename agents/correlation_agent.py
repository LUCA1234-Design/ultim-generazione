"""
agents/correlation_agent.py — Cross-Asset Correlation Agent.

Monitors correlation between the target asset and key market indicators:
  1. BTC dominance proxy (BTC price trend as market leader)
  2. Crypto market breadth (majority of coins moving up/down)
  3. Cross-timeframe momentum divergence

A divergence between the target asset and BTC can signal:
  - Outperformance (target rising faster than BTC) → strong long
  - Underperformance (target lagging BTC rally) → weak long / hold
  - Negative divergence (target down, BTC up) → strong short signal

Note: In the absence of live multi-asset data feeds, this agent uses
the symbol's own internal correlation structure across timeframes
as a proxy for cross-asset analysis.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger("CorrelationAgent")

try:
    from quant.copula_correlations import tail_dependency, rolling_correlation_adaptive
    _COPULA_AVAILABLE = True
except ImportError:
    _COPULA_AVAILABLE = False
    logger.warning("CorrelationAgent: copula_correlations unavailable, using fallback")

_MIN_BARS = 50
_CORR_WINDOW = 30
_MOMENTUM_WINDOW = 14


class CorrelationAgent(BaseAgent):
    """
    Cross-asset correlation agent that detects relative strength
    and momentum divergences as trading signals.
    """

    def __init__(self):
        super().__init__("correlation", initial_weight=0.15)
        self._btc_cache: Dict[str, pd.DataFrame] = {}  # symbol → last df

    def analyse(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        *args,
        **kwargs,
    ) -> Optional[AgentResult]:
        """
        Analyse correlation and divergence signals.

        Parameters
        ----------
        symbol   : trading pair
        interval : timeframe
        df       : OHLCV DataFrame for the target symbol
        btc_df   : optional BTC OHLCV DataFrame for cross-asset analysis

        Returns
        -------
        AgentResult with correlation-based signal.
        """
        if df is None or len(df) < _MIN_BARS:
            return None

        try:
            close = df["close"].values.astype(float)
            log_ret = np.diff(np.log(np.maximum(close, 1e-8)))

            details = []
            metadata = {}
            score = 0.5
            direction = "neutral"
            confidence = 0.3

            # ---- 1. Internal momentum analysis ----
            # Compare short-term vs long-term momentum
            short_ma = float(np.mean(close[-5:])) if len(close) >= 5 else close[-1]
            medium_ma = float(np.mean(close[-14:])) if len(close) >= 14 else close[-1]
            long_ma = float(np.mean(close[-30:])) if len(close) >= 30 else close[-1]

            last_close = float(close[-1])
            short_vs_medium = (short_ma - medium_ma) / max(abs(medium_ma), 1e-8)
            medium_vs_long = (medium_ma - long_ma) / max(abs(long_ma), 1e-8)

            trend_alignment = (
                (short_vs_medium > 0) == (medium_vs_long > 0)
            )  # both bullish or both bearish

            # ---- 2. Volatility-momentum correlation ----
            # When momentum and volatility diverge → potential reversal
            if len(log_ret) >= _CORR_WINDOW:
                ret_window = log_ret[-_CORR_WINDOW:]
                vol_window = np.abs(ret_window)
                momentum_sign = np.sign(np.mean(ret_window))

                # Recent vol spike without price follow-through → uncertain
                vol_ratio = float(np.mean(vol_window[-5:]) / max(np.mean(vol_window), 1e-8))
                vol_spike = vol_ratio > 2.0

            else:
                momentum_sign = 1.0
                vol_spike = False
                vol_ratio = 1.0

            # ---- 3. Cross-asset correlation with BTC (if available) ----
            btc_correlation = None
            btc_divergence = 0.0

            if btc_df is not None and len(btc_df) >= _MIN_BARS and symbol != "BTCUSDT":
                try:
                    btc_close = btc_df["close"].values.astype(float)
                    btc_ret = np.diff(np.log(np.maximum(btc_close, 1e-8)))

                    n = min(len(log_ret), len(btc_ret), _CORR_WINDOW)
                    if n >= 10:
                        corr = float(np.corrcoef(log_ret[-n:], btc_ret[-n:])[0, 1])
                        btc_correlation = corr if np.isfinite(corr) else None

                        if btc_correlation is not None:
                            # Relative performance vs BTC (last 14 periods)
                            n14 = min(14, len(close), len(btc_close))
                            sym_ret_14 = (close[-1] / close[-n14] - 1) if n14 > 1 else 0
                            btc_ret_14 = (btc_close[-1] / btc_close[-n14] - 1) if n14 > 1 else 0
                            btc_divergence = sym_ret_14 - btc_ret_14

                            details.append(
                                f"BTC correlation={btc_correlation:.2f}, "
                                f"divergence={btc_divergence:.3f}"
                            )
                except Exception as e:
                    logger.debug(f"BTC cross-asset analysis error: {e}")

            elif _COPULA_AVAILABLE and len(log_ret) >= 40:
                # Tail dependency analysis on own return distribution
                try:
                    n_half = len(log_ret) // 2
                    td = tail_dependency(log_ret[:n_half], log_ret[n_half:])
                    lambda_lower = float(td.get("lambda_lower", 0.3))
                    lambda_upper = float(td.get("lambda_upper", 0.3))
                    metadata["tail_lower"] = lambda_lower
                    metadata["tail_upper"] = lambda_upper
                    if lambda_lower > 0.5:
                        details.append(f"High lower tail dependency={lambda_lower:.2f}")
                except Exception:
                    pass

            # ---- Composite scoring ----
            score_components = []

            # Trend alignment component
            if trend_alignment:
                if short_vs_medium > 0:
                    score_components.append(0.65)  # aligned bullish
                else:
                    score_components.append(0.35)  # aligned bearish
            else:
                score_components.append(0.50)  # mixed signals

            # BTC divergence component
            if btc_divergence != 0:
                # Outperforming BTC → bullish; underperforming → bearish
                btc_component = 0.50 + np.clip(btc_divergence * 5, -0.20, 0.20)
                score_components.append(float(btc_component))

            # Momentum sign component
            score_components.append(0.55 if momentum_sign > 0 else 0.45)

            score = float(np.mean(score_components))
            score = float(np.clip(score, 0.0, 1.0))

            # Reduce confidence when volatility is spiking
            confidence = 0.5 if not vol_spike else 0.3

            if score > 0.58:
                direction = "long"
            elif score < 0.42:
                direction = "short"
            else:
                direction = "neutral"

            if vol_spike:
                details.append(f"⚠️ Volatility spike ratio={vol_ratio:.2f}")

            details.append(
                f"Trend aligned={trend_alignment}, momentum={momentum_sign:.0f}, "
                f"short_vs_med={short_vs_medium:.3f}"
            )

            metadata.update({
                "trend_alignment": trend_alignment,
                "short_vs_medium": short_vs_medium,
                "medium_vs_long": medium_vs_long,
                "btc_correlation": btc_correlation,
                "btc_divergence": btc_divergence,
                "vol_spike": vol_spike,
            })

            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=score,
                direction=direction,
                confidence=confidence,
                details=details,
                metadata=metadata,
            )

        except Exception as exc:
            logger.warning(f"CorrelationAgent.analyse error [{symbol}/{interval}]: {exc}")
            return None
