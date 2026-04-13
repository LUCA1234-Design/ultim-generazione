"""
agents/orderflow_agent.py — Order Flow Analysis Agent.

Analyses market microstructure using VPIN, Kyle's Lambda,
trade imbalance, and order book proxies to generate directional
signals based on informed trading flow.

High VPIN → uncertainty/informed trading → reduce confidence.
Trade imbalance > 0.6 → buying pressure → long signal.
Absorption score > 0 → bids absorbing sells → long signal.

Integrates:
  - quant/microstructure.py: VPIN, Kyle's Lambda, trade imbalance
  - quant/orderbook_analyzer.py: bid-ask imbalance, iceberg detection

Note: Degrades gracefully if quant modules are unavailable.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger("OrderFlowAgent")

try:
    from quant.microstructure import compute_vpin, compute_trade_imbalance, get_microstructure_score
    from quant.orderbook_analyzer import get_orderbook_signal, get_realtime_orderbook_signal
    _QUANT_AVAILABLE = True
except ImportError:
    _QUANT_AVAILABLE = False
    logger.warning("OrderFlowAgent: quant modules unavailable, using fallback")

_MIN_BARS = 30


class OrderFlowAgent(BaseAgent):
    """
    Agent that scores order flow signals from microstructure analysis.

    Score interpretation:
        > 0.6 → strong buy flow (long signal)
        < 0.4 → strong sell flow (short signal)
        0.4–0.6 → neutral / uncertain
    """

    def __init__(self):
        super().__init__("orderflow", initial_weight=0.20)

    def analyse(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Optional[AgentResult]:
        """
        Analyse order flow and microstructure signals.

        Parameters
        ----------
        symbol   : trading pair
        interval : timeframe
        df       : OHLCV DataFrame

        Returns
        -------
        AgentResult with score, direction, and microstructure metadata.
        """
        if df is None or len(df) < _MIN_BARS:
            return None

        if not _QUANT_AVAILABLE:
            return self._fallback_analysis(symbol, interval, df)

        try:
            details = []
            metadata = {}

            # ---- Microstructure score ----
            micro_score = get_microstructure_score(df)
            if micro_score is None:
                micro_score = 0.0

            # ---- Order book signal (real L2 preferred, OHLCV proxy fallback) ----
            realtime_ob = get_realtime_orderbook_signal(symbol, df=df)
            using_real = bool(realtime_ob.get("using_real_data", False))
            if using_real:
                absorption = float(realtime_ob.get("absorption", 0.0))
                imbalance = float(realtime_ob.get("real_imbalance", 0.5))
                trade_flow = float(realtime_ob.get("order_flow_imbalance", 0.0))  # [-1,1]
                trade_imbalance = float(np.clip((trade_flow + 1.0) / 2.0, 0.0, 1.0))  # [0,1]
                iceberg = False
                details.append("📡 L2 Real Data")
            else:
                ob_signal = get_orderbook_signal(df)
                absorption = float(ob_signal.get("absorption", 0.0))
                imbalance = float(ob_signal.get("imbalance", 0.5))
                iceberg = bool(ob_signal.get("iceberg_recent", False))
                trade_imbalance = 0.5
                details.append("📊 OHLCV Proxy")

            # ---- Trade imbalance ----
            if not using_real:
                imb_series = compute_trade_imbalance(df, window=20)
                if imb_series is not None and not imb_series.dropna().empty:
                    trade_imbalance = float(imb_series.dropna().iloc[-1])

            # ---- VPIN ----
            vpin_series = compute_vpin(df, n_buckets=min(50, len(df) // 2))
            vpin_val = 0.3
            if vpin_series is not None and not vpin_series.dropna().empty:
                vpin_val = float(vpin_series.dropna().iloc[-1])

            # VPIN in [0,1]: high = uncertain (penalise score)
            vpin_confidence_factor = max(0.0, 1.0 - vpin_val)

            # ---- Composite score ----
            # micro_score in [-1, 1] → map to [0, 1]
            micro_component = (micro_score + 1.0) / 2.0

            # Trade imbalance in [0, 1]
            # Absorption in [-1, 1] → map to [0, 1]
            absorption_component = (absorption + 1.0) / 2.0

            # Weighted composite
            raw_score = (
                0.35 * micro_component +
                0.35 * trade_imbalance +
                0.30 * absorption_component
            )

            # Apply VPIN confidence reduction
            score = float(np.clip(raw_score, 0.0, 1.0))

            # Confidence: reduce when VPIN is high (informed trading uncertainty)
            confidence = float(np.clip(vpin_confidence_factor * 0.8, 0.0, 1.0))

            # Direction
            if score > 0.58:
                direction = "long"
                details.append(f"Buy flow: imbalance={trade_imbalance:.2f}, absorption={absorption:.2f}")
            elif score < 0.42:
                direction = "short"
                details.append(f"Sell flow: imbalance={trade_imbalance:.2f}, absorption={absorption:.2f}")
            else:
                direction = "neutral"
                details.append(f"Balanced flow: imbalance={trade_imbalance:.2f}")

            if iceberg:
                details.append(f"⚠️ Iceberg order detected")
                confidence *= 0.8  # reduce confidence when icebergs present

            if vpin_val > 0.6:
                details.append(f"⚠️ High VPIN={vpin_val:.2f} (informed trading risk)")

            metadata.update({
                "microstructure_score": micro_score,
                "trade_imbalance": trade_imbalance,
                "real_orderbook": using_real,
                "absorption": absorption,
                "real_imbalance": imbalance,
                "order_flow_imbalance": float(realtime_ob.get("order_flow_imbalance", 0.0)),
                "vpin": vpin_val,
                "iceberg_detected": iceberg,
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
            logger.warning(f"OrderFlowAgent.analyse error: {exc}")
            return self._fallback_analysis(symbol, interval, df)

    def _fallback_analysis(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
    ) -> Optional[AgentResult]:
        """Simple volume-based fallback when quant modules are unavailable."""
        try:
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            open_p = df["open"].values.astype(float)

            # Simple buy/sell classification from candle direction
            buy_mask = close > open_p
            recent = min(20, len(df))
            buy_vol = volume[-recent:][buy_mask[-recent:]].sum()
            sell_vol = volume[-recent:][~buy_mask[-recent:]].sum()
            total_vol = buy_vol + sell_vol

            if total_vol < 1e-8:
                return None

            imbalance = buy_vol / total_vol
            score = float(np.clip(imbalance, 0.0, 1.0))

            direction = "long" if score > 0.58 else ("short" if score < 0.42 else "neutral")

            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=score,
                direction=direction,
                confidence=0.5,
                details=[f"Volume imbalance={imbalance:.2f} (fallback mode)"],
                metadata={"fallback": True, "imbalance": imbalance},
            )
        except Exception as exc:
            logger.warning(f"OrderFlowAgent fallback error: {exc}")
            return None
