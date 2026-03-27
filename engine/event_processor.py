"""
Event Processor for V17.
Routes market events to agents and orchestrates the decision pipeline.
"""
import logging
import time
import datetime
from typing import Dict, List, Optional, Callable, Any

from agents.base_agent import AgentResult
from agents.pattern_agent import PatternAgent
from agents.regime_agent import RegimeAgent
from agents.confluence_agent import ConfluenceAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.meta_agent import MetaAgent
from engine.decision_fusion import DecisionFusion, FusionResult, DECISION_HOLD
from engine.execution import ExecutionEngine
from data import data_store
from config.settings import (
    ORARI_VIETATI_UTC, ORARI_MIGLIORI_UTC,
    SIGNAL_COOLDOWN_BY_TF, SIGNAL_COOLDOWN,
    MAX_OPEN_POSITIONS,
)

logger = logging.getLogger("EventProcessor")


class EventProcessor:
    """Routes candle close events through the full agent pipeline."""

    def __init__(
        self,
        pattern_agent: PatternAgent,
        regime_agent: RegimeAgent,
        confluence_agent: ConfluenceAgent,
        risk_agent: RiskAgent,
        strategy_agent: StrategyAgent,
        meta_agent: MetaAgent,
        fusion: DecisionFusion,
        execution: ExecutionEngine,
        on_signal: Optional[Callable] = None,
    ):
        self.pattern = pattern_agent
        self.regime = regime_agent
        self.confluence = confluence_agent
        self.risk = risk_agent
        self.strategy = strategy_agent
        self.meta = meta_agent
        self.fusion = fusion
        self.execution = execution
        self.on_signal = on_signal  # callback for notifications

        self._last_signal_time: Dict[str, float] = {}
        self._processed_count = 0
        self._signal_count = 0

    # ------------------------------------------------------------------
    # Time guards
    # ------------------------------------------------------------------

    def _is_forbidden_hour(self) -> bool:
        return datetime.datetime.utcnow().hour in ORARI_VIETATI_UTC

    def _is_signal_cooled(self, symbol: str, interval: str) -> bool:
        key = f"{symbol}_{interval}"
        cooldown = SIGNAL_COOLDOWN_BY_TF.get(interval, SIGNAL_COOLDOWN)
        return (time.time() - self._last_signal_time.get(key, 0)) >= cooldown

    def _mark_signal(self, symbol: str, interval: str) -> None:
        self._last_signal_time[f"{symbol}_{interval}"] = time.time()

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------

    def on_candle_close(self, symbol: str, interval: str, kline: dict) -> Optional[FusionResult]:
        """Process a closed candle event through all agents.

        Returns FusionResult if a trade signal is generated, else None.
        """
        self._processed_count += 1

        # Update realtime data
        data_store.update_realtime(symbol, interval, kline)

        # Guard: forbidden hours
        if self._is_forbidden_hour():
            return None

        # Guard: cooldown
        if not self._is_signal_cooled(symbol, interval):
            return None

        # Guard: max open positions
        open_pos = self.execution.get_open_positions()
        open_for_symbol = [p for p in open_pos if p.symbol == symbol]
        if len(open_pos) >= MAX_OPEN_POSITIONS:
            return None
        if open_for_symbol:
            return None  # Already have a position on this symbol

        df = data_store.get_df(symbol, interval)
        if df is None or len(df) < 50:
            return None

        # ---- Run agents ----
        agent_results: Dict[str, AgentResult] = {}

        # Pattern agent (provides initial direction hint)
        df_btc = data_store.get_df("BTCUSDT", interval)
        pattern_result = self.pattern.safe_analyse(symbol, interval, df, df_btc)
        if pattern_result is not None:
            agent_results["pattern"] = pattern_result
            direction_hint = pattern_result.direction
        else:
            direction_hint = "long"  # default

        # Regime agent
        regime_result = self.regime.safe_analyse(symbol, interval, df)
        if regime_result is not None:
            agent_results["regime"] = regime_result

        # Confluence agent
        confluence_result = self.confluence.safe_analyse(symbol, interval, df, direction_hint)
        if confluence_result is not None:
            agent_results["confluence"] = confluence_result

        # Risk agent
        risk_result = self.risk.safe_analyse(symbol, interval, df, direction_hint)
        if risk_result is not None:
            agent_results["risk"] = risk_result

        # Strategy agent
        strategy_result = self.strategy.safe_analyse(symbol, interval, df, direction_hint)
        if strategy_result is not None:
            agent_results["strategy"] = strategy_result

        if not agent_results:
            return None

        # ---- Fuse decisions ----
        fusion_result = self.fusion.fuse(symbol, interval, agent_results)

        if fusion_result.decision == DECISION_HOLD:
            return None

        # ---- Open position ----
        risk_meta = risk_result.metadata if risk_result else {}
        sl = risk_meta.get("sl", df["close"].iloc[-1] * 0.99)
        tp1 = risk_meta.get("tp1", df["close"].iloc[-1] * 1.02)
        tp2 = risk_meta.get("tp2", df["close"].iloc[-1] * 1.04)
        size = risk_meta.get("size", 0.001)
        entry = risk_meta.get("entry", float(df["close"].iloc[-1]))
        strategy_name = strategy_result.metadata.get("strategy", "") if strategy_result else ""

        position = self.execution.open_position(
            symbol=symbol,
            interval=interval,
            direction=fusion_result.decision,
            entry_price=entry,
            size=size,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            strategy=strategy_name,
            decision_id=fusion_result.decision_id,
        )

        if position:
            self._mark_signal(symbol, interval)
            self._signal_count += 1

            # Notify via callback
            if self.on_signal:
                try:
                    self.on_signal(fusion_result, agent_results, position)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")

        return fusion_result

    def on_price_update(self, symbol: str, current_price: float) -> None:
        """Called on every realtime update to check SL/TP for open positions."""
        self.execution.check_position_levels(symbol, current_price)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "processed": self._processed_count,
            "signals": self._signal_count,
            "execution": self.execution.get_stats(),
        }
