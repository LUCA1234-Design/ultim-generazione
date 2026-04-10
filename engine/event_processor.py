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
from engine.decision_fusion import DecisionFusion, FusionResult, DECISION_HOLD, _SNIPER_MIN_AGREEING_TIMEFRAMES
from engine.execution import ExecutionEngine
from data import data_store
from config.settings import (
    ORARI_VIETATI_UTC,
    ORARI_MIGLIORI_UTC,
    SIGNAL_COOLDOWN_BY_TF,
    SIGNAL_COOLDOWN,
    MAX_OPEN_POSITIONS,
    MIN_FUSION_SCORE,
    MIN_AGENT_CONFIRMATIONS,
    MIN_RR,
    NON_OPTIMAL_HOUR_PENALTY,
)

logger = logging.getLogger("EventProcessor")

# Candle momentum filter thresholds
_VOLUME_THRESHOLD_RATIO = 0.70      # skip if volume < this fraction of recent average
_MIN_EMA_SLOPE_THRESHOLD = 0.0002   # skip if absolute EMA slope is below this (flat market)

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
        # V18 optional agents (graceful degradation)
        orderflow_agent=None,
        sentiment_agent=None,
        correlation_agent=None,
        contrarian_agent=None,
        kill_switch=None,
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

        # V18 optional agents
        self.orderflow_agent = orderflow_agent
        self.sentiment_agent = sentiment_agent
        self.correlation_agent = correlation_agent
        self.contrarian_agent = contrarian_agent
        self.kill_switch = kill_switch

        self._last_signal_time: Dict[str, float] = {}
        self._processed_count = 0
        self._signal_count = 0
        self._last_signal_info: str = ""
        # Risk block log cooldown tracking (per-symbol)
        self._risk_block_log_times: Dict[str, float] = {}
        self._risk_block_counts: Dict[str, int] = {}
        self._risk_block_details: Dict[str, dict] = {}
        # Per-decision context stored on close for feedback loops
        self._decision_contexts: Dict[str, Dict[str, Any]] = {}
        self._skip_reasons: Dict[str, int] = {
            "forbidden_hour": 0,
            "cooldown": 0,
            "max_open_positions": 0,
            "existing_symbol_position": 0,
            "insufficient_data": 0,
            "no_agent_results": 0,
            "insufficient_confirmations": 0,
            "hold_decision": 0,
            "low_fusion_score": 0,
            "low_rr": 0,
            "missing_direction": 0,
            "max_daily_loss_usdt": 0,
            "max_daily_loss_pct": 0,
            "max_consecutive_losses": 0,
            "risk_blocked": 0,
            "high_correlation": 0,
            "unfavorable_regime": 0,
            "weak_confluence": 0,
            "low_volume_candle": 0,
            "flat_ema_slope": 0,
            # V18 kill switch reasons
            "kill_switch_killed": 0,
            "kill_switch_safe_mode": 0,
        }

    # ------------------------------------------------------------------
    # Time guards
    # ------------------------------------------------------------------

    def _is_forbidden_hour(self) -> bool:
        return datetime.datetime.now(datetime.timezone.utc).hour in ORARI_VIETATI_UTC

    def _is_optimal_hour(self) -> bool:
        return datetime.datetime.now(datetime.timezone.utc).hour in ORARI_MIGLIORI_UTC

    def _is_signal_cooled(self, symbol: str, interval: str) -> bool:
        key = f"{symbol}_{interval}"
        cooldown = SIGNAL_COOLDOWN_BY_TF.get(interval, SIGNAL_COOLDOWN)
        return (time.time() - self._last_signal_time.get(key, 0)) >= cooldown

    def _mark_signal(self, symbol: str, interval: str) -> None:
        self._last_signal_time[f"{symbol}_{interval}"] = time.time()
    
    def _skip(self, reason: str) -> None:
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

    def _log_risk_block(self, symbol: str, reason: str, details: dict) -> None:
        """Log risk block events with cooldown to prevent spam.

        Emits one INFO summary per symbol every RISK_LOG_COOLDOWN seconds,
        with count of how many signals were blocked in the interval.
        """
        from config.settings import RISK_LOG_COOLDOWN
        key = f"{symbol}_{reason}"
        now = time.time()
        last_log = self._risk_block_log_times.get(key, 0.0)
        count = self._risk_block_counts.get(key, 0) + 1
        self._risk_block_counts[key] = count
        self._risk_block_details[key] = details

        if now - last_log >= RISK_LOG_COOLDOWN:
            # Build human-readable detail string
            detail_str = ""
            if reason == "max_daily_loss_pct":
                detail_str = (
                    f"daily_loss={details.get('daily_loss_pct', 0):.1f}% "
                    f"> {details.get('daily_loss_pct_max', 0):.1f}% max"
                )
            elif reason == "max_daily_loss_usdt":
                detail_str = (
                    f"daily_loss={details.get('daily_loss_usdt', 0):.2f}$ "
                    f"> {details.get('daily_loss_usdt_max', 0):.2f}$ max"
                )
            elif reason == "max_consecutive_losses":
                detail_str = (
                    f"consecutive_losses={details.get('consecutive_losses', 0)} "
                    f">= {details.get('consecutive_losses_max', 0)} max"
                )
            logger.info(
                f"⚠️ Risk guard active [{symbol}]: {detail_str} ({reason}). "
                f"Blocked {count} signal(s) in last {RISK_LOG_COOLDOWN}s. "
                f"Monitoring continues."
            )
            self._risk_block_log_times[key] = now
            self._risk_block_counts[key] = 0
        else:
            logger.debug(
                f"⛔ {symbol} risk_blocked={reason} "
                f"(blocked {count}x since last log)"
            )

    # ------------------------------------------------------------------
    # Correlation guard
    # ------------------------------------------------------------------

    def _correlation_check(self, symbol: str, interval: str) -> float:
        """Return average correlation between symbol and all open positions.

        Returns 0.0 if no open positions or insufficient data.
        """
        open_pos = self.execution.get_open_positions()
        if not open_pos:
            return 0.0

        df_new = data_store.get_df(symbol, interval)
        if df_new is None or len(df_new) < 20:
            return 0.0

        import numpy as np
        correlations = []
        for pos in open_pos:
            df_existing = data_store.get_df(pos.symbol, interval)
            if df_existing is None or len(df_existing) < 20:
                continue
            try:
                returns_new = df_new["close"].iloc[-21:].pct_change().dropna()
                returns_existing = df_existing["close"].iloc[-21:].pct_change().dropna()
                min_len = min(len(returns_new), len(returns_existing))
                if min_len < 10:
                    continue
                corr = float(
                    np.corrcoef(
                        returns_new.iloc[-min_len:],
                        returns_existing.iloc[-min_len:],
                    )[0, 1]
                )
                correlations.append(corr)
            except Exception:
                continue

        return float(np.mean(correlations)) if correlations else 0.0

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------

    def on_candle_close(self, symbol: str, interval: str, kline: dict) -> Optional[FusionResult]:
        """Process a closed candle event through all agents.

        Returns FusionResult if a trade signal is generated, else None.
        """
        self._processed_count += 1

        # Log skip stats every 100 candles processed
        if self._processed_count % 100 == 0:
            logger.info(
                f"📊 PIPELINE STATS after {self._processed_count} candles: "
                f"signals={self._signal_count} | skips={dict(self._skip_reasons)}"
            )

        # Update realtime data
        data_store.update_realtime(symbol, interval, kline)

        # Guard: forbidden hours
        if self._is_forbidden_hour():
            self._skip("forbidden_hour")
            logger.debug(f"⛔ {symbol}/{interval} SKIP: forbidden_hour")
            return None

        # Guard: cooldown
        if not self._is_signal_cooled(symbol, interval):
            self._skip("cooldown")
            logger.debug(f"⛔ {symbol}/{interval} SKIP: cooldown")
            return None

        # Guard: max open positions
        open_pos = self.execution.get_open_positions()
        open_for_symbol = [p for p in open_pos if p.symbol == symbol]
        if len(open_pos) >= MAX_OPEN_POSITIONS:
            self._skip("max_open_positions")
            logger.info(f"⛔ {symbol}/{interval} SKIP: max_open_positions | open={len(open_pos)}")
            return None
        if open_for_symbol:
            self._skip("existing_symbol_position")
            logger.debug(f"⛔ {symbol}/{interval} SKIP: existing_symbol_position")
            return None  # Already have a position on this symbol
        risk_blocked, risk_reason, risk_details = self.execution.is_risk_blocked()
        if risk_blocked:
            self._log_risk_block(symbol, risk_reason, risk_details)
            self._skip(risk_reason)
            # NON ritornare — continuare la pipeline per analisi, bloccare solo l'esecuzione
            # La posizione non verrà aperta — il blocco avviene in open_position()
            # Ma per efficienza, skippiamo comunque se blocked
            return None

        # Guard: V18 Kill Switch check
        if self.kill_switch is not None:
            try:
                exec_stats = self.execution.get_stats()
                balance = exec_stats.get("balance", exec_stats.get("initial_balance", 1000.0))
                initial_bal = exec_stats.get("initial_balance", balance)
                portfolio_state = {
                    "balance": balance,
                    "initial_balance": initial_bal,
                    "peak_balance": balance,  # simplified: use current balance as peak
                    "daily_pnl": exec_stats.get("daily_pnl", 0),
                    "positions": [{"symbol": p.symbol, "pnl_pct": 0} for p in open_pos],
                    "market_vol": 0,      # not tracked at this layer; L5 trip unlikely
                    "baseline_vol": 0.01,
                    "avg_correlation": 0,
                }
                self.kill_switch.check_all_levels(portfolio_state)
                if self.kill_switch.is_killed():
                    self._skip("kill_switch_killed")
                    logger.warning(f"🔴 {symbol}/{interval} SKIP: Kill Switch KILLED")
                    return None
                if self.kill_switch.is_safe_mode():
                    self._skip("kill_switch_safe_mode")
                    logger.warning(f"🟡 {symbol}/{interval} SKIP: Kill Switch SAFE_MODE")
                    return None
            except Exception as e:
                logger.debug(f"kill_switch check error: {e}")

        df = data_store.get_df(symbol, interval)
        if df is None or len(df) < 50:
            self._skip("insufficient_data")
            logger.debug(f"⛔ {symbol}/{interval} SKIP: insufficient_data | df_len={len(df) if df is not None else 0}")
            return None

        # Guard: correlation with existing positions
        avg_correlation = self._correlation_check(symbol, interval)
        if avg_correlation > 0.80:
            self._skip("high_correlation")
            logger.info(f"⛔ {symbol}/{interval} SKIP: high_correlation={avg_correlation:.2f}")
            return None

        # === FILTRO MOMENTUM CANDELA (CECCHINO) ===
        try:
            if "volume" in df.columns and len(df) >= 21:
                vol_avg = float(df["volume"].iloc[-21:-1].mean())
                vol_current = float(df["volume"].iloc[-1])
                # Candela con volume sotto il 70% della media → segnale debole
                if vol_avg > 0 and vol_current < vol_avg * _VOLUME_THRESHOLD_RATIO:
                    self._skip("low_volume_candle")
                    logger.debug(
                        f"⛔ {symbol}/{interval} SKIP: low_volume_candle "
                        f"vol={vol_current:.0f} avg={vol_avg:.0f} ratio={vol_current/vol_avg:.2f}"
                    )
                    return None
        except Exception as _vol_err:
            logger.debug(f"volume_filter error: {_vol_err}")

        # Filtro EMA slope: richiedere che l'EMA 20 stia accelerando
        try:
            from indicators.technical import ema_slope as _ema_slope_fn
            slope_series = _ema_slope_fn(df["close"], 20, 3)
            if not slope_series.empty:
                current_slope = float(slope_series.iloc[-1])
                # Se slope è quasi zero (mercato piatto), skip
                if abs(current_slope) < _MIN_EMA_SLOPE_THRESHOLD:
                    self._skip("flat_ema_slope")
                    logger.debug(
                        f"⛔ {symbol}/{interval} SKIP: flat_ema_slope={current_slope:.5f}"
                    )
                    return None
        except Exception as _slope_err:
            logger.debug(f"ema_slope_filter error: {_slope_err}")

        # ---- Run agents ----
        agent_results: Dict[str, AgentResult] = {}

        # Pattern agent (provides initial direction hint)
        df_btc = data_store.get_df("BTCUSDT", interval)
        pattern_result = self.pattern.safe_analyse(symbol, interval, df, df_btc)
        if pattern_result is not None:
            agent_results["pattern"] = pattern_result
            direction_hint = pattern_result.direction
        else:
            direction_hint = "neutral"

        # Regime agent
        regime_result = self.regime.safe_analyse(symbol, interval, df)
        current_regime = "unknown"
        if regime_result is not None:
            agent_results["regime"] = regime_result
            current_regime = (
                regime_result.metadata.get("regime", "unknown")
                if regime_result.metadata else "unknown"
            )

        # === FILTRO REGIME VOLATILE POTENZIATO ===
        if current_regime == "volatile" and regime_result is not None:
            # In regime volatile: richiedi almeno score >= 0.80
            # oppure skippa direttamente se score del regime < 0.80
            if regime_result.score < 0.80:
                self._skip("unfavorable_regime")
                logger.info(
                    f"⛔ {symbol}/{interval} SKIP: volatile_regime_strict "
                    f"score={regime_result.score:.2f} < 0.80"
                )
                return None
            # Anche con score >= 0.80, richiedere conferma pattern
            if "pattern" not in agent_results or agent_results["pattern"].score < 0.65:
                self._skip("unfavorable_regime")
                logger.info(
                    f"⛔ {symbol}/{interval} SKIP: volatile_regime_no_pattern_confirm"
                )
                return None

        # Confluence agent
        confluence_result = self.confluence.safe_analyse(symbol, interval, df, direction_hint)
        if confluence_result is not None:
            agent_results["confluence"] = confluence_result
            # ---- SNIPER: Require at least 2/3 TFs agreeing ----
            agreeing_tfs = confluence_result.metadata.get("agreeing_tfs", 0) if confluence_result.metadata else 0
            if agreeing_tfs < _SNIPER_MIN_AGREEING_TIMEFRAMES:
                self._skip("weak_confluence")
                logger.info(
                    f"⛔ {symbol}/{interval} SKIP: weak_confluence "
                    f"agreeing_tfs={agreeing_tfs}/3"
                )
                return None

        # Risk agent
        risk_result = self.risk.safe_analyse(symbol, interval, df, direction_hint, regime=current_regime)
        if risk_result is not None:
            agent_results["risk"] = risk_result

        # Strategy agent
        strategy_result = self.strategy.safe_analyse(symbol, interval, df, direction_hint)
        if strategy_result is not None:
            agent_results["strategy"] = strategy_result

        # Meta agent
        meta_result = self.meta.safe_analyse(symbol, interval, df, agent_results)
        if meta_result is not None:
            agent_results["meta"] = meta_result

        # ---- V18 agents (graceful degradation) ----
        if self.orderflow_agent is not None:
            try:
                of_result = self.orderflow_agent.safe_analyse(symbol, interval, df)
                if of_result is not None:
                    agent_results["orderflow"] = of_result
            except Exception as e:
                logger.debug(f"orderflow_agent error: {e}")

        if self.sentiment_agent is not None:
            try:
                sent_result = self.sentiment_agent.safe_analyse(symbol, interval, df)
                if sent_result is not None:
                    agent_results["sentiment"] = sent_result
            except Exception as e:
                logger.debug(f"sentiment_agent error: {e}")

        if self.correlation_agent is not None:
            try:
                corr_result = self.correlation_agent.safe_analyse(symbol, interval, df, df_btc)
                if corr_result is not None:
                    agent_results["correlation"] = corr_result
            except Exception as e:
                logger.debug(f"correlation_agent error: {e}")

        # Contrarian agent runs LAST — it needs to see consensus direction
        if self.contrarian_agent is not None:
            try:
                contr_result = self.contrarian_agent.safe_analyse(
                    symbol, interval, df,
                    consensus_direction=direction_hint,
                    agent_results=agent_results,
                )
                if contr_result is not None:
                    agent_results["contrarian"] = contr_result
            except Exception as e:
                logger.debug(f"contrarian_agent error: {e}")

        if not agent_results:
            self._skip("no_agent_results")
            logger.info(f"⛔ {symbol}/{interval} SKIP: no_agent_results")
            return None
        if len(agent_results) < MIN_AGENT_CONFIRMATIONS:
            self._skip("insufficient_confirmations")
            logger.info(
                f"⛔ {symbol}/{interval} SKIP: insufficient_confirmations | "
                f"agents={len(agent_results)}/{MIN_AGENT_CONFIRMATIONS} "
                f"present={list(agent_results.keys())}"
            )
            return None

        # ---- Fuse decisions ----
        fusion_result = self.fusion.fuse(symbol, interval, agent_results, regime=current_regime)

        if fusion_result.decision == DECISION_HOLD:
            self._skip("hold_decision")
            logger.info(
                f"⛔ {symbol}/{interval} SKIP: hold_decision | "
                f"agents={len(agent_results)} fusion={fusion_result.final_score:.3f}"
            )
            return None
        if fusion_result.final_score < MIN_FUSION_SCORE:
            self._skip("low_fusion_score")
            logger.info(
                f"⛔ {symbol}/{interval} SKIP: low_fusion_score | "
                f"agents={len(agent_results)} fusion={fusion_result.final_score:.3f} "
                f"threshold={MIN_FUSION_SCORE:.3f}"
            )
            return None

        # ---- Open position ----
        if risk_result and risk_result.metadata:
            risk_meta = risk_result.metadata
        else:
            # ATR-based fallback
            from indicators.technical import atr as calc_atr
            _atr_val = float(calc_atr(df, 14).iloc[-1])
            _close = float(df["close"].iloc[-1])
            risk_meta = {
                "entry": _close,
                "sl": _close - 2.0 * _atr_val if fusion_result.decision == "long" else _close + 2.0 * _atr_val,
                "tp1": _close + 2.0 * _atr_val if fusion_result.decision == "long" else _close - 2.0 * _atr_val,
                "tp2": _close + 4.0 * _atr_val if fusion_result.decision == "long" else _close - 4.0 * _atr_val,
                "size": 0.001,
            }
        sl = risk_meta.get("sl", df["close"].iloc[-1] * 0.99)
        tp1 = risk_meta.get("tp1", df["close"].iloc[-1] * 1.02)
        tp2 = risk_meta.get("tp2", df["close"].iloc[-1] * 1.04)
        size = risk_meta.get("size", 0.001)
        entry = risk_meta.get("entry", float(df["close"].iloc[-1]))
        strategy_name = strategy_result.metadata.get("strategy", "") if strategy_result else ""

        try:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)  # use TP1 to be consistent with RiskAgent
            rr = reward / risk if risk > 0 else 0.0
        except Exception:
            rr = 0.0

        if rr < MIN_RR:
            self._skip("low_rr")
            logger.info(
                f"⛔ {symbol}/{interval} SKIP: low_rr | "
                f"rr={rr:.2f} min={MIN_RR:.2f} entry={entry:.4f} sl={sl:.4f} tp1={tp1:.4f}"
            )
            return None

        # Apply penalty for non-optimal trading hours: require a higher fusion score
        if not self._is_optimal_hour() and fusion_result.final_score < MIN_FUSION_SCORE + NON_OPTIMAL_HOUR_PENALTY:
            self._skip("low_fusion_score")
            logger.info(
                f"⛔ {symbol}/{interval} SKIP: low_fusion_score (non-optimal hour) | "
                f"agents={len(agent_results)} fusion={fusion_result.final_score:.3f} "
                f"threshold={MIN_FUSION_SCORE + NON_OPTIMAL_HOUR_PENALTY:.3f}"
            )
            return None

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
            self._last_signal_info = f"{symbol} {interval} {fusion_result.decision}"

            # Attach signal tags from pattern details
            signal_tags: list = []
            if pattern_result and pattern_result.details:
                signal_tags = list(pattern_result.details)
            fusion_result.signal_tags = signal_tags

            # Store decision context so the evolution engine can access it later
            try:
                self._decision_contexts[fusion_result.decision_id] = {
                    "symbol": symbol,
                    "interval": interval,
                    "agent_scores": {n: r.score for n, r in agent_results.items()},
                    "agent_directions": {n: r.direction for n, r in agent_results.items()},
                    "agent_results": dict(agent_results),
                    "fusion_score": fusion_result.final_score,
                    "regime": current_regime,
                }
            except Exception as _ctx_err:
                logger.debug(f"decision_context store error: {_ctx_err}")

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

    def get_decision_context(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored per-decision context, or None if not found."""
        return self._decision_contexts.get(decision_id)

    def clear_decision_context(self, decision_id: str) -> None:
        """Remove a stored decision context after it has been consumed."""
        self._decision_contexts.pop(decision_id, None)

    def get_stats(self) -> Dict[str, Any]:
        return {
               "processed": self._processed_count,
               "signals": self._signal_count,
               "skip_reasons": dict(self._skip_reasons),
               "execution": self.execution.get_stats(),
               "last_signal": self._last_signal_info,
               "fusion_threshold": self.fusion._threshold,
        } 
