"""
V17 Agentic AI Trading System — Main Orchestrator
Transforms V16 "Cecchino Istituzionale" into a multi-agent adaptive system.
"""
import gc
import logging
import sys
import threading
import time
from typing import Dict, Any, Optional, Set, Tuple

# ---- Config ----
from config.settings import (
    PAPER_TRADING, ACCOUNT_BALANCE, HG_ENABLED, HG_MONITOR_ALL,
    HG_MIN_QUOTE_VOL, SYMBOLS_LIMIT, TELEGRAM_TEST_ON_START,
    STARTUP_TIMEOUT, POLL_CLOSED_ENABLE, DB_PATH,
    HEARTBEAT_INTERVAL, HEARTBEAT_ENABLED,
    ORDERBOOK_STREAM_ENABLED, ORDERBOOK_MAX_SYMBOLS,
    TRAINING_MODE, TRAINING_TARGET_TRADES,
    TRAINING_FUSION_THRESHOLD, TRAINING_MIN_FUSION_SCORE,
    TRAINING_MIN_AGENT_CONFIRMATIONS, TRAINING_MIN_RR, TRAINING_NON_OPTIMAL_HOUR_PENALTY,
    SNIPER_FUSION_THRESHOLD, SNIPER_MIN_FUSION_SCORE,
    SNIPER_MIN_AGENT_CONFIRMATIONS, SNIPER_MIN_RR,
    SNIPER_NON_OPTIMAL_HOUR_PENALTY, SNIPER_SIGNAL_COOLDOWN_BY_TF,
    SNIPER_MAX_OPEN_POSITIONS,
)
import config.settings as _cfg  # Used for runtime threshold updates on Training → Sniper switch

# ---- Data layer ----
from data import data_store
from data.binance_client import get_client, fetch_futures_klines, fetch_exchange_info, fetch_futures_ticker
from data.websocket_manager import (
    start_websockets, startup_health_check, start_rest_fallback,
    register_callbacks,
)

# ---- Indicators (imported so available globally) ----
import indicators.technical  # noqa
import indicators.smart_money  # noqa

# ---- Agents ----
from agents.pattern_agent import PatternAgent
from agents.regime_agent import RegimeAgent
from agents.confluence_agent import ConfluenceAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.meta_agent import MetaAgent

# ---- V18 agents (graceful degradation) ----
try:
    from agents.orderflow_agent import OrderFlowAgent
    _ORDERFLOW_AVAILABLE = True
except ImportError:
    _ORDERFLOW_AVAILABLE = False

try:
    from agents.sentiment_agent import SentimentAgent
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

try:
    from agents.correlation_agent import CorrelationAgent
    _CORRELATION_AVAILABLE = True
except ImportError:
    _CORRELATION_AVAILABLE = False

try:
    from agents.contrarian_agent import ContrarianAgent
    _CONTRARIAN_AVAILABLE = True
except ImportError:
    _CONTRARIAN_AVAILABLE = False

# ---- V18 coordination (graceful degradation) ----
try:
    from coordination.state_machine import StateMachine as _StateMachine
    from coordination.state_machine import (
        STATE_TRAINING, STATE_SNIPER, STATE_SAFE_MODE, STATE_KILLED,
    )
    _STATE_MACHINE_AVAILABLE = True
except ImportError:
    _STATE_MACHINE_AVAILABLE = False

try:
    from coordination.message_bus import (
        get_message_bus,
        TOPIC_SIGNAL_NEW, TOPIC_TRADE_OPEN, TOPIC_TRADE_CLOSE, TOPIC_REGIME_CHANGE,
    )
    _MSG_BUS_AVAILABLE = True
except ImportError:
    _MSG_BUS_AVAILABLE = False

try:
    from risk_institutional.institutional_risk_manager import InstitutionalRiskManager
    _INSTITUTIONAL_RISK_AVAILABLE = True
except ImportError:
    _INSTITUTIONAL_RISK_AVAILABLE = False

# ---- Engine ----
from engine.decision_fusion import DecisionFusion
from engine.execution import ExecutionEngine
from engine.event_processor import EventProcessor

# ---- Memory ----
from memory import experience_db
from memory.performance_tracker import PerformanceTracker

# ---- Evolution engine ----
from evolution.evolution_engine import EvolutionEngine

# ---- Services ----
from services.notification_worker import (
    start_notification_worker,
    enqueue_signal_notification,
)
try:
    from services.latency_monitor import get_latency_report, start_latency_monitor
    _LATENCY_MONITOR_AVAILABLE = True
except ImportError:
    _LATENCY_MONITOR_AVAILABLE = False

# ---- Notifications ----
from notifications.telegram_service import (
    send_message,
    test_connection,
    build_startup_message,
    build_stats_message,
    build_heartbeat_message,
    notify_position_closed,
)


# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v17.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Main")

# ---------------------------------------------------------------------------
# Global symbol universe
# ---------------------------------------------------------------------------

symbols_whitelist = []    # Top N by volume (for divergence/pattern scanning)
symbols_hg_all = []       # All USDT-M perpetual futures (for HG scan)

# Global flag: True once Training Mode has completed and Sniper Mode is active
_sniper_mode_active: bool = False

def load_top_symbols(limit: int = SYMBOLS_LIMIT) -> None:
    global symbols_whitelist
    try:
        info = fetch_exchange_info()
        tickers = fetch_futures_ticker()
        qvol_map = {t.get("symbol"): float(t.get("quoteVolume", 0)) for t in tickers if t.get("symbol")}
        valid = [
            s["symbol"] for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and qvol_map.get(s["symbol"], 0) > 0
        ]
        valid.sort(key=lambda sym: qvol_map.get(sym, 0), reverse=True)
        symbols_whitelist = valid[:limit]
        logger.info(f"✅ Loaded {len(symbols_whitelist)} top USDT-M symbols")
    except Exception as e:
        logger.error(f"load_top_symbols: {e}")
        symbols_whitelist = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_all_futures_symbols() -> None:
    global symbols_hg_all
    try:
        info = fetch_exchange_info()
        symbols_hg_all = sorted([
            s["symbol"] for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ])
        logger.info(f"💎 Loaded {len(symbols_hg_all)} HG symbols")
    except Exception as e:
        logger.error(f"load_all_futures_symbols: {e}")
        symbols_hg_all = list(symbols_whitelist)


def filter_hg_symbols_by_liquidity(min_quote_usdt: float = HG_MIN_QUOTE_VOL) -> None:
    global symbols_hg_all
    try:
        tickers = fetch_futures_ticker()
        qvol = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in tickers if "symbol" in t}
        symbols_hg_all = [s for s in symbols_hg_all if qvol.get(s, 0) >= min_quote_usdt]
        logger.info(f"💎 HG after liquidity filter (>{min_quote_usdt}): {len(symbols_hg_all)}")
    except Exception as e:
        logger.warning(f"filter_hg_symbols_by_liquidity: {e}")


def load_universes() -> None:
    load_top_symbols(SYMBOLS_LIMIT)
    load_all_futures_symbols()
    filter_hg_symbols_by_liquidity(HG_MIN_QUOTE_VOL)


# ---------------------------------------------------------------------------
# Historical data preloading
# ---------------------------------------------------------------------------

def _interpolate_param(training_val, sniper_val, progress: float):
    """Interpola linearmente da training_val a sniper_val.
    progress=0.0 → training_val, progress=1.0 → sniper_val.
    """
    progress = float(max(0.0, min(1.0, progress)))
    return training_val + (sniper_val - training_val) * progress


def preload_historical(symbols, label: str = "") -> None:
    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        # Il 1d viene precaricato via REST — il WebSocket Binance non supporta
        # klines daily in modalità stream continuo, quindi usiamo solo REST polling
        for interval in ["15m", "1h", "4h", "1d"]:
            try:
                klines = fetch_futures_klines(sym, interval, limit=500)
                if klines:
                    data_store.store_historical(sym, interval, klines)
                time.sleep(0.05)
            except Exception as e:
                logger.debug(f"preload {sym} {interval}: {e}")
        if idx % 20 == 0 or idx == total:
            logger.info(f"📊 {label} preload: {idx}/{total}")


# ---------------------------------------------------------------------------
# Agent & engine wiring
# ---------------------------------------------------------------------------

def build_system():
    """Instantiate and wire all V17 + V18 components.

    Returns
    -------
    processor        : EventProcessor
    meta             : MetaAgent
    tracker          : PerformanceTracker
    execution        : ExecutionEngine
    risk             : RiskAgent
    strategy         : StrategyAgent
    confluence       : ConfluenceAgent
    pattern          : PatternAgent
    decision_context : Dict[str, Dict[str, Any]] — runtime signal context cache
    """
    logger.info("🔧 Building V17+V18 agent system...")

    pattern = PatternAgent()
    regime = RegimeAgent()
    confluence = ConfluenceAgent()
    risk = RiskAgent()
    strategy = StrategyAgent()

    # ---- V18 agents (graceful degradation) ----
    orderflow = None
    sentiment = None
    correlation = None
    contrarian = None

    if _ORDERFLOW_AVAILABLE:
        try:
            orderflow = OrderFlowAgent()
            logger.info("🔮 V18 OrderFlowAgent: instantiated")
        except Exception as e:
            logger.warning(f"OrderFlowAgent init error: {e}")

    if _SENTIMENT_AVAILABLE:
        try:
            sentiment = SentimentAgent()
            logger.info("🔮 V18 SentimentAgent: instantiated")
        except Exception as e:
            logger.warning(f"SentimentAgent init error: {e}")

    if _CORRELATION_AVAILABLE:
        try:
            correlation = CorrelationAgent()
            logger.info("🔮 V18 CorrelationAgent: instantiated")
        except Exception as e:
            logger.warning(f"CorrelationAgent init error: {e}")

    if _CONTRARIAN_AVAILABLE:
        try:
            contrarian = ContrarianAgent()
            logger.info("🔮 V18 ContrarianAgent: instantiated")
        except Exception as e:
            logger.warning(f"ContrarianAgent init error: {e}")

    # Wire all agents (V17 + V18) into MetaAgent
    all_agents = [pattern, regime, confluence, risk, strategy]
    for v18_agent in [orderflow, sentiment, correlation, contrarian]:
        if v18_agent is not None:
            all_agents.append(v18_agent)

    meta = MetaAgent(agents=all_agents)

    fusion = DecisionFusion()
    execution = ExecutionEngine(paper_trading=PAPER_TRADING, initial_balance=ACCOUNT_BALANCE)
    tracker = PerformanceTracker()
    supervisor = None

    # decision_id -> context used later on position close
    decision_context: Dict[str, Dict[str, Any]] = {}

    # Load historical win rates from DB into RiskAgent
    try:
        from memory.experience_db import get_agent_win_rates
        db_win_rates = get_agent_win_rates()
        for key, wr in db_win_rates.items():
            risk.set_win_rate(key, wr)
    except Exception as e:
        logger.debug(f"Could not load win rates from DB: {e}")

    def on_signal(fusion_result, agent_results, position):
        """Signal callback: keep fast path light and offload heavy work."""
        try:
            queued = enqueue_signal_notification(
                fusion_result=fusion_result,
                agent_results=agent_results,
                position=position,
            )
            if not queued:
                logger.error(
                    f"Failed to queue signal notification for "
                    f"{fusion_result.symbol} [{fusion_result.interval}]"
                )
        except Exception as e:
            logger.error(f"Signal enqueue error: {e}")

        # Save runtime context for later close handling
        try:
            # Extract regime from agent_results
            _regime = "unknown"
            _regime_result = agent_results.get("regime")
            if _regime_result and hasattr(_regime_result, "metadata") and _regime_result.metadata:
                _regime = _regime_result.metadata.get("regime", "unknown")
            decision_context[fusion_result.decision_id] = {
                "symbol": fusion_result.symbol,
                "interval": fusion_result.interval,
                "decision": fusion_result.decision,
                "agent_scores": dict(fusion_result.agent_scores or {}),
                "agent_directions": {
                    name: getattr(result, "direction", "")
                    for name, result in agent_results.items()
                },
                "agent_results": dict(agent_results),
                "regime": _regime,
            }
        except Exception as e:
            logger.error(f"Decision context cache error: {e}")

        # V18: Publish signal.new on MessageBus
        try:
            if _MSG_BUS_AVAILABLE:
                get_message_bus().publish(TOPIC_SIGNAL_NEW, {
                    "symbol": fusion_result.symbol,
                    "interval": fusion_result.interval,
                    "decision": fusion_result.decision,
                    "score": fusion_result.final_score,
                })
        except Exception:
            pass

    try:
        from coordination.agent_supervisor import AgentSupervisor
        supervisor = AgentSupervisor()
        logger.info("🔎 V18 AgentSupervisor: ON")
    except ImportError:
        supervisor = None

    processor = EventProcessor(
        pattern_agent=pattern,
        regime_agent=regime,
        confluence_agent=confluence,
        risk_agent=risk,
        strategy_agent=strategy,
        meta_agent=meta,
        fusion=fusion,
        execution=execution,
        on_signal=on_signal,
        orderflow_agent=orderflow,
        sentiment_agent=sentiment,
        correlation_agent=correlation,
        contrarian_agent=contrarian,
        supervisor=supervisor,
    )

    logger.info("✅ V17+V18 agent system ready")
    return processor, meta, tracker, execution, risk, strategy, confluence, pattern, decision_context


# ---------------------------------------------------------------------------
# Position monitoring thread
# ---------------------------------------------------------------------------

def _position_monitor(
    processor: EventProcessor,
    tracker: PerformanceTracker,
    decision_context: Dict[str, Dict[str, Any]],
    interval_sec: int = 10,
    evolution_engine: Optional["EvolutionEngine"] = None,
) -> None:
    """Periodically update SL/TP levels for open positions using latest prices."""
    processed_close_ids: Set[Tuple[str, Any]] = set()

    def _handle_closed_position(closed) -> None:
        close_id = (
            str(getattr(closed, "position_id", "") or ""),
            getattr(closed, "close_time", None),
        )
        if close_id in processed_close_ids:
            return

        # Track performance
        try:
            tracker.record_position(closed)
        except Exception as e:
            logger.error(f"tracker.record_position error: {e}")

        # Notify close
        try:
            notify_position_closed(closed)
        except Exception as e:
            logger.error(f"notify_position_closed error: {e}")

        # Update decision outcome in DB
        try:
            if closed.decision_id:
                experience_db.update_decision_outcome(
                    decision_id=closed.decision_id,
                    outcome=closed.status,
                    pnl=closed.pnl or 0.0,
                )
        except Exception as e:
            logger.error(f"update_decision_outcome error: {e}")

        # After updating decision outcome, adapt the fusion threshold
        try:
            correct = (closed.pnl or 0.0) > 0
            processor.fusion.adapt_threshold(correct, 0.0)
        except Exception as e:
            logger.error(f"adapt_threshold error: {e}")

        # Save agent outcomes in DB
        try:
            ctx = decision_context.get(closed.decision_id, {})
            agent_scores = ctx.get("agent_scores", {})
            agent_directions = ctx.get("agent_directions", {})
            correct = (closed.pnl or 0.0) > 0

            # Extract pattern tags for the pattern agent row
            pattern_tags = ""
            pattern_ctx = ctx.get("agent_results", {}).get("pattern")
            if pattern_ctx and hasattr(pattern_ctx, "details"):
                clean_tags = [
                    str(d).split("(")[0].strip()
                    for d in list(pattern_ctx.details)[:10]
                    if d
                ]
                pattern_tags = ",".join(clean_tags)

            for agent_name, score in agent_scores.items():
                experience_db.save_agent_outcome(
                    decision_id=closed.decision_id,
                    agent_name=agent_name,
                    score=float(score),
                    direction=str(agent_directions.get(agent_name, "")),
                    correct=correct,
                    pattern_tags=pattern_tags if agent_name == "pattern" else "",
                )
        except Exception as e:
            logger.error(f"save_agent_outcome error: {e}")

        # NOTE: strategy outcome is forwarded via evolution_engine.on_trade_close()
        # -> StrategyEvolver.record_trade() -> strategy.update_strategy_outcome().
        # Do NOT call update_strategy_outcome() here directly (would double-count).

        # Record outcome in MetaAgent for weight adjustment (Fix 8)
        try:
            ctx = decision_context.get(closed.decision_id, {})
            stored_agent_results = ctx.get("agent_results", {})
            regime = ctx.get("regime", "unknown")
            if stored_agent_results and hasattr(processor.meta, "record_outcome"):
                was_correct = (closed.pnl or 0.0) > 0
                processor.meta.record_outcome(
                    closed.decision_id,
                    stored_agent_results,
                    was_correct,
                    regime=regime,
                )
        except Exception as e:
            logger.error(f"meta.record_outcome error: {e}")

        # Save ctx before pop
        try:
            ctx_for_evolution = decision_context.get(closed.decision_id, {})
        except Exception:
            ctx_for_evolution = {}

        # Clean runtime context (main cache)
        try:
            if closed.decision_id in decision_context:
                decision_context.pop(closed.decision_id, None)
        except Exception as e:
            logger.debug(f"decision_context cleanup error: {e}")

        # Notify evolution engine of closed trade (usa ctx_for_evolution, non decision_context)
        try:
            if evolution_engine is not None:
                # Also try the processor's own context store if ctx is empty
                if not ctx_for_evolution and hasattr(processor, "get_decision_context"):
                    ctx_for_evolution = processor.get_decision_context(
                        getattr(closed, "decision_id", None)
                    ) or {}
                evolution_engine.on_trade_close(closed, ctx_for_evolution)
        except Exception as e:
            logger.error(f"evolution_engine.on_trade_close error: {e}")

        # Cleanup processor-side fallback cache only after evolution callback
        try:
            if hasattr(processor, "clear_decision_context"):
                processor.clear_decision_context(getattr(closed, "decision_id", ""))
        except Exception as e:
            logger.debug(f"processor.decision_context cleanup error: {e}")

        # V18: Publish trade.close event on MessageBus
        try:
            if _MSG_BUS_AVAILABLE:
                get_message_bus().publish(TOPIC_TRADE_CLOSE, {
                    "symbol": closed.symbol,
                    "pnl": closed.pnl,
                    "strategy": closed.strategy,
                })
        except Exception:
            pass

        processed_close_ids.add(close_id)

        # Bound memory in long-running sessions
        if len(processed_close_ids) > 5000:
            recent = processor.execution.get_closed_positions(limit=1000)
            processed_close_ids.clear()
            for p in recent:
                _id = str(getattr(p, "position_id", "") or "")
                if _id:
                    processed_close_ids.add((_id, getattr(p, "close_time", None)))

    while True:
        try:
            open_pos = processor.execution.get_open_positions()
            for pos in open_pos:
                df = data_store.get_df(pos.symbol, pos.interval)
                if df is None or df.empty:
                    continue

                current_price = float(df["close"].iloc[-1])
                closed_positions = processor.execution.check_position_levels(pos.symbol, current_price)

                for closed in closed_positions:
                    _handle_closed_position(closed)

            # Catch closes executed outside check_position_levels (e.g. training timeouts)
            for closed in processor.execution.get_closed_positions(limit=200):
                _handle_closed_position(closed)

        except Exception as e:
            logger.debug(f"position_monitor error: {e}")

        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------

def _heartbeat_loop(processor: EventProcessor, interval_sec: int) -> None:
    """Send periodic heartbeat messages via Telegram so the user knows the bot is alive."""
    import traceback
    logger.info("🫀 Heartbeat thread started, first beat in 120s")
    start_time = time.time()
    time.sleep(120)  # Wait 2 min before first heartbeat
    consecutive_errors = 0
    while True:
        try:
            uptime_sec = time.time() - start_time
            hours = int(uptime_sec // 3600)
            minutes = int((uptime_sec % 3600) // 60)

            stats = processor.get_stats()
            processed = stats.get("processed", 0)
            signals = stats.get("signals", 0)
            skip_reasons = stats.get("skip_reasons", {})
            exec_stats = stats.get("execution", {})
            open_pos = exec_stats.get("open_positions", 0)
            balance = exec_stats.get("balance") or 0
            risk_blocked = exec_stats.get("risk_blocked", False)
            fusion_threshold = stats.get("fusion_threshold", 0.0)
            last_signal_info = stats.get("last_signal", "")

            # Build training-mode status string for heartbeat
            try:
                completed_trades = experience_db.get_completed_trade_count()
            except Exception:
                completed_trades = 0
            if _sniper_mode_active:
                training_status = "🎯 Sniper Mode attivo"
            else:
                training_status = f"📚 Trade: {completed_trades}/{TRAINING_TARGET_TRADES} → Training Mode"
            latency_info = get_latency_report() if _LATENCY_MONITOR_AVAILABLE else None

            msg = build_heartbeat_message(
                uptime_hours=hours,
                uptime_minutes=minutes,
                processed=processed,
                signals=signals,
                open_positions=open_pos,
                balance=balance,
                risk_blocked=risk_blocked,
                skip_reasons=skip_reasons,
                fusion_threshold=fusion_threshold,
                last_signal_info=last_signal_info,
                training_status=training_status,
                latency_info=latency_info,
            )
            send_message(msg)
            logger.info("🫀 Heartbeat sent")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Heartbeat error: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 3:
                try:
                    send_message("🔴 V17 HEARTBEAT — system alive but stats unavailable")
                except Exception:
                    pass
        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Periodic reporting thread
# ---------------------------------------------------------------------------

def _report_loop(processor: EventProcessor, tracker: PerformanceTracker,
                  meta: MetaAgent, interval_sec: int = 3600) -> None:
    """Send periodic performance reports via Telegram."""
    import traceback
    logger.info("📊 Report thread started, first report in 60s")
    time.sleep(60)  # Give system time to start
    consecutive_errors = 0
    while True:
        try:
            exec_stats = processor.execution.get_stats()
            perf_summary = tracker.get_summary()
            try:
                agent_report = meta.get_report() if meta is not None else {}
            except Exception as report_err:
                logger.warning(f"meta.get_report failed, using empty report: {report_err}")
                agent_report = {}
            msg = build_stats_message(exec_stats, perf_summary, agent_report)
            send_message(msg)
            logger.info("📊 Periodic report sent")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"_report_loop error: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 3:
                try:
                    send_message("🔴 V17 REPORT — system alive but report unavailable")
                except Exception:
                    pass
        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("🤖 V18 AGENTIC AI TRADING SYSTEM — ECOSISTEMA VIVENTE")
    logger.info("=" * 60)
    logger.info("🛡️ V17 ACTIVE MODULES:")
    logger.info("   - Regime Agent (GaussianMixture): ON")
    logger.info("   - Pattern Agent (V16 detectors + auto-calibration): ON")
    logger.info("   - Confluence Agent (Probabilistic MTF): ON")
    logger.info("   - Risk Agent (Kelly + real win rates): ON")
    logger.info("   - Strategy Agent (generation + evaluation): ON")
    logger.info("   - Meta Agent (weight adjustment): ON")
    logger.info("   - Decision Fusion (weighted voting): ON")
    logger.info(f"   - Execution: {'PAPER TRADING' if PAPER_TRADING else 'LIVE TRADING'}")
    logger.info("   - Experience DB (SQLite): ON")
    logger.info("=" * 60)

    # ---- Graceful shutdown on SIGTERM (Docker / systemd) ----
    import signal as _signal

    def _sigterm_handler(signum, frame):
        logger.info("⏹️ SIGTERM received — initiating graceful shutdown")
        raise KeyboardInterrupt  # reuse the existing cleanup path

    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    try:
        # ---- DB init ----
        experience_db.init_db(DB_PATH)

        # ---- Async workers ----
        start_notification_worker()

        # ---- Telegram test ----
        if TELEGRAM_TEST_ON_START:
            test_connection()

        # ---- Binance client (initialise early to validate credentials) ----
        _ = get_client()
        if _LATENCY_MONITOR_AVAILABLE:
            start_latency_monitor()

        # ---- Load symbol universes ----
        load_universes()

        if not symbols_whitelist:
            raise ValueError("❌ No symbols loaded for scanning!")

        # Start L2 order book streams for top symbols
        if ORDERBOOK_STREAM_ENABLED:
            try:
                from data.orderbook_stream import start_orderbook_streams
                ob_symbols = symbols_whitelist[:ORDERBOOK_MAX_SYMBOLS]
                start_orderbook_streams(ob_symbols)
                logger.info(f"📡 L2 OrderBook streams started for {len(ob_symbols)} symbols")
            except Exception as e:
                logger.warning(f"OrderBook streams unavailable: {e}")

        # ---- Preload historical data ----
        logger.info(f"📥 Preloading history for {len(symbols_whitelist)} symbols (main list)...")
        preload_historical(symbols_whitelist, "MAIN")

        if HG_ENABLED and HG_MONITOR_ALL and symbols_hg_all:
            # Only preload HG symbols not already in whitelist
            hg_extra = [s for s in symbols_hg_all if s not in set(symbols_whitelist)]
            if hg_extra:
                logger.info(f"📥 Preloading history for {len(hg_extra)} HG-only symbols...")
                preload_historical(hg_extra, "HG")

        # ---- Build V17+V18 system ----
        processor, meta, tracker, execution, risk_agent, strategy_agent, confluence_agent, pattern_agent, decision_context = build_system()

        # ---- V18: Instantiate State Machine ----
        state_machine = None
        if _STATE_MACHINE_AVAILABLE:
            try:
                state_machine = _StateMachine()
                state_machine.transition("boot_complete")  # INITIALIZING → TRAINING
                logger.info(f"🔄 V18 StateMachine: state={state_machine.current_state()}")
            except Exception as e:
                logger.warning(f"StateMachine init error: {e}")

        # ---- V18: Instantiate MessageBus ----
        msg_bus = None
        if _MSG_BUS_AVAILABLE:
            try:
                msg_bus = get_message_bus()
                logger.info("📡 V18 MessageBus: started")
            except Exception as e:
                logger.warning(f"MessageBus init error: {e}")

        # ---- Build & start Evolution Engine ----
        evolution_engine = EvolutionEngine(
            meta_agent=meta,
            fusion=processor.fusion,
            risk_agent=risk_agent,
            strategy_agent=strategy_agent,
            confluence_agent=confluence_agent,
            tracker=tracker,
            pattern_agent=pattern_agent,
            regime_agent=processor.regime,
        )
        evolution_engine.startup()

        # ---- V18: Wire KillSwitch from EvolutionEngine to EventProcessor ----
        try:
            if evolution_engine._kill_switch is not None:
                processor.kill_switch = evolution_engine._kill_switch
                logger.info("🔴 V18 KillSwitch: wired to EventProcessor")
        except Exception as e:
            logger.warning(f"KillSwitch wiring error: {e}")

        # ---- V18: Wire PPO size hints from EvolutionEngine to EventProcessor ----
        try:
            processor._evolution_engine = evolution_engine
            logger.info("🎓 PPO RL size hints: wired to EventProcessor")
        except Exception as _e:
            logger.debug(f"🎓 PPO wiring error: {_e}")

        # ---- V18: Wire institutional risk manager (kill switch + ATR sizing + dynamic trailing) ----
        try:
            if _INSTITUTIONAL_RISK_AVAILABLE:
                processor.institutional_risk = InstitutionalRiskManager(
                    kill_switch=evolution_engine._kill_switch
                )
                processor.risk_alert_callback = send_message
                logger.info("🛡️ InstitutionalRiskManager: wired to EventProcessor")
        except Exception as _risk_wire_err:
            logger.warning(f"InstitutionalRiskManager wiring error: {_risk_wire_err}")

        # ---- V18: Register MessageBus subscribers ----
        if msg_bus is not None:
            try:
                from coordination.message_bus import TOPIC_RISK_ALERT, TOPIC_DRIFT_DETECTED

                def _on_risk_alert(msg):
                    logger.warning(f"📡 MessageBus RISK_ALERT received: {msg}")
                    try:
                        if state_machine is not None:
                            current = state_machine.current_state()
                            if current not in ("KILLED", "SAFE_MODE"):
                                state_machine.transition("risk_alert")
                                logger.warning(f"📡 StateMachine → {state_machine.current_state()} (risk_alert)")
                    except Exception as _sm_err:
                        logger.debug(f"📡 StateMachine risk_alert transition error: {_sm_err}")

                def _on_drift_detected(msg):
                    logger.info(f"📡 MessageBus DRIFT_DETECTED: {msg}")
                    try:
                        if hasattr(evolution_engine, "_drift_detector") and evolution_engine._drift_detector is not None:
                            logger.info("📡 Concept drift detected — flagging for re-train on next tick")
                            evolution_engine._last_drift_retrain = 0.0  # force re-train on next tick
                    except Exception as _drift_err:
                        logger.debug(f"📡 Drift handler error: {_drift_err}")

                msg_bus.subscribe(TOPIC_RISK_ALERT, _on_risk_alert)
                msg_bus.subscribe(TOPIC_DRIFT_DETECTED, _on_drift_detected)
                logger.info("📡 MessageBus: subscribers registered (RISK_ALERT, DRIFT_DETECTED)")
            except Exception as _bus_err:
                logger.warning(f"📡 MessageBus subscriber registration error: {_bus_err}")

        logger.info("🧬 Evolution Engine started")
        logger.info("   - Loop #1 (MetaAgent → Fusion): ON")
        logger.info("   - Loop #2 (Tracker → RiskAgent): ON")
        logger.info("   - Loop #3 (Auto-tune threshold): ON")
        logger.info("   - Loop #5 (Strategy evolution): ON")
        logger.info("   - Loop #6 (MetaAgent persistence): ON")
        logger.info("   - Loop #7 (Confluence TF learning): ON")
        logger.info("🚀 V18 MODULES:")
        logger.info("   - V18 Order Flow Agent: " + ("ON" if processor.orderflow_agent else "OFF"))
        logger.info("   - V18 Sentiment Agent: " + ("ON" if processor.sentiment_agent else "OFF"))
        logger.info("   - V18 Correlation Agent: " + ("ON" if processor.correlation_agent else "OFF"))
        logger.info("   - V18 Contrarian Agent: " + ("ON" if processor.contrarian_agent else "OFF"))
        logger.info("   - V18 Consensus Protocol (3-round): " + ("ON" if processor.fusion._consensus else "OFF"))
        logger.info("   - V18 State Machine: " + ("ON" if state_machine else "OFF"))
        logger.info("   - V18 Message Bus: " + ("ON" if msg_bus else "OFF"))
        logger.info("   - V18 HMM Regime: " + ("ON" if (evolution_engine._hmm is not None) else "OFF"))
        logger.info("   - V18 PPO RL Agent: " + ("ON" if (evolution_engine._ppo is not None) else "OFF"))
        logger.info("   - V18 Kill Switch (5-level): " + ("ON" if (evolution_engine._kill_switch is not None) else "OFF"))

        # ---- Wire WebSocket callbacks ----
        all_symbols = list(set(symbols_whitelist + (symbols_hg_all if HG_MONITOR_ALL else [])))

        def ws_on_update(symbol, interval, kline):
            data_store.update_realtime(symbol, interval, kline)

        def ws_on_closed(symbol, interval, kline):
            processor.on_candle_close(symbol, interval, kline)

        register_callbacks(on_closed=ws_on_closed, on_update=ws_on_update)

        # ---- Start WebSockets ----
        start_websockets(all_symbols, timeframes=["15m", "1h", "4h"])

        # ---- REST fallback ----
        if POLL_CLOSED_ENABLE:
            start_rest_fallback(symbols_whitelist, ws_on_closed)

        # ---- Startup health check ----
        ws_ok = startup_health_check(STARTUP_TIMEOUT)
        if ws_ok:
            logger.info("✅ All WebSockets healthy")
        else:
            logger.warning("⚠️ Some WebSockets not responding — REST fallback active")

        # ---- Background threads ----
        threading.Thread(
            target=lambda: _position_monitor(
                processor, tracker, decision_context,
                interval_sec=10,
                evolution_engine=evolution_engine,
            ),
            daemon=True,
            name="PositionMonitor",
        ).start()

        threading.Thread(
            target=_report_loop,
            args=(processor, tracker, meta),
            daemon=True,
            name="ReportLoop",
        ).start()

        if HEARTBEAT_ENABLED:
            threading.Thread(
                target=_heartbeat_loop,
                args=(processor, HEARTBEAT_INTERVAL),
                daemon=True,
                name="Heartbeat",
            ).start()

        # ---- Send startup notification ----
        send_message(build_startup_message(
            n_symbols=len(symbols_whitelist),
            n_hg=len(symbols_hg_all),
            paper=PAPER_TRADING,
        ))

        logger.info("=" * 60)
        logger.info("🚀 V18 SYSTEM OPERATIONAL — Press Ctrl+C to stop")
        logger.info("=" * 60)

        # ---- Main loop ----
        global _sniper_mode_active
        _sniper_mode_active = not TRAINING_MODE  # True from the start if training is disabled
        _last_evolution_tick = time.time()
        while True:
            time.sleep(30)
            gc.collect()

            # Frequently push fresh win rates to RiskAgent (lightweight)
            tracker.update_risk_agent_win_rates(risk_agent, current_balance=execution.get_stats()["balance"])

            # Auto-switch: Training Mode → Sniper Mode con transizione graduale
            # La transizione avviene tra il trade #TRAINING_TARGET_TRADES e #(TARGET + 50)
            if TRAINING_MODE and not _sniper_mode_active:
                try:
                    completed = experience_db.get_completed_trade_count()
                    _TRANSITION_TRADES = 50  # durata della transizione in trade
                    if completed >= TRAINING_TARGET_TRADES + _TRANSITION_TRADES:
                        # Transizione completata: applica i valori Sniper definitivi
                        _sniper_mode_active = True
                        processor.fusion.threshold = SNIPER_FUSION_THRESHOLD
                        _cfg.FUSION_THRESHOLD_DEFAULT = SNIPER_FUSION_THRESHOLD
                        _cfg.MIN_FUSION_SCORE = SNIPER_MIN_FUSION_SCORE
                        _cfg.MIN_AGENT_CONFIRMATIONS = SNIPER_MIN_AGENT_CONFIRMATIONS
                        _cfg.MIN_RR = SNIPER_MIN_RR
                        _cfg.NON_OPTIMAL_HOUR_PENALTY = SNIPER_NON_OPTIMAL_HOUR_PENALTY
                        _cfg.SIGNAL_COOLDOWN_BY_TF = SNIPER_SIGNAL_COOLDOWN_BY_TF
                        _cfg.MAX_OPEN_POSITIONS = SNIPER_MAX_OPEN_POSITIONS
                        logger.info(
                            f"🎓 TRAINING COMPLETATO ({completed} trade) — "
                            f"Sniper Mode ATTIVO (transizione finita)"
                        )
                        # V18: Transition state machine to SNIPER
                        try:
                            if state_machine is not None:
                                state_machine.transition("training_complete")
                                logger.info(f"🔄 StateMachine → {state_machine.current_state()}")
                        except Exception as _sm_err:
                            logger.debug(f"StateMachine training_complete error: {_sm_err}")
                        try:
                            send_message(
                                f"🎓 *V18 TRAINING COMPLETATO*\n\n"
                                f"✅ {completed} trade completati\n"
                                f"🎯 *Sniper Mode* ATTIVO — soglie finali:\n"
                                f"  • Fusion threshold: {SNIPER_FUSION_THRESHOLD}\n"
                                f"  • Min fusion score: {SNIPER_MIN_FUSION_SCORE}\n"
                                f"  • Min agents: {SNIPER_MIN_AGENT_CONFIRMATIONS}\n"
                                f"  • Min R/R: {SNIPER_MIN_RR}"
                            )
                        except Exception as _notify_err:
                            logger.error(f"Sniper Mode notification error: {_notify_err}")
                    elif completed >= TRAINING_TARGET_TRADES:
                        # In transizione graduale: interpola linearmente i parametri
                        _progress = (completed - TRAINING_TARGET_TRADES) / _TRANSITION_TRADES
                        _new_threshold = _interpolate_param(
                            TRAINING_FUSION_THRESHOLD, SNIPER_FUSION_THRESHOLD, _progress
                        )
                        _new_min_score = _interpolate_param(
                            TRAINING_MIN_FUSION_SCORE, SNIPER_MIN_FUSION_SCORE, _progress
                        )
                        _new_min_rr = _interpolate_param(
                            TRAINING_MIN_RR, SNIPER_MIN_RR, _progress
                        )
                        _new_penalty = _interpolate_param(
                            TRAINING_NON_OPTIMAL_HOUR_PENALTY, SNIPER_NON_OPTIMAL_HOUR_PENALTY, _progress
                        )
                        # MIN_AGENT_CONFIRMATIONS: usa il valore sniper quando progress > 0.5
                        _new_min_agents = (
                            SNIPER_MIN_AGENT_CONFIRMATIONS if _progress > 0.5
                            else TRAINING_MIN_AGENT_CONFIRMATIONS
                        )
                        processor.fusion.threshold = _new_threshold
                        _cfg.FUSION_THRESHOLD_DEFAULT = _new_threshold
                        _cfg.MIN_FUSION_SCORE = _new_min_score
                        _cfg.MIN_AGENT_CONFIRMATIONS = _new_min_agents
                        _cfg.MIN_RR = _new_min_rr
                        _cfg.NON_OPTIMAL_HOUR_PENALTY = _new_penalty
                        logger.info(
                            f"🔄 Transizione Training→Sniper: "
                            f"progress={_progress:.0%} | "
                            f"threshold={_new_threshold:.3f} | "
                            f"min_score={_new_min_score:.3f} | "
                            f"min_rr={_new_min_rr:.2f}"
                        )
                except Exception as _switch_err:
                    logger.error(f"Training→Sniper auto-switch error: {_switch_err}")

            # V18: Check state machine — if KILLED or SAFE_MODE, enforce restrictions
            try:
                if state_machine is not None:
                    sm_state = state_machine.current_state()
                    if sm_state == STATE_KILLED:
                        # Ensure kill switch is also activated
                        if evolution_engine._kill_switch is not None:
                            if not evolution_engine._kill_switch.is_killed():
                                logger.warning("🔴 StateMachine KILLED state active")
                    elif sm_state == STATE_SAFE_MODE:
                        # Raise fusion threshold to discourage new trades
                        if processor.fusion._threshold < 0.75:
                            processor.fusion._threshold = 0.75
                            logger.warning("🟡 StateMachine SAFE_MODE: threshold raised to 0.75")
            except Exception as _sm_err:
                logger.debug(f"StateMachine check error: {_sm_err}")

            # Evolution tick every 30 minutes (handles weight adjust, auto-tune, state save)
            if time.time() - _last_evolution_tick >= 1800:
                try:
                    evolution_engine.tick()
                except Exception as _evo_err:
                    logger.error(f"evolution_engine.tick error: {_evo_err}")
                _last_evolution_tick = time.time()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("⏹️ MANUAL SHUTDOWN (Ctrl+C)")
        logger.info("=" * 60)
        try:
            evolution_engine.shutdown()
        except Exception as e:
            logger.error(f"evolution_engine.shutdown error: {e}")
        try:
            stats = processor.execution.get_stats()
            logger.info(
                f"📊 Final stats: trades={stats['trade_count']} "
                f"wr={stats['win_rate']:.1%} pnl={stats['total_pnl']:+.4f}"
            )
            send_message("⏹️ V17 Agentic AI Trading System — STOPPED")
        except Exception as e:
            logger.error(f"Shutdown cleanup error: {e}")
        logger.info("👋 V17 terminated gracefully")
        sys.exit(0)

    except Exception as e:
        logger.critical("=" * 60)
        logger.critical(f"❌ FATAL ERROR: {e}")
        logger.critical("=" * 60)
        import traceback
        traceback.print_exc()
        try:
            send_message(f"🔴 V17 FATAL ERROR\n\n{str(e)[:300]}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
