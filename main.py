"""
V17 Agentic AI Trading System — Main Orchestrator
Transforms V16 "Cecchino Istituzionale" into a multi-agent adaptive system.
"""
import gc
import logging
import sys
import threading
import time
from typing import Dict, Any

# ---- Config ----
from config.settings import (
    PAPER_TRADING, ACCOUNT_BALANCE, HG_ENABLED, HG_MONITOR_ALL,
    HG_MIN_QUOTE_VOL, SYMBOLS_LIMIT, TELEGRAM_TEST_ON_START,
    STARTUP_TIMEOUT, POLL_CLOSED_ENABLE, DB_PATH,
)

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

# ---- Engine ----
from engine.decision_fusion import DecisionFusion
from engine.execution import ExecutionEngine
from engine.event_processor import EventProcessor

# ---- Memory ----
from memory import experience_db
from memory.performance_tracker import PerformanceTracker

# ---- Services ----
from services.notification_worker import (
    start_notification_worker,
    enqueue_signal_notification,
)

# ---- Notifications ----
from notifications.telegram_service import (
    send_message,
    test_connection,
    build_startup_message,
    build_stats_message,
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

def preload_historical(symbols, label: str = "") -> None:
    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        for interval in ["15m", "1h", "4h"]:
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
    """Instantiate and wire all V17 components."""
    logger.info("🔧 Building V17 agent system...")

    pattern = PatternAgent()
    regime = RegimeAgent()
    confluence = ConfluenceAgent()
    risk = RiskAgent()
    strategy = StrategyAgent()
    meta = MetaAgent(agents=[pattern, regime, confluence, risk, strategy])

    fusion = DecisionFusion()
    execution = ExecutionEngine(paper_trading=PAPER_TRADING, initial_balance=ACCOUNT_BALANCE)
    tracker = PerformanceTracker()

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
            }
        except Exception as e:
            logger.error(f"Decision context cache error: {e}")

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
    )

    logger.info("✅ V17 agent system ready")
    return processor, meta, tracker, execution, risk, decision_context


# ---------------------------------------------------------------------------
# Position monitoring thread
# ---------------------------------------------------------------------------

def _position_monitor(
    processor: EventProcessor,
    tracker: PerformanceTracker,
    decision_context: Dict[str, Dict[str, Any]],
    interval_sec: int = 10,
) -> None:
    """Periodically update SL/TP levels for open positions using latest prices."""
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

                        for agent_name, score in agent_scores.items():
                            experience_db.save_agent_outcome(
                                decision_id=closed.decision_id,
                                agent_name=agent_name,
                                score=float(score),
                                direction=str(agent_directions.get(agent_name, "")),
                                correct=correct,
                            )
                    except Exception as e:
                        logger.error(f"save_agent_outcome error: {e}")

                    # Update StrategyAgent with trade outcome (Fix 4)
                    try:
                        strategy_name = closed.strategy
                        if strategy_name:
                            processor.strategy.update_strategy_outcome(
                                strategy_name,
                                was_profitable=(closed.pnl or 0) > 0,
                            )
                    except Exception as e:
                        logger.error(f"update_strategy_outcome error: {e}")

                    # Record outcome in MetaAgent for weight adjustment (Fix 8)
                    try:
                        ctx = decision_context.get(closed.decision_id, {})
                        stored_agent_results = ctx.get("agent_results", {})
                        if stored_agent_results and hasattr(processor.meta, "record_outcome"):
                            was_correct = (closed.pnl or 0.0) > 0
                            processor.meta.record_outcome(
                                closed.decision_id,
                                stored_agent_results,
                                was_correct,
                            )
                    except Exception as e:
                        logger.error(f"meta.record_outcome error: {e}")

                    # Clean runtime context
                    try:
                        if closed.decision_id in decision_context:
                            decision_context.pop(closed.decision_id, None)
                    except Exception as e:
                        logger.debug(f"decision_context cleanup error: {e}")

        except Exception as e:
            logger.debug(f"position_monitor error: {e}")

        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Periodic reporting thread
# ---------------------------------------------------------------------------

def _report_loop(processor: EventProcessor, tracker: PerformanceTracker,
                  meta: MetaAgent, interval_sec: int = 3600) -> None:
    """Send periodic performance reports via Telegram."""
    time.sleep(60)  # Give system time to start
    while True:
        try:
            exec_stats = processor.execution.get_stats()
            perf_summary = tracker.get_summary()
            agent_report = meta.get_report()
            msg = build_stats_message(exec_stats, perf_summary, agent_report)
            send_message(msg)
            logger.info("📊 Periodic report sent")
        except Exception as e:
            logger.error(f"_report_loop error: {e}")
        time.sleep(interval_sec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("🤖 V17 AGENTIC AI TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info("🛡️ ACTIVE MODULES:")
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

        # ---- Load symbol universes ----
        load_universes()

        if not symbols_whitelist:
            raise ValueError("❌ No symbols loaded for scanning!")

        # ---- Preload historical data ----
        logger.info(f"📥 Preloading history for {len(symbols_whitelist)} symbols (main list)...")
        preload_historical(symbols_whitelist, "MAIN")

        if HG_ENABLED and HG_MONITOR_ALL and symbols_hg_all:
            # Only preload HG symbols not already in whitelist
            hg_extra = [s for s in symbols_hg_all if s not in set(symbols_whitelist)]
            if hg_extra:
                logger.info(f"📥 Preloading history for {len(hg_extra)} HG-only symbols...")
                preload_historical(hg_extra, "HG")

        # ---- Build V17 system ----
        processor, meta, tracker, execution, risk_agent, decision_context = build_system()

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
            target=_position_monitor,
            args=(processor, tracker, decision_context),
            daemon=True,
            name="PositionMonitor",
        ).start()

        threading.Thread(
            target=_report_loop,
            args=(processor, tracker, meta),
            daemon=True,
            name="ReportLoop",
        ).start()

        # ---- Send startup notification ----
        send_message(build_startup_message(
            n_symbols=len(symbols_whitelist),
            n_hg=len(symbols_hg_all),
            paper=PAPER_TRADING,
        ))

        logger.info("=" * 60)
        logger.info("🚀 V17 SYSTEM OPERATIONAL — Press Ctrl+C to stop")
        logger.info("=" * 60)

        # ---- Main loop ----
        _last_weight_adjust = time.time()
        while True:
            time.sleep(30)
            gc.collect()

            # Periodically update risk agent win rates from tracker
            tracker.update_risk_agent_win_rates(risk_agent, current_balance=execution.get_stats()["balance"])

            # Adjust agent weights every 30 minutes
            if time.time() - _last_weight_adjust >= 1800:
                weight_map = meta.adjust_weights()
                processor.fusion.update_weights(weight_map)
                logger.info(f"📐 Agent weights adjusted: {weight_map}")
                _last_weight_adjust = time.time()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("⏹️ MANUAL SHUTDOWN (Ctrl+C)")
        logger.info("=" * 60)
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
