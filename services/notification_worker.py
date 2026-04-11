"""
Notification worker for V17.

Moves Telegram messaging, chart generation, and decision persistence
off the realtime signal path.
"""
import logging
import queue
import threading
import traceback
from typing import Any, Dict

from data import data_store
from memory.experience_db import save_decision
from notifications.chart_generator import generate_signal_chart
from notifications.telegram_service import (
    build_signal_message,
    send_message,
    send_photo,
)
from config.settings import HIGH_MARGIN_ONLY, HIGH_MARGIN_MIN_RR, HIGH_MARGIN_MIN_FUSION_SCORE

logger = logging.getLogger("NotificationWorker")

_signal_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
_worker_started = False
_worker_lock = threading.Lock()


def enqueue_signal_notification(
    fusion_result,
    agent_results,
    position,
) -> bool:
    """Queue a signal notification job for async processing."""
    job = {
        "fusion_result": fusion_result,
        "agent_results": agent_results,
        "position": position,
    }
    try:
        _signal_queue.put_nowait(job)
        return True
    except queue.Full:
        logger.error("Notification queue full: dropping signal job")
        return False


def _process_signal_job(job: Dict[str, Any]) -> None:
    fusion_result = job["fusion_result"]
    agent_results = job["agent_results"]
    position = job["position"]

    # Extract R/R and Kelly from risk agent result if available
    risk_result = agent_results.get("risk")
    rr = risk_result.metadata.get("rr", 0.0) if risk_result else 0.0
    kelly_pct = (risk_result.metadata.get("kelly", 0.0) * 100) if risk_result else 0.0

    # If R/R not in risk metadata, calculate from position
    if rr <= 0 and position:
        try:
            risk = abs(position.entry_price - position.sl)
            reward = abs(position.tp1 - position.entry_price)
            rr = reward / risk if risk > 0 else 0.0
        except Exception as e:
            logger.debug(f"R/R calculation fallback error: {e}")
            rr = 0.0

    # HIGH MARGIN FILTER: skip notifications for signals with low R/R or low fusion score
    if HIGH_MARGIN_ONLY and (rr < HIGH_MARGIN_MIN_RR or fusion_result.final_score < HIGH_MARGIN_MIN_FUSION_SCORE):
        logger.info(
            f"📵 Signal skipped (low_margin): rr={rr:.2f} (min={HIGH_MARGIN_MIN_RR:.2f}) "
            f"fusion={fusion_result.final_score:.3f} (min={HIGH_MARGIN_MIN_FUSION_SCORE:.2f}) "
            f"| {fusion_result.symbol} [{fusion_result.interval}]"
        )
        # Still save the decision to DB for learning purposes
        try:
            save_decision(
                decision_id=fusion_result.decision_id,
                symbol=fusion_result.symbol,
                interval=fusion_result.interval,
                decision=fusion_result.decision,
                final_score=fusion_result.final_score,
                direction=fusion_result.direction,
                threshold=fusion_result.threshold,
                reasoning=fusion_result.reasoning,
                agent_scores=fusion_result.agent_scores,
            )
        except Exception as e:
            logger.error(f"DB save decision error: {e}")
        return

    # 1. Send text message
    try:
        msg = build_signal_message(fusion_result, agent_results, position)
        send_message(msg)
    except Exception as e:
        logger.error(f"Signal notification error: {e}")

    # 2. Send chart
    try:
        df = data_store.get_df(fusion_result.symbol, fusion_result.interval)
        if df is not None and len(df) > 20:
            chart_bytes = generate_signal_chart(
                df=df,
                symbol=fusion_result.symbol,
                interval=fusion_result.interval,
                direction=fusion_result.decision,
                entry=position.entry_price,
                sl=position.sl,
                tp1=position.tp1,
                tp2=position.tp2,
                rr=rr,
                kelly_pct=kelly_pct,
            )
            if chart_bytes:
                send_photo(
                    chart_bytes,
                    caption=f"📊 {fusion_result.symbol} [{fusion_result.interval}]",
                )
    except Exception as e:
        logger.error(f"Chart send error: {e}\n{traceback.format_exc()}")

    # 3. Save decision
    try:
        save_decision(
            decision_id=fusion_result.decision_id,
            symbol=fusion_result.symbol,
            interval=fusion_result.interval,
            decision=fusion_result.decision,
            final_score=fusion_result.final_score,
            direction=fusion_result.direction,
            threshold=fusion_result.threshold,
            reasoning=fusion_result.reasoning,
            agent_scores=fusion_result.agent_scores,
        )
    except Exception as e:
        logger.error(f"DB save decision error: {e}")


def _worker_loop() -> None:
    logger.info("📨 Notification worker started")
    while True:
        try:
            job = _signal_queue.get()
            try:
                _process_signal_job(job)
            finally:
                _signal_queue.task_done()
        except Exception as e:
            logger.error(f"Notification worker loop error: {e}\n{traceback.format_exc()}")


def start_notification_worker() -> None:
    """Start the notification worker once."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(
            target=_worker_loop,
            daemon=True,
            name="NotificationWorker",
        )
        t.start()
        _worker_started = True
