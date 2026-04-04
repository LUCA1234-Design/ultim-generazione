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
