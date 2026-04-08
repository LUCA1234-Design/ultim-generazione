"""
coordination/priority_scheduler.py — Priority Event Scheduler.

Implements a thread-safe priority queue for market events.

Priority levels (lower number = higher priority):
  0 — CRITICAL:  kill switch, risk alerts (immediate action required)
  1 — HIGH:      regime change, 4h/1h candle close
  2 — MEDIUM:    15m candle close, signal evaluation
  3 — LOW:       heartbeat, periodic tasks, background work

Events are dequeued in priority order (FIFO within same priority level).
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PriorityScheduler")

# Priority levels
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3

_PRIORITY_NAMES = {
    0: "CRITICAL",
    1: "HIGH",
    2: "MEDIUM",
    3: "LOW",
}

# Default event type → priority mapping
_EVENT_PRIORITIES: Dict[str, int] = {
    "risk.alert": PRIORITY_CRITICAL,
    "kill.activated": PRIORITY_CRITICAL,
    "kill.recovered": PRIORITY_HIGH,
    "regime.change": PRIORITY_HIGH,
    "candle.4h": PRIORITY_HIGH,
    "candle.1h": PRIORITY_HIGH,
    "candle.15m": PRIORITY_MEDIUM,
    "signal.evaluate": PRIORITY_MEDIUM,
    "drift.detected": PRIORITY_HIGH,
    "heartbeat": PRIORITY_LOW,
    "backtest.run": PRIORITY_LOW,
    "hyperopt.run": PRIORITY_LOW,
}


@dataclass(order=True)
class PrioritisedEvent:
    """Wrapper for priority queue ordering."""
    priority: int
    sequence: int           # tie-breaker: lower = earlier
    timestamp: float = field(compare=False)
    event_type: str = field(compare=False)
    payload: Any = field(compare=False)


class PriorityScheduler:
    """
    Thread-safe priority event queue.

    Usage:
        scheduler = PriorityScheduler()
        scheduler.enqueue("candle.1h", {"symbol": "BTCUSDT"})
        event = scheduler.dequeue(timeout=1.0)
    """

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence = 0
        self._lock = threading.Lock()
        self._enqueue_count = 0
        self._dequeue_count = 0

    # ------------------------------------------------------------------

    def enqueue(
        self,
        event: Any,
        priority: Optional[int] = None,
    ) -> None:
        """
        Enqueue an event.

        Parameters
        ----------
        event    : str event_type or dict with 'type' key
        priority : optional priority override (0=CRITICAL to 3=LOW)
                   If None, inferred from event type.
        """
        if isinstance(event, str):
            event_type = event
            payload = {}
        elif isinstance(event, dict):
            event_type = str(event.get("type", "unknown"))
            payload = event
        else:
            event_type = str(type(event).__name__)
            payload = event

        if priority is None:
            priority = _EVENT_PRIORITIES.get(event_type, PRIORITY_MEDIUM)

        with self._lock:
            seq = self._sequence
            self._sequence += 1
            self._enqueue_count += 1

        item = PrioritisedEvent(
            priority=priority,
            sequence=seq,
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
        )
        self._queue.put(item)

    def dequeue(self, timeout: float = 1.0) -> Optional[PrioritisedEvent]:
        """
        Dequeue the highest-priority event.

        Parameters
        ----------
        timeout : seconds to wait for an event (None = block forever)

        Returns
        -------
        PrioritisedEvent or None if timeout.
        """
        try:
            item = self._queue.get(timeout=timeout)
            with self._lock:
                self._dequeue_count += 1
            return item
        except queue.Empty:
            return None

    def peek(self) -> Optional[PrioritisedEvent]:
        """Peek at the next event without removing it (best-effort)."""
        # PriorityQueue doesn't support non-destructive peek
        # We use a workaround: get then put back
        try:
            item = self._queue.get_nowait()
            self._queue.put(item)
            return item
        except queue.Empty:
            return None

    def size(self) -> int:
        """Return approximate number of events in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "queue_size": self.size(),
                "total_enqueued": self._enqueue_count,
                "total_dequeued": self._dequeue_count,
            }
