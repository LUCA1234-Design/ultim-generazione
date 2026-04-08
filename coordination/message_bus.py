"""
coordination/message_bus.py — Event-Driven Publish/Subscribe Bus.

Thread-safe pub/sub system connecting all V18 modules.

Topics (predefined):
  signal.new         — new trading signal generated
  trade.open         — position opened
  trade.close        — position closed
  regime.change      — regime detection update
  risk.alert         — risk threshold breached
  drift.detected     — concept drift detected
  kill.activated     — kill switch triggered
  kill.recovered     — kill switch recovered
  hyperopt.result    — hyperparameter optimisation result
  backtest.complete  — backtest validation finished

All callbacks are called in the subscriber's own thread.
The bus itself is non-blocking (uses queues for delivery).
"""
import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("MessageBus")

# Predefined topic names
TOPIC_SIGNAL_NEW = "signal.new"
TOPIC_TRADE_OPEN = "trade.open"
TOPIC_TRADE_CLOSE = "trade.close"
TOPIC_REGIME_CHANGE = "regime.change"
TOPIC_RISK_ALERT = "risk.alert"
TOPIC_DRIFT_DETECTED = "drift.detected"
TOPIC_KILL_ACTIVATED = "kill.activated"
TOPIC_KILL_RECOVERED = "kill.recovered"
TOPIC_HYPEROPT_RESULT = "hyperopt.result"
TOPIC_BACKTEST_COMPLETE = "backtest.complete"

ALL_TOPICS = [
    TOPIC_SIGNAL_NEW, TOPIC_TRADE_OPEN, TOPIC_TRADE_CLOSE,
    TOPIC_REGIME_CHANGE, TOPIC_RISK_ALERT, TOPIC_DRIFT_DETECTED,
    TOPIC_KILL_ACTIVATED, TOPIC_KILL_RECOVERED,
    TOPIC_HYPEROPT_RESULT, TOPIC_BACKTEST_COMPLETE,
]

_MAX_QUEUE_SIZE = 1000
_DELIVERY_TIMEOUT = 5.0  # seconds


class MessageBus:
    """
    Thread-safe publish/subscribe event bus.

    Publishers call publish(topic, message) from any thread.
    Subscribers register callbacks that are invoked in the delivery thread.
    """

    def __init__(self, async_delivery: bool = True):
        """
        Parameters
        ----------
        async_delivery : if True, messages are delivered asynchronously
                         in a background thread (non-blocking publish)
        """
        self._subscribers: Dict[str, List[Callable]] = {t: [] for t in ALL_TOPICS}
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._async = async_delivery
        self._message_count = 0
        self._error_count = 0

        if async_delivery:
            self._delivery_thread = threading.Thread(
                target=self._delivery_loop,
                daemon=True,
                name="MessageBus-Delivery",
            )
            self._delivery_thread.start()

    # ------------------------------------------------------------------

    def publish(self, topic: str, message: Any) -> bool:
        """
        Publish a message to a topic.

        Parameters
        ----------
        topic   : str — topic name
        message : any serialisable payload

        Returns
        -------
        bool — True if published successfully
        """
        with self._lock:
            self._message_count += 1

        if self._async:
            try:
                self._queue.put_nowait((topic, message))
                return True
            except queue.Full:
                logger.warning(f"MessageBus queue full, dropping message on topic '{topic}'")
                return False
        else:
            self._deliver(topic, message)
            return True

    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Subscribe a callback to a topic.

        Parameters
        ----------
        topic    : str — topic name (use constants like TOPIC_SIGNAL_NEW)
        callback : callable accepting (message: Any) as first argument
        """
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)
                logger.debug(f"MessageBus: subscribed to '{topic}'")

    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        """
        Remove a subscription.

        Returns True if the callback was found and removed.
        """
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if callback in subs:
                subs.remove(callback)
                return True
        return False

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "messages_published": self._message_count,
                "delivery_errors": self._error_count,
                "queue_size": self._queue.qsize() if self._async else 0,
                "topics": {t: len(s) for t, s in self._subscribers.items()},
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delivery_loop(self) -> None:
        """Background thread: deliver queued messages to subscribers."""
        while True:
            try:
                topic, message = self._queue.get(timeout=1.0)
                self._deliver(topic, message)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"MessageBus delivery_loop error: {exc}")

    def _deliver(self, topic: str, message: Any) -> None:
        """Deliver a message to all subscribers of a topic."""
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))

        for cb in callbacks:
            try:
                cb(message)
            except Exception as exc:
                self._error_count += 1
                logger.warning(f"MessageBus callback error on topic '{topic}': {exc}")


# Global singleton instance
_global_bus: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_message_bus() -> MessageBus:
    """Return the global MessageBus singleton."""
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = MessageBus(async_delivery=True)
    return _global_bus
