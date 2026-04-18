"""
Misura la latenza di rete verso i server Binance e la include nei report.
Utile per valutare se conviene migrare a AWS Tokyo (ap-northeast-1).
"""
import logging
import math
import threading
import time
from collections import deque
from typing import Dict

import requests

logger = logging.getLogger("LatencyMonitor")

_RTT_SAMPLES = deque(maxlen=100)
_WS_DELAY_SAMPLES = deque(maxlen=100)
_MAX_ACCEPTABLE_WS_DELAY_MS = 300_000
_lock = threading.Lock()
_started = False


def _percentile(values, p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def _probe_loop() -> None:
    while True:
        started = time.time()
        try:
            t0 = time.perf_counter()
            requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=5)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            with _lock:
                _RTT_SAMPLES.append(float(rtt_ms))
            report = get_latency_report()
            if report.get("p95_rtt_ms", 0.0) > 100.0:
                logger.warning(
                    "⚠️ High latency detected (p95 RTT=%.1fms). Consider colocation in AWS Tokyo (ap-northeast-1)",
                    report.get("p95_rtt_ms", 0.0),
                )
        except Exception as exc:
            logger.debug(f"Latency ping probe error: {exc}")

        elapsed = time.time() - started
        sleep_sec = max(10.0, 300.0 - elapsed)
        time.sleep(sleep_sec)


def start_latency_monitor() -> None:
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_probe_loop, daemon=True, name="LatencyProbe")
    t.start()
    logger.info("🌐 Latency monitor started")


def record_ws_delay(ws_delay_ms: float) -> None:
    try:
        delay = float(ws_delay_ms)
    except Exception:
        return
    if not math.isfinite(delay):
        return
    # Filter obviously wrong values from timestamp desync / malformed payloads.
    if abs(delay) > _MAX_ACCEPTABLE_WS_DELAY_MS:  # 5 minutes (300,000 ms)
        return
    # Small negative values are possible with clock skew; clamp to zero.
    if delay < 0:
        delay = 0.0
    with _lock:
        _WS_DELAY_SAMPLES.append(delay)


def get_latency_report() -> Dict[str, float]:
    with _lock:
        rtt = list(_RTT_SAMPLES)
        ws = list(_WS_DELAY_SAMPLES)

    mean_rtt = (sum(rtt) / len(rtt)) if rtt else 0.0
    ws_mean = (sum(ws) / len(ws)) if ws else 0.0

    return {
        "mean_rtt_ms": float(mean_rtt),
        "p95_rtt_ms": _percentile(rtt, 0.95),
        "p99_rtt_ms": _percentile(rtt, 0.99),
        "ws_delay_mean_ms": float(ws_mean),
        "samples_rtt": len(rtt),
        "samples_ws_delay": len(ws),
    }
