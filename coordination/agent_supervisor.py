"""
coordination/agent_supervisor.py — Agent Health Supervisor.

Monitors the health of all agents in real-time:
  - Tracks error rate, latency, and output quality
  - Auto-disables agents with error_rate > 30% over recent window
  - Auto-restarts agents after a cooldown period
  - Provides a health score for each agent

Integration: called by EventProcessor after each agent.safe_analyse() call,
and by EvolutionEngine.tick() for periodic health checks.
"""
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("AgentSupervisor")

_WINDOW_SIZE = 50           # rolling window for health metrics
_ERROR_RATE_THRESHOLD = 0.30  # 30% error rate triggers disable
_LATENCY_WARN_MS = 2000     # warn if agent takes > 2s
_RECOVERY_COOLDOWN = 300    # seconds before re-enabling a disabled agent
_MIN_SAMPLES = 10           # minimum calls before applying thresholds


class AgentHealthRecord:
    """Health tracking record for a single agent."""

    def __init__(self, name: str):
        self.name = name
        self._latencies: deque = deque(maxlen=_WINDOW_SIZE)
        self._outcomes: deque = deque(maxlen=_WINDOW_SIZE)  # True=success, False=error
        self._enabled = True
        self._disabled_at: Optional[float] = None
        self._disable_count = 0
        self._total_calls = 0
        self._total_errors = 0

    def record(self, latency_ms: float, error: bool) -> None:
        self._latencies.append(latency_ms)
        self._outcomes.append(not error)
        self._total_calls += 1
        if error:
            self._total_errors += 1

    @property
    def error_rate(self) -> float:
        if len(self._outcomes) < 2:
            return 0.0
        return float(1 - np.mean(list(self._outcomes)))

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(list(self._latencies)))

    @property
    def health_score(self) -> float:
        """Composite health score [0, 1]: 1 = perfect, 0 = completely broken."""
        if len(self._outcomes) < 3:
            return 1.0  # not enough data → assume healthy
        err_penalty = self.error_rate  # [0,1]
        lat_penalty = min(self.avg_latency_ms / (_LATENCY_WARN_MS * 2), 0.5)
        return float(max(0.0, 1.0 - err_penalty * 0.7 - lat_penalty * 0.3))

    def disable(self) -> None:
        self._enabled = False
        self._disabled_at = time.time()
        self._disable_count += 1
        logger.warning(f"AgentSupervisor: agent '{self.name}' DISABLED "
                       f"(error_rate={self.error_rate:.1%})")

    def try_recover(self, cooldown: float = _RECOVERY_COOLDOWN) -> bool:
        if self._enabled:
            return True
        if self._disabled_at and (time.time() - self._disabled_at) >= cooldown:
            self._enabled = True
            self._disabled_at = None
            # Reset error window on recovery
            self._outcomes.clear()
            self._latencies.clear()
            logger.info(f"AgentSupervisor: agent '{self.name}' RECOVERED")
            return True
        return False

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "enabled": self._enabled,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "health_score": self.health_score,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "disable_count": self._disable_count,
            "cooldown_remaining": (
                max(0.0, _RECOVERY_COOLDOWN - (time.time() - self._disabled_at))
                if self._disabled_at else 0.0
            ),
        }


class AgentSupervisor:
    """
    Watchdog supervisor for all agents in the V18 system.

    Usage:
        supervisor = AgentSupervisor()
        supervisor.report_health("pattern", latency_ms=120, error=False)
        if supervisor.is_agent_enabled("pattern"):
            ...
    """

    def __init__(self,
                 error_rate_threshold: float = _ERROR_RATE_THRESHOLD,
                 recovery_cooldown: float = _RECOVERY_COOLDOWN):
        self.error_rate_threshold = error_rate_threshold
        self.recovery_cooldown = recovery_cooldown
        self._records: Dict[str, AgentHealthRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def report_health(
        self,
        agent_name: str,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        """
        Record a health observation for an agent.

        Parameters
        ----------
        agent_name  : str — name of the agent
        latency_ms  : float — time taken for the agent call in milliseconds
        error       : bool — True if the agent call resulted in an error
        """
        with self._lock:
            if agent_name not in self._records:
                self._records[agent_name] = AgentHealthRecord(agent_name)

            record = self._records[agent_name]
            record.record(latency_ms, error)

            # Check if should auto-disable
            if (
                record._enabled
                and len(record._outcomes) >= _MIN_SAMPLES
                and record.error_rate > self.error_rate_threshold
            ):
                record.disable()

            # Log latency warnings
            if latency_ms > _LATENCY_WARN_MS:
                logger.warning(
                    f"AgentSupervisor: '{agent_name}' slow response {latency_ms:.0f}ms"
                )

    def get_agent_health(self) -> Dict[str, Dict]:
        """Return health status for all agents."""
        with self._lock:
            return {name: r.get_status() for name, r in self._records.items()}

    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is currently enabled."""
        with self._lock:
            record = self._records.get(agent_name)
            if record is None:
                return True  # unknown agents are assumed enabled
            return record._enabled

    def disable_agent(self, agent_name: str, reason: str = "manual") -> None:
        """Manually disable an agent."""
        with self._lock:
            if agent_name not in self._records:
                self._records[agent_name] = AgentHealthRecord(agent_name)
            self._records[agent_name].disable()

    def enable_agent(self, agent_name: str) -> None:
        """Manually enable/re-enable an agent."""
        with self._lock:
            if agent_name not in self._records:
                self._records[agent_name] = AgentHealthRecord(agent_name)
            record = self._records[agent_name]
            record._enabled = True
            record._disabled_at = None
            record._outcomes.clear()
            record._latencies.clear()
            logger.info(f"AgentSupervisor: '{agent_name}' manually ENABLED")

    def attempt_recovery(self) -> List[str]:
        """
        Try to recover all disabled agents with expired cooldown.

        Returns list of agent names that were re-enabled.
        """
        recovered = []
        with self._lock:
            for name, record in self._records.items():
                if not record._enabled:
                    if record.try_recover(self.recovery_cooldown):
                        recovered.append(name)
        return recovered

    def get_enabled_agents(self) -> List[str]:
        """Return list of currently enabled agent names."""
        with self._lock:
            return [name for name, r in self._records.items() if r._enabled]

    def get_summary(self) -> Dict:
        """Return system-level health summary."""
        with self._lock:
            all_scores = [r.health_score for r in self._records.values()]
            disabled = [name for name, r in self._records.items() if not r._enabled]
        return {
            "overall_health": float(np.mean(all_scores)) if all_scores else 1.0,
            "n_agents": len(self._records),
            "n_disabled": len(disabled),
            "disabled_agents": disabled,
        }
