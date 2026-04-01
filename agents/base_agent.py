"""
Abstract base class for all V17 agents.
Every agent exposes:
  - analyse(symbol, interval, df) -> AgentResult
  - name property
  - weight property (adjusted by MetaAgent)
"""
import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    """Standardised output from any agent."""
    agent_name: str
    symbol: str
    interval: str
    score: float                          # 0.0 – 1.0
    direction: str = "neutral"            # "long" | "short" | "neutral"
    confidence: float = 0.0              # 0.0 – 1.0
    details: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"AgentResult({self.agent_name}, {self.symbol}/{self.interval}, "
            f"score={self.score:.3f}, dir={self.direction}, conf={self.confidence:.3f})"
        )


class BaseAgent(abc.ABC):
    """Abstract base class for all agents in the V17 system."""

    def __init__(self, name: str, initial_weight: float = 1.0):
        self._name = name
        self._weight = initial_weight
        self._call_count = 0
        self._error_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = max(0.01, min(value, 10.0))

    @abc.abstractmethod
    def analyse(self, symbol: str, interval: str, df, *args, **kwargs) -> Optional[AgentResult]:
        """Analyse the given DataFrame and return a scored result.

        Parameters
        ----------
        symbol : str   – e.g. "BTCUSDT"
        interval : str – e.g. "1h"
        df : pd.DataFrame – OHLCV DataFrame
        *args, **kwargs – extra arguments forwarded to subclass implementations

        Returns
        -------
        AgentResult or None if insufficient data.
        """

    def safe_analyse(self, symbol: str, interval: str, df, *args, **kwargs) -> Optional[AgentResult]:
        """Wrapper around analyse() that catches exceptions."""
        self._call_count += 1
        try:
            return self.analyse(symbol, interval, df, *args, **kwargs)
        except Exception as exc:
            self._error_count += 1
            import logging
            logging.getLogger(self._name).warning(
                f"analyse error [{symbol}/{interval}]: {exc}"
            )
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "weight": self._weight,
            "calls": self._call_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(self._call_count, 1),
        }
