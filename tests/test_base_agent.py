"""
Unit tests for BaseAgent and AgentResult.
"""
import time
import pytest
from agents.base_agent import BaseAgent, AgentResult


# ---------------------------------------------------------------------------
# Concrete stub for testing abstract BaseAgent
# ---------------------------------------------------------------------------

class StubAgent(BaseAgent):
    """Minimal concrete agent used only in tests."""

    def __init__(self, name="stub", weight=1.0, return_value=None):
        super().__init__(name, initial_weight=weight)
        self._return_value = return_value

    def analyse(self, symbol, interval, df, *args, **kwargs):
        return self._return_value


class RaisingAgent(BaseAgent):
    """Agent whose analyse() always raises."""

    def analyse(self, symbol, interval, df, *args, **kwargs):
        raise ValueError("simulated error")


# ---------------------------------------------------------------------------
# AgentResult tests
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(agent_name="x", symbol="BTCUSDT", interval="1h", score=0.5)
        assert r.direction == "neutral"
        assert r.confidence == 0.0
        assert r.details == []
        assert r.metadata == {}
        assert r.timestamp <= time.time()

    def test_repr_contains_key_fields(self):
        r = AgentResult(
            agent_name="pattern", symbol="ETHUSDT", interval="15m",
            score=0.75, direction="long", confidence=0.8,
        )
        text = repr(r)
        assert "pattern" in text
        assert "ETHUSDT" in text
        assert "15m" in text
        assert "long" in text

    def test_score_stored_correctly(self):
        r = AgentResult(agent_name="a", symbol="X", interval="1h", score=0.123)
        assert abs(r.score - 0.123) < 1e-9

    def test_metadata_and_details(self):
        r = AgentResult(
            agent_name="a", symbol="X", interval="1h", score=0.5,
            details=["d1", "d2"], metadata={"key": 42},
        )
        assert r.details == ["d1", "d2"]
        assert r.metadata["key"] == 42


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------

class TestBaseAgent:
    def test_name_property(self):
        agent = StubAgent("my_agent")
        assert agent.name == "my_agent"

    def test_weight_property_default(self):
        agent = StubAgent(weight=1.5)
        assert agent.weight == 1.5

    def test_weight_setter_clamps_low(self):
        agent = StubAgent()
        agent.weight = -999
        assert agent.weight == 0.01

    def test_weight_setter_clamps_high(self):
        agent = StubAgent()
        agent.weight = 999
        assert agent.weight == 10.0

    def test_weight_setter_valid(self):
        agent = StubAgent()
        agent.weight = 2.5
        assert agent.weight == 2.5

    def test_analyse_called(self):
        expected = AgentResult(
            agent_name="stub", symbol="BTCUSDT", interval="1h", score=0.6
        )
        agent = StubAgent(return_value=expected)
        result = agent.analyse("BTCUSDT", "1h", None)
        assert result is expected

    def test_safe_analyse_returns_none_on_exception(self):
        agent = RaisingAgent("raiser")
        result = agent.safe_analyse("BTCUSDT", "1h", None)
        assert result is None

    def test_safe_analyse_increments_error_count(self):
        agent = RaisingAgent("raiser")
        agent.safe_analyse("BTCUSDT", "1h", None)
        agent.safe_analyse("BTCUSDT", "1h", None)
        stats = agent.get_stats()
        assert stats["errors"] == 2
        assert stats["calls"] == 2

    def test_get_stats_structure(self):
        agent = StubAgent("s", weight=0.5)
        stats = agent.get_stats()
        assert "name" in stats
        assert "weight" in stats
        assert "calls" in stats
        assert "errors" in stats
        assert "error_rate" in stats

    def test_error_rate_is_zero_on_success(self):
        result = AgentResult(agent_name="stub", symbol="X", interval="1h", score=0.5)
        agent = StubAgent(return_value=result)
        agent.safe_analyse("X", "1h", None)
        agent.safe_analyse("X", "1h", None)
        assert agent.get_stats()["error_rate"] == 0.0
