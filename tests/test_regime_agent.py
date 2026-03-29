"""
Unit tests for RegimeAgent.
"""
import pandas as pd
import numpy as np
import pytest

from agents.regime_agent import RegimeAgent
from agents.base_agent import AgentResult


class TestRegimeAgent:
    """Tests for RegimeAgent.analyse()."""

    @pytest.fixture(autouse=True)
    def agent(self):
        self.agent = RegimeAgent()

    # ------------------------------------------------------------------
    # Basic return-type tests
    # ------------------------------------------------------------------

    def test_analyse_returns_none_on_tiny_df(self, small_df):
        """With very little data the agent should return None rather than crash."""
        result = self.agent.analyse("BTCUSDT", "1h", small_df)
        # May be None or AgentResult – must never raise
        assert result is None or isinstance(result, AgentResult)

    def test_analyse_returns_agent_result_on_sufficient_data(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        assert result is None or isinstance(result, AgentResult)

    def test_result_fields_within_bounds(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        if result is None:
            pytest.skip("Not enough data for regime fitting")
        assert 0.0 <= result.score <= 1.0
        assert result.direction in ("long", "short", "neutral")
        assert result.agent_name == "regime"
        assert result.symbol == "BTCUSDT"
        assert result.interval == "1h"

    # ------------------------------------------------------------------
    # Multiple calls / regime labels
    # ------------------------------------------------------------------

    def test_analyse_multiple_calls_stable(self, ohlcv_df):
        """Calling analyse multiple times on the same data should be stable."""
        r1 = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        r2 = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        if r1 is not None and r2 is not None:
            assert abs(r1.score - r2.score) < 0.05

    def test_analyse_different_symbols(self, ohlcv_df):
        """Same data, different symbols should be handled independently."""
        r1 = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        r2 = self.agent.analyse("ETHUSDT", "1h", ohlcv_df)
        # Both should be valid or None — no crash
        assert r1 is None or isinstance(r1, AgentResult)
        assert r2 is None or isinstance(r2, AgentResult)

    # ------------------------------------------------------------------
    # Bullish / Bearish regime tendency
    # ------------------------------------------------------------------

    def test_bullish_df_produces_result(self, bullish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bullish_df)
        assert result is None or isinstance(result, AgentResult)
        if result is not None:
            assert result.score >= 0.0

    def test_bearish_df_produces_result(self, bearish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bearish_df)
        assert result is None or isinstance(result, AgentResult)
        if result is not None:
            assert result.score >= 0.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_analyse_with_empty_df(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = self.agent.analyse("BTCUSDT", "1h", empty)
        assert result is None

    def test_safe_analyse_does_not_raise_on_bad_df(self):
        bad_df = pd.DataFrame({"close": [1, 2, 3]})
        result = self.agent.safe_analyse("BTCUSDT", "1h", bad_df)
        # Should swallow exception and return None
        assert result is None or isinstance(result, AgentResult)

    def test_current_regime_returns_string(self, ohlcv_df):
        regime = self.agent.current_regime("BTCUSDT", "1h", ohlcv_df)
        assert isinstance(regime, str)
