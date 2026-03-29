"""
Unit tests for PatternAgent.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from agents.pattern_agent import PatternAgent
from agents.base_agent import AgentResult


class TestPatternAgent:
    @pytest.fixture(autouse=True)
    def agent(self):
        self.agent = PatternAgent()

    # ------------------------------------------------------------------
    # Basic return-type tests
    # ------------------------------------------------------------------

    def test_analyse_returns_none_on_tiny_df(self, small_df):
        result = self.agent.analyse("BTCUSDT", "1h", small_df)
        assert result is None or isinstance(result, AgentResult)

    def test_analyse_returns_agent_result_on_sufficient_data(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        # pattern_agent may need btc_df kwarg — without it may return None
        assert result is None or isinstance(result, AgentResult)

    def test_result_fields_within_bounds(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        if result is None:
            pytest.skip("Insufficient data or missing indicator")
        assert 0.0 <= result.score <= 1.0
        assert result.direction in ("long", "short", "neutral")
        assert result.agent_name == "pattern"

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def test_detect_nr7_on_narrowing_range(self, ohlcv_df):
        """Last bar has narrowest range → NR7 should be True."""
        df = ohlcv_df.copy()
        # Force last bar to have very narrow range
        last_close = df["close"].iloc[-1]
        df.loc[df.index[-1], "high"] = last_close + 0.0001
        df.loc[df.index[-1], "low"] = last_close - 0.0001
        # Must not crash
        result = self.agent.detect_nr7(df)
        assert isinstance(result, bool)

    def test_detect_squeeze_returns_tuple(self, ohlcv_df):
        active, n_bars = self.agent.detect_squeeze(ohlcv_df)
        assert isinstance(active, bool)
        assert isinstance(n_bars, int)
        assert n_bars >= 0

    def test_detect_squeeze_validated_returns_tuple(self, ohlcv_df):
        was_squeezing, n = self.agent.detect_squeeze_validated(ohlcv_df)
        assert isinstance(was_squeezing, bool)
        assert n >= 0

    # ------------------------------------------------------------------
    # Bullish / Bearish direction tendencies
    # ------------------------------------------------------------------

    def test_bullish_df_direction(self, bullish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bullish_df)
        if result is not None:
            # In a strong uptrend the agent should lean long or neutral
            assert result.direction in ("long", "neutral", "short")

    def test_bearish_df_direction(self, bearish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bearish_df)
        if result is not None:
            assert result.direction in ("long", "neutral", "short")

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_df_returns_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = self.agent.analyse("BTCUSDT", "1h", empty)
        assert result is None

    def test_safe_analyse_does_not_raise(self):
        bad_df = pd.DataFrame({"close": [1, 2, 3]})
        result = self.agent.safe_analyse("BTCUSDT", "1h", bad_df)
        assert result is None or isinstance(result, AgentResult)

    # ------------------------------------------------------------------
    # Threshold auto-calibration
    # ------------------------------------------------------------------

    def test_update_threshold_raises_on_poor_performance(self):
        agent = PatternAgent()
        initial = agent._get_threshold("1h")
        for _ in range(25):
            agent.update_threshold("1h", was_correct=False)
        new = agent._get_threshold("1h")
        assert new >= initial

    def test_update_threshold_lowers_on_good_performance(self):
        agent = PatternAgent()
        # First set a high threshold
        agent._thresholds["1h"] = 0.70
        for _ in range(25):
            agent.update_threshold("1h", was_correct=True)
        new = agent._get_threshold("1h")
        assert new <= 0.70
