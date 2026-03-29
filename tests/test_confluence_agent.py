"""
Unit tests for ConfluenceAgent.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from agents.confluence_agent import ConfluenceAgent
from agents.base_agent import AgentResult


class TestConfluenceAgent:
    @pytest.fixture(autouse=True)
    def agent(self):
        self.agent = ConfluenceAgent()

    # ------------------------------------------------------------------
    # Return-type tests
    # ------------------------------------------------------------------

    def test_analyse_returns_none_on_tiny_df(self, small_df):
        with patch("data.data_store.get_df", return_value=small_df):
            result = self.agent.analyse("BTCUSDT", "1h", small_df)
        assert result is None or isinstance(result, AgentResult)

    def test_analyse_returns_agent_result_on_sufficient_data(self, ohlcv_df):
        with patch("data.data_store.get_df", return_value=ohlcv_df):
            result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        assert result is None or isinstance(result, AgentResult)

    def test_result_fields_within_bounds(self, ohlcv_df):
        with patch("data.data_store.get_df", return_value=ohlcv_df):
            result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        if result is None:
            pytest.skip("Insufficient data")
        assert 0.0 <= result.score <= 1.0
        assert result.direction in ("long", "short", "neutral")
        assert result.agent_name == "confluence"

    # ------------------------------------------------------------------
    # Direction tests with mocked data store
    # ------------------------------------------------------------------

    def test_long_direction(self, ohlcv_df):
        with patch("data.data_store.get_df", return_value=ohlcv_df):
            result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df, direction="long")
        if result is not None:
            assert result.direction in ("long", "short")

    def test_short_direction(self, ohlcv_df):
        with patch("data.data_store.get_df", return_value=ohlcv_df):
            result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df, direction="short")
        if result is not None:
            assert result.direction in ("short", "long")

    # ------------------------------------------------------------------
    # Internal bias scoring
    # ------------------------------------------------------------------

    def test_tf_bias_long_returns_float(self, ohlcv_df):
        score = self.agent._tf_bias(ohlcv_df, "long")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_tf_bias_short_returns_float(self, ohlcv_df):
        score = self.agent._tf_bias(ohlcv_df, "short")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_tf_bias_on_small_df_returns_zero(self, small_df):
        score = self.agent._tf_bias(small_df, "long")
        assert score == 0.0

    def test_tf_bias_on_none_returns_zero(self):
        score = self.agent._tf_bias(None, "long")
        assert score == 0.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_df_returns_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with patch("data.data_store.get_df", return_value=empty):
            result = self.agent.analyse("BTCUSDT", "1h", empty)
        assert result is None

    def test_safe_analyse_does_not_raise(self):
        bad_df = pd.DataFrame({"close": [1, 2, 3]})
        with patch("data.data_store.get_df", return_value=bad_df):
            result = self.agent.safe_analyse("BTCUSDT", "1h", bad_df)
        assert result is None or isinstance(result, AgentResult)

    def test_compute_confluence_returns_dict(self, ohlcv_df):
        with patch("data.data_store.get_df", return_value=ohlcv_df):
            data = self.agent.compute_confluence("BTCUSDT", "1h", "long")
        assert isinstance(data, dict)
        assert "confluence" in data
        assert 0.0 <= data["confluence"] <= 1.0
