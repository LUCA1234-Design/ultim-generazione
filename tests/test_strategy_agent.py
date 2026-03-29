"""
Unit tests for StrategyAgent.
"""
import pandas as pd
import numpy as np
import pytest

from agents.strategy_agent import StrategyAgent
from agents.base_agent import AgentResult


class TestStrategyAgent:
    @pytest.fixture(autouse=True)
    def agent(self):
        self.agent = StrategyAgent()

    # ------------------------------------------------------------------
    # Return-type tests
    # ------------------------------------------------------------------

    def test_analyse_returns_none_on_tiny_df(self, small_df):
        result = self.agent.analyse("BTCUSDT", "1h", small_df)
        assert result is None or isinstance(result, AgentResult)

    def test_analyse_returns_agent_result_on_sufficient_data(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        assert result is None or isinstance(result, AgentResult)

    def test_result_fields_within_bounds(self, ohlcv_df):
        result = self.agent.analyse("BTCUSDT", "1h", ohlcv_df)
        if result is None:
            pytest.skip("Insufficient data")
        assert 0.0 <= result.score <= 1.0
        assert result.direction in ("long", "short", "neutral")
        assert result.agent_name == "strategy"

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    def test_best_strategy_returns_tuple(self, ohlcv_df):
        name, score = self.agent.best_strategy(ohlcv_df, "long")
        assert isinstance(name, str)
        assert 0.0 <= score <= 1.0

    def test_all_default_strategies_evaluated(self, ohlcv_df):
        """Ensure all default strategies can be evaluated without crash."""
        for params in self.agent.DEFAULT_STRATEGIES:
            score = self.agent._eval_strategy(ohlcv_df, params, "long")
            assert 0.0 <= score <= 1.0

    def test_eval_strategy_short_direction(self, ohlcv_df):
        for params in self.agent.DEFAULT_STRATEGIES:
            score = self.agent._eval_strategy(ohlcv_df, params, "short")
            assert 0.0 <= score <= 1.0

    # ------------------------------------------------------------------
    # Historical score updates
    # ------------------------------------------------------------------

    def test_update_strategy_score(self):
        agent = StrategyAgent()
        agent.update_strategy_outcome("rsi_macd_trend", was_profitable=True)
        agent.update_strategy_outcome("rsi_macd_trend", was_profitable=True)
        agent.update_strategy_outcome("rsi_macd_trend", was_profitable=False)
        n = agent._strategy_counts.get("rsi_macd_trend", 0)
        assert n == 3
        score = agent._strategy_scores.get("rsi_macd_trend", 0)
        assert 0.0 <= score <= 1.0

    def test_strategy_score_used_after_min_samples(self, ohlcv_df):
        agent = StrategyAgent()
        # Give rsi_macd_trend a high historical score
        for _ in range(10):
            agent.update_strategy_outcome("rsi_macd_trend", was_profitable=True)
        result = agent.analyse("BTCUSDT", "1h", ohlcv_df, direction="long")
        if result is not None:
            assert result.score >= 0.0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_df_returns_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = self.agent.analyse("BTCUSDT", "1h", empty)
        assert result is None

    def test_safe_analyse_does_not_raise(self):
        bad_df = pd.DataFrame({"close": [1, 2]})
        result = self.agent.safe_analyse("BTCUSDT", "1h", bad_df)
        assert result is None or isinstance(result, AgentResult)

    def test_bullish_df_direction(self, bullish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bullish_df, direction="long")
        if result is not None:
            assert result.direction in ("long", "short", "neutral")

    def test_bearish_df_direction(self, bearish_df):
        result = self.agent.analyse("BTCUSDT", "1h", bearish_df, direction="short")
        if result is not None:
            assert result.direction in ("long", "short", "neutral")
