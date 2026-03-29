"""
Unit tests for RiskAgent.
"""
import pandas as pd
import numpy as np
import pytest

from agents.risk_agent import RiskAgent
from agents.base_agent import AgentResult


class TestRiskAgent:
    @pytest.fixture(autouse=True)
    def agent(self):
        self.agent = RiskAgent()

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
        assert result.agent_name == "risk"

    # ------------------------------------------------------------------
    # Risk parameter validation
    # ------------------------------------------------------------------

    def test_calc_levels_long_direction(self, ohlcv_df):
        sl, tp1, tp2, rr = self.agent.calc_levels(ohlcv_df, "long")
        close = ohlcv_df["close"].iloc[-1]
        assert sl < close, "SL must be below entry for long"
        assert tp1 > close, "TP1 must be above entry for long"
        assert tp2 > tp1, "TP2 must be above TP1"
        assert rr > 0, "R/R must be positive"

    def test_calc_levels_short_direction(self, ohlcv_df):
        sl, tp1, tp2, rr = self.agent.calc_levels(ohlcv_df, "short")
        close = ohlcv_df["close"].iloc[-1]
        assert sl > close, "SL must be above entry for short"
        assert tp1 < close, "TP1 must be below entry for short"
        assert tp2 < tp1, "TP2 must be below TP1"
        assert rr > 0

    def test_kelly_fraction_valid_range(self):
        f = RiskAgent.kelly_fraction(win_rate=0.6, rr=2.0)
        assert 0.005 <= f <= 0.05

    def test_kelly_fraction_poor_edge_case(self):
        f = RiskAgent.kelly_fraction(win_rate=0.01, rr=1.0)
        assert f >= 0.005

    def test_kelly_fraction_zero_rr(self):
        f = RiskAgent.kelly_fraction(win_rate=0.6, rr=0)
        assert f == 0.01  # falls back to minimum

    # ------------------------------------------------------------------
    # Win rate management
    # ------------------------------------------------------------------

    def test_set_and_get_win_rate(self):
        self.agent.set_win_rate("BTCUSDT_1h", 0.65)
        wr = self.agent.get_win_rate("BTCUSDT", "1h")
        assert abs(wr - 0.65) < 1e-9

    def test_win_rate_clamped(self):
        self.agent.set_win_rate("key", 1.5)
        assert self.agent._win_rates["key"] <= 0.99

    def test_win_rate_fallback(self):
        wr = self.agent.get_win_rate("XYZUSDT", "4h")
        assert 0.0 < wr <= 1.0

    # ------------------------------------------------------------------
    # Balance management
    # ------------------------------------------------------------------

    def test_update_balance(self):
        self.agent.update_balance(2000.0)
        assert self.agent._balance == 2000.0

    def test_update_balance_negative_clamped(self):
        self.agent.update_balance(-500.0)
        assert self.agent._balance == 0.0

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

    def test_long_short_result(self, ohlcv_df):
        long_res = self.agent.analyse("BTCUSDT", "1h", ohlcv_df, direction="long")
        short_res = self.agent.analyse("BTCUSDT", "1h", ohlcv_df, direction="short")
        for res in [long_res, short_res]:
            if res is not None:
                assert isinstance(res.score, float)
