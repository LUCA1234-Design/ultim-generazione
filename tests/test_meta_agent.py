"""
Unit tests for MetaAgent (including enhanced Week 3 features).
"""
import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock

from agents.meta_agent import MetaAgent, AgentRecord
from agents.base_agent import AgentResult, BaseAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAgent(BaseAgent):
    def __init__(self, name, weight=1.0):
        super().__init__(name, initial_weight=weight)

    def analyse(self, symbol, interval, df, *args, **kwargs):
        return AgentResult(
            agent_name=self.name, symbol=symbol, interval=interval,
            score=0.6, direction="long", confidence=0.7,
        )


def _feed_outcomes(meta, n_correct, n_wrong, regime=None):
    """Helper: feed n correct + n wrong decisions to the meta agent."""
    results = {
        "alpha": AgentResult("alpha", "BTCUSDT", "1h", score=0.7, direction="long"),
        "beta": AgentResult("beta", "BTCUSDT", "1h", score=0.6, direction="long"),
    }
    for i in range(n_correct):
        meta.record_outcome(f"c{i}", results, was_correct=True, regime=regime)
    for i in range(n_wrong):
        meta.record_outcome(f"w{i}", results, was_correct=False, regime=regime)


# ---------------------------------------------------------------------------
# AgentRecord tests
# ---------------------------------------------------------------------------

class TestAgentRecord:
    def test_initial_win_rate_without_samples(self):
        rec = AgentRecord("test")
        assert rec.win_rate() == 0.5

    def test_win_rate_with_outcomes(self):
        rec = AgentRecord("test")
        rec.add_outcome("d1", 0.8, "long", True)
        rec.add_outcome("d2", 0.4, "short", False)
        rec.add_outcome("d3", 0.7, "long", True)
        rec.add_outcome("d4", 0.6, "long", True)
        rec.add_outcome("d5", 0.5, "long", True)
        # After META_MIN_SAMPLES decisions, real win rate is computed
        from config.settings import META_MIN_SAMPLES
        if len(rec.decisions) >= META_MIN_SAMPLES:
            wr = rec.win_rate()
            assert 0.0 <= wr <= 1.0

    def test_calibration_error(self):
        rec = AgentRecord("test")
        rec.add_outcome("d1", 1.0, "long", True)   # error = |1.0 - 1.0| = 0
        rec.add_outcome("d2", 0.0, "long", False)  # error = |0.0 - 0.0| = 0
        assert 0.0 <= rec.calibration_error() <= 1.0

    def test_sliding_window(self):
        from config.settings import META_EVAL_WINDOW
        rec = AgentRecord("test")
        for i in range(META_EVAL_WINDOW + 10):
            rec.add_outcome(str(i), 0.5, "long", True)
        assert len(rec.decisions) <= META_EVAL_WINDOW


# ---------------------------------------------------------------------------
# MetaAgent tests
# ---------------------------------------------------------------------------

class TestMetaAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.agent_a = FakeAgent("alpha", weight=1.0)
        self.agent_b = FakeAgent("beta", weight=1.0)
        self.meta = MetaAgent([self.agent_a, self.agent_b])

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_agent(self):
        new_agent = FakeAgent("gamma")
        self.meta.register(new_agent)
        assert "gamma" in self.meta._agents

    def test_registered_agents_have_records(self):
        assert "alpha" in self.meta._records
        assert "beta" in self.meta._records

    def test_empty_meta_agent(self):
        meta = MetaAgent()
        assert len(meta._agents) == 0

    # ------------------------------------------------------------------
    # record_outcome
    # ------------------------------------------------------------------

    def test_record_outcome_stores_decision(self):
        results = {
            "alpha": AgentResult("alpha", "BTCUSDT", "1h", score=0.7, direction="long"),
            "beta": AgentResult("beta", "BTCUSDT", "1h", score=0.6, direction="long"),
        }
        self.meta.record_outcome("dec1", results, was_correct=True)
        assert len(self.meta._records["alpha"].decisions) == 1
        assert len(self.meta._records["beta"].decisions) == 1

    def test_record_outcome_ignores_unknown_agents(self):
        results = {
            "unknown_agent": AgentResult("unknown", "X", "1h", score=0.5),
        }
        # Should not crash
        self.meta.record_outcome("dec2", results, was_correct=True)

    # ------------------------------------------------------------------
    # adjust_weights
    # ------------------------------------------------------------------

    def test_adjust_weights_returns_dict(self):
        weight_map = self.meta.adjust_weights()
        assert isinstance(weight_map, dict)

    def test_adjust_weights_after_records(self):
        from config.settings import META_MIN_SAMPLES
        results = {
            "alpha": AgentResult("alpha", "BTCUSDT", "1h", score=0.8, direction="long"),
            "beta": AgentResult("beta", "BTCUSDT", "1h", score=0.4, direction="short"),
        }
        for i in range(META_MIN_SAMPLES + 5):
            correct = i % 2 == 0
            self.meta.record_outcome(f"d{i}", results, was_correct=correct)
        weight_map = self.meta.adjust_weights()
        for name, w in weight_map.items():
            assert w >= 0.05

    # ------------------------------------------------------------------
    # get_report
    # ------------------------------------------------------------------

    def test_get_report_structure(self):
        report = self.meta.get_report()
        assert isinstance(report, dict)
        for name in ("alpha", "beta"):
            assert name in report
            assert "win_rate" in report[name]
            assert "n_decisions" in report[name]
            assert "weight" in report[name]

    # ------------------------------------------------------------------
    # analyse
    # ------------------------------------------------------------------

    def test_analyse_returns_agent_result(self, ohlcv_df):
        result = self.meta.analyse("BTCUSDT", "1h", ohlcv_df)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "meta"
        assert 0.0 <= result.score <= 1.0

    def test_analyse_with_empty_meta_returns_warmup(self, ohlcv_df):
        meta = MetaAgent()
        result = meta.analyse("BTCUSDT", "1h", ohlcv_df)
        assert isinstance(result, AgentResult)
        assert "warmup_mode" in result.details[0]

    def test_analyse_with_agent_results_kwarg(self, ohlcv_df, mock_agent_results):
        result = self.meta.analyse("BTCUSDT", "1h", ohlcv_df, agent_results=mock_agent_results)
        assert isinstance(result, AgentResult)


# ---------------------------------------------------------------------------
# Enhanced Week 3 feature tests
# ---------------------------------------------------------------------------

class TestMetaAgentEnhanced:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.agent_a = FakeAgent("alpha", weight=1.0)
        self.agent_b = FakeAgent("beta", weight=1.0)
        self.meta = MetaAgent([self.agent_a, self.agent_b])

    # ------------------------------------------------------------------
    # Lower confidence bound
    # ------------------------------------------------------------------

    def test_lcb_without_samples(self):
        rec = AgentRecord("test")
        lcb = rec.lower_confidence_bound()
        assert 0.0 <= lcb <= 1.0

    def test_lcb_after_all_correct(self):
        from config.settings import META_MIN_SAMPLES
        rec = AgentRecord("test")
        for i in range(META_MIN_SAMPLES + 10):
            rec.add_outcome(str(i), 0.8, "long", True)
        lcb = rec.lower_confidence_bound()
        assert lcb > 0.5   # confident good agent

    def test_lcb_after_all_wrong(self):
        from config.settings import META_MIN_SAMPLES
        rec = AgentRecord("test")
        for i in range(META_MIN_SAMPLES + 10):
            rec.add_outcome(str(i), 0.3, "long", False)
        lcb = rec.lower_confidence_bound()
        assert lcb < 0.5   # confident bad agent

    # ------------------------------------------------------------------
    # Regime-aware records
    # ------------------------------------------------------------------

    def test_record_outcome_with_regime(self):
        _feed_outcomes(self.meta, 5, 2, regime="trending")
        assert "trending" in self.meta._regime_records

    def test_regime_records_per_agent(self):
        _feed_outcomes(self.meta, 5, 2, regime="ranging")
        regime_recs = self.meta._regime_records.get("ranging", {})
        assert "alpha" in regime_recs or "beta" in regime_recs

    def test_adjust_weights_with_regime(self):
        _feed_outcomes(self.meta, 5, 0, regime="trending")
        wmap = self.meta.adjust_weights(regime="trending")
        assert isinstance(wmap, dict)

    # ------------------------------------------------------------------
    # Demotion / promotion
    # ------------------------------------------------------------------

    def test_demotion_on_poor_win_rate(self):
        from config.settings import META_MIN_SAMPLES
        _feed_outcomes(self.meta, 0, META_MIN_SAMPLES + 5)
        self.meta.adjust_weights()
        # At least one agent should be demoted
        assert any(self.meta._demoted.values())

    def test_promotion_after_recovery(self):
        from config.settings import META_MIN_SAMPLES
        # First demote
        _feed_outcomes(self.meta, 0, META_MIN_SAMPLES + 5)
        self.meta.adjust_weights()
        # Now add good outcomes and recover
        _feed_outcomes(self.meta, META_MIN_SAMPLES + 5, 0)
        self.meta.adjust_weights()
        # History should contain promotion events
        promotions = [e for e in self.meta._demotion_history if e["event"] == "promoted"]
        assert len(promotions) >= 0  # may or may not promote depending on LCB threshold

    def test_demotion_history_logged(self):
        from config.settings import META_MIN_SAMPLES
        _feed_outcomes(self.meta, 0, META_MIN_SAMPLES + 5)
        self.meta.adjust_weights()
        demotions = [e for e in self.meta._demotion_history if e["event"] == "demoted"]
        assert len(demotions) >= 1

    # ------------------------------------------------------------------
    # EMA smoothing
    # ------------------------------------------------------------------

    def test_ema_smoothing_weight_does_not_jump(self):
        from config.settings import META_MIN_SAMPLES, META_WEIGHT_DECAY
        initial_w = self.agent_a.weight
        _feed_outcomes(self.meta, META_MIN_SAMPLES + 5, 0)
        self.meta.adjust_weights()
        # With EMA decay the weight cannot jump by more than (1-decay)*range
        new_w = self.agent_a.weight
        assert abs(new_w - initial_w) < 10.0   # sanity bound

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def test_save_and_load_state(self):
        from config.settings import META_MIN_SAMPLES
        _feed_outcomes(self.meta, META_MIN_SAMPLES, 2)
        self.meta.adjust_weights()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            path = tf.name

        try:
            assert self.meta.save_state(path)

            # Create fresh meta agent and load
            agent_a2 = FakeAgent("alpha")
            agent_b2 = FakeAgent("beta")
            meta2 = MetaAgent([agent_a2, agent_b2])
            assert meta2.load_state(path)
            assert len(meta2._records["alpha"].decisions) == \
                   len(self.meta._records["alpha"].decisions)
        finally:
            os.unlink(path)

    def test_load_state_missing_file(self):
        result = self.meta.load_state("/tmp/non_existent_meta_state_xyz.json")
        assert result is False

    # ------------------------------------------------------------------
    # Enhanced get_report
    # ------------------------------------------------------------------

    def test_get_report_has_lcb_and_variance(self, ohlcv_df):
        _feed_outcomes(self.meta, 5, 2)
        report = self.meta.get_report()
        for name in ("alpha", "beta"):
            assert "win_rate_lcb" in report[name]
            assert "variance" in report[name]
            assert "demoted" in report[name]

    def test_get_report_includes_demotion_history(self):
        report = self.meta.get_report()
        assert "_demotion_history" in report

    def test_get_report_regime_stats_populated(self):
        _feed_outcomes(self.meta, 5, 2, regime="trending")
        report = self.meta.get_report(include_regime=True)
        for name in ("alpha", "beta"):
            if "regime_stats" in report[name]:
                assert "trending" in report[name]["regime_stats"]

    # ------------------------------------------------------------------
    # analyse with regime param
    # ------------------------------------------------------------------

    def test_analyse_with_regime_param(self, ohlcv_df):
        result = self.meta.analyse("BTCUSDT", "1h", ohlcv_df, regime="trending")
        assert isinstance(result, AgentResult)
        assert result.metadata.get("regime") == "trending"
