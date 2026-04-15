from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import engine.event_processor as event_processor_mod
from agents.base_agent import AgentResult
from config.settings import RL_N_FEATURES
from engine.event_processor import EventProcessor
from evolution.evolution_engine import EvolutionEngine, _THRESHOLD_MAX


def _mk_result(name: str, score: float, direction: str = "long", **metadata):
    return AgentResult(
        agent_name=name,
        symbol="BTCUSDT",
        interval="1h",
        score=score,
        direction=direction,
        confidence=score,
        metadata=metadata,
    )


def test_event_processor_fallback_rr_and_context_pruning(monkeypatch):
    class _DummyExecution:
        def __init__(self):
            self.last_open = None

        def get_open_positions(self):
            return []

        def is_risk_blocked(self):
            return False, "", {}

        def open_position(self, **kwargs):
            self.last_open = kwargs
            return SimpleNamespace(**kwargs)

    class _DummyAgent:
        def __init__(self, result):
            self._result = result

        def safe_analyse(self, *_args, **_kwargs):
            return self._result

    class _DummyFusion:
        _threshold = 0.5

        def fuse(self, symbol, interval, agent_results, regime=None):
            return SimpleNamespace(
                decision_id="d-new",
                symbol=symbol,
                interval=interval,
                decision="long",
                final_score=0.9,
                signal_tags=[],
            )

    class _DummyMeta(_DummyAgent):
        pass

    df = pd.DataFrame(
        {
            "close": np.linspace(100.0, 120.0, 80),
            "volume": np.full(80, 1000.0),
        }
    )

    monkeypatch.setattr(event_processor_mod.data_store, "update_realtime", lambda *_a, **_k: None)
    monkeypatch.setattr(event_processor_mod.data_store, "get_df", lambda *_a, **_k: df)
    monkeypatch.setattr("indicators.technical.atr", lambda *_a, **_k: pd.Series(np.full(len(df), 2.0)))
    monkeypatch.setattr(event_processor_mod.EventProcessor, "_is_forbidden_hour", lambda self: False)
    monkeypatch.setattr(event_processor_mod.EventProcessor, "_is_signal_cooled", lambda self, *_a: True)
    monkeypatch.setattr(event_processor_mod.EventProcessor, "_correlation_check", lambda self, *_a: 0.0)
    monkeypatch.setattr(event_processor_mod, "_cfg", SimpleNamespace(
        ORARI_VIETATI_UTC=[],
        ORARI_MIGLIORI_UTC=list(range(24)),
        SIGNAL_COOLDOWN=0,
        SIGNAL_COOLDOWN_BY_TF={},
        MAX_OPEN_POSITIONS=10,
        MIN_AGENT_CONFIRMATIONS=1,
        MIN_FUSION_SCORE=0.5,
        MIN_RR=1.2,
        HIGH_MARGIN_ONLY=False,
        HIGH_MARGIN_MIN_RR=2.0,
        NON_OPTIMAL_HOUR_PENALTY=0.0,
        RL_SIZE_HINT_ENABLED=False,
    ))

    execution = _DummyExecution()
    processor = EventProcessor(
        pattern_agent=_DummyAgent(_mk_result("pattern", 0.9, "long")),
        regime_agent=_DummyAgent(_mk_result("regime", 0.9, "long", regime="trending")),
        confluence_agent=_DummyAgent(_mk_result("confluence", 0.9, "long", agreeing_tfs=3)),
        risk_agent=_DummyAgent(None),
        strategy_agent=_DummyAgent(_mk_result("strategy", 0.9, "long", strategy="s1")),
        meta_agent=_DummyMeta(_mk_result("meta", 0.9, "long")),
        fusion=_DummyFusion(),
        execution=execution,
    )
    processor._decision_contexts = {f"old-{i}": {"x": i} for i in range(500)}

    processor.on_candle_close("BTCUSDT", "1h", {"close": "120"})

    assert execution.last_open is not None
    close = float(df["close"].iloc[-1])
    assert execution.last_open["sl"] == pytest.approx(close - 1.5 * 2.0)
    assert execution.last_open["tp1"] == pytest.approx(close + 2.5 * 2.0)
    assert execution.last_open["tp2"] == pytest.approx(close + 5.0 * 2.0)
    assert len(processor._decision_contexts) == 451
    assert "old-0" not in processor._decision_contexts


def test_evolution_engine_populates_full_state_buffer_and_hyperopt(monkeypatch):
    class _DummyMeta:
        def adjust_weights(self):
            return {}

        def save_state(self):
            return None

    class _DummyFusion:
        _threshold = 0.5

    class _DummyRisk:
        def set_win_rate(self, *_args, **_kwargs):
            return None

    class _DummyTracker:
        def get_summary(self):
            return {"max_drawdown": 0.0}

        def update_risk_agent_win_rates(self, *_args, **_kwargs):
            return None

    class _DummyStrategy:
        def update_strategy_outcome(self, *_args, **_kwargs):
            return None

    class _DummyConfluence:
        pass

    class _DummyPPO:
        def select_action(self, _state):
            return 4, -0.1

        def get_value(self, _state):
            return 0.2

        def train_step(self, *_args, **_kwargs):
            return None

    class _DummyHyperopt:
        def __init__(self):
            self.reports = []

        def report_result(self, params, score):
            self.reports.append((params, score))

        def suggest_params(self):
            return {"fusion_threshold": 0.88}

    engine = EvolutionEngine(
        meta_agent=_DummyMeta(),
        fusion=_DummyFusion(),
        risk_agent=_DummyRisk(),
        strategy_agent=_DummyStrategy(),
        confluence_agent=_DummyConfluence(),
        tracker=_DummyTracker(),
        ppo_agent=_DummyPPO(),
    )
    engine._hyperopt = _DummyHyperopt()
    engine._last_hyperopt_params = {"fusion_threshold": 0.45}

    closed = SimpleNamespace(
        decision_id="d-evo",
        strategy="trend",
        pnl=10.0,
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        open_time=0.0,
        close_time=3600.0,
    )
    ctx = {
        "fusion_score": 0.8,
        "regime": "trending",
        "agent_directions": {"pattern": "long", "meta": "short"},
        "agent_scores": {"pattern": 0.9, "meta": 0.7},
        "agent_results": {},
    }
    engine.on_trade_close(closed, ctx)

    last_trade = engine._closed_trades_buffer[-1]
    assert last_trade["symbol"] == "BTCUSDT"
    assert last_trade["interval"] == "1h"
    assert last_trade["direction"] == "long"

    state = engine._ppo_experience_buffer[-1]["state"]
    assert len(state) == RL_N_FEATURES
    assert state[3] == pytest.approx(0.8)
    assert state[4] == 1.0
    assert state[7] == 1.0
    assert state[8] == pytest.approx(0.5)
    assert state[9] == pytest.approx(0.8)
    assert state[10] == pytest.approx(3600.0 / 86400.0)
    assert state[11] == 1.0

    monkeypatch.setattr(
        "evolution.evolution_engine.experience_db.get_recent_decisions",
        lambda limit=50: [{"outcome": "tp1", "pnl": 1.0}] * 20,
    )
    monkeypatch.setattr("evolution.evolution_engine.experience_db.save_param", lambda *_a, **_k: None)
    engine._auto_tune_params()

    assert engine._hyperopt.reports
    assert engine._last_hyperopt_params == {"fusion_threshold": 0.88}
    assert engine._fusion._threshold == pytest.approx(min(0.88, _THRESHOLD_MAX))
