from types import SimpleNamespace

import numpy as np
import pytest

import main as main_mod
import evolution.evolution_engine as evolution_engine_mod
from evolution.evolution_engine import EvolutionEngine


def test_position_monitor_processes_closed_positions_outside_price_checks(monkeypatch):
    updates = []
    outcomes = []
    evo_calls = []
    tracked = []

    closed = SimpleNamespace(
        position_id="p-1",
        decision_id="d-1",
        status="training_timeout",
        pnl=5.0,
        symbol="BTCUSDT",
        strategy="trend",
        close_time=123.0,
    )

    class _DummyExec:
        def get_open_positions(self):
            return []

        def get_closed_positions(self, limit=200):
            return [closed]

        def check_position_levels(self, *_args, **_kwargs):
            return []

    class _DummyFusion:
        def adapt_threshold(self, *_args, **_kwargs):
            return None

    class _DummyMeta:
        def record_outcome(self, *_args, **_kwargs):
            return None

    processor = SimpleNamespace(
        execution=_DummyExec(),
        fusion=_DummyFusion(),
        meta=_DummyMeta(),
        get_decision_context=lambda _decision_id: {},
        clear_decision_context=lambda _decision_id: None,
    )
    tracker = SimpleNamespace(record_position=lambda pos: tracked.append(pos))
    decision_context = {
        "d-1": {
            "agent_scores": {"pattern": 0.9},
            "agent_directions": {"pattern": "long"},
            "agent_results": {"pattern": SimpleNamespace(details=["sweep"])},
            "regime": "trending",
        }
    }
    evolution_engine = SimpleNamespace(
        on_trade_close=lambda closed_pos, ctx: evo_calls.append((closed_pos, ctx))
    )

    monkeypatch.setattr(main_mod, "notify_position_closed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_mod.experience_db,
        "update_decision_outcome",
        lambda **kwargs: updates.append(kwargs),
    )
    monkeypatch.setattr(
        main_mod.experience_db,
        "save_agent_outcome",
        lambda **kwargs: outcomes.append(kwargs),
    )
    def _raise_keyboard_interrupt(_sec):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_mod.time, "sleep", _raise_keyboard_interrupt)

    with pytest.raises(KeyboardInterrupt):
        main_mod._position_monitor(
            processor=processor,
            tracker=tracker,
            decision_context=decision_context,
            interval_sec=1,
            evolution_engine=evolution_engine,
        )

    assert tracked and tracked[0] is closed
    assert updates and updates[0]["decision_id"] == "d-1"
    assert outcomes and outcomes[0]["decision_id"] == "d-1"
    assert evo_calls and evo_calls[0][0] is closed
    assert "d-1" not in decision_context


def test_evolution_engine_persists_and_restores_runtime_learning_state(tmp_path, monkeypatch):
    ppo_model_path = tmp_path / "ppo_runtime_state.npz"
    runtime_state_path = tmp_path / "evolution_runtime_state.json"
    monkeypatch.setattr(evolution_engine_mod, "_PPO_MODEL_PATH", str(ppo_model_path))
    monkeypatch.setattr(evolution_engine_mod, "_RUNTIME_STATE_PATH", str(runtime_state_path))
    monkeypatch.setattr(evolution_engine_mod.experience_db, "save_param", lambda *_a, **_k: None)
    monkeypatch.setattr(evolution_engine_mod.experience_db, "get_param", lambda *_a, **_k: None)

    class _DummyMeta:
        def adjust_weights(self):
            return {}

        def save_state(self):
            return True

        def load_state(self):
            return False

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
        pass

    class _DummyConfluence:
        pass

    class _DummyPPO:
        def __init__(self):
            self.saved_path = None
            self.loaded_path = None

        def save(self, path):
            self.saved_path = path
            with open(path, "wb") as fh:
                fh.write(b"ok")
            return True

        def load(self, path):
            self.loaded_path = path
            return True

    ppo_a = _DummyPPO()
    engine_a = EvolutionEngine(
        meta_agent=_DummyMeta(),
        fusion=_DummyFusion(),
        risk_agent=_DummyRisk(),
        strategy_agent=_DummyStrategy(),
        confluence_agent=_DummyConfluence(),
        tracker=_DummyTracker(),
        ppo_agent=ppo_a,
    )
    engine_a._closed_trades_buffer = [{"pnl": 2.0, "win": True, "symbol": "BTCUSDT"}]
    engine_a._last_hyperopt_params = {"fusion_threshold": 0.77}
    engine_a._ppo_experience_buffer = [
        {
            "symbol": "BTCUSDT",
            "trade_reward": 1.0,
            "pnl": 2.0,
            "state": np.array([0.1] * 12, dtype=float),
            "action": 4,
            "log_prob": -0.2,
            "reward": 0.2,
            "value": 0.1,
            "done": 1.0,
            "next_state": np.array([0.2] * 12, dtype=float),
        }
    ]

    engine_a._save_state()

    assert runtime_state_path.exists()
    assert ppo_model_path.exists()
    assert ppo_a.saved_path == str(ppo_model_path)

    ppo_b = _DummyPPO()
    engine_b = EvolutionEngine(
        meta_agent=_DummyMeta(),
        fusion=_DummyFusion(),
        risk_agent=_DummyRisk(),
        strategy_agent=_DummyStrategy(),
        confluence_agent=_DummyConfluence(),
        tracker=_DummyTracker(),
        ppo_agent=ppo_b,
    )
    engine_b.startup()

    assert ppo_b.loaded_path == str(ppo_model_path)
    assert engine_b._last_hyperopt_params == {"fusion_threshold": 0.77}
    assert len(engine_b._closed_trades_buffer) == 1
    assert len(engine_b._ppo_experience_buffer) == 1
    assert np.allclose(engine_b._ppo_experience_buffer[0]["state"], np.array([0.1] * 12))
