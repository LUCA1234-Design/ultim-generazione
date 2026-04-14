import numpy as np
import pandas as pd

from engine.execution import ExecutionEngine
from evolution.evolution_engine import EvolutionEngine


class _DummyMeta:
    def adjust_weights(self):
        return {}


class _DummyFusion:
    _threshold = 0.5

    def update_weights(self, _):
        return None


class _DummyRisk:
    def set_win_rate(self, *_args, **_kwargs):
        return None


class _DummyTracker:
    def update_risk_agent_win_rates(self, *_args, **_kwargs):
        return None


class _DummyStrategy:
    pass


class _DummyConfluence:
    pass


class _DummyPPO:
    def act(self, _state):
        return 2


class _DummyMarginMonitor:
    def __init__(self, leverage=10.0):
        self.leverage = leverage

    def compute_margin_usage(self, _positions, _balance):
        return {"can_open_new": False, "utilisation_pct": 0.9, "status": "CRITICAL"}


def _make_engine(ppo=None):
    return EvolutionEngine(
        meta_agent=_DummyMeta(),
        fusion=_DummyFusion(),
        risk_agent=_DummyRisk(),
        strategy_agent=_DummyStrategy(),
        confluence_agent=_DummyConfluence(),
        tracker=_DummyTracker(),
        ppo_agent=ppo,
    )


def test_get_rl_size_hint_returns_default_when_ppo_missing():
    engine = _make_engine(ppo=None)
    df = pd.DataFrame({"close": np.linspace(100.0, 110.0, 20)})
    assert engine.get_rl_size_hint("BTCUSDT", "1h", df, "long") == 1.0


def test_get_rl_size_hint_uses_ppo_action_map():
    engine = _make_engine(ppo=_DummyPPO())
    df = pd.DataFrame({"close": np.linspace(100.0, 110.0, 20)})
    assert engine.get_rl_size_hint("BTCUSDT", "1h", df, "long") == 1.4


def test_open_position_blocked_by_regulatory_limits(monkeypatch):
    monkeypatch.setattr(
        "risk_institutional.regulatory_limits.check_position_limits",
        lambda *_args, **_kwargs: {"allowed": False, "violations": ["test"]},
    )
    engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)

    pos = engine.open_position(
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        entry_price=100.0,
        size=0.1,
        sl=95.0,
        tp1=110.0,
        tp2=120.0,
    )

    assert pos is None


def test_open_position_blocked_by_margin_monitor(monkeypatch):
    monkeypatch.setattr(
        "risk_institutional.regulatory_limits.check_position_limits",
        lambda *_args, **_kwargs: {"allowed": True, "violations": []},
    )
    monkeypatch.setattr("risk_institutional.margin_monitor.MarginMonitor", _DummyMarginMonitor)
    engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)

    pos = engine.open_position(
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        entry_price=100.0,
        size=0.1,
        sl=95.0,
        tp1=110.0,
        tp2=120.0,
    )

    assert pos is None
