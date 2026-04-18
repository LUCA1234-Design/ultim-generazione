import numpy as np
import pandas as pd

from engine.execution import ExecutionEngine
from risk_institutional.institutional_risk_manager import InstitutionalRiskManager
from risk_institutional.kill_switch import KillSwitch


def test_apply_atr_position_sizing_reduces_size_when_volatility_rises():
    irm = InstitutionalRiskManager(atr_risk_fraction=0.01, atr_stop_mult=1.5)
    current_size = 10.0
    low_atr_size = irm.apply_atr_position_sizing(
        current_size=current_size,
        balance=1000.0,
        entry_price=100.0,
        atr_value=1.0,
    )
    high_atr_size = irm.apply_atr_position_sizing(
        current_size=current_size,
        balance=1000.0,
        entry_price=100.0,
        atr_value=4.0,
    )
    assert high_atr_size < low_atr_size
    assert low_atr_size <= current_size


def test_trailing_stop_from_std_is_monotonic_in_favor():
    irm = InstitutionalRiskManager(trailing_window=10, trailing_std_mult=2.0)
    closes = np.linspace(100.0, 120.0, 30)
    long_sl = irm.trailing_stop_from_std("long", current_price=120.0, existing_sl=110.0, closes=closes)
    assert long_sl >= 110.0

    descending = np.linspace(120.0, 100.0, 30)
    short_sl = irm.trailing_stop_from_std("short", current_price=100.0, existing_sl=110.0, closes=descending)
    assert short_sl <= 110.0


def test_compute_market_state_detects_flash_crash():
    irm = InstitutionalRiskManager(flash_crash_pct=0.05, flash_crash_lookback=3)
    closes = list(np.linspace(100.0, 101.0, 20)) + [95.0]
    state = irm.compute_market_state(closes)
    assert state["flash_crash"] is True


def test_should_kill_globally_for_daily_loss_or_level5():
    irm = InstitutionalRiskManager(kill_switch=KillSwitch())
    assert irm.should_kill_globally({"triggered_levels": [2]}) is True
    assert irm.should_kill_globally({"triggered_levels": [5]}) is True
    assert irm.should_kill_globally({"triggered_levels": [2, 3]}) is True
    assert irm.should_kill_globally({"triggered_levels": [1, 5]}) is True
    assert irm.should_kill_globally({"triggered_levels": [1, 3]}) is False


def test_execution_standby_blocks_new_positions():
    engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
    engine.set_standby(True, reason="test")
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


def test_execution_dynamic_trailing_fallback_without_data(monkeypatch):
    monkeypatch.setattr(
        "risk_institutional.regulatory_limits.check_position_limits",
        lambda *_args, **_kwargs: {"allowed": True, "violations": []},
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
    assert pos is not None
    pos.tp1_hit = True
    def _raise_fallback(*_args, **_kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr("data.data_store.get_df", lambda *_args, **_kwargs: pd.DataFrame({"close": [100.0]}))
    monkeypatch.setattr(
        "risk_institutional.institutional_risk_manager.InstitutionalRiskManager.trailing_stop_from_std",
        _raise_fallback,
    )
    new_sl = engine._compute_dynamic_trailing_sl(pos, current_price=120.0)
    assert new_sl == 115.0


def test_close_all_positions_closes_all_open_positions(monkeypatch):
    monkeypatch.setattr(
        "risk_institutional.regulatory_limits.check_position_limits",
        lambda *_args, **_kwargs: {"allowed": True, "violations": []},
    )
    engine = ExecutionEngine(paper_trading=True, initial_balance=1000.0)
    p1 = engine.open_position(
        symbol="BTCUSDT",
        interval="1h",
        direction="long",
        entry_price=100.0,
        size=0.1,
        sl=95.0,
        tp1=110.0,
        tp2=120.0,
    )
    p2 = engine.open_position(
        symbol="ETHUSDT",
        interval="1h",
        direction="short",
        entry_price=200.0,
        size=0.1,
        sl=210.0,
        tp1=190.0,
        tp2=180.0,
    )
    assert p1 is not None and p2 is not None

    monkeypatch.setattr(
        "data.data_store.get_df",
        lambda *_args, **_kwargs: pd.DataFrame({"close": [101.0]}),
    )
    closed = engine.close_all_positions(reason="kill_switch")
    assert len(closed) == 2
    assert len(engine.get_open_positions()) == 0
    assert all(p.status == "kill_switch" for p in closed)
