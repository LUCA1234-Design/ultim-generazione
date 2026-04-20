import json
from collections import deque

import pandas as pd

import config.settings as settings
import data.websocket_manager as ws_manager
import services.latency_monitor as latency_monitor
import agents.confluence_agent as confluence_mod
from agents.confluence_agent import ConfluenceAgent
from agents.regime_agent import RegimeAgent
from notifications.telegram_service import build_stats_message


def test_build_stats_message_skips_non_agent_entries():
    msg = build_stats_message(
        exec_stats={"balance": 1000.0, "total_pnl": 0.0, "pnl_pct": 0.0},
        perf_summary={"win_rate": 0.5, "wins": 1, "losses": 1, "sharpe": 0.0},
        agent_report={
            "pattern": {"weight": 1.2, "win_rate": 0.6, "n_decisions": 10},
            "_demotion_history": [{"agent": "pattern", "event": "demoted"}],
        },
    )
    assert "pattern: w=1.20 wr=60.0% n=10" in msg


def test_build_stats_message_handles_empty_inputs():
    msg = build_stats_message(
        exec_stats={"balance": None, "total_pnl": None, "pnl_pct": None},
        perf_summary={"win_rate": None, "wins": None, "losses": None, "sharpe": None},
        agent_report=None,
    )
    assert "Balance: `0.00`" in msg
    assert "Total P&L: `+0.0000` (+0.00%)" in msg
    assert "Win Rate: `0.0%` (0W / 0L)" in msg
    assert "Sharpe: `0.00`" in msg
    assert "no agent stats yet (0 trades recorded)" in msg


def test_record_ws_delay_filters_bad_values():
    latency_monitor._WS_DELAY_SAMPLES = deque(maxlen=100)
    latency_monitor.record_ws_delay(-2_008_255)
    latency_monitor.record_ws_delay(50.0)
    latency_monitor.record_ws_delay(-25.0)
    report = latency_monitor.get_latency_report()
    assert report["samples_ws_delay"] == 2
    assert report["ws_delay_mean_ms"] >= 0.0


def test_ws_delay_uses_event_timestamp_for_open_candles(monkeypatch):
    captured = []
    monkeypatch.setattr(ws_manager.time, "time", lambda: 1_700_000_000.2)
    monkeypatch.setattr(latency_monitor, "record_ws_delay", lambda v: captured.append(float(v)))

    msg = json.dumps(
        {
            "data": {
                "E": 1_700_000_000_000,
                "k": {
                    "s": "BTCUSDT",
                    "i": "1m",
                    "x": False,
                    "t": 1_700_000_000_000,
                    "T": 1_700_000_060_000,  # future close time for still-open candle
                },
            }
        }
    )
    ws_manager._handle_message("ws-test", msg)
    assert captured
    assert captured[-1] == 200.0


def test_training_mode_thresholds_relaxed():
    assert settings.TRAINING_FUSION_THRESHOLD <= 0.30
    assert settings.TRAINING_MIN_FUSION_SCORE <= 0.25
    assert settings.TRAINING_MIN_RR <= 0.90


def test_confluence_training_thresholds_are_relaxed(monkeypatch):
    monkeypatch.setattr(confluence_mod, "TRAINING_MODE", True)
    agent = ConfluenceAgent()

    def _fake_compute_confluence(symbol, primary_interval, direction):
        return {
            "tf_scores": {"15m": 0.40, "1h": 0.32, "4h": 0.40, "1d": 0.40},
            "confluence": 0.40,
            "direction_agreement": 1.0,
            "agreement_mult": 1.0,
        }

    monkeypatch.setattr(agent, "compute_confluence", _fake_compute_confluence)
    df = pd.DataFrame({"close": [float(i) for i in range(60)]})
    res = agent.analyse("BTCUSDT", "1h", df, direction="long")
    assert res is not None
    assert res.metadata["agreeing_tfs"] == 3


def test_regime_training_relaxes_capitulation_penalty():
    agent = RegimeAgent()
    df = pd.DataFrame({
        "open": [1.0] * 80,
        "high": [1.0] * 80,
        "low": [1.0] * 80,
        "close": [1.0] * 80,
        "volume": [1.0] * 80,
    })
    agent._fitted_keys.add(("BTCUSDT", "1h"))
    agent.get_regime_probs = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "capitulation": 0.9,
        "ranging": 0.1,
    }
    res = agent.analyse("BTCUSDT", "1h", df)
    assert res is not None
    assert res.score >= 0.60
