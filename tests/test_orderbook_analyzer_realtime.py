import pandas as pd

from quant import orderbook_analyzer as oba


def _df():
    rows = 50
    base = pd.Series(range(1, rows + 1), dtype=float)
    return pd.DataFrame({
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.2,
        "volume": base * 10,
    })


def test_realtime_orderbook_signal_fallback(monkeypatch):
    monkeypatch.setattr(oba, "_REAL_ORDERBOOK_AVAILABLE", False)
    out = oba.get_realtime_orderbook_signal("BTCUSDT", df=_df())
    assert out["using_real_data"] is False
    assert "real_imbalance" in out


def test_realtime_orderbook_signal_real_data(monkeypatch):
    monkeypatch.setattr(oba, "_REAL_ORDERBOOK_AVAILABLE", True)
    monkeypatch.setattr(
        oba,
        "get_orderbook_snapshot",
        lambda _sym: {
            "last_update_ts": __import__("time").time(),
            "bid_ask_imbalance": 0.8,
            "cumulative_buy_volume": 150.0,
            "cumulative_sell_volume": 50.0,
        },
    )
    out = oba.get_realtime_orderbook_signal("BTCUSDT", df=_df())
    assert out["using_real_data"] is True
    assert out["real_imbalance"] == 0.8
    assert out["direction"] == "long"
