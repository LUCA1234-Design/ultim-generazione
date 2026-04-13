from notifications.telegram_service import build_heartbeat_message


def test_heartbeat_message_includes_latency():
    msg = build_heartbeat_message(
        uptime_hours=1,
        uptime_minutes=2,
        processed=10,
        signals=1,
        open_positions=0,
        balance=1000.0,
        risk_blocked=False,
        skip_reasons={},
        latency_info={
            "mean_rtt_ms": 45.0,
            "p95_rtt_ms": 78.0,
            "ws_delay_mean_ms": 23.0,
        },
    )
    assert "Latency" in msg
    assert "RTT=45ms" in msg
