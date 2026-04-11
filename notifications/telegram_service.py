"""
Enhanced Telegram service for V17.
Sends rich messages with agent reasoning, regime probabilities, and confidence.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_RATE_LIMIT, HIGH_MARGIN_MIN_RR, HIGH_MARGIN_BADGE
from agents.base_agent import AgentResult
from engine.decision_fusion import FusionResult
from engine.execution import Position

logger = logging.getLogger("TelegramService")

_last_message_time: Dict[str, float] = {}
_send_lock = __import__("threading").Lock()  # prevents concurrent sends racing the rate limiter


# ---------------------------------------------------------------------------
# Core send helpers
# ---------------------------------------------------------------------------

def send_message(text: str, parse_mode: str = "Markdown",
                 chat_id: str = TELEGRAM_CHAT_ID) -> Optional[dict]:
    """Send a text message with rate limiting (thread-safe)."""
    with _send_lock:
        now = time.time()
        last = _last_message_time.get(chat_id, 0)
        if now - last < TELEGRAM_RATE_LIMIT:
            time.sleep(TELEGRAM_RATE_LIMIT - (now - last))
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
            resp = requests.post(url, json=payload, timeout=10)
            _last_message_time[chat_id] = time.time()
            result = resp.json()
            if not result.get("ok"):
                logger.warning(f"Telegram send failed: {result.get('description')}")
            return result
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return None


def send_photo(image_bytes: bytes, caption: str = "",
               chat_id: str = TELEGRAM_CHAT_ID) -> Optional[dict]:
    """Send a photo/chart."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes, "image/png")}
        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}
        resp = requests.post(url, files=files, data=data, timeout=20)
        return resp.json()
    except Exception as e:
        logger.error(f"Telegram photo error: {e}")
        return None


def test_connection() -> bool:
    """Test Telegram bot connection."""
    result = send_message("🔧 V17 Agentic AI Trading System — connection test")
    ok = result is not None and result.get("ok", False)
    if ok:
        logger.info("✅ Telegram connection OK")
    else:
        logger.warning("⚠️ Telegram connection failed")
    return ok


# ---------------------------------------------------------------------------
# Signal message builders
# ---------------------------------------------------------------------------

def build_signal_message(
    fusion: FusionResult,
    agent_results: Dict[str, AgentResult],
    position: Position,
) -> str:
    """Build a rich signal notification message."""
    import datetime
    direction = fusion.decision
    symbol = fusion.symbol
    interval = fusion.interval
    score = fusion.final_score

    dir_emoji = "📈" if direction == "long" else "📉"
    hour = datetime.datetime.now(datetime.timezone.utc).hour

    from config.settings import ORARI_MIGLIORI_UTC
    time_quality = "🟢" if hour in ORARI_MIGLIORI_UTC else "🟡"

    # Extract R/R for title line
    risk_result = agent_results.get("risk")
    rr = risk_result.metadata.get("rr", 0) if risk_result else 0

    # Calculate % gain for TP1 and TP2
    entry = position.entry_price
    tp1 = position.tp1
    tp2 = position.tp2
    if direction == "long":
        pct_tp1 = ((tp1 - entry) / entry) * 100 if entry > 0 else 0.0
        pct_tp2 = ((tp2 - entry) / entry) * 100 if entry > 0 else 0.0
    else:
        pct_tp1 = ((entry - tp1) / entry) * 100 if entry > 0 else 0.0
        pct_tp2 = ((entry - tp2) / entry) * 100 if entry > 0 else 0.0

    is_high_margin = rr >= HIGH_MARGIN_MIN_RR

    header_line = f"{dir_emoji} *{symbol}* [{interval}] — *{direction.upper()}* (R/R {rr:.1f}x)"
    if is_high_margin:
        header_line = f"{HIGH_MARGIN_BADGE}\n{header_line}"

    lines = [
        header_line,
        "",
        f"💰 Entry: `{entry:.4f}`",
        f"🛑 SL: `{position.sl:.4f}`",
        f"🎯 TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`",
        "",
        f"📊 Fusion Score: `{score:.3f}`",
        f"{'📄 PAPER TRADE' if position.paper else '⚡ LIVE TRADE'}",
        "",
    ]

    # MARGINE DI GUADAGNO section
    margin_lines = [
        "💰 *MARGINE DI GUADAGNO*",
        f"  TP1: +{pct_tp1:.2f}%",
        f"  TP2: +{pct_tp2:.2f}%",
        f"  R/R: {rr:.1f}x",
    ]
    lines.extend(margin_lines)
    lines.append("")

    # Signal type badges
    pattern_result = agent_results.get("pattern")
    badges: List[str] = []
    if pattern_result and pattern_result.details:
        details_joined = " ".join(pattern_result.details)
        if "squeeze_breakout" in details_joined:
            badges.append("💥 SQUEEZE BREAKOUT")
        if "rsi_div_bullish" in details_joined or "rsi_div_bearish" in details_joined:
            badges.append("📊 RSI DIVERGENCE")
        if "RS_leader" in details_joined:
            badges.append("💎 RS LEADER")
        if "breakout_long" in details_joined or "breakout_short" in details_joined:
            badges.append("🚀 BREAKOUT")
        if "NR7" in details_joined:
            badges.append("📐 NR7")
        if "hammer_bullish" in details_joined:
            badges.append("🕯️ hammer_bullish")
        if "shooting_star_bearish" in details_joined:
            badges.append("🌠 shooting_star_bearish")
    if badges:
        lines.append(f"🏷️ {' | '.join(badges)}")

    # Agent agreement count
    fusion_direction = fusion.direction or "neutral"
    total_agents = len(agent_results)
    agreeing = sum(
        1 for r in agent_results.values()
        if r is not None and r.direction == fusion_direction
    )
    lines.append(f"🤝 Agents: {agreeing}/{total_agents} agree on {fusion_direction}")
    lines.append("")

    # Regime section
    regime_result = agent_results.get("regime")
    if regime_result:
        regime = regime_result.metadata.get("regime", "unknown")
        probs = regime_result.metadata.get("regime_probs", {})
        prob_str = " | ".join(f"{k}={v:.0%}" for k, v in probs.items())
        lines.append(f"🌡️ Regime: *{regime}* ({prob_str})")

    # Pattern section
    if pattern_result and pattern_result.details:
        lines.append(f"🧩 Patterns: {', '.join(pattern_result.details[:4])}")
        lines.append(f"   RSI={pattern_result.metadata.get('rsi', 0):.1f} "
                     f"ADX={pattern_result.metadata.get('adx', 0):.1f}")

    # Confluence section
    conf_result = agent_results.get("confluence")
    if conf_result:
        tf_scores = conf_result.metadata.get("tf_scores", {})
        tf_str = " | ".join(f"{tf}={v:.2f}" for tf, v in tf_scores.items())
        lines.append(f"🔗 Confluence: {tf_str}")

    # Risk section
    if risk_result:
        meta = risk_result.metadata
        kelly = meta.get("kelly", 0)
        win_rate = meta.get("win_rate", 0)
        lines.append(f"⚖️ R/R: `{rr:.2f}x` | Kelly: `{kelly*100:.1f}%` | WR: `{win_rate:.0%}`")

    # Strategy section
    strategy_result = agent_results.get("strategy")
    if strategy_result:
        strat = strategy_result.metadata.get("strategy", "")
        lines.append(f"🎲 Strategy: `{strat}`")

    lines.append(f"\n{time_quality} UTC {hour:02d}:xx | ID: `{fusion.decision_id}`")
    return "\n".join(lines)


def build_heartbeat_message(uptime_hours: int, uptime_minutes: int,
                            processed: int, signals: int,
                            open_positions: int, balance: float,
                            risk_blocked: bool,
                            skip_reasons: dict,
                            fusion_threshold: float = 0.0,
                            last_signal_info: str = "",
                            training_status: str = "") -> str:
    """Build the periodic heartbeat message string."""
    skip_sorted = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)
    skip_lines = "\n".join(
        f"  • {reason}: {count}" for reason, count in skip_sorted[:5] if count > 0
    )
    if not skip_lines:
        skip_lines = "  • Nessuno"

    lines = [
        f"🫀 *V17 HEARTBEAT*\n",
        f"⏱ Uptime: {uptime_hours}h {uptime_minutes}m",
        f"📊 Candele processate: {processed:,}",
        f"📈 Segnali generati: {signals}",
        f"📂 Posizioni aperte: {open_positions}",
        f"💰 Balance: {balance:.2f} USDT",
        f"📏 Fusion threshold: {fusion_threshold:.3f}",
        f"{'🔴 RISK BLOCKED' if risk_blocked else '🟢 Risk OK'}",
        "",
        f"🚫 *Top motivi skip:*\n{skip_lines}",
    ]
    if training_status:
        lines.append(f"{training_status}")
    if last_signal_info:
        lines.append(f"📡 Ultimo segnale: {last_signal_info}")
    lines.append("")
    lines.append("🔋 Status: *ATTIVO*")
    return "\n".join(lines)


def build_startup_message(n_symbols: int, n_hg: int, paper: bool) -> str:
    mode_str = "📄 PAPER TRADING" if paper else "⚡ LIVE TRADING"
    return (
        f"🚀 *V17 Agentic AI Trading System — STARTED*\n\n"
        f"🤖 Multi-Agent System: ONLINE\n"
        f"{'📄' if paper else '⚡'} Mode: *{mode_str}*\n\n"
        f"✅ Divergenze: {n_symbols} simboli\n"
        f"💎 Hidden Gems: {n_hg} simboli\n\n"
        f"🧠 Agents: Regime | Pattern | Confluence | Risk | Strategy | Meta\n"
        f"🔀 Decision Fusion: Probabilistic Weighted Voting\n"
        f"📈 Experience DB: Active\n\n"
        f"⏰ In attesa del setup perfetto..."
    )


def build_stats_message(exec_stats: Dict[str, Any],
                         perf_summary: Dict[str, Any],
                         agent_report: Dict[str, Any]) -> str:
    lines = [
        "📊 *V17 Performance Report*\n",
        f"💰 Balance: `{exec_stats.get('balance', 0):.2f}` USDT",
        f"📈 Total P&L: `{exec_stats.get('total_pnl', 0):+.4f}` ({exec_stats.get('pnl_pct', 0):+.2f}%)",
        f"🏆 Win Rate: `{perf_summary.get('win_rate', 0):.1%}` "
        f"({perf_summary.get('wins', 0)}W / {perf_summary.get('losses', 0)}L)",
        f"📉 Sharpe: `{perf_summary.get('sharpe', 0):.2f}`",
        "",
        "🤖 *Agent Weights:*",
    ]
    for name, info in agent_report.items():
        w = info.get("weight", 1.0) or 1.0
        wr = info.get("win_rate", 0.5)
        n = info.get("n_decisions", 0)
        lines.append(f"  • {name}: w={w:.2f} wr={wr:.1%} n={n}")
    return "\n".join(lines)


def notify_position_closed(pos: Position) -> None:
    """Send notification when a position is closed."""
    emoji = "✅" if (pos.pnl or 0) > 0 else "❌"
    direction = pos.direction.upper()
    msg = (
        f"{emoji} *Position Closed* — {pos.symbol} [{pos.interval}]\n\n"
        f"Direction: *{direction}*\n"
        f"Entry: `{pos.entry_price:.4f}` → Close: `{pos.close_price:.4f}`\n"
        f"P&L: `{(pos.pnl or 0):+.4f}` USDT\n"
        f"Status: `{pos.status}`\n"
        f"{'📄 Paper' if pos.paper else '⚡ Live'}"
    )
    send_message(msg)
