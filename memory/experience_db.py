"""
Experience Database for V17.
SQLite database with tables:
  - decisions      : every fusion decision with reasoning
  - agent_performance : per-agent outcome records
  - optimal_params : auto-tuned parameters
  - trade_outcomes : position open/close records
"""
import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from config.settings import DB_PATH

logger = logging.getLogger("ExperienceDB")

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(path: str = DB_PATH) -> None:
    """Initialise the SQLite database and create tables if they don't exist."""
    global _conn
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _create_tables()
    logger.info(f"✅ Experience DB initialised at {path}")


def _create_tables() -> None:
    c = _conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT    UNIQUE,
            ts          REAL    NOT NULL,
            symbol      TEXT    NOT NULL,
            interval    TEXT    NOT NULL,
            decision    TEXT    NOT NULL,
            final_score REAL,
            direction   TEXT,
            threshold   REAL,
            reasoning   TEXT,
            agent_scores TEXT,
            outcome     TEXT    DEFAULT NULL,
            pnl         REAL    DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_performance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            decision_id TEXT    NOT NULL,
            agent_name  TEXT    NOT NULL,
            score       REAL,
            direction   TEXT,
            correct     INTEGER,
            pattern_tags TEXT   DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS optimal_params (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            param_key   TEXT    UNIQUE NOT NULL,
            param_value TEXT    NOT NULL,
            source      TEXT
        );

        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id  TEXT    UNIQUE,
            ts_open      REAL,
            ts_close     REAL,
            symbol       TEXT,
            interval     TEXT,
            direction    TEXT,
            entry_price  REAL,
            close_price  REAL,
            size         REAL,
            pnl          REAL,
            status       TEXT,
            strategy     TEXT,
            decision_id  TEXT,
            paper        INTEGER
        );
    """)
    _conn.commit()
    # Migrate existing databases: add pattern_tags column if absent
    try:
        _conn.execute("ALTER TABLE agent_performance ADD COLUMN pattern_tags TEXT DEFAULT ''")
        _conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            logger.warning(f"_create_tables migration warning: {e}")


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def save_decision(decision_id: str, symbol: str, interval: str, decision: str,
                   final_score: float, direction: str, threshold: float,
                   reasoning: List[str], agent_scores: Dict[str, float]) -> None:
    if _conn is None:
        return
    with _lock:
        try:
            _conn.execute(
                """INSERT OR IGNORE INTO decisions
                   (decision_id, ts, symbol, interval, decision, final_score,
                    direction, threshold, reasoning, agent_scores)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (decision_id, time.time(), symbol, interval, decision,
                 final_score, direction, threshold,
                 json.dumps(reasoning), json.dumps(agent_scores))
            )
            _conn.commit()
        except Exception as e:
            logger.error(f"save_decision error: {e}")


def update_decision_outcome(decision_id: str, outcome: str, pnl: float) -> None:
    if _conn is None:
        return
    with _lock:
        try:
            _conn.execute(
                "UPDATE decisions SET outcome=?, pnl=? WHERE decision_id=?",
                (outcome, pnl, decision_id)
            )
            _conn.commit()
        except Exception as e:
            logger.error(f"update_decision_outcome error: {e}")


def get_recent_decisions(limit: int = 20) -> List[Dict[str, Any]]:
    if _conn is None:
        return []
    with _lock:
        try:
            rows = _conn.execute(
                "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"get_recent_decisions error: {e}")
            return []


# ---------------------------------------------------------------------------
# Agent performance
# ---------------------------------------------------------------------------

def save_agent_outcome(decision_id: str, agent_name: str, score: float,
                        direction: str, correct: bool,
                        pattern_tags: str = "") -> None:
    if _conn is None:
        return
    with _lock:
        try:
            _conn.execute(
                """INSERT INTO agent_performance
                   (ts, decision_id, agent_name, score, direction, correct, pattern_tags)
                   VALUES (?,?,?,?,?,?,?)""",
                (time.time(), decision_id, agent_name, score, direction, int(correct), pattern_tags)
            )
            _conn.commit()
        except Exception as e:
            logger.error(f"save_agent_outcome error: {e}")


def get_agent_win_rates() -> Dict[str, float]:
    """Return win rate per agent over all recorded outcomes."""
    if _conn is None:
        return {}
    with _lock:
        try:
            rows = _conn.execute(
                """SELECT agent_name,
                          AVG(correct) as win_rate,
                          COUNT(*)     as n
                   FROM agent_performance
                   GROUP BY agent_name"""
            ).fetchall()
            return {row["agent_name"]: float(row["win_rate"]) for row in rows if row["n"] >= 5}
        except Exception as e:
            logger.error(f"get_agent_win_rates error: {e}")
            return {}


# ---------------------------------------------------------------------------
# Optimal parameters
# ---------------------------------------------------------------------------

def save_param(key: str, value: Any, source: str = "auto") -> None:
    if _conn is None:
        return
    with _lock:
        try:
            _conn.execute(
                """INSERT OR REPLACE INTO optimal_params (ts, param_key, param_value, source)
                   VALUES (?,?,?,?)""",
                (time.time(), key, json.dumps(value), source)
            )
            _conn.commit()
        except Exception as e:
            logger.error(f"save_param error: {e}")


def get_param(key: str, default: Any = None) -> Any:
    if _conn is None:
        return default
    with _lock:
        try:
            row = _conn.execute(
                "SELECT param_value FROM optimal_params WHERE param_key=?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["param_value"])
        except Exception as e:
            logger.error(f"get_param error: {e}")
    return default


# ---------------------------------------------------------------------------
# Trade outcomes
# ---------------------------------------------------------------------------

def save_trade_outcome(position_id: str, ts_open: float, ts_close: float,
                        symbol: str, interval: str, direction: str,
                        entry_price: float, close_price: float, size: float,
                        pnl: float, status: str, strategy: str,
                        decision_id: str, paper: bool) -> None:
    if _conn is None:
        return
    with _lock:
        try:
            _conn.execute(
                """INSERT OR REPLACE INTO trade_outcomes
                   (position_id, ts_open, ts_close, symbol, interval, direction,
                    entry_price, close_price, size, pnl, status, strategy, decision_id, paper)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (position_id, ts_open, ts_close, symbol, interval, direction,
                 entry_price, close_price, size, pnl, status, strategy, decision_id, int(paper))
            )
            _conn.commit()
        except Exception as e:
            logger.error(f"save_trade_outcome error: {e}")


def get_win_rate_by_symbol(symbol: str) -> Optional[float]:
    if _conn is None:
        return None
    with _lock:
        try:
            row = _conn.execute(
                """SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as wr
                   FROM trade_outcomes WHERE symbol=? AND pnl IS NOT NULL""",
                (symbol,)
            ).fetchone()
            if row and row["wr"] is not None:
                return float(row["wr"])
        except Exception as e:
            logger.error(f"get_win_rate_by_symbol error: {e}")
    return None


def get_win_rate_by_interval(interval: str) -> Optional[float]:
    if _conn is None:
        return None
    with _lock:
        try:
            row = _conn.execute(
                """SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as wr
                   FROM trade_outcomes WHERE interval=? AND pnl IS NOT NULL""",
                (interval,)
            ).fetchone()
            if row and row["wr"] is not None:
                return float(row["wr"])
        except Exception as e:
            logger.error(f"get_win_rate_by_interval error: {e}")
    return None


def get_completed_trade_count() -> int:
    """Return the total number of completed trades (those with a non-null pnl)."""
    if _conn is None:
        return 0
    with _lock:
        try:
            row = _conn.execute(
                "SELECT COUNT(*) AS cnt FROM trade_outcomes WHERE pnl IS NOT NULL"
            ).fetchone()
            if row:
                return int(row["cnt"])
        except Exception as e:
            logger.error(f"get_completed_trade_count error: {e}")
    return 0
