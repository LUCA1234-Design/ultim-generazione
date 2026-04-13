"""
Experience Database for V17/V18.
Supports SQLite (default) and optional TimescaleDB backend.
"""
import json
import logging
import queue
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from config.settings import DB_PATH, DB_BACKEND, DB_TIMESCALE_URL

logger = logging.getLogger("ExperienceDB")

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2.extras import RealDictCursor
    _PSYCOPG2_AVAILABLE = True
except Exception:
    _PSYCOPG2_AVAILABLE = False


class DatabaseBackend(ABC):
    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def execute(self, query: str, params: Sequence[Any] = ()):
        ...

    @abstractmethod
    def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]):
        ...

    @abstractmethod
    def fetchone(self, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def fetchall(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def commit(self) -> None:
        ...


class SQLiteBackend(DatabaseBackend):
    def __init__(self, path: str):
        self.path = path
        self.conn: Optional[sqlite3.Connection] = None

    def init(self) -> None:
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.commit()

    def execute(self, query: str, params: Sequence[Any] = ()):
        return self.conn.execute(query, tuple(params))

    def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]):
        return self.conn.executemany(query, [tuple(p) for p in params_seq])

    def fetchone(self, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(query, tuple(params)).fetchone()
        return dict(row) if row else None

    def fetchall(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def commit(self) -> None:
        self.conn.commit()


class TimescaleBackend(DatabaseBackend):
    def __init__(self, url: str):
        self.url = url
        self.pool: Optional[ThreadedConnectionPool] = None

    @staticmethod
    def _q(sql: str) -> str:
        return sql.replace("?", "%s")

    def init(self) -> None:
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available")
        self.pool = ThreadedConnectionPool(minconn=2, maxconn=10, dsn=self.url)

    def execute(self, query: str, params: Sequence[Any] = ()):
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._q(query), tuple(params))
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]):
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.executemany(self._q(query), [tuple(p) for p in params_seq])
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def fetchone(self, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._q(query), tuple(params))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self.pool.putconn(conn)

    def fetchall(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._q(query), tuple(params))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            self.pool.putconn(conn)

    def commit(self) -> None:
        return


_lock = threading.Lock()
_backend: Optional[DatabaseBackend] = None
_write_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=10000)
_write_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _is_timescale() -> bool:
    return isinstance(_backend, TimescaleBackend)


def _create_tables() -> None:
    if _backend is None:
        return

    if _is_timescale():
        statements = [
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id BIGSERIAL PRIMARY KEY,
                decision_id TEXT UNIQUE,
                ts DOUBLE PRECISION NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                decision TEXT NOT NULL,
                final_score DOUBLE PRECISION,
                direction TEXT,
                threshold DOUBLE PRECISION,
                reasoning TEXT,
                agent_scores TEXT,
                outcome TEXT DEFAULT NULL,
                pnl DOUBLE PRECISION DEFAULT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id BIGSERIAL PRIMARY KEY,
                ts DOUBLE PRECISION NOT NULL,
                decision_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                score DOUBLE PRECISION,
                direction TEXT,
                correct INTEGER,
                pattern_tags TEXT DEFAULT ''
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS optimal_params (
                id BIGSERIAL PRIMARY KEY,
                ts DOUBLE PRECISION NOT NULL,
                param_key TEXT UNIQUE NOT NULL,
                param_value TEXT NOT NULL,
                source TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id BIGSERIAL PRIMARY KEY,
                position_id TEXT UNIQUE,
                ts_open DOUBLE PRECISION,
                ts_close DOUBLE PRECISION,
                symbol TEXT,
                interval TEXT,
                direction TEXT,
                entry_price DOUBLE PRECISION,
                close_price DOUBLE PRECISION,
                size DOUBLE PRECISION,
                pnl DOUBLE PRECISION,
                status TEXT,
                strategy TEXT,
                decision_id TEXT,
                paper INTEGER
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_outcomes (symbol, ts_open DESC)",
        ]
        for sql in statements:
            _backend.execute(sql)
        try:
            _backend.execute("SELECT create_hypertable('trade_outcomes', 'ts_open', if_not_exists => TRUE)")
        except Exception as e:
            logger.debug(f"trade_outcomes hypertable warning: {e}")
        try:
            _backend.execute("SELECT create_hypertable('agent_performance', 'ts', if_not_exists => TRUE)")
        except Exception as e:
            logger.debug(f"agent_performance hypertable warning: {e}")
    else:
        _backend.execute(
            """
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
            )
            """
        )
        _backend.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                decision_id TEXT    NOT NULL,
                agent_name  TEXT    NOT NULL,
                score       REAL,
                direction   TEXT,
                correct     INTEGER,
                pattern_tags TEXT   DEFAULT ''
            )
            """
        )
        _backend.execute(
            """
            CREATE TABLE IF NOT EXISTS optimal_params (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                param_key   TEXT    UNIQUE NOT NULL,
                param_value TEXT    NOT NULL,
                source      TEXT
            )
            """
        )
        _backend.execute(
            """
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
            )
            """
        )
        try:
            _backend.execute("ALTER TABLE agent_performance ADD COLUMN pattern_tags TEXT DEFAULT ''")
        except Exception:
            pass


def _start_write_worker() -> None:
    global _write_worker_thread
    if _write_worker_thread and _write_worker_thread.is_alive():
        return

    _stop_event.clear()

    def _worker():
        while not _stop_event.is_set():
            try:
                item = _write_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                _write_queue.task_done()
                break
            op, args = item
            try:
                if op == "save_decision":
                    _save_decision_sync(*args)
                elif op == "save_agent_outcome":
                    _save_agent_outcome_sync(*args)
                elif op == "save_trade_outcome":
                    _save_trade_outcome_sync(*args)
            except Exception as e:
                logger.error(f"write worker error ({op}): {e}")
            finally:
                _write_queue.task_done()

    _write_worker_thread = threading.Thread(target=_worker, daemon=True, name="ExperienceDB-Writer")
    _write_worker_thread.start()


def init_db(path: str = DB_PATH) -> None:
    global _backend

    backend_mode = (DB_BACKEND or "sqlite").strip().lower()
    if backend_mode == "timescaledb":
        try:
            _backend = TimescaleBackend(DB_TIMESCALE_URL)
            _backend.init()
            _create_tables()
            _start_write_worker()
            logger.info("✅ Experience DB initialised with TimescaleDB backend")
            return
        except Exception as e:
            logger.warning(f"⚠️ TimescaleDB unavailable ({e}) — falling back to SQLite")

    _backend = SQLiteBackend(path)
    _backend.init()
    _create_tables()
    _start_write_worker()
    logger.info(f"✅ Experience DB initialised at {path} (SQLite WAL mode)")


def _enqueue_write(op: str, args: tuple) -> None:
    if _backend is None:
        return
    try:
        _write_queue.put_nowait((op, args))
    except queue.Full:
        logger.warning(f"write queue full for op={op}; executing synchronously")
        if op == "save_decision":
            _save_decision_sync(*args)
        elif op == "save_agent_outcome":
            _save_agent_outcome_sync(*args)
        elif op == "save_trade_outcome":
            _save_trade_outcome_sync(*args)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def _save_decision_sync(decision_id: str, symbol: str, interval: str, decision: str,
                        final_score: float, direction: str, threshold: float,
                        reasoning: List[str], agent_scores: Dict[str, float]) -> None:
    if _backend is None:
        return
    with _lock:
        if _is_timescale():
            sql = """INSERT INTO decisions
                     (decision_id, ts, symbol, interval, decision, final_score,
                      direction, threshold, reasoning, agent_scores)
                     VALUES (?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT (decision_id) DO NOTHING"""
        else:
            sql = """INSERT OR IGNORE INTO decisions
                     (decision_id, ts, symbol, interval, decision, final_score,
                      direction, threshold, reasoning, agent_scores)
                     VALUES (?,?,?,?,?,?,?,?,?,?)"""
        _backend.execute(
            sql,
            (decision_id, time.time(), symbol, interval, decision,
             final_score, direction, threshold,
             json.dumps(reasoning), json.dumps(agent_scores)),
        )


def save_decision(decision_id: str, symbol: str, interval: str, decision: str,
                  final_score: float, direction: str, threshold: float,
                  reasoning: List[str], agent_scores: Dict[str, float]) -> None:
    _enqueue_write("save_decision", (decision_id, symbol, interval, decision, final_score,
                                      direction, threshold, reasoning, agent_scores))


def update_decision_outcome(decision_id: str, outcome: str, pnl: float) -> None:
    if _backend is None:
        return
    with _lock:
        try:
            _backend.execute(
                "UPDATE decisions SET outcome=?, pnl=? WHERE decision_id=?",
                (outcome, pnl, decision_id),
            )
        except Exception as e:
            logger.error(f"update_decision_outcome error: {e}")


def get_recent_decisions(limit: int = 20) -> List[Dict[str, Any]]:
    if _backend is None:
        return []
    with _lock:
        try:
            return _backend.fetchall("SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
        except Exception as e:
            logger.error(f"get_recent_decisions error: {e}")
            return []


# ---------------------------------------------------------------------------
# Agent performance
# ---------------------------------------------------------------------------

def _save_agent_outcome_sync(decision_id: str, agent_name: str, score: float,
                             direction: str, correct: bool, pattern_tags: str = "") -> None:
    if _backend is None:
        return
    with _lock:
        _backend.execute(
            """INSERT INTO agent_performance
               (ts, decision_id, agent_name, score, direction, correct, pattern_tags)
               VALUES (?,?,?,?,?,?,?)""",
            (time.time(), decision_id, agent_name, score, direction, int(correct), pattern_tags),
        )


def save_agent_outcome(decision_id: str, agent_name: str, score: float,
                       direction: str, correct: bool, pattern_tags: str = "") -> None:
    _enqueue_write("save_agent_outcome", (decision_id, agent_name, score, direction, correct, pattern_tags))


def get_agent_win_rates() -> Dict[str, float]:
    if _backend is None:
        return {}
    with _lock:
        try:
            rows = _backend.fetchall(
                """SELECT agent_name, AVG(correct) as win_rate, COUNT(*) as n
                   FROM agent_performance GROUP BY agent_name"""
            )
            return {row["agent_name"]: float(row["win_rate"]) for row in rows if row.get("n", 0) >= 5}
        except Exception as e:
            logger.error(f"get_agent_win_rates error: {e}")
            return {}


# ---------------------------------------------------------------------------
# Optimal parameters
# ---------------------------------------------------------------------------

def save_param(key: str, value: Any, source: str = "auto") -> None:
    if _backend is None:
        return
    with _lock:
        try:
            if _is_timescale():
                _backend.execute(
                    """INSERT INTO optimal_params (ts, param_key, param_value, source)
                       VALUES (?,?,?,?)
                       ON CONFLICT (param_key) DO UPDATE SET
                       ts=EXCLUDED.ts, param_value=EXCLUDED.param_value, source=EXCLUDED.source""",
                    (time.time(), key, json.dumps(value), source),
                )
            else:
                _backend.execute(
                    """INSERT OR REPLACE INTO optimal_params (ts, param_key, param_value, source)
                       VALUES (?,?,?,?)""",
                    (time.time(), key, json.dumps(value), source),
                )
        except Exception as e:
            logger.error(f"save_param error: {e}")


def get_param(key: str, default: Any = None) -> Any:
    if _backend is None:
        return default
    with _lock:
        try:
            row = _backend.fetchone("SELECT param_value FROM optimal_params WHERE param_key=?", (key,))
            if row and row.get("param_value") is not None:
                return json.loads(row["param_value"])
        except Exception as e:
            logger.error(f"get_param error: {e}")
    return default


# ---------------------------------------------------------------------------
# Trade outcomes
# ---------------------------------------------------------------------------

def _save_trade_outcome_sync(position_id: str, ts_open: float, ts_close: float,
                             symbol: str, interval: str, direction: str,
                             entry_price: float, close_price: float, size: float,
                             pnl: float, status: str, strategy: str,
                             decision_id: str, paper: bool) -> None:
    if _backend is None:
        return
    with _lock:
        if _is_timescale():
            sql = """INSERT INTO trade_outcomes
                     (position_id, ts_open, ts_close, symbol, interval, direction,
                      entry_price, close_price, size, pnl, status, strategy, decision_id, paper)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT (position_id) DO UPDATE SET
                       ts_open=EXCLUDED.ts_open, ts_close=EXCLUDED.ts_close,
                       symbol=EXCLUDED.symbol, interval=EXCLUDED.interval,
                       direction=EXCLUDED.direction, entry_price=EXCLUDED.entry_price,
                       close_price=EXCLUDED.close_price, size=EXCLUDED.size,
                       pnl=EXCLUDED.pnl, status=EXCLUDED.status, strategy=EXCLUDED.strategy,
                       decision_id=EXCLUDED.decision_id, paper=EXCLUDED.paper"""
        else:
            sql = """INSERT OR REPLACE INTO trade_outcomes
                     (position_id, ts_open, ts_close, symbol, interval, direction,
                      entry_price, close_price, size, pnl, status, strategy, decision_id, paper)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
        _backend.execute(
            sql,
            (position_id, ts_open, ts_close, symbol, interval, direction,
             entry_price, close_price, size, pnl, status, strategy, decision_id, int(paper)),
        )


def save_trade_outcome(position_id: str, ts_open: float, ts_close: float,
                       symbol: str, interval: str, direction: str,
                       entry_price: float, close_price: float, size: float,
                       pnl: float, status: str, strategy: str,
                       decision_id: str, paper: bool) -> None:
    _enqueue_write("save_trade_outcome", (position_id, ts_open, ts_close, symbol, interval, direction,
                                           entry_price, close_price, size, pnl, status, strategy,
                                           decision_id, paper))


def get_win_rate_by_symbol(symbol: str) -> Optional[float]:
    if _backend is None:
        return None
    with _lock:
        try:
            row = _backend.fetchone(
                """SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as wr
                   FROM trade_outcomes WHERE symbol=? AND pnl IS NOT NULL""",
                (symbol,),
            )
            if row and row.get("wr") is not None:
                return float(row["wr"])
        except Exception as e:
            logger.error(f"get_win_rate_by_symbol error: {e}")
    return None


def get_win_rate_by_interval(interval: str) -> Optional[float]:
    if _backend is None:
        return None
    with _lock:
        try:
            row = _backend.fetchone(
                """SELECT AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as wr
                   FROM trade_outcomes WHERE interval=? AND pnl IS NOT NULL""",
                (interval,),
            )
            if row and row.get("wr") is not None:
                return float(row["wr"])
        except Exception as e:
            logger.error(f"get_win_rate_by_interval error: {e}")
    return None


def get_completed_trade_count() -> int:
    if _backend is None:
        return 0
    with _lock:
        try:
            row = _backend.fetchone("SELECT COUNT(*) AS cnt FROM trade_outcomes WHERE pnl IS NOT NULL")
            if row:
                return int(row.get("cnt", 0))
        except Exception as e:
            logger.error(f"get_completed_trade_count error: {e}")
    return 0
