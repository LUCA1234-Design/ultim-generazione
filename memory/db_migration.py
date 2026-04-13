"""SQLite -> TimescaleDB migration helper."""
import argparse
import logging
import sqlite3
from typing import Dict, List

logger = logging.getLogger("DBMigration")

TABLES = ["decisions", "agent_performance", "optimal_params", "trade_outcomes"]


def _to_pg_insert(table: str, cols: List[str]) -> str:
    placeholders = ",".join(["%s"] * len(cols))
    col_sql = ",".join(cols)
    if table == "decisions":
        return f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) ON CONFLICT (decision_id) DO NOTHING"
    if table == "optimal_params":
        return f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) ON CONFLICT (param_key) DO UPDATE SET ts=EXCLUDED.ts,param_value=EXCLUDED.param_value,source=EXCLUDED.source"
    if table == "trade_outcomes":
        return f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) ON CONFLICT (position_id) DO UPDATE SET ts_open=EXCLUDED.ts_open,ts_close=EXCLUDED.ts_close,symbol=EXCLUDED.symbol,interval=EXCLUDED.interval,direction=EXCLUDED.direction,entry_price=EXCLUDED.entry_price,close_price=EXCLUDED.close_price,size=EXCLUDED.size,pnl=EXCLUDED.pnl,status=EXCLUDED.status,strategy=EXCLUDED.strategy,decision_id=EXCLUDED.decision_id,paper=EXCLUDED.paper"
    return f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"


def migrate(sqlite_path: str, timescale_url: str) -> Dict[str, dict]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    report = {}
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    dst = psycopg2.connect(timescale_url)

    try:
        with dst.cursor(cursor_factory=RealDictCursor) as cur:
            for table in TABLES:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
                n_src = len(rows)
                if not rows:
                    report[table] = {"source": 0, "target": 0}
                    continue

                cols = list(rows[0].keys())
                insert_sql = _to_pg_insert(table, cols)
                payload = [tuple(r[c] for c in cols) for r in rows]
                cur.executemany(insert_sql, payload)
                dst.commit()

                cur.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                n_dst = int(cur.fetchone()["cnt"])
                report[table] = {"source": n_src, "target": n_dst}
    finally:
        src.close()
        dst.close()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate V17 SQLite data to TimescaleDB")
    parser.add_argument("--sqlite-path", default="v17_experience.db")
    parser.add_argument("--timescale-url", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    rep = migrate(args.sqlite_path, args.timescale_url)

    print("\n=== MIGRATION REPORT ===")
    ok = True
    for table, stats in rep.items():
        src = stats.get("source", 0)
        dst = stats.get("target", 0)
        status = "OK" if dst >= src else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"{table:20s} source={src:8d} target={dst:8d} [{status}]")

    print("\nIntegrity:", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
