"""Download and cache historical futures kline datasets."""
import argparse
import os
import time
from typing import List

import pandas as pd

from data.binance_client import fetch_futures_klines


def _fetch_paginated(symbol: str, interval: str, years: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365 * 24 * 3600 * 1000)
    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        batch = fetch_futures_klines(symbol, interval, limit=1000, start_time=cursor)
        if not batch:
            break
        all_rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].dropna()


def _save_df(df: pd.DataFrame, path_without_ext: str) -> str:
    try:
        df.to_parquet(path_without_ext + ".parquet", index=False)
        return path_without_ext + ".parquet"
    except Exception:
        csv_path = path_without_ext + ".csv"
        df.to_csv(csv_path, index=False)
        return csv_path


def download(symbols: List[str], intervals: List[str], years: int, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for symbol in symbols:
        for interval in intervals:
            df = _fetch_paginated(symbol, interval, years)
            if df.empty:
                print(f"{symbol} {interval}: no data")
                continue
            base = os.path.join(cache_dir, f"{symbol}_{interval}")
            path = _save_df(df, base)
            size = os.path.getsize(path) / 1024.0
            print(
                f"{symbol} {interval}: candles={len(df)} period={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} "
                f"file={os.path.basename(path)} size={size:.1f}KB"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical Binance futures klines")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", required=True)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/historical_cache")
    args = parser.parse_args()
    download([s.upper() for s in args.symbols], args.intervals, args.years, args.cache_dir)


if __name__ == "__main__":
    main()
