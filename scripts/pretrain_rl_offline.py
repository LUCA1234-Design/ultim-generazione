"""Offline RL pretraining for PPO on historical replay datasets."""
import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from scripts.download_historical import download as download_historical


def classify_market_regime(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 220:
        return "choppy"
    close = df["close"].astype(float)
    ema200 = close.ewm(span=200, adjust=False).mean()
    slope = float((ema200.iloc[-1] - ema200.iloc[-50]) / max(ema200.iloc[-50], 1e-8))
    atr_rel = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1] / max(close.iloc[-1], 1e-8))
    if slope > 0.03 and atr_rel < 0.02:
        return "bull"
    if slope < -0.03:
        return "bear"
    if atr_rel > 0.04:
        return "crash"
    return "choppy"


def _load_cached(cache_dir: str, symbol: str, interval: str) -> pd.DataFrame:
    pq = os.path.join(cache_dir, f"{symbol}_{interval}.parquet")
    csv = os.path.join(cache_dir, f"{symbol}_{interval}.csv")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return pd.DataFrame()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    close = out["close"].astype(float)
    delta = close.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    out["atr"] = (out["high"].astype(float) - out["low"].astype(float)).rolling(14).mean().fillna(0.0)
    out["volume_ratio"] = (
        out["volume"].astype(float) / out["volume"].astype(float).rolling(20).mean().replace(0, np.nan)
    ).fillna(1.0)
    return out[["timestamp", "open", "high", "low", "close", "volume", "rsi", "atr", "volume_ratio"]]


def _split_train_val(df: pd.DataFrame, validation_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    cut = df["timestamp"].max() - pd.Timedelta(days=validation_months * 30)
    train = df[df["timestamp"] < cut].reset_index(drop=True)
    val = df[df["timestamp"] >= cut].reset_index(drop=True)
    return train, val


def _rollout_episode(env: TradingEnv, ppo: PPOAgent, df_train: pd.DataFrame, window_size: int) -> Dict:
    if len(df_train) > window_size + 120:
        start = random.randint(0, len(df_train) - window_size - 1)
        ep_df = df_train.iloc[start:start + window_size].reset_index(drop=True)
    else:
        ep_df = df_train

    state, _ = env.reset(df=ep_df)
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    done = False
    while not done:
        action, logp = ppo.select_action(state)
        value = ppo.get_value(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        states.append(state)
        actions.append(action)
        log_probs.append(logp)
        rewards.append(float(reward))
        values.append(float(value))
        dones.append(float(done))
        state = next_state

    trajectory = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }
    stats = env.get_episode_stats()
    return {"trajectory": trajectory, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PPO pretraining")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", required=True)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--cache-dir", default="data/historical_cache")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    download_historical([s.upper() for s in args.symbols], args.intervals, args.years, args.cache_dir)

    datasets = []
    for symbol in [s.upper() for s in args.symbols]:
        for interval in args.intervals:
            raw = _load_cached(args.cache_dir, symbol, interval)
            if raw.empty:
                continue
            df = _prepare_df(raw)
            train_df, val_df = _split_train_val(df, args.validation_months)
            if len(train_df) < 120:
                continue
            regime = classify_market_regime(train_df)
            difficulty = {"bull": 0, "bear": 1, "choppy": 2, "crash": 3}.get(regime, 2)
            datasets.append((difficulty, symbol, interval, regime, train_df, val_df))

    datasets.sort(key=lambda x: x[0])
    if not datasets:
        raise SystemExit("No datasets available for pretraining")

    ppo = PPOAgent()
    env = TradingEnv()

    t0 = time.time()
    all_ep_returns = []
    for ep in range(1, args.episodes + 1):
        _, symbol, interval, regime, train_df, _ = datasets[(ep - 1) % len(datasets)]
        rolled = _rollout_episode(env, ppo, train_df, args.window_size)
        train_stats = ppo.update([rolled["trajectory"]])
        ep_stats = rolled["stats"]
        all_ep_returns.append(ep_stats.get("total_return", 0.0))

        if ep % 50 == 0:
            print(
                f"ep={ep} {symbol}/{interval} [{regime}] ret={ep_stats.get('total_return', 0):+.4f} "
                f"sharpe={ep_stats.get('sharpe_ratio', 0):+.3f} mdd={ep_stats.get('max_drawdown', 0):+.3f} "
                f"actor_loss={train_stats.get('actor_loss', 0):+.5f} entropy={train_stats.get('entropy', 0):+.5f}"
            )

        if ep % args.checkpoint_every == 0:
            ppo.save(f"models/ppo_checkpoint_{ep}.pt")

    validations = []
    for _, symbol, interval, _, _, val_df in datasets:
        if len(val_df) < 100:
            continue
        try:
            val_ep = _rollout_episode(env, ppo, val_df, window_size=min(len(val_df), args.window_size))
            validations.append(val_ep["stats"])
        except Exception as e:
            print(f"  Warning: validation failed for {symbol}/{interval}: {e}")

    final_sharpe = float(np.mean([v.get("sharpe_ratio", 0.0) for v in validations])) if validations else 0.0
    final_win_rate = float(np.mean([
        1.0 if v.get("total_return", 0.0) > 0 else 0.0
        for v in validations
    ])) if validations else 0.0

    report = {
        "total_episodes": args.episodes,
        "symbols_trained": sorted(list({d[1] for d in datasets})),
        "intervals_trained": sorted(list({d[2] for d in datasets})),
        "final_sharpe": final_sharpe,
        "final_win_rate": final_win_rate,
        "training_duration_min": round((time.time() - t0) / 60.0, 2),
    }

    with open("models/pretrain_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    ppo.save("models/ppo_pretrained.pt")
    print("Saved models/ppo_pretrained.pt and models/pretrain_report.json")


if __name__ == "__main__":
    main()
