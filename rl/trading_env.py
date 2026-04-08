"""
rl/trading_env.py — Gymnasium-Compatible Trading Environment.

Implements a single-asset trading environment that follows the
Gymnasium (OpenAI Gym successor) interface.

State space:
    [log_return, realised_vol, rsi_norm, adx_norm, volume_ratio,
     regime_trending, regime_ranging, regime_volatile,
     position_side, position_pct, balance_pct, drawdown]
    Dimension: 12

Action space (discrete, 8 actions):
    0: HOLD
    1: LONG_SMALL  (0.25x sizing)
    2: LONG_MEDIUM (0.50x)
    3: LONG_LARGE  (1.00x)
    4: SHORT_SMALL
    5: SHORT_MEDIUM
    6: SHORT_LARGE
    7: CLOSE

Episodes are constructed by replaying historical OHLCV data.

Integration: used by PPOAgent for offline training and online inference.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("TradingEnv")

# Action constants
HOLD = 0
LONG_SMALL = 1
LONG_MEDIUM = 2
LONG_LARGE = 3
SHORT_SMALL = 4
SHORT_MEDIUM = 5
SHORT_LARGE = 6
CLOSE = 7

ACTION_NAMES = ["HOLD", "LONG_S", "LONG_M", "LONG_L",
                "SHORT_S", "SHORT_M", "SHORT_L", "CLOSE"]
ACTION_SIZES = [0.0, 0.25, 0.50, 1.0, 0.25, 0.50, 1.0, 0.0]

N_ACTIONS = 8
N_FEATURES = 12

_INITIAL_BALANCE = 10_000.0
_TRANSACTION_COST = 0.0006   # 0.06% per trade (maker fee)
_LEVERAGE = 5


class Position:
    """Simple position tracker."""

    def __init__(self):
        self.side: str = "none"          # 'long' | 'short' | 'none'
        self.size: float = 0.0           # fraction of balance
        self.entry_price: float = 0.0
        self.holding_steps: int = 0

    def is_open(self) -> bool:
        return self.side != "none"

    def pnl(self, current_price: float) -> float:
        if not self.is_open() or self.entry_price <= 0:
            return 0.0
        if self.side == "long":
            return (current_price / self.entry_price - 1.0) * self.size * _LEVERAGE
        else:
            return (1.0 - current_price / self.entry_price) * self.size * _LEVERAGE


class TradingEnv:
    """Gymnasium-compatible trading environment for backtesting and RL training."""

    def __init__(self,
                 df: Optional[pd.DataFrame] = None,
                 initial_balance: float = _INITIAL_BALANCE,
                 transaction_cost: float = _TRANSACTION_COST):
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self._n_features = N_FEATURES
        self._n_actions = N_ACTIONS

        self._t: int = 0
        self._balance: float = initial_balance
        self._peak_balance: float = initial_balance
        self._position = Position()
        self._done: bool = False
        self._history: List[float] = []
        self._last_obs: Optional[np.ndarray] = None  # cached last valid observation

        # Observation/action space metadata (Gymnasium-compatible)
        self.observation_space = _SpaceSpec(shape=(N_FEATURES,),
                                            low=-np.inf, high=np.inf)
        self.action_space = _SpaceSpec(shape=(1,), low=0, high=N_ACTIONS - 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for a new episode.

        Parameters
        ----------
        df : optional new DataFrame to use for this episode

        Returns
        -------
        (observation, info)
        """
        if df is not None:
            self.df = df
        if self.df is None:
            raise ValueError("TradingEnv: no data provided. Pass df to reset().")

        self._t = 60   # start after burn-in period for indicator calculation
        self._balance = self.initial_balance
        self._peak_balance = self.initial_balance
        self._position = Position()
        self._done = False
        self._history = [self.initial_balance]
        self._last_obs = None

        obs = self._get_observation()
        self._last_obs = obs
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : int in [0, N_ACTIONS-1]

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        if self._done:
            # Return the last cached observation when episode is already done
            obs = self._last_obs if self._last_obs is not None else np.zeros(N_FEATURES, dtype=np.float32)
            return obs, 0.0, True, False, {}

        prev_balance = self._balance
        current_price = float(self.df["close"].iloc[self._t])

        # Apply action
        reward = self._apply_action(action, current_price)

        # Step time
        self._t += 1
        self._position.holding_steps += 1 if self._position.is_open() else 0

        # Update balance with unrealised P&L
        unreal_pnl = self._position.pnl(current_price) * self._balance
        current_value = self._balance + unreal_pnl
        self._peak_balance = max(self._peak_balance, current_value)
        self._history.append(current_value)

        done = self._t >= len(self.df) - 1
        if done:
            # Close position on episode end
            if self._position.is_open():
                reward += self._close_position(current_price)
            self._done = True

        obs = self._get_observation()
        self._last_obs = obs  # cache observation for done-state returns
        info = {
            "balance": self._balance,
            "step": self._t,
            "position_side": self._position.side,
        }
        return obs, reward, self._done, False, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Return a text summary of the current state."""
        msg = (
            f"Step={self._t}/{len(self.df) if self.df is not None else '?'} | "
            f"Balance={self._balance:.2f} | "
            f"Position={self._position.side} "
            f"(size={self._position.size:.2f}) | "
            f"PnL={self._position.pnl(float(self.df['close'].iloc[self._t])) if self.df is not None else 0:.4f}"
        )
        if mode == "human":
            logger.debug(msg)
        return msg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: int, price: float) -> float:
        """Apply action and return immediate reward."""
        reward = 0.0
        size = ACTION_SIZES[action]

        if action == HOLD:
            pass

        elif action in (LONG_SMALL, LONG_MEDIUM, LONG_LARGE):
            if self._position.side == "short":
                reward += self._close_position(price)
            if not self._position.is_open():
                self._position.side = "long"
                self._position.size = size
                self._position.entry_price = price
                self._position.holding_steps = 0
                self._balance -= self._balance * self.transaction_cost

        elif action in (SHORT_SMALL, SHORT_MEDIUM, SHORT_LARGE):
            if self._position.side == "long":
                reward += self._close_position(price)
            if not self._position.is_open():
                self._position.side = "short"
                self._position.size = size
                self._position.entry_price = price
                self._position.holding_steps = 0
                self._balance -= self._balance * self.transaction_cost

        elif action == CLOSE:
            if self._position.is_open():
                reward += self._close_position(price)

        return reward

    def _close_position(self, price: float) -> float:
        """Close position and return realised P&L as fraction of balance."""
        if not self._position.is_open():
            return 0.0
        pnl_frac = self._position.pnl(price)
        self._balance += self._balance * pnl_frac
        self._balance -= self._balance * self.transaction_cost
        self._position = Position()
        return pnl_frac

    def _get_observation(self) -> np.ndarray:
        """Build normalised observation vector."""
        if self.df is None or self._t >= len(self.df):
            return np.zeros(N_FEATURES, dtype=np.float32)

        try:
            t = self._t
            df = self.df
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)

            # Log return
            log_ret = float(np.log(close[t] / max(close[t - 1], 1e-8))) if t > 0 else 0.0

            # Realised vol (20-period)
            if t >= 20:
                rv = float(np.std(np.log(close[t - 19:t + 1] / close[t - 20:t]))) * np.sqrt(365)
            else:
                rv = 0.0

            # RSI (14-period) normalised to [0,1]
            rsi_norm = self._rsi_norm(close, t, period=14)

            # ADX proxy (range ratio) normalised to [0,1]
            if t >= 14:
                highs = df["high"].values[t - 14:t]
                lows = df["low"].values[t - 14:t]
                atr = float(np.mean(highs - lows))
                adx_norm = min(atr / max(close[t] * 0.02, 1e-8), 1.0)
            else:
                adx_norm = 0.0

            # Volume ratio
            if t >= 20:
                vol_ratio = float(volume[t] / max(np.mean(volume[t - 20:t]), 1e-8))
                vol_ratio = min(vol_ratio / 5.0, 1.0)  # normalise
            else:
                vol_ratio = 0.2

            # Regime (simplified: use volatility level)
            regime_trending = 1.0 if rv < 0.5 and adx_norm > 0.3 else 0.0
            regime_ranging = 1.0 if rv < 0.3 and adx_norm < 0.2 else 0.0
            regime_volatile = 1.0 if rv > 0.8 else 0.0

            # Position info
            pos_side = {"long": 1.0, "short": -1.0, "none": 0.0}[self._position.side]
            pos_pct = self._position.size if self._position.is_open() else 0.0

            # Balance ratio
            balance_pct = min(self._balance / self.initial_balance, 2.0) - 1.0

            # Drawdown
            current_val = self._balance
            drawdown = min((self._peak_balance - current_val) / max(self._peak_balance, 1), 1.0)

            obs = np.array([
                np.clip(log_ret * 100, -5, 5),   # scaled
                np.clip(rv, 0, 5),
                rsi_norm,
                adx_norm,
                vol_ratio,
                regime_trending,
                regime_ranging,
                regime_volatile,
                pos_side,
                pos_pct,
                np.clip(balance_pct, -1, 1),
                drawdown,
            ], dtype=np.float32)

            return obs
        except Exception as exc:
            logger.debug(f"_get_observation error: {exc}")
            return np.zeros(N_FEATURES, dtype=np.float32)

    @staticmethod
    def _rsi_norm(close: np.ndarray, t: int, period: int = 14) -> float:
        """Compute normalised RSI [0,1]."""
        if t < period:
            return 0.5
        diffs = np.diff(close[t - period:t + 1])
        gains = diffs[diffs > 0]
        losses = -diffs[diffs < 0]
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 1e-8
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi / 100.0

    def get_episode_stats(self) -> Dict:
        """Return episode performance statistics."""
        if not self._history:
            return {}
        arr = np.array(self._history)
        returns = np.diff(arr) / np.maximum(arr[:-1], 1e-8)
        total_return = (arr[-1] / arr[0] - 1.0) if arr[0] > 0 else 0.0
        max_dd = float(np.max(1 - arr / np.maximum.accumulate(arr))) if len(arr) > 1 else 0.0
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if np.std(returns) > 1e-8 else 0.0
        return {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "final_balance": float(arr[-1]),
            "n_steps": len(arr),
        }


class _SpaceSpec:
    """Minimal space descriptor (replaces gymnasium.spaces.*)."""

    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high
        self.n = shape[0] if len(shape) == 1 else None
