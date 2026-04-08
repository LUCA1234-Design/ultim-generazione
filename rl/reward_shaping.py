"""
rl/reward_shaping.py — Reward Engineering for RL Trading.

Implements a composite reward function that shapes the RL agent's
behaviour beyond simple P&L maximisation:

  R = R_pnl + R_sharpe_bonus - R_drawdown_penalty
        - R_inactivity_penalty - R_overtrading_penalty

Components:
  - Base:        realised P&L of the closed/current step
  - Drawdown:    -λ × incremental max drawdown (penalises losses)
  - Sharpe:      +bonus proportional to rolling Sharpe ratio
  - Inactivity:  small negative reward for holding too long without action
  - Overtrading: penalises high trade frequency (churning)

Integration: TradingEnv.step() calls compute_reward() to shape the
signal used to train the PPOAgent.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("RewardShaping")

# Default coefficients (all tuneable via config)
_LAMBDA_DRAWDOWN = 2.0         # drawdown penalty multiplier
_LAMBDA_SHARPE = 0.1           # Sharpe bonus multiplier
_INACTIVITY_PENALTY = -0.0005  # per-step penalty for holding too long
_INACTIVITY_THRESHOLD = 20     # steps before inactivity penalty kicks in
_OVERTRADE_PENALTY = -0.002    # per trade when frequency too high
_OVERTRADE_WINDOW = 20         # steps window for frequency check
_OVERTRADE_MAX_FREQ = 5        # max trades per window before penalising
_SHARPE_WINDOW = 30            # rolling window for Sharpe estimate


class RewardShaper:
    """
    Stateful reward shaper that tracks history across steps.

    Usage:
        shaper = RewardShaper()
        shaper.reset()
        for each step:
            reward = shaper.compute_reward(pnl, drawdown, is_trade, is_holding)
    """

    def __init__(self,
                 lambda_drawdown: float = _LAMBDA_DRAWDOWN,
                 lambda_sharpe: float = _LAMBDA_SHARPE,
                 inactivity_penalty: float = _INACTIVITY_PENALTY,
                 inactivity_threshold: int = _INACTIVITY_THRESHOLD,
                 overtrade_penalty: float = _OVERTRADE_PENALTY):
        self.lambda_drawdown = lambda_drawdown
        self.lambda_sharpe = lambda_sharpe
        self.inactivity_penalty = inactivity_penalty
        self.inactivity_threshold = inactivity_threshold
        self.overtrade_penalty = overtrade_penalty

        self._pnl_history: deque = deque(maxlen=_SHARPE_WINDOW)
        self._trade_times: deque = deque(maxlen=_OVERTRADE_WINDOW)
        self._holding_counter: int = 0
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._step: int = 0

    def reset(self, initial_value: float = 1.0) -> None:
        self._pnl_history.clear()
        self._trade_times.clear()
        self._holding_counter = 0
        self._peak_value = initial_value
        self._current_value = initial_value
        self._step = 0

    def compute_reward(
        self,
        pnl: float,
        drawdown: float = 0.0,
        is_trade: bool = False,
        is_holding: bool = False,
        current_value: Optional[float] = None,
    ) -> float:
        """
        Compute composite reward for one environment step.

        Parameters
        ----------
        pnl           : realised P&L this step (fraction of portfolio)
        drawdown      : current drawdown from peak (0 to 1)
        is_trade      : whether a new trade was opened this step
        is_holding    : whether the agent is in an open position and doing nothing
        current_value : current portfolio value (for Sharpe calculation)

        Returns
        -------
        float reward
        """
        self._step += 1

        # ---- Base P&L reward ----
        r_pnl = float(pnl)

        # ---- Drawdown penalty ----
        # Only penalise *incremental* drawdown (new lows)
        r_drawdown = 0.0
        if current_value is not None:
            self._current_value = current_value
            self._peak_value = max(self._peak_value, current_value)
            incremental_dd = max(0.0, (self._peak_value - current_value) / max(self._peak_value, 1e-8))
            if incremental_dd > drawdown * 0.99:  # new drawdown low
                r_drawdown = -self.lambda_drawdown * incremental_dd
        else:
            r_drawdown = -self.lambda_drawdown * max(0.0, drawdown)

        # ---- Sharpe bonus ----
        self._pnl_history.append(pnl)
        r_sharpe = 0.0
        if len(self._pnl_history) >= 10:
            arr = np.array(self._pnl_history)
            std = np.std(arr)
            if std > 1e-8:
                sharpe = float(np.mean(arr) / std * np.sqrt(365))
                r_sharpe = self.lambda_sharpe * np.clip(sharpe, -2.0, 2.0)

        # ---- Inactivity penalty ----
        r_inactivity = 0.0
        if is_holding:
            self._holding_counter += 1
        else:
            self._holding_counter = 0

        if self._holding_counter > self.inactivity_threshold:
            r_inactivity = self.inactivity_penalty

        # ---- Overtrading penalty ----
        r_overtrade = 0.0
        if is_trade:
            self._trade_times.append(self._step)

        if len(self._trade_times) >= _OVERTRADE_MAX_FREQ:
            window_start = self._step - _OVERTRADE_WINDOW
            recent_trades = sum(1 for t in self._trade_times if t >= window_start)
            if recent_trades >= _OVERTRADE_MAX_FREQ:
                r_overtrade = self.overtrade_penalty

        total_reward = r_pnl + r_drawdown + r_sharpe + r_inactivity + r_overtrade
        return float(np.clip(total_reward, -10.0, 10.0))


def compute_reward(
    pnl: float,
    drawdown: float = 0.0,
    sharpe: float = 0.0,
    holding_time: int = 0,
    trade_frequency: int = 0,
    lambda_drawdown: float = _LAMBDA_DRAWDOWN,
    lambda_sharpe: float = _LAMBDA_SHARPE,
) -> float:
    """
    Stateless reward function for single-call use.

    Parameters
    ----------
    pnl            : realised P&L fraction
    drawdown       : current drawdown fraction [0,1]
    sharpe         : rolling Sharpe ratio (can be negative)
    holding_time   : steps held in current position
    trade_frequency: trades in recent window

    Returns
    -------
    float reward
    """
    r_pnl = float(pnl)
    r_drawdown = -lambda_drawdown * float(max(drawdown, 0.0))
    r_sharpe = lambda_sharpe * float(np.clip(sharpe, -3.0, 3.0))

    r_inactivity = 0.0
    if holding_time > _INACTIVITY_THRESHOLD:
        r_inactivity = _INACTIVITY_PENALTY * (holding_time - _INACTIVITY_THRESHOLD)

    r_overtrade = 0.0
    if trade_frequency > _OVERTRADE_MAX_FREQ:
        r_overtrade = _OVERTRADE_PENALTY * (trade_frequency - _OVERTRADE_MAX_FREQ)

    total = r_pnl + r_drawdown + r_sharpe + r_inactivity + r_overtrade
    return float(np.clip(total, -10.0, 10.0))
