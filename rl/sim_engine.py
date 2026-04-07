"""
rl/sim_engine.py — Monte Carlo Simulation Engine.

Generates synthetic market episodes for offline RL training via:

1. **Parametric Simulation**: draws returns from estimated distributions
   (e.g., normal, Student-t) with optional regime switching.

2. **Block Bootstrap**: resamples contiguous blocks from historical data
   preserving local autocorrelation structure.

Integration:
  - Used by PPOAgent for offline training when live data is limited.
  - Feeds diverse scenario variations into the RL environment.
  - Results contribute to Monte Carlo validation in backtesting/.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SimEngine")

_DEFAULT_N_STEPS = 200
_BLOCK_SIZE = 20      # block bootstrap block size
_PRICE_INIT = 100.0


def simulate_episode(
    n_steps: int = _DEFAULT_N_STEPS,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV episode from specified parameters.

    Parameters
    ----------
    n_steps : number of candles to generate
    params  : dict with optional keys:
        'mu'          : float — drift (default 0.0)
        'sigma'       : float — daily volatility (default 0.02)
        'vol_of_vol'  : float — volatility of volatility (default 0.2)
        'nu'          : float — Student-t degrees of freedom (default 10)
        'regime'      : str — 'trending' | 'volatile' | 'ranging'
        'init_price'  : float — starting price (default 100.0)

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    """
    p = params or {}
    mu = float(p.get("mu", 0.0))
    sigma = float(p.get("sigma", 0.02))
    vol_of_vol = float(p.get("vol_of_vol", 0.2))
    nu = float(p.get("nu", 10.0))
    regime = str(p.get("regime", "trending"))
    init_price = float(p.get("init_price", _PRICE_INIT))

    # Regime-specific adjustments
    if regime == "volatile":
        sigma *= 2.0
        mu = 0.0
    elif regime == "ranging":
        sigma *= 0.5
        mu = 0.0
    elif regime == "trending":
        mu = abs(mu) if mu >= 0 else 0.001  # slight positive drift

    # Stochastic volatility: sigma follows a log-normal process
    log_sigma = np.log(sigma)
    prices = [init_price]
    volumes = []

    for t in range(n_steps):
        # Update volatility (GARCH-like stochastic vol)
        log_sigma += np.random.normal(0, vol_of_vol * sigma)
        log_sigma = np.clip(log_sigma, np.log(sigma * 0.1), np.log(sigma * 5.0))
        local_sigma = np.exp(log_sigma)

        # Draw return from Student-t with fat tails
        z = float(np.random.standard_t(nu))
        ret = mu / 252.0 + local_sigma / np.sqrt(252) * z
        ret = np.clip(ret, -0.2, 0.2)

        new_price = prices[-1] * (1 + ret)
        new_price = max(new_price, 0.01)
        prices.append(new_price)

        # Simulate intrabar OHLC
        range_pct = abs(np.random.normal(0, local_sigma * 0.5)) + local_sigma * 0.1
        open_p = prices[-2]
        close_p = new_price
        direction = np.sign(ret)
        high_p = max(open_p, close_p) * (1 + range_pct)
        low_p = min(open_p, close_p) * (1 - range_pct)

        # Volume: correlated with range
        vol_base = 1_000_000 * (1 + abs(ret) * 10)
        volume = abs(np.random.normal(vol_base, vol_base * 0.3))
        volumes.append((open_p, high_p, low_p, close_p, volume))

    prices_arr = prices[1:]
    df = pd.DataFrame(volumes, columns=["open", "high", "low", "close", "volume"])
    return df


def bootstrap_episodes(
    historical_data: pd.DataFrame,
    n_episodes: int = 10,
    episode_length: int = _DEFAULT_N_STEPS,
    block_size: int = _BLOCK_SIZE,
) -> List[pd.DataFrame]:
    """
    Generate synthetic episodes via block bootstrap from historical data.

    Block bootstrap preserves local autocorrelation (momentum, mean
    reversion patterns) while shuffling blocks to create diversity.

    Parameters
    ----------
    historical_data : OHLCV DataFrame of historical prices
    n_episodes      : number of synthetic episodes to generate
    episode_length  : number of candles per episode
    block_size      : length of each bootstrap block

    Returns
    -------
    List of pd.DataFrames, each of length episode_length.
    """
    if historical_data is None or len(historical_data) < block_size * 2:
        logger.warning("bootstrap_episodes: insufficient historical data, using simulation")
        return [simulate_episode(episode_length) for _ in range(n_episodes)]

    n_rows = len(historical_data)
    episodes = []

    for _ in range(n_episodes):
        blocks = []
        total = 0
        while total < episode_length:
            # Random starting point for this block
            start = np.random.randint(0, max(n_rows - block_size, 1))
            end = min(start + block_size, n_rows)
            block = historical_data.iloc[start:end].copy()
            blocks.append(block)
            total += len(block)

        episode = pd.concat(blocks, ignore_index=True).iloc[:episode_length]
        episodes.append(episode)

    return episodes


def run_batch_simulation(
    n_sims: int = 100,
    n_steps: int = _DEFAULT_N_STEPS,
    params: Optional[Dict] = None,
    regimes: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Run a batch of Monte Carlo simulations.

    Parameters
    ----------
    n_sims   : number of simulation episodes
    n_steps  : steps per episode
    params   : base parameters (passed to simulate_episode)
    regimes  : if provided, cycle through these regime types

    Returns
    -------
    List of n_sims OHLCV DataFrames.
    """
    _regimes = regimes or ["trending", "ranging", "volatile"]
    episodes = []

    for i in range(n_sims):
        regime = _regimes[i % len(_regimes)]
        ep_params = dict(params or {})
        ep_params["regime"] = regime
        # Randomise parameters for diversity
        ep_params["sigma"] = float(ep_params.get("sigma", 0.02)) * np.random.uniform(0.5, 2.0)
        ep_params["mu"] = np.random.normal(0, 0.001)
        episodes.append(simulate_episode(n_steps, ep_params))

    return episodes


def estimate_params_from_data(df: pd.DataFrame) -> Dict:
    """
    Estimate simulation parameters from historical OHLCV data.

    Returns
    -------
    dict suitable for passing to simulate_episode().
    """
    if df is None or len(df) < 20:
        return {}

    close = df["close"].values.astype(float)
    log_ret = np.diff(np.log(np.maximum(close, 1e-8)))

    mu = float(np.mean(log_ret)) * 252
    sigma = float(np.std(log_ret)) * np.sqrt(252)

    # Estimate Student-t degrees of freedom via moment matching
    # For t-distribution: kurtosis = 6/(nu-4) for nu>4
    excess_kurt = float(pd.Series(log_ret).kurtosis())
    if excess_kurt > 0.5:
        nu = max(4.5, 6.0 / excess_kurt + 4.0)
    else:
        nu = 30.0  # approximately normal

    return {
        "mu": mu,
        "sigma": sigma,
        "nu": nu,
        "init_price": float(close[-1]),
    }
