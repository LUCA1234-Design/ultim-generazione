"""
rl/multi_objective_rl.py — Multi-Objective Reinforcement Learning.

Optimises trading strategies across three competing objectives:
  1. Total profit (maximise)
  2. Max drawdown (minimise)
  3. Trade frequency (minimise to reduce costs/risk)

The Pareto frontier is computed from a set of candidate solutions,
and a preferred solution is selected based on user-defined weights.

Integration: used by StrategyEvolver and HyperoptEngine to guide
the evolution of trading strategies toward non-dominated solutions.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MultiObjectiveRL")


def compute_pareto_front(solutions: List[Dict]) -> List[Dict]:
    """
    Identify non-dominated solutions (Pareto frontier).

    A solution is Pareto-optimal if no other solution is better
    on at least one objective while no worse on all others.

    Parameters
    ----------
    solutions : list of dicts, each with:
        'profit'     : float (higher is better)
        'drawdown'   : float (lower is better, positive value)
        'frequency'  : float (lower is better, trades per episode)
        ... other fields are preserved

    Returns
    -------
    List of Pareto-optimal solution dicts.
    """
    if not solutions:
        return []

    n = len(solutions)
    # Convert to maximisation: negate drawdown and frequency
    scores = np.array([
        [s.get("profit", 0.0),
         -s.get("drawdown", 0.0),
         -s.get("frequency", 0.0)]
        for s in solutions
    ])

    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all objectives and > i on at least one
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                dominated[i] = True
                break

    pareto = [solutions[i] for i in range(n) if not dominated[i]]
    return pareto


def select_solution(
    solutions: List[Dict],
    preferences: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Select the best solution from a set given preference weights.

    Parameters
    ----------
    solutions   : list of solution dicts (Pareto or full set)
    preferences : dict with weight keys:
        'profit_weight'    (default 0.5)
        'drawdown_weight'  (default 0.3)
        'frequency_weight' (default 0.2)

    Returns
    -------
    The highest-scoring solution dict, or None if empty.
    """
    if not solutions:
        return None

    prefs = preferences or {}
    w_profit = float(prefs.get("profit_weight", 0.5))
    w_dd = float(prefs.get("drawdown_weight", 0.3))
    w_freq = float(prefs.get("frequency_weight", 0.2))

    # Normalise weights
    total_w = w_profit + w_dd + w_freq + 1e-8
    w_profit /= total_w
    w_dd /= total_w
    w_freq /= total_w

    # Extract objective values
    profits = np.array([s.get("profit", 0.0) for s in solutions])
    drawdowns = np.array([s.get("drawdown", 0.0) for s in solutions])
    freqs = np.array([s.get("frequency", 0.0) for s in solutions])

    # Normalise each objective to [0, 1]
    def _norm(arr: np.ndarray) -> np.ndarray:
        r = arr.max() - arr.min()
        if r < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr.min()) / r

    profit_norm = _norm(profits)
    dd_norm = 1.0 - _norm(drawdowns)    # lower is better → invert
    freq_norm = 1.0 - _norm(freqs)      # lower is better → invert

    scores = w_profit * profit_norm + w_dd * dd_norm + w_freq * freq_norm
    best_idx = int(np.argmax(scores))
    return solutions[best_idx]


class MultiObjectiveOptimiser:
    """
    Stateful multi-objective optimiser that maintains a population
    of solutions and updates preference weights based on performance.
    """

    def __init__(self,
                 profit_weight: float = 0.5,
                 drawdown_weight: float = 0.3,
                 frequency_weight: float = 0.2):
        self._weights = {
            "profit_weight": profit_weight,
            "drawdown_weight": drawdown_weight,
            "frequency_weight": frequency_weight,
        }
        self._population: List[Dict] = []
        self._history: List[Tuple[Dict, float]] = []  # (solution, composite_score)

    def add_solution(self, solution: Dict) -> None:
        """Add a new evaluated solution to the population."""
        self._population.append(solution)

    def get_pareto_front(self) -> List[Dict]:
        """Return current Pareto-optimal solutions."""
        return compute_pareto_front(self._population)

    def select_best(self) -> Optional[Dict]:
        """Select best solution from the Pareto front given current weights."""
        pareto = self.get_pareto_front()
        return select_solution(pareto or self._population, self._weights)

    def update_weights(self, performance: Dict) -> None:
        """
        Adapt objective weights based on recent performance.

        If recent drawdown is high → increase drawdown weight.
        If recent profit is low    → increase profit weight.

        Parameters
        ----------
        performance : dict with 'recent_drawdown', 'recent_profit', 'recent_frequency'
        """
        recent_dd = float(performance.get("recent_drawdown", 0.0))
        recent_profit = float(performance.get("recent_profit", 0.0))

        # Adaptive reweighting: steer toward what matters most now
        if recent_dd > 0.15:
            # Drawdown too high: shift weight toward risk management
            self._weights["drawdown_weight"] = min(self._weights["drawdown_weight"] + 0.05, 0.6)
            self._weights["profit_weight"] = max(self._weights["profit_weight"] - 0.025, 0.2)
            self._weights["frequency_weight"] = max(self._weights["frequency_weight"] - 0.025, 0.1)
        elif recent_profit < 0:
            # Losing money: shift weight toward profitability
            self._weights["profit_weight"] = min(self._weights["profit_weight"] + 0.05, 0.7)
            self._weights["drawdown_weight"] = max(self._weights["drawdown_weight"] - 0.025, 0.1)
            self._weights["frequency_weight"] = max(self._weights["frequency_weight"] - 0.025, 0.1)

        # Normalise
        total = sum(self._weights.values())
        for k in self._weights:
            self._weights[k] /= total

        logger.debug(f"MultiObjective weights updated: {self._weights}")

    def get_weights(self) -> Dict:
        return dict(self._weights)

    def clear_population(self) -> None:
        self._population.clear()
