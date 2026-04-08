"""
metalearning/hyperopt_engine.py — Bayesian Hyperparameter Optimisation.

Implements a simplified TPE (Tree-structured Parzen Estimator) for
black-box optimisation of trading system hyperparameters.

TPE models the distribution of good configurations (l(x)) and bad
ones (g(x)) separately using KDE, and samples from l(x)/g(x).

Parameters optimised:
  - FUSION_THRESHOLD (decision gate threshold)
  - Agent weights
  - ATR multipliers for SL/TP
  - Cooldown periods

Integration (Loop #11): Meta-Learning → Hyperparameters
  HyperoptEngine.suggest_params() proposes new configurations.
  HyperoptEngine.report_result() stores outcomes.
  EvolutionEngine periodically applies the best params found.
"""
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("HyperoptEngine")

# Default parameter search space
_DEFAULT_SPACE = {
    "fusion_threshold": (0.30, 0.80),
    "pattern_weight": (0.10, 0.60),
    "confluence_weight": (0.10, 0.60),
    "regime_weight": (0.05, 0.30),
    "risk_weight": (0.05, 0.30),
    "atr_sl_mult": (1.0, 3.0),
    "atr_tp_mult": (1.5, 5.0),
    "cooldown_15m": (300, 1800),
    "cooldown_1h": (1200, 7200),
}

_N_STARTUP = 10       # random samples before fitting KDE
_GAMMA = 0.25         # fraction of best configs to use as "good"
_KDE_BW = 0.2         # KDE bandwidth parameter


class TPEHyperoptEngine:
    """
    Tree-structured Parzen Estimator for hyperparameter optimisation.

    Usage:
        engine = TPEHyperoptEngine()
        params = engine.suggest_params()
        # ... evaluate params ...
        engine.report_result(params, score)
        best = engine.get_best_params()
    """

    def __init__(self,
                 search_space: Optional[Dict[str, Tuple[float, float]]] = None,
                 gamma: float = _GAMMA,
                 budget: int = 200):
        self.space = search_space or _DEFAULT_SPACE
        self.gamma = gamma
        self.budget = budget
        self._history: List[Tuple[Dict, float]] = []
        self._n_evals = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def suggest_params(self) -> Dict[str, Any]:
        """
        Suggest a new hyperparameter configuration.

        During startup phase: uniform random sampling.
        After startup: TPE-guided sampling.

        Returns
        -------
        dict mapping param_name → value
        """
        with self._lock:
            if self._n_evals < _N_STARTUP:
                return self._random_sample()
            return self._tpe_sample()

    def report_result(self, params: Dict, score: float) -> None:
        """
        Record the result of evaluating a configuration.

        Parameters
        ----------
        params : parameter dict (from suggest_params)
        score  : evaluation score (higher = better, e.g. Sharpe ratio)
        """
        with self._lock:
            self._history.append((params.copy(), float(score)))
            self._n_evals += 1
            logger.debug(f"HyperoptEngine: eval #{self._n_evals}, score={score:.4f}")

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Return the configuration with the highest observed score."""
        with self._lock:
            if not self._history:
                return None
            best_params, best_score = max(self._history, key=lambda x: x[1])
            logger.debug(f"HyperoptEngine: best score={best_score:.4f}")
            return best_params.copy()

    def get_optimization_history(self) -> List[Tuple[Dict, float]]:
        """Return list of (params, score) tuples in order of evaluation."""
        with self._lock:
            return [(p.copy(), s) for p, s in self._history]

    def get_top_k(self, k: int = 5) -> List[Dict]:
        """Return top-k configurations sorted by score descending."""
        with self._lock:
            sorted_h = sorted(self._history, key=lambda x: x[1], reverse=True)
            return [p.copy() for p, _ in sorted_h[:k]]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_sample(self) -> Dict[str, Any]:
        """Uniform random sample from search space."""
        return {
            name: float(np.random.uniform(lo, hi))
            for name, (lo, hi) in self.space.items()
        }

    def _tpe_sample(self) -> Dict[str, Any]:
        """TPE-guided sample."""
        scores = np.array([s for _, s in self._history])
        threshold = np.percentile(scores, 100 * (1 - self.gamma))

        good_configs = [p for p, s in self._history if s >= threshold]
        bad_configs = [p for p, s in self._history if s < threshold]

        if len(good_configs) < 3:
            return self._random_sample()

        suggestion: Dict[str, Any] = {}

        for name, (lo, hi) in self.space.items():
            good_vals = np.array([c.get(name, (lo + hi) / 2) for c in good_configs])
            bad_vals = np.array([c.get(name, (lo + hi) / 2) for c in bad_configs])

            # Normalise to [0, 1]
            rng = hi - lo
            if rng < 1e-10:
                suggestion[name] = float((lo + hi) / 2)
                continue

            good_norm = (good_vals - lo) / rng
            bad_norm = (bad_vals - lo) / rng if len(bad_vals) > 0 else np.array([0.5])

            # Sample candidates from good distribution
            candidates_norm = np.linspace(0, 1, 25)

            # KDE densities
            good_density = self._kde_density(candidates_norm, good_norm, _KDE_BW)
            bad_density = self._kde_density(candidates_norm, bad_norm, _KDE_BW)

            # EI proxy: l(x) / g(x)
            ratio = good_density / (bad_density + 1e-10)

            # Add prior (uniform noise) for exploration
            ratio += np.random.uniform(0, 0.1, len(ratio))

            best_candidate_norm = float(candidates_norm[np.argmax(ratio)])
            best_candidate = lo + best_candidate_norm * rng

            suggestion[name] = float(np.clip(best_candidate, lo, hi))

        return suggestion

    @staticmethod
    def _kde_density(x: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
        """Evaluate Gaussian KDE density at points x."""
        if len(data) == 0:
            return np.ones(len(x))
        densities = np.zeros(len(x))
        for xi in data:
            densities += np.exp(-0.5 * ((x - xi) / bandwidth) ** 2)
        densities /= (len(data) * bandwidth * np.sqrt(2 * np.pi) + 1e-10)
        return densities + 1e-10
