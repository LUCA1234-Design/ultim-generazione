"""
metalearning/maml_adapter.py — Model-Agnostic Meta-Learning Adapter.

Implements a simplified MAML-inspired approach for few-shot adaptation
of agent parameters to new market regimes.

Concept:
  - Each market regime has a set of "adapted parameters" derived from
    a set of meta-parameters via a small gradient update.
  - When a new regime is detected, use the last N_SHOTS trade outcomes
    to quickly adapt the meta-parameters to that regime.
  - Result: all agents receive regime-specific weight adjustments.

Integration (Loop #11): Meta-Learning → Hyperparameters
  EvolutionEngine calls meta_update() after each trade close and
  adapt_to_regime() when a regime change is detected.
"""
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("MAMLAdapter")

_META_LR = 0.01        # meta learning rate (outer loop)
_ADAPT_LR = 0.1        # task adaptation learning rate (inner loop)
_N_SHOTS = 10          # few-shot adaptation: trades per regime
_REGIMES = ["trending", "ranging", "volatile"]
_AGENT_NAMES = ["regime", "pattern", "confluence", "risk", "strategy", "meta",
                "orderflow", "sentiment", "correlation", "contrarian"]


class MAMLAdapter:
    """
    MAML-inspired few-shot adaptation for agent parameter tuning.

    Meta-parameters: a weight vector for each agent (normalised importance).
    Inner loop: gradient step on the current regime's recent outcomes.
    Outer loop: update meta-parameters to generalise across regimes.
    """

    def __init__(self,
                 agent_names: List[str] = None,
                 meta_lr: float = _META_LR,
                 adapt_lr: float = _ADAPT_LR,
                 n_shots: int = _N_SHOTS):
        self._agents = agent_names or _AGENT_NAMES
        self.meta_lr = meta_lr
        self.adapt_lr = adapt_lr
        self.n_shots = n_shots

        n = len(self._agents)

        # Meta-parameters: uniform initialisation
        self._meta_params: np.ndarray = np.ones(n) / n

        # Per-regime adapted parameters
        self._adapted_params: Dict[str, np.ndarray] = {
            r: self._meta_params.copy() for r in _REGIMES
        }

        # Per-regime recent outcomes buffer
        self._regime_buffers: Dict[str, List[Dict]] = {r: [] for r in _REGIMES}

        self._lock = threading.Lock()
        self._update_count = 0

    # ------------------------------------------------------------------

    def meta_update(
        self,
        regime: str,
        agent_weights: Dict[str, float],
        outcomes: List[Dict],
    ) -> None:
        """
        Outer meta-update: incorporate new outcomes from a regime into
        the meta-parameters so future adaptation is faster.

        Parameters
        ----------
        regime        : str — current market regime
        agent_weights : dict mapping agent_name → current weight
        outcomes      : list of recent trade outcomes (each has 'win', 'pnl', 'agents_used')
        """
        if not outcomes:
            return

        regime = regime.lower() if regime else "ranging"
        if regime not in _REGIMES:
            regime = "ranging"

        with self._lock:
            # Store outcomes in regime buffer
            buffer = self._regime_buffers.setdefault(regime, [])
            buffer.extend(outcomes)
            # Keep only last n_shots
            self._regime_buffers[regime] = buffer[-self.n_shots * 5:]

            # Compute gradient signal from outcomes
            gradient = self._compute_gradient(outcomes, agent_weights)

            # Meta outer update (MAML outer loop)
            self._meta_params += self.meta_lr * gradient
            self._meta_params = np.clip(self._meta_params, 0.01, 1.0)
            self._meta_params /= self._meta_params.sum()

            self._update_count += 1
            logger.debug(f"MAMLAdapter meta_update #{self._update_count}, regime={regime}")

    def adapt_to_regime(self, regime: str, n_shots: Optional[int] = None) -> Dict[str, float]:
        """
        Inner loop adaptation: compute regime-specific agent weights
        from recent outcomes in that regime.

        Parameters
        ----------
        regime  : str — target regime
        n_shots : override for number of shots (default: self.n_shots)

        Returns
        -------
        dict mapping agent_name → adapted weight
        """
        n_shots = n_shots or self.n_shots
        regime = regime.lower() if regime else "ranging"
        if regime not in _REGIMES:
            regime = "ranging"

        with self._lock:
            shots = self._regime_buffers.get(regime, [])[-n_shots:]

            if len(shots) < 3:
                # Not enough data: return meta-parameters
                adapted = self._meta_params.copy()
            else:
                # Inner loop: gradient step from regime-specific shots
                adapted = self._meta_params.copy()
                gradient = self._compute_gradient(shots, {})
                adapted += self.adapt_lr * gradient
                adapted = np.clip(adapted, 0.01, 1.0)
                adapted /= adapted.sum()

            self._adapted_params[regime] = adapted

        return self.get_adapted_weights(regime)

    def get_adapted_weights(self, regime: str) -> Dict[str, float]:
        """
        Return adapted weight dict for the specified regime.

        Returns
        -------
        dict mapping agent_name → weight (sums to 1)
        """
        with self._lock:
            params = self._adapted_params.get(
                regime, self._meta_params.copy()
            )
            return {
                name: float(params[i])
                for i, name in enumerate(self._agents)
                if i < len(params)
            }

    def get_meta_weights(self) -> Dict[str, float]:
        """Return current meta-parameters as agent weights."""
        with self._lock:
            return {
                name: float(self._meta_params[i])
                for i, name in enumerate(self._agents)
                if i < len(self._meta_params)
            }

    def get_stats(self) -> Dict:
        return {
            "update_count": self._update_count,
            "meta_weights": self.get_meta_weights(),
            "n_shots_available": {
                r: len(buf) for r, buf in self._regime_buffers.items()
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_gradient(
        self,
        outcomes: List[Dict],
        current_weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Compute gradient signal from trade outcomes.

        Intuition: agents that contributed to wins get positive gradient;
        agents that contributed to losses get negative gradient.
        """
        n = len(self._agents)
        gradient = np.zeros(n)

        for outcome in outcomes:
            win = bool(outcome.get("win", False))
            pnl = float(outcome.get("pnl", 0.0))
            agents_used = outcome.get("agents_used", {})  # agent_name → score used

            signal = 1.0 if win else -1.0
            signal *= (1.0 + abs(pnl))  # scale by magnitude

            for i, agent_name in enumerate(self._agents):
                contribution = float(agents_used.get(agent_name, 0.5))
                gradient[i] += signal * (contribution - 0.5)

        if len(outcomes) > 0:
            gradient /= len(outcomes)

        return gradient
