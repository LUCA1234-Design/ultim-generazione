"""
metalearning/learning_rate_scheduler.py — Adaptive Learning Rate Scheduler.

Assigns an independent learning rate to each feedback loop in the
EvolutionEngine. Each loop's learning rate adapts based on recent
performance using cosine annealing with warm restarts (SGDR).

Concept:
  - Each feedback loop has a "temperature" — how aggressively it
    updates its parameters.
  - During good performance: gradual cooling (lower LR → stable)
  - When concept drift is detected: warm restart (higher LR → fast adaptation)
  - Cosine schedule between restarts for smooth transitions

Integration (Loop #12): Concept Drift → Re-training
  When concept_drift_detector fires, all schedulers are reset()
  to trigger warm restarts and accelerate re-adaptation.
"""
import logging
import math
import threading
from typing import Dict, Optional

logger = logging.getLogger("LRScheduler")

# Default parameters
_LR_INIT = 0.01
_LR_MIN = 0.001
_LR_MAX = 0.1
_T0 = 50          # initial restart period (steps)
_T_MULT = 2       # restart period multiplier
_WARMUP_STEPS = 5


class CosineAnnealingScheduler:
    """
    Cosine annealing with warm restarts (SGDR) for a single learning rate.

    LR(t) = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * t/T))
    After T steps: restart with T *= T_mult
    """

    def __init__(self,
                 lr_init: float = _LR_INIT,
                 lr_min: float = _LR_MIN,
                 lr_max: float = _LR_MAX,
                 t0: int = _T0,
                 t_mult: int = _T_MULT):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_init = lr_init
        self._t0 = t0
        self._t_mult = t_mult
        self._t_cur = 0
        self._t_period = t0
        self._current_lr = lr_init
        self._n_restarts = 0

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self._current_lr

    def step(self, performance: Optional[float] = None) -> float:
        """
        Advance one step and return new LR.

        Parameters
        ----------
        performance : optional signal in [-1, 1]:
            > 0 = good performance (dampen LR)
            < 0 = bad performance (raise LR slightly)

        Returns
        -------
        float — new learning rate
        """
        self._t_cur += 1

        # Cosine annealing
        cos_val = math.cos(math.pi * self._t_cur / self._t_period)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + cos_val)

        # Performance adjustment
        if performance is not None:
            if performance > 0.2:
                lr *= 0.9  # doing well → slow down updates
            elif performance < -0.2:
                lr *= 1.1  # struggling → try larger steps
            lr = max(self.lr_min, min(lr, self.lr_max))

        self._current_lr = lr

        # Check restart
        if self._t_cur >= self._t_period:
            self._restart()

        return lr

    def reset(self) -> float:
        """Warm restart: reset to maximum LR."""
        self._t_cur = 0
        self._t_period = self._t0
        self._n_restarts += 1
        self._current_lr = self.lr_max
        logger.debug(f"CosineAnnealingScheduler warm restart #{self._n_restarts}")
        return self._current_lr

    def _restart(self) -> None:
        self._t_cur = 0
        self._t_period = self._t_period * self._t_mult
        self._n_restarts += 1
        self._current_lr = self.lr_max

    def get_state(self) -> Dict:
        return {
            "current_lr": self._current_lr,
            "t_cur": self._t_cur,
            "t_period": self._t_period,
            "n_restarts": self._n_restarts,
        }


class LearningRateSchedulerManager:
    """
    Manages independent learning rate schedulers for each feedback loop.

    Usage:
        manager = LearningRateSchedulerManager()
        lr = manager.get_lr("loop_1_meta_agent")
        manager.step("loop_1_meta_agent", performance=0.6)
        manager.reset("loop_1_meta_agent")  # on concept drift
    """

    # Default loop names matching EvolutionEngine
    _DEFAULT_LOOPS = [
        "loop_1_meta_agent",
        "loop_2_risk_winrate",
        "loop_3_threshold_tune",
        "loop_4_pattern_adapt",
        "loop_5_strategy_evolver",
        "loop_6_meta_persist",
        "loop_7_confluence_adapt",
        "loop_8_rl_execution",
        "loop_9_hmm_regime",
        "loop_10_bayesian",
        "loop_11_maml",
        "loop_12_concept_drift",
        "loop_13_var_kill",
        "loop_14_contrarian_fusion",
        "loop_15_backtest_validator",
    ]

    def __init__(self, loops: Optional[list] = None):
        self._loops = loops or self._DEFAULT_LOOPS
        self._schedulers: Dict[str, CosineAnnealingScheduler] = {}
        self._lock = threading.Lock()
        self._init_schedulers()

    def _init_schedulers(self) -> None:
        for loop in self._loops:
            self._schedulers[loop] = CosineAnnealingScheduler()

    def get_lr(self, loop_name: str) -> float:
        """Get current learning rate for a specific loop."""
        with self._lock:
            sched = self._schedulers.get(loop_name)
            if sched is None:
                # Auto-create scheduler for unknown loops
                self._schedulers[loop_name] = CosineAnnealingScheduler()
                sched = self._schedulers[loop_name]
            return sched.get_lr()

    def step(self, loop_name: str, performance: Optional[float] = None) -> float:
        """Advance one step for a specific loop and return new LR."""
        with self._lock:
            sched = self._schedulers.get(loop_name)
            if sched is None:
                self._schedulers[loop_name] = CosineAnnealingScheduler()
                sched = self._schedulers[loop_name]
            return sched.step(performance)

    def reset(self, loop_name: Optional[str] = None) -> None:
        """
        Trigger warm restart for one or all loops.

        Parameters
        ----------
        loop_name : if None, reset ALL loops (e.g., on major concept drift)
        """
        with self._lock:
            if loop_name is None:
                for sched in self._schedulers.values():
                    sched.reset()
                logger.info("LRScheduler: warm restart for ALL loops")
            else:
                sched = self._schedulers.get(loop_name)
                if sched:
                    sched.reset()
                    logger.debug(f"LRScheduler: warm restart for {loop_name}")

    def get_all_lrs(self) -> Dict[str, float]:
        """Return current LR for all loops."""
        with self._lock:
            return {name: s.get_lr() for name, s in self._schedulers.items()}

    def get_state(self) -> Dict:
        """Return full state for all schedulers (for persistence)."""
        with self._lock:
            return {name: s.get_state() for name, s in self._schedulers.items()}
