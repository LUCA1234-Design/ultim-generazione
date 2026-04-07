"""
coordination/state_machine.py — Global Finite State Machine.

The system-wide FSM controls which modules are active and with what
parameters at each stage of the trading lifecycle.

States:
  INITIALIZING → system booting up, loading state, fitting models
  TRAINING     → accumulating trades, reduced thresholds, all learning enabled
  LEARNING     → transition mode: moderate thresholds, meta-learning active
  SNIPER       → fully operational: high thresholds, maximum precision
  SAFE_MODE    → drawdown/risk alert: only close positions, no new trades
  KILLED       → Level 5 kill switch active: no actions allowed
  RECOVERY     → transitioning from KILLED/SAFE_MODE back to SNIPER

Transitions are event-driven (e.g., TRAINING→SNIPER after 200 trades).
Each state has a configuration dict that overrides relevant settings.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("StateMachine")

# ---- State names ------------------------------------------------------------

STATE_INITIALIZING = "INITIALIZING"
STATE_TRAINING = "TRAINING"
STATE_LEARNING = "LEARNING"
STATE_SNIPER = "SNIPER"
STATE_SAFE_MODE = "SAFE_MODE"
STATE_KILLED = "KILLED"
STATE_RECOVERY = "RECOVERY"

ALL_STATES = [
    STATE_INITIALIZING, STATE_TRAINING, STATE_LEARNING,
    STATE_SNIPER, STATE_SAFE_MODE, STATE_KILLED, STATE_RECOVERY,
]

# ---- State configurations ---------------------------------------------------

_STATE_CONFIGS: Dict[str, Dict] = {
    STATE_INITIALIZING: {
        "allow_trading": False,
        "allow_learning": True,
        "fusion_threshold_override": None,
        "description": "System starting up and loading state",
    },
    STATE_TRAINING: {
        "allow_trading": True,
        "allow_learning": True,
        "fusion_threshold_override": 0.35,
        "min_confirmations": 3,
        "description": "Accumulating trades for learning algorithms",
    },
    STATE_LEARNING: {
        "allow_trading": True,
        "allow_learning": True,
        "fusion_threshold_override": 0.45,
        "min_confirmations": 3,
        "description": "Meta-learning actively adapting parameters",
    },
    STATE_SNIPER: {
        "allow_trading": True,
        "allow_learning": True,
        "fusion_threshold_override": None,  # use auto-tuned threshold
        "min_confirmations": 4,
        "description": "Full sniper mode: high precision, all systems active",
    },
    STATE_SAFE_MODE: {
        "allow_trading": False,  # no new positions
        "allow_close": True,     # can close existing positions
        "allow_learning": True,
        "fusion_threshold_override": 0.75,
        "description": "Risk circuit breaker: close-only mode",
    },
    STATE_KILLED: {
        "allow_trading": False,
        "allow_close": False,
        "allow_learning": False,
        "fusion_threshold_override": 1.0,
        "description": "Level 5 kill switch active: all actions halted",
    },
    STATE_RECOVERY: {
        "allow_trading": False,
        "allow_close": True,
        "allow_learning": True,
        "fusion_threshold_override": 0.65,
        "description": "Recovering from kill switch or safe mode",
    },
}

# ---- Default transition table -----------------------------------------------
# (from_state, event) → to_state

_DEFAULT_TRANSITIONS: List[Tuple[str, str, str]] = [
    (STATE_INITIALIZING, "boot_complete", STATE_TRAINING),
    (STATE_TRAINING, "training_complete", STATE_SNIPER),
    (STATE_TRAINING, "drawdown_critical", STATE_SAFE_MODE),
    (STATE_LEARNING, "learning_stable", STATE_SNIPER),
    (STATE_LEARNING, "drawdown_critical", STATE_SAFE_MODE),
    (STATE_SNIPER, "drawdown_critical", STATE_SAFE_MODE),
    (STATE_SNIPER, "kill_level5", STATE_KILLED),
    (STATE_SNIPER, "drift_detected", STATE_LEARNING),
    (STATE_SAFE_MODE, "recovery_complete", STATE_RECOVERY),
    (STATE_SAFE_MODE, "kill_level5", STATE_KILLED),
    (STATE_KILLED, "kill_recovered", STATE_RECOVERY),
    (STATE_RECOVERY, "recovery_stable", STATE_SNIPER),
    (STATE_RECOVERY, "drawdown_critical", STATE_SAFE_MODE),
]


class StateMachine:
    """
    Thread-safe global finite state machine for the V18 trading system.

    Usage:
        fsm = StateMachine()
        fsm.transition("boot_complete")
        config = fsm.get_state_config()
        print(fsm.current_state())
    """

    def __init__(self, initial_state: str = STATE_INITIALIZING):
        self._current = initial_state
        self._transitions: Dict[Tuple[str, str], str] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._history: List[Tuple[float, str, str, str]] = []  # (time, from, event, to)
        self._lock = threading.RLock()

        # Register default transitions
        for from_s, event, to_s in _DEFAULT_TRANSITIONS:
            self.register_transition(from_s, event, to_s)

    # ------------------------------------------------------------------

    def current_state(self) -> str:
        """Return current state name."""
        with self._lock:
            return self._current

    def transition(self, event: str) -> Optional[str]:
        """
        Attempt a state transition based on an event.

        Parameters
        ----------
        event : str — event name (e.g. 'boot_complete', 'kill_level5')

        Returns
        -------
        New state if transition occurred, None if no valid transition.
        """
        with self._lock:
            key = (self._current, event)
            new_state = self._transitions.get(key)

            if new_state is None:
                logger.debug(
                    f"StateMachine: no transition from '{self._current}' on '{event}'"
                )
                return None

            old_state = self._current
            self._current = new_state
            self._history.append((time.time(), old_state, event, new_state))

            logger.info(
                f"🔄 StateMachine: {old_state} --[{event}]--> {new_state}"
            )

            # Fire callbacks
            for cb in self._callbacks.get(new_state, []):
                try:
                    cb(old_state, event, new_state)
                except Exception as exc:
                    logger.error(f"StateMachine callback error: {exc}")

            return new_state

    def register_transition(
        self,
        from_state: str,
        event: str,
        to_state: str,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Register a state transition.

        Parameters
        ----------
        from_state : source state
        event      : triggering event
        to_state   : destination state
        callback   : optional callable(from, event, to) called on transition
        """
        with self._lock:
            self._transitions[(from_state, event)] = to_state
            if callback:
                if to_state not in self._callbacks:
                    self._callbacks[to_state] = []
                self._callbacks[to_state].append(callback)

    def get_allowed_actions(self) -> Dict[str, bool]:
        """Return allowed actions in the current state."""
        with self._lock:
            config = _STATE_CONFIGS.get(self._current, {})
            return {
                "allow_trading": bool(config.get("allow_trading", True)),
                "allow_close": bool(config.get("allow_close", config.get("allow_trading", True))),
                "allow_learning": bool(config.get("allow_learning", True)),
            }

    def get_state_config(self) -> Dict[str, Any]:
        """Return full configuration for the current state."""
        with self._lock:
            return dict(_STATE_CONFIGS.get(self._current, {}))

    def get_history(self, last_n: int = 10) -> List[Dict]:
        """Return the last N state transitions."""
        with self._lock:
            recent = self._history[-last_n:]
            return [
                {"time": t, "from": f, "event": e, "to": to}
                for t, f, e, to in recent
            ]

    def is_trading_allowed(self) -> bool:
        """Convenience: True if new trades can be opened."""
        return self.get_allowed_actions()["allow_trading"]

    def is_closing_allowed(self) -> bool:
        """Convenience: True if positions can be closed."""
        return self.get_allowed_actions()["allow_close"]

    def get_fusion_threshold_override(self) -> Optional[float]:
        """Return the threshold override for the current state, or None."""
        with self._lock:
            config = _STATE_CONFIGS.get(self._current, {})
            return config.get("fusion_threshold_override")
