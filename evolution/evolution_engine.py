"""
Evolution Engine for V17 — Central Brain.

Closes all broken feedback loops:
  Loop #1: MetaAgent → DecisionFusion weight updates
  Loop #2: PerformanceTracker → RiskAgent win rates
  Loop #3: Auto-tunes FUSION_THRESHOLD via optimal_params DB
  Loop #4: Pattern threshold adaptation
  Loop #5: Strategy evolution (delegated to StrategyEvolver)
  Loop #6: MetaAgent state persistence (save/load)
  Loop #7: Confluence TF weight learning (delegated to ConfluenceAdapter)

Usage in main.py:
    engine = EvolutionEngine(meta, fusion, risk, strategy, confluence, tracker, pattern)
    engine.startup()                                # on boot
    engine.on_trade_close(closed_pos, ctx)          # in position monitor
    engine.tick()                                   # every 30 min in main loop
    engine.shutdown()                               # on Ctrl+C
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from memory import experience_db
from evolution.strategy_evolver import StrategyEvolver
from evolution.confluence_adapter import ConfluenceAdapter

logger = logging.getLogger("EvolutionEngine")

# Tuning constants
_TUNE_INTERVAL_SEC = 1800   # 30 min between auto-tune runs
_SAVE_INTERVAL_SEC = 900    # 15 min between state saves
_MIN_COMPLETED = 10         # minimum completed trades before tuning
_THRESHOLD_STEP_UP = 0.02   # raise threshold by this when win-rate is too low
_THRESHOLD_STEP_DOWN = 0.01 # lower threshold by this when win-rate is excellent
_THRESHOLD_LOW_WR = 0.50    # win-rate below this triggers a raise
_THRESHOLD_HIGH_WR = 0.65   # win-rate above this triggers a lower
_THRESHOLD_MIN = 0.45
_THRESHOLD_MAX = 0.85
_DRAWDOWN_WARN = 0.08       # 8% drawdown → raise threshold
_DRAWDOWN_CRITICAL = 0.15   # 15% → safe mode


class EvolutionEngine:
    """Central orchestrator that wires all V17 feedback loops."""

    def __init__(self,
        meta_agent,
        fusion,
        risk_agent,
        strategy_agent,
        confluence_agent,
        tracker,
        pattern_agent=None,
    ):
        self._meta = meta_agent
        self._fusion = fusion
        self._risk = risk_agent
        self._tracker = tracker
        self._pattern = pattern_agent

        # Sub-engines for loop #5 and #7
        self._strategy_evolver = StrategyEvolver(strategy_agent)
        self._confluence_adapter = ConfluenceAdapter(confluence_agent)

        self._last_tune: float = 0.0
        self._last_save: float = 0.0

        # Thread safety lock
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Restore previous learning state — call once on bot boot."""

        # Loop #6: restore MetaAgent weights from disk
        try:
            loaded = self._meta.load_state()
            if loaded:
                logger.info("🧠 EvolutionEngine: MetaAgent state restored from disk")
            else:
                logger.info("🧠 EvolutionEngine: no saved MetaAgent state, starting fresh")
        except Exception as exc:
            logger.error(f"EvolutionEngine.startup load_state error: {exc}")

        # Loop #3: restore auto-tuned fusion threshold from DB
        try:
            saved_threshold = experience_db.get_param("fusion_threshold")
            if saved_threshold is not None:
                clamped = float(np.clip(float(saved_threshold), _THRESHOLD_MIN, _THRESHOLD_MAX))
                self._fusion._threshold = clamped
                logger.info(f"🔧 EvolutionEngine: restored fusion_threshold={{clamped:.3f}}")
        except Exception as exc:
            logger.error(f"EvolutionEngine.startup restore_threshold error: {exc}")

        # Loop #7: restore confluence TF performance counters from DB
        try:
            tf_data = experience_db.get_param("confluence_tf_performance")
            if tf_data and isinstance(tf_data, dict):
                self._confluence_adapter.load_state(tf_data)
                logger.info("🌊 EvolutionEngine: confluence TF performance data restored")
        except Exception as exc:
            logger.debug(f"EvolutionEngine.startup TF data restore error: {exc}")

        # Loop #5: restore strategy evolver trade count
        try:
            saved_count = experience_db.get_param("strategy_evolver_trade_count")
            if saved_count is not None:
                self._strategy_evolver.trade_count = int(saved_count)
                logger.info(
                    f"🧬 EvolutionEngine: restored strategy_evolver trade_count="
                    f"{{self._strategy_evolver.trade_count}}"
                )
        except Exception as exc:
            logger.debug(f"EvolutionEngine.startup strategy trade_count restore error: {exc}")

    def on_trade_close(
        self,
        closed_position: Any,
        decision_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify the evolution engine that a position just closed.

        Parameters
        ----------
        closed_position : Position object with at minimum ``decision_id``,
                          ``strategy``, and ``pnl`` attributes.
        decision_ctx    : the dict stored in decision_context[decision_id],
                          including ``agent_results``.
        """
        with self._lock:
            was_profitable = (getattr(closed_position, "pnl", None) or 0.0) > 0
            ctx = decision_ctx or {}

            # Loop #5: strategy evolution (accumulate + possibly prune/mutate)
            try:
                strategy_name = getattr(closed_position, "strategy", None) or ""
                self._strategy_evolver.record_trade(strategy_name, was_profitable)
            except Exception as exc:
                logger.error(f"EvolutionEngine strategy_evolver error: {exc}")

            # Loop #7: confluence TF tracking
            try:
                agent_results = ctx.get("agent_results", {})
                confluence_result = agent_results.get("confluence")
                if confluence_result and hasattr(confluence_result, "metadata"):
                    tf_scores: Dict[str, float] = confluence_result.metadata.get("tf_scores", {})
                    self._confluence_adapter.record_trade(tf_scores, was_profitable)
            except Exception as exc:
                logger.debug(f"EvolutionEngine confluence_adapter error: {exc}")

            # Loop #4: pattern threshold adaptation
            try:
                if self._pattern is not None:
                    agent_results = ctx.get("agent_results", {})
                    pattern_result = agent_results.get("pattern")
                    interval = ctx.get("interval", "1h")
                    self._pattern.update_threshold(interval, was_profitable)
                    if pattern_result and hasattr(pattern_result, "details"):
                        # details is a list of human-readable strings like
                        # "squeeze_active(5b)", "RSI_active(38)", etc.
                        # Extract the tag portion (before any parenthesis)
                        # so that record_pattern_outcome receives clean
                        # pattern names instead of raw detail strings.
                        raw_details = list(pattern_result.details)
                        pattern_tags = [
                            d.split("(")[0] for d in raw_details if d
                        ]
                        if pattern_tags and hasattr(self._pattern, "record_pattern_outcome"):
                            self._pattern.record_pattern_outcome(pattern_tags, was_profitable)
            except Exception as exc:
                logger.error(f"EvolutionEngine pattern_threshold error: {exc}")

    def tick(self) -> None:
        """Periodic evolution step — call every ~30 minutes from main loop.

        Designed to be non-blocking: all heavy operations use cached DB data.
        """
        with self._lock:
            now = time.time()

            # Loop #1: push updated agent weights to DecisionFusion
            try:
                weight_map = self._meta.adjust_weights()
                if weight_map:
                    self._fusion.update_weights(weight_map)
                    logger.info(f"🎚️ EvolutionEngine: agent weights updated → {{weight_map}}")
            except Exception as exc:
                logger.error(f"EvolutionEngine weight_update error: {exc}")

            # Loop #2: push real win rates into RiskAgent
            try:
                self._tracker.update_risk_agent_win_rates(self._risk)
            except Exception as exc:
                logger.error(f"EvolutionEngine win_rate_update error: {exc}")

            # Loop #3: auto-tune FUSION_THRESHOLD
            if now - self._last_tune >= _TUNE_INTERVAL_SEC:
                self._auto_tune_params()
                self._last_tune = now

            # Loop #6: periodically persist MetaAgent state
            if now - self._last_save >= _SAVE_INTERVAL_SEC:
                self._save_state()
                self._last_save = now

            # Loop #7: adapt confluence TF weights
            try:
                self._confluence_adapter.maybe_adapt()
            except Exception as exc:
                logger.error(f"EvolutionEngine confluence_adapt error: {exc}")

            # Drawdown circuit breaker
            self._check_drawdown()

    def shutdown(self) -> None:
        """Persist all state — call on graceful shutdown."""
        try:
            self._meta.save_state()
            experience_db.save_param(
                "fusion_threshold", self._fusion._threshold, "shutdown"
            )
            experience_db.save_param(
                "confluence_tf_performance",
                self._confluence_adapter.dump_state(),
                "shutdown",
            )
            experience_db.save_param(
                "strategy_evolver_trade_count",
                self._strategy_evolver.trade_count,
                "shutdown",
            )
            logger.info("💾 EvolutionEngine: state saved on shutdown")
        except Exception as exc:
            logger.error(f"EvolutionEngine.shutdown error: {exc}")

    def get_report(self) -> Dict[str, Any]:
        """Return a human-readable summary of the current evolution state."""
        return {
            "fusion_threshold": self._fusion._threshold,
            "strategy_trade_count": self._strategy_evolver.trade_count,
            "tf_performance": self._confluence_adapter.get_performance_summary(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_drawdown(self) -> None:
        """Monitor drawdown and activate circuit breaker if needed."""
        try:
            stats = self._tracker.get_summary()
            max_dd = abs(stats.get("max_drawdown", 0.0))
            current_threshold = self._fusion._threshold

            if max_dd >= _DRAWDOWN_CRITICAL:
                # Safe mode: only very high quality signals
                safe_threshold = float(np.clip(current_threshold + 0.15, 0.50, 0.95))
                if self._fusion._threshold < safe_threshold:
                    self._fusion._threshold = safe_threshold
                    logger.warning(
                        f"🔴 DRAWDOWN CRITICAL ({{max_dd:.1%}}) → safe mode, "
                        f"threshold={{safe_threshold:.3f}}"
                    )
            elif max_dd >= _DRAWDOWN_WARN:
                warn_threshold = float(np.clip(current_threshold + 0.05, 0.30, 0.90))
                if self._fusion._threshold < warn_threshold:
                    self._fusion._threshold = warn_threshold
                    logger.warning(
                        f"🟡 DRAWDOWN WARNING ({{max_dd:.1%}}) → threshold raised to "
                        f"{{warn_threshold:.3f}}"
                    )
        except Exception as exc:
            logger.debug(f"_check_drawdown error: {exc}")

    def _auto_tune_params(self) -> None:
        """Adjust FUSION_THRESHOLD based on recent completed trade outcomes."""
        try:
            recent = experience_db.get_recent_decisions(limit=50)
            completed = [
                d for d in recent
                if d.get("outcome") is not None and d.get("pnl") is not None
            ]
            if len(completed) < _MIN_COMPLETED:
                return

            wins = sum(1 for d in completed if (d.get("pnl") or 0) > 0)
            win_rate = wins / len(completed)
            current = self._fusion._threshold
            # Default: no change
            new_threshold = current

            if win_rate < _THRESHOLD_LOW_WR:
                new_threshold = float(np.clip(current + _THRESHOLD_STEP_UP,
                                              _THRESHOLD_MIN, _THRESHOLD_MAX))
            elif win_rate > _THRESHOLD_HIGH_WR:
                new_threshold = float(np.clip(current - _THRESHOLD_STEP_DOWN,
                                              _THRESHOLD_MIN, _THRESHOLD_MAX))

            if new_threshold != current:
                logger.info(
                    f"🔧 Auto-tune: win_rate={{win_rate:.1%}} → "
                    f"threshold {{current:.3f}} → {{new_threshold:.3f}}"
                )

            self._fusion._threshold = new_threshold
            experience_db.save_param("fusion_threshold", new_threshold, "auto_tune")
        except Exception as exc:
            logger.error(f"_auto_tune_params error: {exc}")

    def _save_state(self) -> None:
        """Persist MetaAgent state and current tuned parameters."""
        try:
            self._meta.save_state()
            experience_db.save_param(
                "fusion_threshold", self._fusion._threshold, "periodic"
            )
            experience_db.save_param(
                "confluence_tf_performance",
                self._confluence_adapter.dump_state(),
                "periodic",
            )
            experience_db.save_param(
                "strategy_evolver_trade_count",
                self._strategy_evolver.trade_count,
                "periodic",
            )
            logger.info("💾 EvolutionEngine: periodic state save complete")
        except Exception as exc:
            logger.error(f"_save_state error: {exc}")
