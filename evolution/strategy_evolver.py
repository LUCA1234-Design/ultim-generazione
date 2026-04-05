"""
Strategy Evolver for V17.

Handles Loop #5: genetic strategy evolution.
Every EVOLVE_EVERY closed trades the worst-performing strategy (below
MIN_WIN_RATE with enough samples) is pruned, and if the pool is below
MAX_STRATEGIES a mutation of the best strategy is added in its place.

The heavy lifting lives in StrategyAgent.prune_and_evolve(); this class
is responsible only for the *timing* (counting trades) and logging.
"""
import logging
from typing import Optional

logger = logging.getLogger("StrategyEvolver")

_EVOLVE_EVERY = 20      # trigger evolution every N closed trades
_MIN_SAMPLES = 10       # minimum recorded outcomes before pruning eligibility
_MIN_WIN_RATE = 0.35    # strategies below this win-rate are candidates for pruning
_MAX_STRATEGIES = 10    # cap on total strategies in the pool


class StrategyEvolver:
    """Counts closed trades and triggers strategy evolution at the right cadence."""

    def __init__(self, strategy_agent, evolve_every: int = _EVOLVE_EVERY):
        self._strategy = strategy_agent
        self._evolve_every = evolve_every
        self._trade_count = 0

    @property
    def trade_count(self) -> int:
        return self._trade_count

    def record_trade(self, strategy_name: str, was_profitable: bool) -> None:
        """Register a closed trade outcome and trigger evolution if due.

        Parameters
        ----------
        strategy_name  : name of the strategy that generated the signal
        was_profitable : whether the trade closed with positive P&L
        """
        self._trade_count += 1

        if self._trade_count % self._evolve_every == 0:
            self._run_evolution()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_evolution(self) -> None:
        """Call prune_and_evolve on the StrategyAgent and log results."""
        try:
            changes = self._strategy.prune_and_evolve(
                min_samples=_MIN_SAMPLES,
                min_win_rate=_MIN_WIN_RATE,
                max_strategies=_MAX_STRATEGIES,
            )
            if changes:
                logger.info(
                    f"🧬 StrategyEvolver (trade #{self._trade_count}): "
                    + " | ".join(changes)
                )
            else:
                logger.debug(
                    f"🧬 StrategyEvolver (trade #{self._trade_count}): "
                    "pool healthy, no changes"
                )
        except Exception as exc:
            logger.error(f"StrategyEvolver._run_evolution error: {exc}")
