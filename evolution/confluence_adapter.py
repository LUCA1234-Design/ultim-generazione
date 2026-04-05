"""
Confluence Adapter for V17.

Handles Loop #7: dynamic learning of per-timeframe weights.

After each closed trade the adapter records which timeframes were
"active" (score > SCORE_THRESHOLD) and whether the trade was profitable.
When enough samples accumulate the adapter rebalances TF_WEIGHTS in the
ConfluenceAgent so timeframes that correlate with winning trades receive
higher weight.

All weights are clipped and normalised so the adapted values stay within
[TF_WEIGHT_MIN, TF_WEIGHT_MAX] and always sum to 1.0.
"""
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("ConfluenceAdapter")

_TF_ORDER = ["15m", "1h", "4h"]
_SCORE_THRESHOLD = 0.40   # a TF is considered "active" if its bias score exceeds this
_MIN_SAMPLES = 5          # minimum active samples per TF before adapting
_TF_WEIGHT_MIN = 0.10
_TF_WEIGHT_MAX = 0.70
_ADAPT_ALPHA = 0.20       # EMA blend: new_weight = (1-alpha)*old + alpha*computed
# Weight formula: target = WR * _WR_SCALE_FACTOR + _WR_BASELINE
# A 50 % win-rate TF gets weight 0.50, a 100 % TF gets 0.90 (capped at _TF_WEIGHT_MAX)
_WR_SCALE_FACTOR = 0.80
_WR_BASELINE = 0.10


class ConfluenceAdapter:
    """Tracks per-TF signal quality and adapts TF weights in ConfluenceAgent."""

    def __init__(self, confluence_agent):
        self._confluence = confluence_agent
        # Active sample counters (only incremented when score > threshold)
        self._tf_wins: Dict[str, int] = {tf: 0 for tf in _TF_ORDER}
        self._tf_total: Dict[str, int] = {tf: 0 for tf in _TF_ORDER}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, tf_scores: Dict[str, float], was_profitable: bool) -> None:
        """Register the TF scores at signal time and the resulting outcome.

        Parameters
        ----------
        tf_scores      : {"15m": 0.6, "1h": 0.8, "4h": 0.5} from ConfluenceAgent metadata
        was_profitable : whether the trade closed with positive P&L
        """
        for tf in _TF_ORDER:
            score = tf_scores.get(tf, 0.0)
            if score > _SCORE_THRESHOLD:
                self._tf_total[tf] += 1
                if was_profitable:
                    self._tf_wins[tf] += 1

    def maybe_adapt(self) -> None:
        """Rebalance TF weights based on per-TF win rates.

        Only timeframes with enough samples are included; the rest retain
        their current weights. Normalisation is applied across all TFs so
        the adapted weights always sum to 1.0.
        """
        current_weights = dict(self._confluence._tf_weights)
        new_weights: Dict[str, float] = {}

        for tf in _TF_ORDER:
            total = self._tf_total[tf]
            if total < _MIN_SAMPLES:
                # Not enough data: keep the current weight unchanged
                new_weights[tf] = current_weights.get(tf, 1.0 / len(_TF_ORDER))
                continue
            wr = self._tf_wins[tf] / total
            # EMA blend between old weight and win-rate-derived target.
            old_weight = current_weights.get(tf, 1.0 / len(_TF_ORDER))
            target = float(np.clip(wr * _WR_SCALE_FACTOR + _WR_BASELINE,
                                   _TF_WEIGHT_MIN, _TF_WEIGHT_MAX))
            blended = float(np.clip(
                (1.0 - _ADAPT_ALPHA) * old_weight + _ADAPT_ALPHA * target,
                _TF_WEIGHT_MIN, _TF_WEIGHT_MAX,
            ))
            new_weights[tf] = blended

        # Only push an update if at least one TF has been adapted
        adapted = [tf for tf in _TF_ORDER if self._tf_total[tf] >= _MIN_SAMPLES]
        if adapted:
            self._confluence.update_tf_weights(new_weights)
            logger.info(
                f"🌊 ConfluenceAdapter: TF weights adapted (sampled={adapted}) → "
                f"{ {k: f'{v:.3f}' for k, v in self._confluence._tf_weights.items()} }"
            )

    # ------------------------------------------------------------------
    # State persistence helpers (called by EvolutionEngine)
    # ------------------------------------------------------------------

    def dump_state(self) -> Dict[str, Any]:
        """Serialise internal counters for storage in the DB."""
        return {
            "wins": dict(self._tf_wins),
            "total": dict(self._tf_total),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        """Restore internal counters from previously dumped state."""
        wins = data.get("wins", {})
        total = data.get("total", {})
        for tf in _TF_ORDER:
            self._tf_wins[tf] = int(wins.get(tf, 0))
            self._tf_total[tf] = int(total.get(tf, 0))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of per-TF win rates."""
        return {
            tf: {
                "wins": self._tf_wins[tf],
                "total": self._tf_total[tf],
                "wr": (self._tf_wins[tf] / self._tf_total[tf])
                if self._tf_total[tf] > 0 else None,
            }
            for tf in _TF_ORDER
        }
