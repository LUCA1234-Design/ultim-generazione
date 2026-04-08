"""
metalearning/concept_drift_detector.py — Concept Drift Detection.

Detects non-stationarity in trading performance distributions using:

1. **Page-Hinkley Test**
   - Sequential change-point detection for the mean of a stream.
   - Detects when the cumulative deviation from initial mean exceeds λ.
   - Fast O(1) per sample, suitable for streaming data.

2. **ADWIN (Adaptive Windowing) — simplified**
   - Maintains adaptive window sizes for win rate estimation.
   - Splits window and tests if means differ significantly.
   - Automatically shrinks window size when drift is detected.

Integration (Loop #12): Concept Drift → Re-training
  When is_drift_detected() returns True:
    - EvolutionEngine triggers re-training of GMM and HMM
    - LRScheduler resets all loops for warm restarts
    - MAMLAdapter's regime buffers are cleared for re-adaptation
"""
import logging
import math
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger("ConceptDriftDetector")

# Page-Hinkley parameters
_PH_DELTA = 0.005    # acceptable change in mean (small = sensitive)
_PH_LAMBDA = 50.0    # detection threshold
_PH_ALPHA = 0.9999   # forgetting factor (1 = no forgetting)

# ADWIN parameters
_ADWIN_DELTA = 0.002  # confidence for drift detection
_ADWIN_MAX_BUCKETS = 5


class PageHinkleyTest:
    """
    Page-Hinkley change-point detection test.

    Detects upward shifts in the mean of a sequence.
    Two detectors (positive and negative) are run in parallel
    to catch both increase and decrease drifts.
    """

    def __init__(self,
                 delta: float = _PH_DELTA,
                 threshold: float = _PH_LAMBDA,
                 alpha: float = _PH_ALPHA):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._reset()

    def update(self, x: float) -> bool:
        """
        Update detector with a new observation.

        Returns True if drift is detected.
        """
        self._n += 1
        self._cumsum += x

        # Upward drift detector (PHT+)
        self._m_up = self.alpha * self._m_up + x
        self._ph_up = max(0.0, self._ph_up + x - (self._cumsum / self._n) - self.delta)

        # Downward drift detector (PHT-)
        self._m_down = self.alpha * self._m_down + x
        self._ph_down = max(0.0, self._ph_down - x + (self._cumsum / self._n) - self.delta)

        self._drift_detected = (
            self._ph_up > self.threshold or
            self._ph_down > self.threshold
        )

        if self._drift_detected:
            self._drift_magnitude = max(self._ph_up, self._ph_down)
            logger.debug(
                f"PageHinkley drift detected: ph_up={self._ph_up:.2f}, "
                f"ph_down={self._ph_down:.2f}"
            )

        return self._drift_detected

    def is_drift_detected(self) -> bool:
        return self._drift_detected

    def get_drift_magnitude(self) -> float:
        return self._drift_magnitude

    def reset(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._cumsum = 0.0
        self._m_up = 0.0
        self._m_down = 0.0
        self._ph_up = 0.0
        self._ph_down = 0.0
        self._drift_detected = False
        self._drift_magnitude = 0.0


class ADWINDetector:
    """
    Simplified ADWIN (Adaptive Windowing) for win rate drift.

    Maintains a sliding window of win/loss outcomes and tests
    if the win rate has changed significantly using Hoeffding bounds.
    """

    def __init__(self, delta: float = _ADWIN_DELTA, max_window: int = 500):
        self.delta = delta
        self.max_window = max_window
        self._window: List[float] = []
        self._drift_detected = False
        self._drift_magnitude = 0.0

    def update(self, x: float) -> bool:
        """
        Update detector with a new binary or continuous observation.

        Returns True if drift is detected.
        """
        self._window.append(float(x))
        if len(self._window) > self.max_window:
            self._window.pop(0)

        self._drift_detected = False
        self._drift_magnitude = 0.0

        if len(self._window) < 20:
            return False

        # Test all possible splits of the window
        arr = np.array(self._window)
        n = len(arr)
        mu_total = float(np.mean(arr))

        # Try different split points
        for split in range(10, n - 10, max(1, n // 10)):
            w0 = arr[:split]
            w1 = arr[split:]
            mu0 = float(np.mean(w0))
            mu1 = float(np.mean(w1))

            # Hoeffding bound on difference of means
            eps = math.sqrt(
                math.log(4 * n / self.delta)
                / (2 * (1 / len(w0) + 1 / len(w1)) * len(arr))
            )

            diff = abs(mu0 - mu1)
            if diff >= eps:
                self._drift_detected = True
                self._drift_magnitude = diff
                # Shrink window: keep only the newer part
                self._window = list(arr[split:])
                logger.debug(
                    f"ADWIN drift detected: diff={diff:.4f}, eps={eps:.4f}, "
                    f"window_split={split}/{n}"
                )
                break

        return self._drift_detected

    def is_drift_detected(self) -> bool:
        return self._drift_detected

    def get_drift_magnitude(self) -> float:
        return self._drift_magnitude

    def reset(self) -> None:
        self._window.clear()
        self._drift_detected = False
        self._drift_magnitude = 0.0

    def get_current_estimate(self) -> float:
        """Return current mean estimate from the adaptive window."""
        if not self._window:
            return 0.5
        return float(np.mean(self._window))


class ConceptDriftDetector:
    """
    Combined concept drift detector using both PH test and ADWIN.

    Monitors:
    - Returns distribution (Page-Hinkley)
    - Win rate distribution (ADWIN)

    Drift is flagged if either test fires.
    """

    def __init__(self):
        self._ph_returns = PageHinkleyTest()
        self._adwin_winrate = ADWINDetector()
        self._drift_detected = False
        self._drift_count = 0
        self._lock = threading.Lock()

    def update(self, value: float, is_win: Optional[bool] = None) -> bool:
        """
        Update detectors with a new observation.

        Parameters
        ----------
        value  : float — recent P&L or return
        is_win : optional bool — whether the trade was a win

        Returns
        -------
        bool — True if any drift is detected
        """
        with self._lock:
            ph_drift = self._ph_returns.update(value)

            adwin_drift = False
            if is_win is not None:
                adwin_drift = self._adwin_winrate.update(float(is_win))

            self._drift_detected = ph_drift or adwin_drift
            if self._drift_detected:
                self._drift_count += 1
                logger.warning(
                    f"ConceptDrift detected (#{self._drift_count}): "
                    f"PH={ph_drift}, ADWIN={adwin_drift}"
                )

            return self._drift_detected

    def is_drift_detected(self) -> bool:
        with self._lock:
            return self._drift_detected

    def get_drift_magnitude(self) -> float:
        """Return the maximum drift magnitude across all detectors."""
        with self._lock:
            return max(
                self._ph_returns.get_drift_magnitude(),
                self._adwin_winrate.get_drift_magnitude(),
            )

    def reset(self) -> None:
        """Reset all detectors after re-training is triggered."""
        with self._lock:
            self._ph_returns.reset()
            self._adwin_winrate.reset()
            self._drift_detected = False
            logger.info("ConceptDriftDetector reset after re-training")

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "drift_count": self._drift_count,
                "current_drift": self._drift_detected,
                "ph_magnitude": self._ph_returns.get_drift_magnitude(),
                "adwin_magnitude": self._adwin_winrate.get_drift_magnitude(),
                "win_rate_estimate": self._adwin_winrate.get_current_estimate(),
            }
