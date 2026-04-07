"""
backtesting/ab_tester.py — Live A/B Strategy Testing.

Enables controlled live comparison between two strategy variants:
  - Strategy A: current production strategy
  - Strategy B: proposed new strategy

Traffic (signals) are randomly assigned based on split_ratio.
Statistical tests (t-test, chi-squared) determine the winner.

Usage:
    tester = ABTester()
    test_id = tester.create_test("threshold_test", config_a, config_b, split=0.5)
    variant = tester.assign_signal(test_id, signal_id)
    tester.record_outcome(test_id, variant, pnl=0.02, win=True)
    if tester.is_significant(test_id):
        print(tester.get_results(test_id))
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("ABTester")

_DEFAULT_SPLIT = 0.50
_MIN_SAMPLES_PER_VARIANT = 30


class ABTest:
    """A single A/B test tracking outcomes for two variants."""

    def __init__(self, name: str, strategy_a: Dict, strategy_b: Dict, split_ratio: float):
        self.name = name
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.split_ratio = split_ratio

        self._a_outcomes: List[Dict] = []
        self._b_outcomes: List[Dict] = []
        self._assignment_count = 0

    def assign(self) -> str:
        """Randomly assign a variant based on split_ratio."""
        self._assignment_count += 1
        return "A" if np.random.random() < self.split_ratio else "B"

    def record(self, variant: str, outcome: Dict) -> None:
        """Record an outcome for a variant."""
        if variant == "A":
            self._a_outcomes.append(outcome)
        else:
            self._b_outcomes.append(outcome)

    @property
    def n_a(self) -> int:
        return len(self._a_outcomes)

    @property
    def n_b(self) -> int:
        return len(self._b_outcomes)

    def get_metrics(self, variant: str) -> Dict:
        outcomes = self._a_outcomes if variant == "A" else self._b_outcomes
        if not outcomes:
            return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        pnls = np.array([float(o.get("pnl", 0)) for o in outcomes])
        wins = np.array([float(o.get("win", pnl > 0)) for pnl, o in zip(pnls, outcomes)])

        return {
            "n": len(outcomes),
            "win_rate": float(np.mean(wins)),
            "avg_pnl": float(np.mean(pnls)),
            "pnl_std": float(np.std(pnls)),
            "total_pnl": float(np.sum(pnls)),
        }

    def statistical_test(self, p_value_threshold: float = 0.05) -> Dict:
        """
        Run statistical tests to determine if results are significant.

        Returns
        -------
        dict with t-test and chi-squared results plus recommendation.
        """
        if self.n_a < _MIN_SAMPLES_PER_VARIANT or self.n_b < _MIN_SAMPLES_PER_VARIANT:
            return {
                "significant": False,
                "reason": f"Insufficient samples (A={self.n_a}, B={self.n_b})",
                "winner": None,
            }

        a_pnls = np.array([float(o.get("pnl", 0)) for o in self._a_outcomes])
        b_pnls = np.array([float(o.get("pnl", 0)) for o in self._b_outcomes])

        # Welch's t-test for P&L difference
        t_stat, p_value_ttest = scipy_stats.ttest_ind(a_pnls, b_pnls, equal_var=False)
        p_ttest = float(p_value_ttest)

        # Chi-squared test for win rate difference
        a_wins = sum(1 for o in self._a_outcomes if o.get("win", o.get("pnl", 0) > 0))
        b_wins = sum(1 for o in self._b_outcomes if o.get("win", o.get("pnl", 0) > 0))
        a_losses = self.n_a - a_wins
        b_losses = self.n_b - b_wins

        contingency = np.array([[a_wins, a_losses], [b_wins, b_losses]])
        if contingency.min() >= 5:
            chi2, p_chi2 = scipy_stats.chi2_contingency(contingency)[:2]
            p_chi2 = float(p_chi2)
        else:
            p_chi2 = 1.0  # insufficient data for chi2

        significant = p_ttest < p_value_threshold

        # Determine winner
        a_metrics = self.get_metrics("A")
        b_metrics = self.get_metrics("B")
        winner = None
        if significant:
            if a_metrics["avg_pnl"] > b_metrics["avg_pnl"]:
                winner = "A"
            elif b_metrics["avg_pnl"] > a_metrics["avg_pnl"]:
                winner = "B"

        return {
            "significant": significant,
            "p_value_ttest": p_ttest,
            "p_value_chi2": p_chi2,
            "t_stat": float(t_stat),
            "winner": winner,
            "a_metrics": a_metrics,
            "b_metrics": b_metrics,
        }


class ABTester:
    """
    Manager for multiple simultaneous A/B tests.
    """

    def __init__(self):
        self._tests: Dict[str, ABTest] = {}
        self._lock = threading.Lock()

    def create_test(
        self,
        name: str,
        strategy_a: Dict,
        strategy_b: Dict,
        split_ratio: float = _DEFAULT_SPLIT,
    ) -> str:
        """
        Create a new A/B test.

        Returns the test name (ID).
        """
        with self._lock:
            if name in self._tests:
                logger.warning(f"ABTester: test '{name}' already exists, overwriting")
            self._tests[name] = ABTest(name, strategy_a, strategy_b, split_ratio)
            logger.info(f"ABTester: created test '{name}' (split={split_ratio:.0%})")
        return name

    def assign_signal(self, test_name: str, signal_id: str = "") -> Optional[str]:
        """
        Assign a signal to a variant (A or B).

        Returns the variant ('A' or 'B') or None if test not found.
        """
        with self._lock:
            test = self._tests.get(test_name)
            if test is None:
                return None
            return test.assign()

    def record_outcome(
        self,
        test_name: str,
        variant: str,
        outcome: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """
        Record a trade outcome for a variant.

        Parameters
        ----------
        test_name : str — test name
        variant   : 'A' or 'B'
        outcome   : dict with 'pnl', 'win' keys (or pass as kwargs)
        """
        o = outcome or kwargs
        with self._lock:
            test = self._tests.get(test_name)
            if test:
                test.record(variant, o)

    def get_results(self, test_name: str) -> Optional[Dict]:
        """Return current results for a test."""
        with self._lock:
            test = self._tests.get(test_name)
            if test is None:
                return None
            return {
                "name": test.name,
                "n_a": test.n_a,
                "n_b": test.n_b,
                "a_metrics": test.get_metrics("A"),
                "b_metrics": test.get_metrics("B"),
                "statistical_test": test.statistical_test(),
            }

    def is_significant(
        self,
        test_name: str,
        p_value: float = 0.05,
    ) -> bool:
        """True if the test has reached statistical significance."""
        with self._lock:
            test = self._tests.get(test_name)
            if test is None:
                return False
            result = test.statistical_test(p_value)
            return bool(result.get("significant", False))

    def list_tests(self) -> List[str]:
        with self._lock:
            return list(self._tests.keys())
