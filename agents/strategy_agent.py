"""
Strategy Agent for V17.
Generates and evaluates trading strategies dynamically.
Strategies are parameter sets (RSI level, ATR mult, min score, etc.) that are
scored based on historical performance stored in the experience DB.
"""
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import rsi, atr, macd, adx, bollinger_bands, zscore

logger = logging.getLogger("StrategyAgent")


# Strategy definition: a set of parameters to evaluate
StrategyParams = Dict[str, float]


class StrategyAgent(BaseAgent):
    """Generates strategies and scores them against historical data."""

    DEFAULT_STRATEGIES = [
        # RSI oversold + MACD cross + trend
        {"name": "rsi_macd_trend", "rsi_threshold": 35, "adx_min": 20,
         "macd_cross": True, "vol_mult": 1.0, "bb_touch": False},
        # Squeeze breakout
        {"name": "squeeze_breakout", "rsi_threshold": 50, "adx_min": 15,
         "macd_cross": False, "vol_mult": 1.5, "bb_touch": False},
        # BB lower touch + volume spike
        {"name": "bb_bounce", "rsi_threshold": 40, "adx_min": 10,
         "macd_cross": False, "vol_mult": 2.0, "bb_touch": True},
        # Divergence entry
        {"name": "divergence", "rsi_threshold": 40, "adx_min": 10,
         "macd_cross": False, "vol_mult": 0.8, "bb_touch": False},
        # Momentum breakout
        {"name": "momentum", "rsi_threshold": 55, "adx_min": 25,
         "macd_cross": True, "vol_mult": 1.8, "bb_touch": False},
    ]

    def __init__(self):
        super().__init__("strategy", initial_weight=0.10)
        self._strategies: List[StrategyParams] = list(self.DEFAULT_STRATEGIES)
        self._strategy_scores: Dict[str, float] = {}  # name -> avg score
        self._strategy_counts: Dict[str, int] = {}    # name -> sample count

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    def _eval_strategy(self, df: pd.DataFrame, params: StrategyParams,
                        direction: str) -> float:
        """Evaluate a single strategy against current market conditions.
        Returns 0.0 – 1.0 match score.
        """
        try:
            close = df["close"]
            rsi_val = float(rsi(close, 14).iloc[-1])
            adx_s, di_p, di_m = adx(df, 14)
            last_adx = float(adx_s.iloc[-1])
            macd_l, macd_sig, macd_hist = macd(close)
            last_hist = float(macd_hist.iloc[-1])
            prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 2 else last_hist
            bb_up, bb_mid, bb_lo = bollinger_bands(close, 20, 2.0)
            avg_vol = float(df["volume"].iloc[-20:-1].mean())
            last_vol = float(df["volume"].iloc[-1])
            rvol = last_vol / avg_vol if avg_vol > 0 else 1.0

            conditions_met = 0
            total_conditions = 0

            # RSI threshold
            total_conditions += 1
            threshold = params.get("rsi_threshold", 50)
            if direction == "long":
                if rsi_val <= threshold:
                    conditions_met += 1
            else:
                if rsi_val >= (100 - threshold):
                    conditions_met += 1

            # ADX min
            total_conditions += 1
            if last_adx >= params.get("adx_min", 15):
                conditions_met += 1

            # MACD cross OR momentum
            if params.get("macd_cross", False):
                total_conditions += 1
                if direction == "long" and last_hist > 0 and prev_hist <= 0:
                    conditions_met += 1          # Perfect cross
                elif direction == "long" and last_hist > prev_hist and last_hist > 0:
                    conditions_met += 0.7        # Momentum: MACD hist rising and positive
                elif direction == "short" and last_hist < 0 and prev_hist >= 0:
                    conditions_met += 1          # Perfect cross
                elif direction == "short" and last_hist < prev_hist and last_hist < 0:
                    conditions_met += 0.7        # Momentum: MACD hist falling and negative

            # Volume multiplier
            total_conditions += 1
            if rvol >= params.get("vol_mult", 1.0):
                conditions_met += 1

            # BB touch
            if params.get("bb_touch", False):
                total_conditions += 1
                last_close = float(close.iloc[-1])
                if direction == "long" and last_close <= float(bb_lo.iloc[-1]) * 1.005:
                    conditions_met += 1
                elif direction == "short" and last_close >= float(bb_up.iloc[-1]) * 0.995:
                    conditions_met += 1

            return conditions_met / max(total_conditions, 1)
        except Exception as e:
            logger.debug(f"_eval_strategy error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Strategy selection and mutation
    # ------------------------------------------------------------------

    def best_strategy(self, df: pd.DataFrame, direction: str) -> Tuple[str, float]:
        """Return (strategy_name, match_score) for the best-matching strategy."""
        best_name = "default"
        best_score = 0.0
        for params in self._strategies:
            score = self._eval_strategy(df, params, direction)
            name = params.get("name", "unknown")
            # Blend with historical performance if available
            hist_score = self._strategy_scores.get(name, 0.5)
            n = self._strategy_counts.get(name, 0)
            blended = score * 0.7 + hist_score * 0.3 if n >= 5 else score
            if blended > best_score:
                best_score = blended
                best_name = name
        return best_name, float(best_score)

    def update_strategy_outcome(self, name: str, was_profitable: bool) -> None:
        """Update historical score for a strategy based on trade outcome."""
        old_score = self._strategy_scores.get(name, 0.5)
        n = self._strategy_counts.get(name, 0)
        # Incremental mean update
        new_score = (old_score * n + float(was_profitable)) / (n + 1)
        self._strategy_scores[name] = new_score
        self._strategy_counts[name] = n + 1

    def mutate_strategy(self, base_name: str) -> StrategyParams:
        """Generate a mutated version of a strategy for exploration."""
        base = next((p for p in self._strategies if p.get("name") == base_name),
                    self.DEFAULT_STRATEGIES[0])
        mutated = dict(base)
        mutated["name"] = f"{base_name}_mut_{random.randint(100, 999)}"
        # Mutate numeric params
        for key in ["rsi_threshold", "adx_min", "vol_mult"]:
            if key in mutated:
                delta = random.uniform(-0.1, 0.1) * mutated[key]
                mutated[key] = float(np.clip(mutated[key] + delta, 5, 200))
        return mutated

    # ------------------------------------------------------------------
    # Strategy evolution
    # ------------------------------------------------------------------

    def prune_and_evolve(
        self,
        min_samples: int = 10,
        min_win_rate: float = 0.35,
        max_strategies: int = 10,
    ) -> List[str]:
        """Prune under-performing strategies and optionally generate a mutation.

        Parameters
        ----------
        min_samples   : minimum recorded trades before a strategy is eligible for pruning
        min_win_rate  : strategies below this win-rate (and with enough samples) are removed
        max_strategies: cap on total strategy count; new mutations are added only below cap

        Returns
        -------
        A list of human-readable change descriptions (empty if nothing changed).
        """
        changes: List[str] = []

        # --- Find the best and worst strategies (by historical score) ---
        scored = [
            (p.get("name", "unknown"), self._strategy_scores.get(p.get("name", ""), 0.5),
             self._strategy_counts.get(p.get("name", ""), 0))
            for p in self._strategies
        ]
        # Sort: highest win-rate first
        scored.sort(key=lambda x: x[1], reverse=True)

        # --- Prune losers ---
        to_prune = [
            name for name, wr, n in scored
            if n >= min_samples and wr < min_win_rate
        ]
        # Never remove the last strategy
        for name in to_prune:
            if len(self._strategies) <= 1:
                break
            self._strategies = [p for p in self._strategies if p.get("name") != name]
            self._strategy_scores.pop(name, None)
            self._strategy_counts.pop(name, None)
            changes.append(f"❌ pruned '{name}' (wr<{min_win_rate:.0%})")
            logger.info(f"StrategyAgent: pruned low-performing strategy '{name}'")

        # --- Generate a mutation of the best strategy if below cap ---
        if len(self._strategies) < max_strategies and scored:
            # Diversity check: count unique base names
            base_names = set()
            for p in self._strategies:
                name = p.get("name", "")
                base = name.split("_mut_")[0].split("_fresh_")[0] if ("_mut_" in name or "_fresh_" in name) else name
                base_names.add(base)

            if len(base_names) < 3 and len(self.DEFAULT_STRATEGIES) > 0:
                # Low diversity: inject a random DEFAULT strategy instead of mutating
                available = [
                    s for s in self.DEFAULT_STRATEGIES
                    if s.get("name") not in [p.get("name") for p in self._strategies]
                ]
                if available:
                    fresh = dict(random.choice(available))
                    fresh["name"] = f"{fresh['name']}_fresh_{random.randint(100, 999)}"
                    self._strategies.append(fresh)
                    changes.append(f"🌱 injected fresh '{fresh['name']}' for diversity")
                    logger.info(f"StrategyAgent: injected fresh strategy '{fresh['name']}' for diversity")
                else:
                    # All defaults present, mutate as usual
                    best_name = scored[0][0]
                    mutated = self.mutate_strategy(best_name)
                    self._strategies.append(mutated)
                    changes.append(f"🧬 added mutation '{mutated['name']}' from '{best_name}'")
                    logger.info(f"StrategyAgent: added mutated strategy '{mutated['name']}'")
            else:
                best_name = scored[0][0]
                mutated = self.mutate_strategy(best_name)
                self._strategies.append(mutated)
                changes.append(f"🧬 added mutation '{mutated['name']}' from '{best_name}'")
                logger.info(f"StrategyAgent: added mutated strategy '{mutated['name']}'")

        return changes

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df,
                direction: str = "long") -> Optional[AgentResult]:
        if df is None or len(df) < 30:
            return None

        strategy_name, match_score = self.best_strategy(df, direction)
        details = [f"strategy={strategy_name}", f"match={match_score:.2f}"]

        # Historical performance of this strategy
        hist_score = self._strategy_scores.get(strategy_name, 0.5)
        n = self._strategy_counts.get(strategy_name, 0)
        if n > 0:
            details.append(f"hist_win={hist_score:.2%}({n})")

        # Combined score
        score = match_score * 0.7 + hist_score * 0.3 if n >= 5 else match_score

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(score, 0.0, 1.0)),
            direction=direction,
            confidence=float(match_score),
            details=details,
            metadata={
                "strategy": strategy_name,
                "match_score": match_score,
                "hist_score": hist_score,
                "n_samples": n,
            },
        )
