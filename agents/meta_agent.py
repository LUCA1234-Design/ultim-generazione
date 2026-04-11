"""
Meta Agent for V17.
Monitors performance of all other agents and adjusts their weights dynamically.
Implements feedback loop: trade outcome → weight adjustment.

Enhanced (Week 3 improvements):
- Regime-aware weight adjustment with per-regime performance records
- EMA smoothing for weight updates (META_WEIGHT_DECAY)
- Confidence intervals (lower confidence bound) for conservative weighting
- Automatic agent demotion / promotion based on win-rate thresholds
- State persistence via save_state() / load_state()
- Enhanced get_report() with regime stats, CI, and demotion history
"""
import json
import logging
import math
import os
import time
import numpy as np
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, AgentResult
from config.settings import META_EVAL_WINDOW, META_MIN_SAMPLES, META_WEIGHT_DECAY

logger = logging.getLogger("MetaAgent")

# Demotion / promotion thresholds
_DEMOTION_WIN_RATE = 0.40
_PROMOTION_WIN_RATE = 0.55
_DEMOTED_WEIGHT = 0.05
_DEFAULT_SAVE_PATH = "data/meta_agent_state.json"


class AgentRecord:
    """Performance record for a single agent."""

    def __init__(self, name: str):
        self.name = name
        self.decisions: List[dict] = []   # {decision_id, score, direction, correct}

    def add_outcome(self, decision_id: str, score: float,
                    direction: str, correct: bool) -> None:
        self.decisions.append({
            "id": decision_id,
            "ts": time.time(),
            "score": score,
            "direction": direction,
            "correct": correct,
        })
        if len(self.decisions) > META_EVAL_WINDOW:
            self.decisions.pop(0)

    def win_rate(self) -> float:
        if len(self.decisions) < META_MIN_SAMPLES:
            return 0.5
        return sum(1 for d in self.decisions if d["correct"]) / len(self.decisions)

    def variance(self) -> float:
        """Variance of correctness (binary outcomes)."""
        if len(self.decisions) < META_MIN_SAMPLES:
            return 0.25  # maximum variance for unknown
        wr = self.win_rate()
        return wr * (1.0 - wr)

    def lower_confidence_bound(self, confidence: float = 0.95) -> float:
        """
        Wilson lower confidence bound for win rate.
        Conservative estimate: penalises agents with high variance / few samples.
        """
        n = len(self.decisions)
        if n < META_MIN_SAMPLES:
            return 0.5
        p = self.win_rate()
        z = 1.96  # 95% CI
        denominator = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denominator
        margin = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denominator
        return float(np.clip(centre - margin, 0.0, 1.0))

    def avg_score_when_correct(self) -> float:
        correct = [d["score"] for d in self.decisions if d["correct"]]
        return np.mean(correct) if correct else 0.5

    def avg_score_when_wrong(self) -> float:
        wrong = [d["score"] for d in self.decisions if not d["correct"]]
        return np.mean(wrong) if wrong else 0.5

    def calibration_error(self) -> float:
        """Mean absolute calibration error: |score - outcome|."""
        errors = [abs(d["score"] - float(d["correct"])) for d in self.decisions]
        return float(np.mean(errors)) if errors else 0.5

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"name": self.name, "decisions": list(self.decisions)}

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRecord":
        rec = cls(data["name"])
        rec.decisions = list(data.get("decisions", []))
        return rec


class MetaAgent(BaseAgent):
    """Monitors other agents and adjusts their weights based on performance."""

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        super().__init__("meta", initial_weight=1.0)
        self._agents: Dict[str, BaseAgent] = {}
        self._records: Dict[str, AgentRecord] = {}

        # Per-regime records: {regime_name: {agent_name: AgentRecord}}
        self._regime_records: Dict[str, Dict[str, AgentRecord]] = {}

        # Demotion / promotion tracking
        self._demoted: Dict[str, bool] = {}           # agent_name → is_demoted
        self._demotion_history: List[dict] = []       # chronological log

        if agents:
            for agent in agents:
                self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent for monitoring."""
        self._agents[agent.name] = agent
        self._records[agent.name] = AgentRecord(agent.name)
        self._demoted[agent.name] = False
        logger.info(f"MetaAgent: registered agent '{agent.name}'")

    # ------------------------------------------------------------------
    # Outcome recording & weight adjustment
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        decision_id: str,
        agent_results: Dict[str, AgentResult],
        was_correct: bool,
        regime: Optional[str] = None,
    ) -> None:
        """Record whether a decision was correct for each participating agent."""
        for name, result in agent_results.items():
            record = self._records.get(name)
            if record is None:
                continue
            record.add_outcome(
                decision_id=decision_id,
                score=result.score,
                direction=result.direction,
                correct=was_correct,
            )

            # Regime-specific record
            if regime:
                regime_agent_records = self._regime_records.setdefault(regime, {})
                if name not in regime_agent_records:
                    regime_agent_records[name] = AgentRecord(name)
                regime_agent_records[name].add_outcome(
                    decision_id=decision_id,
                    score=result.score,
                    direction=result.direction,
                    correct=was_correct,
                )

    def adjust_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Recalculate agent weights based on recent performance.

        Enhancements over v1
        --------------------
        1. Uses lower confidence bound (LCB) instead of raw win rate
        2. EMA smoothing: new_weight = decay * old + (1 - decay) * computed
        3. Regime-specific records used when ``regime`` is provided
        4. Automatic demotion / promotion

        Returns the new weight map.
        """
        decay = float(META_WEIGHT_DECAY)
        weight_map: Dict[str, float] = {}

        for name, record in self._records.items():
            agent = self._agents.get(name)
            if agent is None:
                continue

            # Pick the regime-specific record if available
            if regime and regime in self._regime_records:
                regime_rec = self._regime_records[regime].get(name, record)
            else:
                regime_rec = record

            if len(regime_rec.decisions) < META_MIN_SAMPLES:
                weight_map[name] = agent.weight
                continue

            # --- LCB-based performance score ---
            lcb = regime_rec.lower_confidence_bound()
            cal_error = regime_rec.calibration_error()
            cal_factor = 1.0 - cal_error            # well-calibrated → higher

            # Computed weight: scaled around 1.0, bounded to [0.05, 5.0]
            computed_weight = float(np.clip(
                lcb * 2.0 * cal_factor,             # range ~0.0 – 2.0
                0.05, 5.0
            ))

            # --- EMA smoothing ---
            new_weight = decay * agent.weight + (1.0 - decay) * computed_weight
            new_weight = float(np.clip(new_weight, 0.05, 5.0))

            # --- Demotion / promotion ---
            win_rate = regime_rec.win_rate()
            is_demoted = self._demoted.get(name, False)

            if not is_demoted and win_rate < _DEMOTION_WIN_RATE:
                # Demote
                new_weight = _DEMOTED_WEIGHT
                self._demoted[name] = True
                event = {
                    "ts": time.time(), "agent": name, "event": "demoted",
                    "win_rate": win_rate, "regime": regime,
                }
                self._demotion_history.append(event)
                logger.warning(
                    f"MetaAgent: DEMOTED '{name}' (wr={win_rate:.2%} < "
                    f"{_DEMOTION_WIN_RATE:.0%}) → weight={_DEMOTED_WEIGHT}"
                )
            elif is_demoted and win_rate >= _PROMOTION_WIN_RATE:
                # Promote
                self._demoted[name] = False
                event = {
                    "ts": time.time(), "agent": name, "event": "promoted",
                    "win_rate": win_rate, "regime": regime,
                }
                self._demotion_history.append(event)
                logger.info(
                    f"MetaAgent: PROMOTED '{name}' (wr={win_rate:.2%} ≥ "
                    f"{_PROMOTION_WIN_RATE:.0%}) → weight={new_weight:.3f}"
                )
            elif is_demoted:
                # Stay demoted
                new_weight = _DEMOTED_WEIGHT

            agent.weight = new_weight
            weight_map[name] = new_weight
            logger.debug(
                f"MetaAgent: {name} wr={win_rate:.2%} lcb={lcb:.3f} "
                f"cal={cal_error:.3f} → weight={new_weight:.3f}"
                + (f" [demoted]" if self._demoted.get(name) else "")
            )

        return weight_map

    def get_report(self, include_regime: bool = True) -> Dict[str, Any]:
        """Return a performance report for all monitored agents.

        Enhanced to include:
        - Confidence intervals (lower confidence bound)
        - Per-regime statistics
        - Demotion / promotion history
        """
        report: Dict[str, Any] = {}
        for name, record in self._records.items():
            agent = self._agents.get(name)
            entry: Dict[str, Any] = {
                "weight": agent.weight if agent else None,
                "win_rate": record.win_rate(),
                "win_rate_lcb": record.lower_confidence_bound(),
                "variance": record.variance(),
                "n_decisions": len(record.decisions),
                "cal_error": record.calibration_error(),
                "avg_score_correct": record.avg_score_when_correct(),
                "avg_score_wrong": record.avg_score_when_wrong(),
                "demoted": self._demoted.get(name, False),
            }

            if include_regime and self._regime_records:
                regime_stats: Dict[str, Any] = {}
                for regime_name, regime_agents in self._regime_records.items():
                    rec = regime_agents.get(name)
                    if rec and len(rec.decisions) >= 1:
                        regime_stats[regime_name] = {
                            "win_rate": rec.win_rate(),
                            "win_rate_lcb": rec.lower_confidence_bound(),
                            "n_decisions": len(rec.decisions),
                        }
                entry["regime_stats"] = regime_stats

            report[name] = entry

        report["_demotion_history"] = self._demotion_history[-20:]  # last 20 events
        return report

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str = _DEFAULT_SAVE_PATH) -> bool:
        """Serialise MetaAgent records and weights to a JSON file."""
        try:
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path if dir_path else ".", exist_ok=True)
            state = {
                "records": {name: rec.to_dict() for name, rec in self._records.items()},
                "weights": {name: agent.weight for name, agent in self._agents.items()},
                "demoted": dict(self._demoted),
                "demotion_history": self._demotion_history,
                "regime_records": {
                    regime: {
                        name: rec.to_dict()
                        for name, rec in agents.items()
                    }
                    for regime, agents in self._regime_records.items()
                },
                "saved_at": time.time(),
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            logger.info(f"MetaAgent: state saved to '{path}'")
            return True
        except Exception as exc:
            logger.error(f"MetaAgent: failed to save state to '{path}': {exc}")
            return False

    def load_state(self, path: str = _DEFAULT_SAVE_PATH) -> bool:
        """Load MetaAgent records and weights from a JSON file."""
        if not os.path.exists(path):
            logger.info(f"MetaAgent: no state file at '{path}', starting fresh")
            return False
        try:
            with open(path, "r", encoding="utf-8") as fh:
                state = json.load(fh)

            # Restore records
            for name, rec_data in state.get("records", {}).items():
                if name in self._records:
                    self._records[name] = AgentRecord.from_dict(rec_data)

            # Restore weights
            for name, w in state.get("weights", {}).items():
                agent = self._agents.get(name)
                if agent:
                    agent.weight = w

            # Restore demotion state
            self._demoted.update(state.get("demoted", {}))
            self._demotion_history = list(state.get("demotion_history", []))

            # Restore regime records
            for regime, agents in state.get("regime_records", {}).items():
                self._regime_records[regime] = {
                    name: AgentRecord.from_dict(rec_data)
                    for name, rec_data in agents.items()
                }

            logger.info(
                f"MetaAgent: state loaded from '{path}' "
                f"(saved_at={state.get('saved_at', 'unknown')})"
            )
            return True
        except Exception as exc:
            logger.error(f"MetaAgent: failed to load state from '{path}': {exc}")
            return False

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(
        self,
        symbol: str,
        interval: str,
        df,
        agent_results: Optional[Dict[str, AgentResult]] = None,
        regime: Optional[str] = None,
    ) -> Optional[AgentResult]:
        """Return a meta-score based on current agent performance.

        Parameters
        ----------
        symbol, interval, df : standard BaseAgent signature
        agent_results        : optional results from other agents (not used for scoring here)
        regime               : optional current market regime name from RegimeAgent
        """
        report = self.get_report(include_regime=False)
        if not report or all(k.startswith("_") for k in report):
            return AgentResult(
                agent_name=self.name,
                symbol=symbol,
                interval=interval,
                score=0.5,
                direction="neutral",
                confidence=0.3,
                details=["meta:warmup_mode"],
                metadata={},
            )

        # Use LCB-weighted average as meta-score
        lcb_values = [
            v["win_rate_lcb"]
            for k, v in report.items()
            if not k.startswith("_") and isinstance(v, dict) and "win_rate_lcb" in v
        ]
        meta_score = float(np.mean(lcb_values)) if lcb_values else 0.5

        details = [
            f"{name}:wr={v['win_rate']:.2%},lcb={v['win_rate_lcb']:.2%}"
            for name, v in report.items()
            if not name.startswith("_") and isinstance(v, dict)
        ]
        if regime:
            details.append(f"regime={regime}")

        # Direction voting: analizza i record recenti per il regime corrente
        direction = "neutral"
        direction_confidence = meta_score
        if regime and regime in self._regime_records:
            # Raccogli tutte le decisioni recenti per il regime corrente
            all_decisions = []
            for rec in self._regime_records[regime].values():
                all_decisions.extend(rec.decisions[-META_EVAL_WINDOW:])

            # Conta win rate long vs short nelle ultime decisioni
            long_wins = sum(1 for d in all_decisions if d.get("direction") == "long" and d.get("correct"))
            long_total = sum(1 for d in all_decisions if d.get("direction") == "long")
            short_wins = sum(1 for d in all_decisions if d.get("direction") == "short" and d.get("correct"))
            short_total = sum(1 for d in all_decisions if d.get("direction") == "short")

            min_samples = META_MIN_SAMPLES
            long_wr = long_wins / long_total if long_total >= min_samples else 0.5
            short_wr = short_wins / short_total if short_total >= min_samples else 0.5

            if long_total >= min_samples or short_total >= min_samples:
                wr_gap = long_wr - short_wr
                if wr_gap > 0.10:
                    direction = "long"
                    direction_confidence = float(np.clip(meta_score + wr_gap * 0.5, 0.0, 1.0))
                    details.append(f"dir_vote=long(gap={wr_gap:.2f})")
                elif wr_gap < -0.10:
                    direction = "short"
                    direction_confidence = float(np.clip(meta_score + abs(wr_gap) * 0.5, 0.0, 1.0))
                    details.append(f"dir_vote=short(gap={wr_gap:.2f})")
                else:
                    details.append(f"dir_vote=neutral(gap={wr_gap:.2f})")

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(meta_score, 0.0, 1.0)),
            direction=direction,
            confidence=direction_confidence,
            details=details,
            metadata={"report": report, "regime": regime},
        )
