"""
Meta Agent for V17.
Monitors performance of all other agents and adjusts their weights dynamically.
Implements feedback loop: trade outcome → weight adjustment.
"""
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, AgentResult
from config.settings import META_EVAL_WINDOW, META_MIN_SAMPLES, META_WEIGHT_DECAY

logger = logging.getLogger("MetaAgent")


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


class MetaAgent(BaseAgent):
    """Monitors other agents and adjusts their weights based on performance."""

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        super().__init__("meta", initial_weight=1.0)
        self._agents: Dict[str, BaseAgent] = {}
        self._records: Dict[str, AgentRecord] = {}
        if agents:
            for agent in agents:
                self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent for monitoring."""
        self._agents[agent.name] = agent
        self._records[agent.name] = AgentRecord(agent.name)
        logger.info(f"MetaAgent: registered agent '{agent.name}'")

    # ------------------------------------------------------------------
    # Outcome recording & weight adjustment
    # ------------------------------------------------------------------

    def record_outcome(self, decision_id: str, agent_results: Dict[str, AgentResult],
                        was_correct: bool) -> None:
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

    def adjust_weights(self) -> Dict[str, float]:
        """Recalculate agent weights based on recent performance.

        Returns the new weight map.
        """
        weight_map: Dict[str, float] = {}
        for name, record in self._records.items():
            agent = self._agents.get(name)
            if agent is None:
                continue
            if len(record.decisions) < META_MIN_SAMPLES:
                weight_map[name] = agent.weight
                continue

            win_rate = record.win_rate()
            cal_error = record.calibration_error()

            # Performance factor: based on win rate above chance (0.5)
            perf_factor = 1.0 + 2.0 * (win_rate - 0.5)  # range 0.0 – 2.0
            # Calibration factor: well-calibrated agents score higher
            cal_factor = 1.0 - cal_error  # range 0.0 – 1.0

            new_weight = float(np.clip(agent.weight * perf_factor * cal_factor, 0.05, 5.0))
            agent.weight = new_weight
            weight_map[name] = new_weight
            logger.debug(
                f"MetaAgent: {name} wr={win_rate:.2%} cal={cal_error:.3f} "
                f"→ weight={new_weight:.3f}"
            )
        return weight_map

    def get_report(self) -> Dict[str, Any]:
        """Return a performance report for all monitored agents."""
        report = {}
        for name, record in self._records.items():
            agent = self._agents.get(name)
            report[name] = {
                "weight": agent.weight if agent else None,
                "win_rate": record.win_rate(),
                "n_decisions": len(record.decisions),
                "cal_error": record.calibration_error(),
                "avg_score_correct": record.avg_score_when_correct(),
                "avg_score_wrong": record.avg_score_when_wrong(),
            }
        return report

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df,
                agent_results: Optional[Dict[str, AgentResult]] = None) -> Optional[AgentResult]:
        """Return a meta-score based on current agent performance."""
        report = self.get_report()
        if not report:
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

        # Meta-score = average win rate across all agents
        win_rates = [v["win_rate"] for v in report.values() if v["win_rate"] is not None]
        meta_score = float(np.mean(win_rates)) if win_rates else 0.5

        details = [f"{name}:wr={v['win_rate']:.2%}" for name, v in report.items()]

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(meta_score, 0.0, 1.0)),
            direction="neutral",
            confidence=meta_score,
            details=details,
            metadata={"report": report},
        )
