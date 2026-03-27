"""
Decision Fusion Engine for V17.
Replaces the binary IF/RETURN cascade of V16 with weighted voting.
Each agent returns a score 0.0 – 1.0; final decision = weighted average vs adaptive threshold.
Every decision is logged with full reasoning.
"""
import logging
import time
import uuid
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

from agents.base_agent import AgentResult
from config.settings import FUSION_THRESHOLD_DEFAULT, FUSION_AGENT_WEIGHTS

logger = logging.getLogger("DecisionFusion")

# Decision outcomes
DECISION_LONG = "long"
DECISION_SHORT = "short"
DECISION_HOLD = "hold"


class FusionResult:
    """Final fused decision."""

    def __init__(self, decision_id: str, symbol: str, interval: str,
                 decision: str, final_score: float, direction: str,
                 agent_scores: Dict[str, float], agent_results: Dict[str, AgentResult],
                 threshold: float, reasoning: List[str]):
        self.decision_id = decision_id
        self.symbol = symbol
        self.interval = interval
        self.decision = decision          # LONG / SHORT / HOLD
        self.final_score = final_score
        self.direction = direction
        self.agent_scores = agent_scores
        self.agent_results = agent_results
        self.threshold = threshold
        self.reasoning = reasoning
        self.timestamp = time.time()

    def should_trade(self) -> bool:
        return self.decision in (DECISION_LONG, DECISION_SHORT)

    def __repr__(self) -> str:
        return (
            f"FusionResult({self.symbol}/{self.interval}, {self.decision}, "
            f"score={self.final_score:.3f}, threshold={self.threshold:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": self.decision,
            "final_score": self.final_score,
            "direction": self.direction,
            "agent_scores": self.agent_scores,
            "threshold": self.threshold,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


class DecisionFusion:
    """Weighted vote fusion with adaptive threshold."""

    def __init__(self, agent_weights: Optional[Dict[str, float]] = None,
                 threshold: float = FUSION_THRESHOLD_DEFAULT):
        self._weights = dict(agent_weights or FUSION_AGENT_WEIGHTS)
        self._threshold = threshold
        self._threshold_history: List[float] = []
        self._decision_log: List[FusionResult] = []

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weight(self, agent_name: str, new_weight: float) -> None:
        self._weights[agent_name] = float(np.clip(new_weight, 0.01, 10.0))

    def update_weights(self, weight_map: Dict[str, float]) -> None:
        for name, w in weight_map.items():
            self.update_weight(name, w)

    # ------------------------------------------------------------------
    # Threshold adaptation
    # ------------------------------------------------------------------

    def adapt_threshold(self, was_correct: bool, score: float) -> None:
        """Adjust threshold based on whether the last decision was correct."""
        self._threshold_history.append(float(was_correct))
        if len(self._threshold_history) > 50:
            self._threshold_history.pop(0)
        if len(self._threshold_history) >= 20:
            recent_acc = sum(self._threshold_history[-20:]) / 20
            if recent_acc < 0.40:
                # Too many wrong — raise threshold
                self._threshold = float(np.clip(self._threshold + 0.02, 0.30, 0.90))
            elif recent_acc > 0.65:
                # Good accuracy — slightly lower threshold
                self._threshold = float(np.clip(self._threshold - 0.01, 0.30, 0.90))

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def fuse(self, symbol: str, interval: str,
             agent_results: Dict[str, AgentResult]) -> FusionResult:
        """Compute weighted fusion of agent results.

        Parameters
        ----------
        agent_results : dict  {agent_name: AgentResult}

        Returns
        -------
        FusionResult with the fused decision.
        """
        decision_id = str(uuid.uuid4())[:8]
        reasoning: List[str] = []
        agent_scores: Dict[str, float] = {}

        if not agent_results:
            return FusionResult(
                decision_id=decision_id,
                symbol=symbol,
                interval=interval,
                decision=DECISION_HOLD,
                final_score=0.0,
                direction="neutral",
                agent_scores={},
                agent_results=agent_results,
                threshold=self._threshold,
                reasoning=["No agent results available"],
            )

        # --- Direction voting ---
        direction_votes: Dict[str, float] = {"long": 0.0, "short": 0.0}
        total_weight = 0.0
        weighted_score = 0.0

        for name, result in agent_results.items():
            if result is None:
                continue
            w = self._weights.get(name, 1.0)
            agent_scores[name] = result.score
            weighted_score += result.score * w
            total_weight += w
            if result.direction in direction_votes:
                direction_votes[result.direction] += w * result.confidence
            reasoning.append(
                f"{name}: score={result.score:.3f} dir={result.direction} "
                f"conf={result.confidence:.2f} w={w:.2f} | {', '.join(result.details[:3])}"
            )

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        direction = max(direction_votes, key=direction_votes.get) if direction_votes else "neutral"

        reasoning.append(
            f"FUSION: score={final_score:.3f} threshold={self._threshold:.3f} "
            f"direction={direction} ({direction_votes})"
        )

        if final_score >= self._threshold:
            decision = DECISION_LONG if direction == "long" else DECISION_SHORT
        else:
            decision = DECISION_HOLD

        result = FusionResult(
            decision_id=decision_id,
            symbol=symbol,
            interval=interval,
            decision=decision,
            final_score=float(final_score),
            direction=direction,
            agent_scores=agent_scores,
            agent_results=agent_results,
            threshold=self._threshold,
            reasoning=reasoning,
        )

        # Log
        self._decision_log.append(result)
        if len(self._decision_log) > 1000:
            self._decision_log.pop(0)

        if decision != DECISION_HOLD:
            logger.info(
                f"🎯 DECISION [{decision_id}] {symbol}/{interval}: {decision.upper()} "
                f"(score={final_score:.3f} ≥ {self._threshold:.3f})"
            )
        else:
            logger.debug(
                f"HOLD [{decision_id}] {symbol}/{interval}: "
                f"score={final_score:.3f} < {self._threshold:.3f}"
            )

        return result

    def get_decision_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent decisions as dicts."""
        return [r.to_dict() for r in self._decision_log[-limit:]]
