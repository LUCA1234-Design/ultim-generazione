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
from config.settings import TRAINING_MODE

logger = logging.getLogger("DecisionFusion")

try:
    from coordination.consensus_protocol import ConsensusProtocol
    _CONSENSUS_AVAILABLE = True
except ImportError:
    _CONSENSUS_AVAILABLE = False

# Sniper calibration constants
_SNIPER_MIN_DIRECTION_AGREEMENT = 0.60   # minimum fraction of directional agents that must agree
SNIPER_MIN_AGREEING_TIMEFRAMES = 1 if TRAINING_MODE else 2
_SNIPER_MIN_STRONG_AGENTS = 1 if TRAINING_MODE else 3
_SNIPER_AGENT_MIN_SCORE = 0.40 if TRAINING_MODE else 0.55
DECISION_LONG = "long"
DECISION_SHORT = "short"
DECISION_HOLD = "hold"

# Regime-aware threshold multipliers — 5 regimi + backward compat
_REGIME_THRESHOLD_MULTIPLIERS = {
    "trending_up": 0.85,
    "trending_down": 0.90,   # Slightly harder for shorts (trend bias)
    "ranging": 1.20,
    "volatile": 1.10,
    "capitulation": 1.50,    # Molto penalizzato — evitare di tradare in capitulation
    "unknown": 1.0,
    "trending": 0.85,        # Backward compat
}


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
        self.signal_tags: list = []

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
            "signal_tags": self.signal_tags,
        }


class DecisionFusion:
    """Weighted vote fusion with adaptive threshold."""

    def __init__(self, agent_weights: Optional[Dict[str, float]] = None,
                 threshold: float = FUSION_THRESHOLD_DEFAULT):
        self._weights = dict(agent_weights or FUSION_AGENT_WEIGHTS)
        self._threshold = threshold
        self._threshold_history: List[float] = []
        self._decision_log: List[FusionResult] = []

        # V18 Consensus Protocol (3-round)
        self._consensus: Optional["ConsensusProtocol"] = None
        if _CONSENSUS_AVAILABLE:
            try:
                self._consensus = ConsensusProtocol()
            except Exception:
                pass

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
            if recent_acc < 0.45:
                # Too many wrong — raise threshold more aggressively
                self._threshold = float(np.clip(self._threshold + 0.03, 0.50, 0.85))
            elif recent_acc > 0.70:
                # Good accuracy — slightly lower threshold
                self._threshold = float(np.clip(self._threshold - 0.005, 0.50, 0.85))

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def fuse(self, symbol: str, interval: str,
             agent_results: Dict[str, AgentResult],
             regime: str = "unknown") -> FusionResult:
        """Compute weighted fusion of agent results.

        Parameters
        ----------
        agent_results : dict  {agent_name: AgentResult}
        regime        : current market regime string for threshold adjustment

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

            # ContrarianAgent uses inverted semantics: high score = "don't trade".
            # Invert its score before it enters the weighted average so that a
            # strong contrarian signal suppresses rather than boosts the total.
            effective_score = (1.0 - result.score) if name == "contrarian" else result.score

            weighted_score += effective_score * w
            total_weight += w
            if result.direction in direction_votes:
                direction_votes[result.direction] += w * result.confidence
            reasoning.append(
                f"{name}: score={result.score:.3f} dir={result.direction} "
                f"conf={result.confidence:.2f} w={w:.2f} | {', '.join(result.details[:3])}"
            )

        direction = max(direction_votes, key=direction_votes.get) if direction_votes else "neutral"

        # ---- SNIPER: Direction unanimity filter ----
        directional_agents = [
            name for name, r in agent_results.items()
            if r is not None and r.direction in ("long", "short")
        ]
        agreeing_agents = [
            name for name, r in agent_results.items()
            if r is not None and r.direction == direction
        ]
        if len(directional_agents) >= 3:
            agreement_ratio = len(agreeing_agents) / len(directional_agents)
            if agreement_ratio < _SNIPER_MIN_DIRECTION_AGREEMENT:
                reasoning.append(
                    f"SNIPER_VETO: direction_agreement={agreement_ratio:.0%} "
                    f"({len(agreeing_agents)}/{len(directional_agents)}) — need >={_SNIPER_MIN_DIRECTION_AGREEMENT:.0%}"
                )
                final_score = weighted_score / total_weight if total_weight > 0 else 0.0
                return FusionResult(
                    decision_id=decision_id, symbol=symbol, interval=interval,
                    decision=DECISION_HOLD, final_score=float(final_score),
                    direction=direction, agent_scores=agent_scores,
                    agent_results=agent_results, threshold=self._threshold,
                    reasoning=reasoning,
                )

        # --- Direction consensus bonus/penalty ---
        total_agents_with_direction = sum(
            1 for _, r in agent_results.items()
            if r is not None and r.direction in ("long", "short")
        )
        agents_agreeing = sum(
            1 for _, r in agent_results.items()
            if r is not None and r.direction == direction
        )
        agreement_ratio = agents_agreeing / max(total_agents_with_direction, 1)

        if agreement_ratio >= 0.70:
            consensus_bonus = 0.05
            reasoning.append(f"CONSENSUS_BONUS: {agreement_ratio:.0%} agree → +{consensus_bonus}")
        elif agreement_ratio < 0.40:
            consensus_bonus = -0.10
            reasoning.append(f"CONSENSUS_PENALTY: {agreement_ratio:.0%} agree → {consensus_bonus}")
        else:
            consensus_bonus = 0.0

        final_score = (weighted_score / total_weight if total_weight > 0 else 0.0) + consensus_bonus
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # V18: Blend with 3-round Consensus Protocol when enough agents are present
        if self._consensus is not None and len(agent_results) >= 4:
            try:
                consensus_result = self._consensus.full_consensus(agent_results, self._weights)
                consensus_score = float(consensus_result.get("score", consensus_result.get("final_score", final_score)))
                # 70% weighted voting (proven) + 30% consensus (new)
                final_score = float(np.clip(
                    0.70 * final_score + 0.30 * consensus_score, 0.0, 1.0
                ))
                reasoning.append(
                    f"CONSENSUS_3R: blended score={final_score:.3f} "
                    f"(consensus_level={consensus_result.get('consensus_level', '?')})"
                )
            except Exception as e:
                reasoning.append(f"CONSENSUS_3R: fallback to 1-round (error: {e})")

        # === CECCHINO: RICHIEDERE ALMENO 3 AGENTI SOPRA SOGLIA ===
        strong_agents = [
            name for name, result in agent_results.items()
            if result is not None and result.score >= _SNIPER_AGENT_MIN_SCORE
        ]
        if len(strong_agents) < _SNIPER_MIN_STRONG_AGENTS:
            reasoning.append(
                f"SNIPER_VETO_WEAK_AGENTS: only {len(strong_agents)}/{_SNIPER_MIN_STRONG_AGENTS} "
                f"agents above {_SNIPER_AGENT_MIN_SCORE:.0%} threshold "
                f"(strong={strong_agents})"
            )
            return FusionResult(
                decision_id=decision_id, symbol=symbol, interval=interval,
                decision=DECISION_HOLD, final_score=float(final_score),
                direction=direction, agent_scores=agent_scores,
                agent_results=agent_results, threshold=self._threshold,
                reasoning=reasoning,
            )
        reasoning.append(f"SNIPER_OK: {len(strong_agents)} strong agents (>={_SNIPER_AGENT_MIN_SCORE:.0%})")

        # --- Regime-aware threshold adjustment ---
        effective_threshold = self._threshold * _REGIME_THRESHOLD_MULTIPLIERS.get(regime, 1.0)
        effective_threshold = float(np.clip(effective_threshold, 0.20, 0.95))
        reasoning.append(
            f"REGIME_THRESHOLD: regime={regime}, base={self._threshold:.3f}, "
            f"effective={effective_threshold:.3f}"
        )

        reasoning.append(
            f"FUSION: score={final_score:.3f} threshold={effective_threshold:.3f} "
            f"direction={direction} ({direction_votes})"
        )

        if final_score >= effective_threshold:
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
            threshold=effective_threshold,
            reasoning=reasoning,
        )

        # Log
        self._decision_log.append(result)
        if len(self._decision_log) > 1000:
            self._decision_log.pop(0)

        if decision != DECISION_HOLD:
            logger.info(
                f"🎯 DECISION [{decision_id}] {symbol}/{interval}: {decision.upper()} "
                f"(score={final_score:.3f} ≥ {effective_threshold:.3f})"
            )
        else:
            # Log near-miss decisions at INFO level for debugging
            if final_score >= effective_threshold * 0.70:
                logger.info(
                    f"📊 NEAR-MISS [{decision_id}] {symbol}/{interval}: "
                    f"score={final_score:.3f} < {effective_threshold:.3f} "
                    f"(gap={effective_threshold - final_score:.3f}) agents={agent_scores}"
                )
            else:
                logger.debug(
                    f"HOLD [{decision_id}] {symbol}/{interval}: "
                    f"score={final_score:.3f} < {effective_threshold:.3f}"
                )

        return result

    def get_decision_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent decisions as dicts."""
        return [r.to_dict() for r in self._decision_log[-limit:]]
