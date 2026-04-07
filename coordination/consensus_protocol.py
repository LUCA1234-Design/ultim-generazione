"""
coordination/consensus_protocol.py — Multi-Round Consensus Protocol.

Implements a 3-round voting protocol for robust signal confirmation:

  Round 1 — PROPOSE:  each agent submits independent score + direction
  Round 2 — DEBATE:   agents see other proposals; contrarian agent challenges;
                       scores can be revised based on peer signals
  Round 3 — COMMIT:   final weighted vote incorporating debate adjustments

The contrarian agent has special veto power in Round 2:
  - If contrarian score > 0.7, it reduces all other agents' scores by 30%
  - This implements a devil's advocate mechanism that prevents groupthink

Integration: wraps the existing DecisionFusion to add the debate round.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ConsensusProtocol")

# Debate parameters
_CONTRARIAN_VETO_THRESHOLD = 0.70   # contrarian score above this triggers penalty
_CONTRARIAN_PENALTY = 0.30          # fraction to reduce other scores by
_CONSENSUS_BONUS = 0.05             # bonus when all agents agree
_DISSENT_PENALTY = 0.03             # penalty when significant dissent exists
_DISSENT_THRESHOLD = 0.40           # fraction of disagreeing agents to trigger penalty


class ConsensusProtocol:
    """
    Multi-round consensus protocol for robust signal generation.

    Usage:
        protocol = ConsensusProtocol()
        result = protocol.full_consensus(agent_results)
        print(result['final_score'], result['direction'])
    """

    def __init__(self,
                 contrarian_veto_threshold: float = _CONTRARIAN_VETO_THRESHOLD,
                 contrarian_penalty: float = _CONTRARIAN_PENALTY):
        self.contrarian_veto_threshold = contrarian_veto_threshold
        self.contrarian_penalty = contrarian_penalty

    # ------------------------------------------------------------------

    def full_consensus(
        self,
        agent_results: Dict[str, Any],
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Execute full 3-round consensus protocol.

        Parameters
        ----------
        agent_results : dict mapping agent_name → AgentResult (or dict with score/direction)
        agent_weights : optional weight overrides per agent

        Returns
        -------
        dict with:
            'final_score'     : float [0,1]
            'direction'       : str 'long' | 'short' | 'neutral'
            'confidence'      : float [0,1]
            'round_1_score'   : float
            'round_2_score'   : float
            'contrarian_flag' : bool
            'consensus_level' : str 'strong' | 'moderate' | 'weak'
        """
        # Round 1: Collect proposals
        proposals = self.propose(agent_results, agent_weights)

        if not proposals:
            return self._empty_result()

        # Round 2: Debate (contrarian check + peer adjustment)
        debated = self.debate(proposals, agent_results)

        # Round 3: Commit (final vote)
        final = self.commit(debated)

        return final

    def propose(
        self,
        agent_results: Dict[str, Any],
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """
        Round 1: Collect proposals from all non-contrarian agents.

        Returns dict mapping agent_name → proposal dict.
        """
        proposals: Dict[str, Dict] = {}
        weights = agent_weights or {}

        for name, result in agent_results.items():
            if name == "contrarian":
                continue  # contrarian participates in Round 2

            score, direction, confidence = self._extract(result)
            weight = float(weights.get(name, 1.0))

            proposals[name] = {
                "score": score,
                "direction": direction,
                "confidence": confidence,
                "weight": weight,
                "original_score": score,
            }

        return proposals

    def debate(
        self,
        proposals: Dict[str, Dict],
        agent_results: Dict[str, Any],
    ) -> Dict[str, Dict]:
        """
        Round 2: Debate — agents adjust based on peer signals.

        Contrarian agent reviews the proposals and can penalise
        those that show excessive groupthink or extreme signals.
        """
        if not proposals:
            return proposals

        debated = {k: dict(v) for k, v in proposals.items()}

        # Determine consensus direction from Round 1
        directions = [p["direction"] for p in debated.values() if p["direction"] != "neutral"]
        if directions:
            long_count = sum(1 for d in directions if d == "long")
            short_count = sum(1 for d in directions if d == "short")
            total = len(directions)
            consensus_dir = "long" if long_count > short_count else "short"
            consensus_strength = max(long_count, short_count) / max(total, 1)
        else:
            consensus_dir = "neutral"
            consensus_strength = 0.0

        # Apply contrarian agent's Round 2 challenge
        contrarian_result = agent_results.get("contrarian")
        contrarian_flag = False

        if contrarian_result is not None:
            c_score, c_dir, c_conf = self._extract(contrarian_result)
            if c_score > self.contrarian_veto_threshold:
                contrarian_flag = True
                logger.info(
                    f"ConsensusProtocol: contrarian veto! score={c_score:.3f}, "
                    f"penalising all agents by {self.contrarian_penalty:.0%}"
                )
                for name in debated:
                    debated[name]["score"] *= (1 - self.contrarian_penalty)
                    debated[name]["contrarian_flag"] = True

        # Peer adjustment: reward agents that agree with consensus
        for name, prop in debated.items():
            if prop["direction"] == consensus_dir and consensus_strength > 0.7:
                # Strong consensus → slight confidence boost
                debated[name]["score"] = min(1.0, prop["score"] * 1.05)

        # Add debate metadata
        for name in debated:
            debated[name]["consensus_direction"] = consensus_dir
            debated[name]["consensus_strength"] = consensus_strength
            debated[name]["contrarian_flag"] = contrarian_flag

        return debated

    def commit(self, debated_proposals: Dict[str, Dict]) -> Dict:
        """
        Round 3: Final weighted vote to produce committed signal.

        Returns
        -------
        Final consensus result dict.
        """
        if not debated_proposals:
            return self._empty_result()

        long_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        all_confidences = []

        for name, prop in debated_proposals.items():
            w = float(prop.get("weight", 1.0))
            s = float(prop.get("score", 0.5))
            d = prop.get("direction", "neutral")
            c = float(prop.get("confidence", 0.5))

            total_weight += w
            all_confidences.append(c)

            if d == "long":
                long_score += w * s
            elif d == "short":
                short_score += w * s

        if total_weight < 1e-8:
            return self._empty_result()

        long_score /= total_weight
        short_score /= total_weight

        # Determine final direction and score
        if long_score > short_score:
            direction = "long"
            final_score = float(long_score)
        elif short_score > long_score:
            direction = "short"
            final_score = float(short_score)
        else:
            direction = "neutral"
            final_score = 0.0

        # Consensus bonus/penalty
        directions = [p.get("direction", "neutral") for p in debated_proposals.values()]
        agreement_count = sum(1 for d in directions if d == direction)
        agreement_rate = agreement_count / max(len(directions), 1)

        if agreement_rate > 1 - _DISSENT_THRESHOLD:
            final_score = min(1.0, final_score + _CONSENSUS_BONUS)
            consensus_level = "strong" if agreement_rate > 0.8 else "moderate"
        else:
            final_score = max(0.0, final_score - _DISSENT_PENALTY)
            consensus_level = "weak"

        avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.5
        round_1_scores = [p.get("original_score", p.get("score", 0)) for p in debated_proposals.values()]
        round_1_avg = float(np.mean(round_1_scores)) if round_1_scores else 0.0
        contrarian_flag = any(p.get("contrarian_flag", False) for p in debated_proposals.values())

        return {
            "final_score": final_score,
            "direction": direction,
            "confidence": avg_confidence * (0.9 if contrarian_flag else 1.0),
            "round_1_score": round_1_avg,
            "round_2_score": float(np.mean([p.get("score", 0) for p in debated_proposals.values()])),
            "contrarian_flag": contrarian_flag,
            "consensus_level": consensus_level,
            "agreement_rate": agreement_rate,
            "n_agents": len(debated_proposals),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(result: Any) -> Tuple[float, str, float]:
        """Extract (score, direction, confidence) from an AgentResult or dict."""
        if result is None:
            return 0.5, "neutral", 0.0
        if hasattr(result, "score"):
            return (
                float(result.score),
                str(getattr(result, "direction", "neutral")),
                float(getattr(result, "confidence", 0.5)),
            )
        if isinstance(result, dict):
            return (
                float(result.get("score", 0.5)),
                str(result.get("direction", "neutral")),
                float(result.get("confidence", 0.5)),
            )
        return 0.5, "neutral", 0.0

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "final_score": 0.0,
            "direction": "neutral",
            "confidence": 0.0,
            "round_1_score": 0.0,
            "round_2_score": 0.0,
            "contrarian_flag": False,
            "consensus_level": "weak",
            "agreement_rate": 0.0,
            "n_agents": 0,
        }
