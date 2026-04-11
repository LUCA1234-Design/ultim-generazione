"""
rl/ppo_agent.py — Proximal Policy Optimization (Actor-Critic).

Pure-numpy implementation of PPO for discrete action spaces.

Architecture:
  - Actor  (policy): 2-layer MLP → softmax over actions
  - Critic (value) : 2-layer MLP → scalar state value

Training:
  - Collects trajectories via rollout
  - Computes GAE (Generalised Advantage Estimation)
  - Updates with clipped PPO surrogate objective
  - No external deep-learning framework required

Integration (Loop #8): RL → Execution
  The trained policy provides a suggested action/sizing signal that
  the ExecutionEngine can use to confirm or scale trade size.
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PPOAgent")

# Hyperparameters (overridable via config)
_HIDDEN_SIZE = 64
_LEARNING_RATE = 3e-4
_GAMMA = 0.99           # discount factor
_LAMBDA_GAE = 0.95      # GAE lambda
_CLIP_EPSILON = 0.2     # PPO clip
_VALUE_COEF = 0.5       # value loss coefficient
_ENTROPY_COEF = 0.01    # entropy bonus
_MAX_GRAD_NORM = 0.5
_EPOCHS = 4             # optimisation epochs per update


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max())
    return ex / (ex.sum() + 1e-10)


class MLP:
    """
    Simple multi-layer perceptron with numpy.

    Layers: input → hidden → hidden → output (no activation on output).
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int, lr: float = _LEARNING_RATE):
        self.lr = lr
        scale1 = np.sqrt(2.0 / in_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.W1 = np.random.randn(in_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, out_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(out_size)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass. Returns output and cache for backward."""
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        cache = {"x": x, "h1": h1, "h2": h2}
        return out, cache

    def backward(self, d_out: np.ndarray, cache: Dict) -> None:
        """Backward pass with gradient descent update."""
        x, h1, h2 = cache["x"], cache["h1"], cache["h2"]

        # Layer 3
        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0) if d_out.ndim == 2 else d_out
        d_h2 = d_out @ self.W3.T

        # Layer 2 (ReLU backprop)
        d_h2_pre = d_h2 * (h2 > 0)
        dW2 = h1.T @ d_h2_pre
        db2 = d_h2_pre.sum(axis=0) if d_h2_pre.ndim == 2 else d_h2_pre
        d_h1 = d_h2_pre @ self.W2.T

        # Layer 1 (ReLU backprop)
        d_h1_pre = d_h1 * (h1 > 0)
        dW1 = x.T @ d_h1_pre
        db1 = d_h1_pre.sum(axis=0) if d_h1_pre.ndim == 2 else d_h1_pre

        # Clip gradients
        for grad in [dW1, dW2, dW3]:
            np.clip(grad, -_MAX_GRAD_NORM, _MAX_GRAD_NORM, out=grad)

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3


class PPOAgent:
    """
    Proximal Policy Optimization agent with numpy Actor-Critic.

    The actor outputs a probability distribution over discrete actions.
    The critic estimates the state value V(s).
    """

    def __init__(self,
                 n_features: int = 12,
                 n_actions: int = 8,
                 hidden_size: int = _HIDDEN_SIZE,
                 lr: float = _LEARNING_RATE,
                 gamma: float = _GAMMA,
                 gae_lambda: float = _LAMBDA_GAE,
                 clip_epsilon: float = _CLIP_EPSILON):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self._actor = MLP(n_features, hidden_size, n_actions, lr=lr)
        self._critic = MLP(n_features, hidden_size, 1, lr=lr)
        self._is_trained = False
        self._lock = threading.Lock()
        self._total_updates = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Sample action from policy given observation.

        Returns
        -------
        (action_idx, log_prob)
        """
        with self._lock:
            logits, _ = self._actor.forward(state.reshape(1, -1))
            probs = _softmax(logits[0])
            action = int(np.random.choice(self.n_actions, p=probs))
            log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return full action probability distribution."""
        with self._lock:
            logits, _ = self._actor.forward(state.reshape(1, -1))
            return _softmax(logits[0])

    def get_value(self, state: np.ndarray) -> float:
        """Estimate state value V(s)."""
        with self._lock:
            val, _ = self._critic.forward(state.reshape(1, -1))
            return float(val[0, 0])

    def best_action(self, state: np.ndarray) -> int:
        """Return greedy (argmax) action — used for online inference."""
        probs = self.get_action_probabilities(state)
        return int(np.argmax(probs))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, trajectories: List[Dict]) -> Dict:
        """
        PPO update from a list of trajectories.

        Each trajectory dict contains:
            'states'      : (T, n_features)
            'actions'     : (T,)
            'log_probs'   : (T,)  — old log probs
            'rewards'     : (T,)
            'values'      : (T,)
            'dones'       : (T,)

        Returns
        -------
        dict with training statistics (loss, entropy, etc.)
        """
        if not trajectories:
            return {}

        # Concatenate all trajectories
        states = np.vstack([t["states"] for t in trajectories])
        actions = np.concatenate([t["actions"] for t in trajectories])
        old_log_probs = np.concatenate([t["log_probs"] for t in trajectories])
        rewards = np.concatenate([t["rewards"] for t in trajectories])
        values = np.concatenate([t["values"] for t in trajectories])
        dones = np.concatenate([t["dones"] for t in trajectories])

        # Compute GAE advantages
        advantages, returns_target = self._compute_gae(rewards, values, dones)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []

        with self._lock:
            for _ in range(_EPOCHS):
                # Shuffle mini-batch
                idx = np.random.permutation(len(states))
                for i in idx:
                    s = states[i:i+1]
                    a = int(actions[i])
                    adv = float(advantages[i])
                    ret = float(returns_target[i])
                    old_lp = float(old_log_probs[i])

                    # Actor forward
                    logits, actor_cache = self._actor.forward(s)
                    probs = _softmax(logits[0])
                    new_log_prob = float(np.log(probs[a] + 1e-10))

                    # PPO ratio and clipped objective
                    ratio = np.exp(new_log_prob - old_lp)
                    unclipped = ratio * adv
                    clipped = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                    actor_loss = -min(unclipped, clipped)

                    # Entropy bonus (encourages exploration)
                    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                    actor_loss -= _ENTROPY_COEF * entropy

                    # Actor backward
                    d_logits = probs.copy()
                    d_logits[a] -= 1.0
                    # Scale gradient by PPO clipped objective.
                    # The gradient is suppressed (scale=0) when the ratio IS outside
                    # the clip bounds — not when it is inside (the previous implementation
                    # was inverted, zeroing the gradient for in-range samples).
                    ratio_clipped = ratio < (1 - self.clip_epsilon) or ratio > (1 + self.clip_epsilon)
                    if ratio_clipped:
                        # Outside clip bounds: gradient is blocked
                        scale = 0.0
                    else:
                        # Inside clip bounds: normal PPO gradient
                        scale = -adv * ratio
                    d_logits = d_logits.reshape(1, -1) * scale
                    self._actor.backward(d_logits, actor_cache)

                    # Critic forward and backward
                    val_pred, critic_cache = self._critic.forward(s)
                    val_err = float(val_pred[0, 0]) - ret
                    critic_loss = 0.5 * val_err ** 2

                    d_val = np.array([[val_err]]) * _VALUE_COEF
                    self._critic.backward(d_val, critic_cache)

                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    entropies.append(entropy)

            self._is_trained = True
            self._total_updates += 1

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "n_samples": len(states),
            "total_updates": self._total_updates,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> bool:
        """Save model parameters to a numpy .npz file."""
        try:
            np.savez(
                path,
                actor_W1=self._actor.W1, actor_b1=self._actor.b1,
                actor_W2=self._actor.W2, actor_b2=self._actor.b2,
                actor_W3=self._actor.W3, actor_b3=self._actor.b3,
                critic_W1=self._critic.W1, critic_b1=self._critic.b1,
                critic_W2=self._critic.W2, critic_b2=self._critic.b2,
                critic_W3=self._critic.W3, critic_b3=self._critic.b3,
            )
            return True
        except Exception as exc:
            logger.error(f"PPOAgent.save error: {exc}")
            return False

    def load(self, path: str) -> bool:
        """Load model parameters from a .npz file."""
        try:
            data = np.load(path)
            self._actor.W1 = data["actor_W1"]
            self._actor.b1 = data["actor_b1"]
            self._actor.W2 = data["actor_W2"]
            self._actor.b2 = data["actor_b2"]
            self._actor.W3 = data["actor_W3"]
            self._actor.b3 = data["actor_b3"]
            self._critic.W1 = data["critic_W1"]
            self._critic.b1 = data["critic_b1"]
            self._critic.W2 = data["critic_W2"]
            self._critic.b2 = data["critic_b2"]
            self._critic.W3 = data["critic_W3"]
            self._critic.b3 = data["critic_b3"]
            self._is_trained = True
            return True
        except Exception as exc:
            logger.error(f"PPOAgent.load error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalised Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0

        # Append bootstrap value 0 for terminal step
        values_ext = np.append(values, 0.0)

        for t in reversed(range(T)):
            next_val = values_ext[t + 1] * (1.0 - float(dones[t]))
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        returns_target = advantages + values
        return advantages, returns_target
