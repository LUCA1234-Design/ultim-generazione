"""
rl/experience_replay.py — Prioritized Experience Replay Buffer.

Implements a circular replay buffer with priority-based sampling
using a sum-tree data structure for O(log n) operations.

Priority is based on TD-error: transitions with high prediction
error are sampled more frequently, improving sample efficiency.

Integration: used by PPOAgent during offline training and supports
batch sampling for policy gradient updates.
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ExperienceReplay")

_ALPHA = 0.6       # priority exponent (0 = uniform, 1 = full priority)
_BETA_START = 0.4  # IS correction exponent (anneals to 1.0)
_BETA_END = 1.0
_EPSILON = 1e-6    # minimum priority


class SumTree:
    """
    Binary sum-tree for efficient priority sampling.

    Leaf nodes store priorities; internal nodes store sums.
    O(log n) for update and sampling operations.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity)  # sum-tree array
        self._data_ptr = 0
        self._n_entries = 0

    def add(self, priority: float, idx: int) -> None:
        """Add/update priority at circular buffer position idx."""
        tree_idx = idx + self.capacity
        self._update(tree_idx, priority)
        self._data_ptr = (self._data_ptr + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at position idx."""
        tree_idx = idx + self.capacity
        self._update(tree_idx, priority)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample `batch_size` indices proportional to priority.

        Returns
        -------
        (indices, priorities) arrays of shape (batch_size,)
        """
        total = self._tree[1]
        if total < 1e-10:
            indices = np.random.randint(0, max(self._n_entries, 1), size=batch_size)
            priorities = np.ones(batch_size)
            return indices, priorities

        segment = total / batch_size
        indices = np.zeros(batch_size, dtype=int)
        priorities = np.zeros(batch_size)

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx = self._retrieve(1, value)
            data_idx = leaf_idx - self.capacity
            indices[i] = data_idx % max(self._n_entries, 1)
            priorities[i] = self._tree[leaf_idx]

        return indices, priorities

    def total_priority(self) -> float:
        return float(self._tree[1])

    # ------------------------------------------------------------------

    def _update(self, tree_idx: int, priority: float) -> None:
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self._tree[tree_idx] += change

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= len(self._tree):
            return idx
        if value <= self._tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._tree[left])


class PrioritizedReplayBuffer:
    """
    Circular replay buffer with prioritized sampling.

    Transitions are stored as (state, action, reward, next_state, done).
    Sampling is proportional to priority^alpha with IS correction.
    """

    def __init__(self,
                 capacity: int = 10_000,
                 alpha: float = _ALPHA,
                 beta_start: float = _BETA_START):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self._beta_increment = (_BETA_END - beta_start) / 100_000  # anneal over time

        self._tree = SumTree(capacity)
        self._states: List[Optional[np.ndarray]] = [None] * capacity
        self._actions: np.ndarray = np.zeros(capacity, dtype=int)
        self._rewards: np.ndarray = np.zeros(capacity, dtype=float)
        self._next_states: List[Optional[np.ndarray]] = [None] * capacity
        self._dones: np.ndarray = np.zeros(capacity, dtype=bool)

        self._write_ptr = 0
        self._n_entries = 0
        self._max_priority = 1.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with maximum current priority."""
        with self._lock:
            idx = self._write_ptr
            self._states[idx] = state.copy()
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._next_states[idx] = next_state.copy()
            self._dones[idx] = done

            priority = float(self._max_priority ** self.alpha)
            self._tree.add(priority, idx)

            self._write_ptr = (self._write_ptr + 1) % self.capacity
            self._n_entries = min(self._n_entries + 1, self.capacity)

    def sample(self, batch_size: int) -> Optional[Dict]:
        """
        Sample a batch of transitions with IS weights.

        Returns
        -------
        dict with keys:
            'states', 'actions', 'rewards', 'next_states', 'dones'
            'weights' (IS importance sampling weights)
            'indices' (buffer indices for priority update)
        or None if buffer too small.
        """
        with self._lock:
            if self._n_entries < batch_size:
                return None

            indices, priorities = self._tree.sample(batch_size)

            # Clamp indices
            indices = np.clip(indices, 0, self._n_entries - 1)

            # Importance sampling weights
            total = self._tree.total_priority()
            probs = priorities / (total + 1e-10)
            weights = (self._n_entries * probs) ** (-self.beta)
            weights /= weights.max() + 1e-10  # normalise

            # Anneal beta
            self.beta = min(self.beta + self._beta_increment, _BETA_END)

            states = np.vstack([self._states[i] for i in indices])
            actions = self._actions[indices]
            rewards = self._rewards[indices]
            next_states = np.vstack([self._next_states[i] for i in indices])
            dones = self._dones[indices]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones.astype(float),
            "weights": weights.astype(float),
            "indices": indices,
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on absolute TD errors."""
        with self._lock:
            for idx, err in zip(indices, td_errors):
                priority = float((abs(err) + _EPSILON) ** self.alpha)
                self._tree.update(int(idx), priority)
                self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._n_entries

    def is_ready(self, min_size: int) -> bool:
        """True if buffer has at least min_size transitions."""
        return self._n_entries >= min_size
