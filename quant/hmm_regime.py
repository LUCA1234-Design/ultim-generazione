"""
quant/hmm_regime.py — Hidden Markov Model Regime Detection.

Implements an HMM with Gaussian emissions using the Baum-Welch algorithm
(via numpy/scipy) to model sequential market regime transitions.

Unlike the GMM-based RegimeAgent (which clusters static feature vectors),
this HMM models the *sequence* of regimes and exposes:
  - Viterbi-decoded most likely regime path
  - Forward-backward posterior probabilities
  - Transition matrix (e.g., P(volatile | trending))

Integration: the RegimeAgent can consult get_regime_probs() as a second
opinion, and the EvolutionEngine runs Loop #9 to push HMM posterior into
the RegimeAgent's confidence scores.
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("HMMRegime")

# ---- Constants ---------------------------------------------------------------

_N_STATES = 3          # trending / ranging / volatile
_STATE_NAMES = ["trending", "ranging", "volatile"]
_MIN_SAMPLES = 50      # minimum rows to fit
_MAX_ITER = 100        # EM iterations
_TOL = 1e-4            # convergence tolerance
_EPSILON = 1e-300      # numerical floor


class GaussianHMM:
    """
    Pure-numpy Gaussian Hidden Markov Model.

    Each state emits a multivariate Gaussian observation.
    Parameters are estimated with the Baum-Welch (EM) algorithm.
    """

    def __init__(self, n_states: int = _N_STATES, max_iter: int = _MAX_ITER, tol: float = _TOL):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self._is_fitted = False
        self._lock = threading.Lock()

        # HMM parameters (set after fit)
        self.pi: Optional[np.ndarray] = None      # initial distribution  (n_states,)
        self.A: Optional[np.ndarray] = None       # transition matrix     (n_states, n_states)
        self.mu: Optional[np.ndarray] = None      # emission means        (n_states, n_features)
        self.sigma: Optional[np.ndarray] = None   # emission covariances  (n_states, n_features, n_features)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit HMM parameters via Baum-Welch EM algorithm.

        Parameters
        ----------
        X : (T, n_features) observation sequence
        """
        T, n_features = X.shape
        K = self.n_states

        # ---- Initialise parameters with k-means-style split ----
        indices = np.random.choice(T, K, replace=False)
        mu = X[indices].copy().astype(float)
        sigma = np.array([np.eye(n_features) * 0.01 for _ in range(K)])
        pi = np.ones(K) / K
        A = np.ones((K, K)) / K

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # E-step: forward-backward
            log_emission = self._log_emission(X, mu, sigma)
            alpha, log_ll = self._forward(pi, A, log_emission)
            beta = self._backward(A, log_emission)
            gamma, xi = self._compute_gamma_xi(alpha, beta, A, log_emission)

            # M-step: update parameters
            pi = gamma[0] + _EPSILON
            pi /= pi.sum()

            A_new = xi.sum(axis=0) + _EPSILON
            A_new /= A_new.sum(axis=1, keepdims=True)

            for k in range(K):
                gk = gamma[:, k]
                Nk = gk.sum() + _EPSILON
                mu[k] = (gk[:, None] * X).sum(axis=0) / Nk
                diff = X - mu[k]
                sigma[k] = (gk[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / Nk
                # Regularise to avoid singular covariance
                sigma[k] += np.eye(n_features) * 1e-6

            A = A_new

            if abs(log_ll - prev_ll) < self.tol:
                logger.debug(f"HMM converged at iteration {iteration}, ll={log_ll:.4f}")
                break
            prev_ll = log_ll

        with self._lock:
            self.pi = pi
            self.A = A
            self.mu = mu
            self.sigma = sigma
            self._is_fitted = True
            self._n_features = n_features

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding: return most likely state sequence."""
        self._check_fitted()
        return self._viterbi(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities at each time step (T, n_states)."""
        self._check_fitted()
        log_emission = self._log_emission(X, self.mu, self.sigma)
        alpha, _ = self._forward(self.pi, self.A, log_emission)
        beta = self._backward(self.A, log_emission)
        gamma, _ = self._compute_gamma_xi(alpha, beta, self.A, log_emission)
        return gamma

    def get_transition_matrix(self) -> np.ndarray:
        """Return the (n_states, n_states) transition matrix A."""
        self._check_fitted()
        return self.A.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")

    @staticmethod
    def _log_multivariate_gaussian(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Log probability of each row in X under N(mu, sigma). Returns (T,)."""
        n = mu.shape[0]
        diff = X - mu
        try:
            L = np.linalg.cholesky(sigma)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff.T).T  # (T, n)
            maha = np.sum(solved ** 2, axis=1)
        except np.linalg.LinAlgError:
            # fallback: diagonal
            diag = np.maximum(np.diag(sigma), 1e-10)
            log_det = np.sum(np.log(diag))
            maha = np.sum(diff ** 2 / diag, axis=1)
        return -0.5 * (n * np.log(2 * np.pi) + log_det + maha)

    def _log_emission(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Return (T, K) matrix of log emission probabilities."""
        T = len(X)
        K = self.n_states
        log_b = np.zeros((T, K))
        for k in range(K):
            log_b[:, k] = self._log_multivariate_gaussian(X, mu[k], sigma[k])
        return log_b

    def _forward(self, pi: np.ndarray, A: np.ndarray,
                 log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        """Log-scale forward algorithm. Returns (alpha, log_likelihood)."""
        T, K = log_b.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(pi + _EPSILON) + log_b[0]
        log_A = np.log(A + _EPSILON)

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_b[t, k] + self._logsumexp(
                    log_alpha[t - 1] + log_A[:, k]
                )

        log_ll = self._logsumexp(log_alpha[-1])
        # Convert back to (normalised) probabilities for gamma/xi
        alpha = np.exp(log_alpha - log_ll)
        return alpha, log_ll

    def _backward(self, A: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        """Log-scale backward algorithm. Returns beta in probability space."""
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(A + _EPSILON)

        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = self._logsumexp(
                    log_A[k] + log_b[t + 1] + log_beta[t + 1]
                )

        # Normalise
        max_val = log_beta.max()
        beta = np.exp(log_beta - max_val)
        return beta

    def _compute_gamma_xi(
        self, alpha: np.ndarray, beta: np.ndarray,
        A: np.ndarray, log_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gamma (T, K) and xi (T-1, K, K)."""
        T, K = alpha.shape
        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True) + _EPSILON
        gamma /= row_sums

        b = np.exp(log_b)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * b[t + 1, j] * beta[t + 1, j]
            xi[t] /= (xi[t].sum() + _EPSILON)

        return gamma, xi

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for MAP state sequence."""
        log_b = self._log_emission(X, self.mu, self.sigma)
        T, K = log_b.shape
        log_A = np.log(self.A + _EPSILON)
        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)

        delta[0] = np.log(self.pi + _EPSILON) + log_b[0]
        for t in range(1, T):
            for k in range(K):
                trans = delta[t - 1] + log_A[:, k]
                psi[t, k] = np.argmax(trans)
                delta[t, k] = trans[psi[t, k]] + log_b[t, k]

        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    @staticmethod
    def _logsumexp(a: np.ndarray) -> float:
        max_a = np.max(a)
        return float(max_a + np.log(np.sum(np.exp(a - max_a)) + _EPSILON))


# ---- High-level wrapper used by the trading system --------------------------

class HMMRegimeDetector:
    """
    High-level interface wrapping GaussianHMM for market regime detection.

    Features extracted from OHLCV DataFrame:
      - Log returns
      - Realised volatility (rolling std)
      - Volume ratio (volume / rolling-mean volume)

    State mapping is determined post-hoc by sorting states by
    mean volatility: low → trending, medium → ranging, high → volatile.
    """

    def __init__(self, n_states: int = _N_STATES):
        self._hmm = GaussianHMM(n_states=n_states)
        self._is_fitted = False
        self._state_map: Dict[int, str] = {}  # internal state idx → regime name
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit the HMM on historical OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame with columns [open, high, low, close, volume]

        Returns
        -------
        True on success, False if insufficient data.
        """
        X = self._extract_features(df)
        if X is None or len(X) < _MIN_SAMPLES:
            logger.warning(f"HMMRegimeDetector.fit: insufficient data ({len(df) if df is not None else 0} rows)")
            return False
        try:
            self._hmm.fit(X)
            self._build_state_map(X)
            with self._lock:
                self._is_fitted = True
            logger.info(f"HMMRegimeDetector fitted on {len(X)} samples, state_map={self._state_map}")
            return True
        except Exception as exc:
            logger.error(f"HMMRegimeDetector.fit error: {exc}")
            return False

    def predict_regime(self, df: pd.DataFrame) -> Optional[str]:
        """
        Return the most likely current regime for the last observation.

        Returns
        -------
        str | None  — one of 'trending', 'ranging', 'volatile'
        """
        if not self._is_fitted:
            return None
        X = self._extract_features(df)
        if X is None or len(X) == 0:
            return None
        try:
            states = self._hmm.predict(X)
            last_state = int(states[-1])
            return self._state_map.get(last_state, "ranging")
        except Exception as exc:
            logger.debug(f"predict_regime error: {exc}")
            return None

    def get_regime_probs(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Return posterior probabilities over regimes for the last observation.

        Returns
        -------
        dict  e.g. {'trending': 0.6, 'ranging': 0.3, 'volatile': 0.1}
        """
        if not self._is_fitted:
            return None
        X = self._extract_features(df)
        if X is None or len(X) == 0:
            return None
        try:
            gamma = self._hmm.predict_proba(X)
            last = gamma[-1]
            result: Dict[str, float] = {}
            for k, p in enumerate(last):
                name = self._state_map.get(k, f"state_{k}")
                result[name] = float(p)
            return result
        except Exception as exc:
            logger.debug(f"get_regime_probs error: {exc}")
            return None

    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        """
        Return the transition matrix as a DataFrame indexed by regime names.
        """
        if not self._is_fitted:
            return None
        try:
            A = self._hmm.get_transition_matrix()
            names = [self._state_map.get(i, f"state_{i}") for i in range(self._hmm.n_states)]
            return pd.DataFrame(A, index=names, columns=names)
        except Exception as exc:
            logger.debug(f"get_transition_matrix error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(df: pd.DataFrame) -> Optional[np.ndarray]:
        """Build (T, 3) feature matrix: log_return, realised_vol, volume_ratio."""
        if df is None or len(df) < 20:
            return None
        try:
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)

            log_ret = np.diff(np.log(np.maximum(close, 1e-8)))
            vol = pd.Series(log_ret).rolling(10).std().fillna(0).values
            vol_ratio = volume[1:] / (pd.Series(volume[1:]).rolling(20).mean().fillna(1).values + 1e-8)

            X = np.column_stack([log_ret, vol, vol_ratio])
            # Remove NaN/Inf rows
            mask = np.isfinite(X).all(axis=1)
            return X[mask]
        except Exception as exc:
            logger.debug(f"_extract_features error: {exc}")
            return None

    def _build_state_map(self, X: np.ndarray) -> None:
        """
        Map internal state indices to regime names by sorting
        states on mean realised volatility (feature index 1).
        """
        vol_means = [float(self._hmm.mu[k, 1]) for k in range(self._hmm.n_states)]
        sorted_states = np.argsort(vol_means)  # ascending volatility
        # low vol → trending, medium → ranging, high → volatile
        regime_labels = ["trending", "ranging", "volatile"]
        self._state_map = {int(s): regime_labels[i] for i, s in enumerate(sorted_states)}
