"""
quant/bayesian_inference.py — Bayesian Online Learning for Trading.

Implements conjugate-prior Bayesian updating for two key quantities:

1. **Win Rate** — Beta-Binomial model
   Prior: Beta(alpha, beta)
   Likelihood: Binomial
   Posterior: Beta(alpha + wins, beta + losses)

2. **Returns** — Normal-Inverse-Gamma (NIG) model
   Prior: NIG(mu0, kappa0, alpha0, beta0)
   Likelihood: Normal with unknown mean and variance
   Posterior: updated NIG parameters

Integration (Loop #10): the EvolutionEngine calls `update_prior(obs)`
after every trade close; all agents can request `get_posterior()` to
obtain calibrated Bayesian estimates rather than frequentist point estimates.
"""
import logging
import math
import threading
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger("BayesianInference")


class BetaBinomialModel:
    """
    Bayesian win-rate estimator using Beta-Binomial conjugacy.

    Prior  : Beta(alpha, beta)
    Update : observe win (+1 to alpha) or loss (+1 to beta)
    Output : posterior mean, credible interval, predictive probability
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        """
        Parameters
        ----------
        alpha0 : float — prior pseudo-wins (1 = uniform / Bayes-Laplace)
        beta0  : float — prior pseudo-losses
        """
        self._alpha = float(alpha0)
        self._beta = float(beta0)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def update(self, win: bool) -> None:
        """Update posterior with a single trade outcome."""
        with self._lock:
            if win:
                self._alpha += 1.0
            else:
                self._beta += 1.0

    def update_batch(self, wins: int, losses: int) -> None:
        """Batch update with aggregate counts."""
        with self._lock:
            self._alpha += float(wins)
            self._beta += float(losses)

    def posterior_mean(self) -> float:
        """Posterior mean of win rate = alpha / (alpha + beta)."""
        with self._lock:
            return self._alpha / (self._alpha + self._beta)

    def credible_interval(self, credibility: float = 0.95) -> Tuple[float, float]:
        """
        Bayesian credible interval for the win rate.

        Returns (lower, upper) bounds.
        """
        with self._lock:
            lo = float(stats.beta.ppf((1 - credibility) / 2, self._alpha, self._beta))
            hi = float(stats.beta.ppf((1 + credibility) / 2, self._alpha, self._beta))
        return lo, hi

    def predictive_probability(self) -> float:
        """Probability that the *next* trade is a win (posterior predictive)."""
        return self.posterior_mean()

    def get_summary(self) -> Dict:
        lo, hi = self.credible_interval()
        return {
            "win_rate_bayes": self.posterior_mean(),
            "credible_lo_95": lo,
            "credible_hi_95": hi,
            "alpha": self._alpha,
            "beta": self._beta,
            "n_trades": int(self._alpha + self._beta - 2),  # subtract priors
        }

    def reset(self, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        with self._lock:
            self._alpha = float(alpha0)
            self._beta = float(beta0)


class NormalInverseGammaModel:
    """
    Bayesian return estimator using Normal-Inverse-Gamma conjugacy.

    Models: X ~ Normal(mu, sigma²), with prior:
        mu    | sigma² ~ Normal(mu0, sigma² / kappa0)
        sigma²         ~ InvGamma(alpha0, beta0)

    Predictive distribution: Student-t with 2*alpha0 degrees of freedom.
    """

    def __init__(self,
                 mu0: float = 0.0,
                 kappa0: float = 1.0,
                 alpha0: float = 1.0,
                 beta0: float = 1e-4):
        """
        Parameters
        ----------
        mu0    : prior mean
        kappa0 : prior confidence in mean (higher = more confident)
        alpha0 : prior shape for variance (higher = more data in prior)
        beta0  : prior rate for variance
        """
        self._mu0 = float(mu0)
        self._kappa0 = float(kappa0)
        self._alpha0 = float(alpha0)
        self._beta0 = float(beta0)

        self._mu_n = self._mu0
        self._kappa_n = self._kappa0
        self._alpha_n = self._alpha0
        self._beta_n = self._beta0
        self._n = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """Update posterior with a single return observation."""
        with self._lock:
            self._n += 1
            n = 1  # single observation
            x_bar = x

            kappa_new = self._kappa_n + n
            mu_new = (self._kappa_n * self._mu_n + n * x_bar) / kappa_new
            alpha_new = self._alpha_n + n / 2.0
            beta_new = (
                self._beta_n
                + 0.5 * n * (x_bar - self._mu_n) ** 2 * self._kappa_n / kappa_new
            )

            self._mu_n = mu_new
            self._kappa_n = kappa_new
            self._alpha_n = alpha_new
            self._beta_n = beta_new

    def update_batch(self, returns: np.ndarray) -> None:
        """Batch update with an array of returns."""
        for r in returns:
            self.update(float(r))

    def posterior_mean(self) -> float:
        """Posterior predictive mean of returns."""
        with self._lock:
            return self._mu_n

    def posterior_std(self) -> float:
        """
        Posterior predictive standard deviation (Student-t scale).
        """
        with self._lock:
            df = 2.0 * self._alpha_n
            if df <= 0:
                return float("inf")
            scale = math.sqrt(self._beta_n * (self._kappa_n + 1) / (self._kappa_n * self._alpha_n))
        return scale

    def predictive_interval(self, credibility: float = 0.95) -> Tuple[float, float]:
        """
        Predictive credible interval for the next return.
        Uses Student-t predictive distribution.
        """
        with self._lock:
            df = 2.0 * self._alpha_n
            scale = math.sqrt(
                max(self._beta_n, 1e-12)
                * (self._kappa_n + 1)
                / (max(self._kappa_n, 1e-12) * max(self._alpha_n, 1e-12))
            )
            mu = self._mu_n

        lo = float(stats.t.ppf((1 - credibility) / 2, df, loc=mu, scale=scale))
        hi = float(stats.t.ppf((1 + credibility) / 2, df, loc=mu, scale=scale))
        return lo, hi

    def get_summary(self) -> Dict:
        lo, hi = self.predictive_interval()
        return {
            "expected_return_bayes": self.posterior_mean(),
            "return_std_bayes": self.posterior_std(),
            "predictive_lo_95": lo,
            "predictive_hi_95": hi,
            "n_observations": self._n,
        }

    def reset(self) -> None:
        with self._lock:
            self._mu_n = self._mu0
            self._kappa_n = self._kappa0
            self._alpha_n = self._alpha0
            self._beta_n = self._beta0
            self._n = 0


# ---- High-level facade -------------------------------------------------------

class BayesianOnlineLearner:
    """
    Combined Bayesian learner that tracks both win rate and returns.

    Used by the EvolutionEngine for Loop #10:
        learner.update_prior(pnl=0.02, win=True)
        summary = learner.get_posterior()
    """

    def __init__(self):
        self._win_rate_model = BetaBinomialModel(alpha0=2.0, beta0=2.0)
        self._return_model = NormalInverseGammaModel()
        self._lock = threading.Lock()

    def update_prior(self, observation: Dict) -> None:
        """
        Update all Bayesian models with a new trade observation.

        Parameters
        ----------
        observation : dict with keys:
            'pnl'  : float — trade P&L (can be None)
            'win'  : bool  — was the trade profitable
        """
        pnl = observation.get("pnl")
        win = observation.get("win", (pnl or 0) > 0)

        self._win_rate_model.update(bool(win))
        if pnl is not None:
            self._return_model.update(float(pnl))

    def get_posterior(self) -> Dict:
        """Return combined posterior summary from all models."""
        wr = self._win_rate_model.get_summary()
        ret = self._return_model.get_summary()
        return {**wr, **ret}

    def predictive_probability(self) -> float:
        """Probability next trade wins (Bayesian win rate posterior mean)."""
        return self._win_rate_model.predictive_probability()

    def expected_return(self) -> float:
        """Bayesian expected return for next trade."""
        return self._return_model.posterior_mean()

    def reset(self) -> None:
        self._win_rate_model.reset()
        self._return_model.reset()
