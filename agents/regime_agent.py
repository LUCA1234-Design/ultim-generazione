"""
Regime Agent for V17.
Uses sklearn GaussianMixture to identify market regimes from feature vectors.
Features: ADX, BB/KC ratio, Z-score, RSI, volume ratio, EMA slope.
Outputs: probability distribution over N regimes.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import rsi, atr, adx, bollinger_bands, keltner_channels, zscore, ema_slope, volume_ratio
from config.settings import REGIME_N_COMPONENTS, REGIME_NAMES, REGIME_LOOKBACK

logger = logging.getLogger("RegimeAgent")


class RegimeAgent(BaseAgent):
    """Detects market regime using a fitted Gaussian Mixture Model."""

    REGIMES = REGIME_NAMES  # ["trending", "ranging", "volatile"]

    def __init__(self, n_components: int = REGIME_N_COMPONENTS):
        super().__init__("regime", initial_weight=0.20)
        self.n_components = n_components
        # Per-symbol/interval models: (symbol, interval) -> (scaler, gmm, regime_labels)
        self._scalers: Dict[tuple, StandardScaler] = {}
        self._gmms: Dict[tuple, GaussianMixture] = {}
        self._regime_labels_map: Dict[tuple, Dict[int, str]] = {}
        self._fitted_keys: set = set()
        self._feature_cache: Dict[str, np.ndarray] = {}  # key -> last features

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract regime features from a DataFrame.

        Returns array of shape (n_samples, n_features).
        """
        if len(df) < REGIME_LOOKBACK + 30:
            return None
        try:
            close = df["close"]
            rsi_series = rsi(close, 14)
            adx_series, di_plus, di_minus = adx(df, 14)
            bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
            kc_upper, kc_mid, kc_lower = keltner_channels(df, 20, 1.5)
            z = zscore(close, 20)
            vol_r = volume_ratio(df, 20)
            ema_slp = ema_slope(close, 20, 5)

            bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
            kc_width = (kc_upper - kc_lower) / kc_mid.replace(0, np.nan)
            bb_kc_ratio = bb_width / kc_width.replace(0, np.nan)

            combined = pd.DataFrame({
                "adx": adx_series,
                "bb_kc_ratio": bb_kc_ratio,
                "zscore": z,
                "rsi": rsi_series,
                "vol_ratio": vol_r,
                "ema_slope": ema_slp,
            }).dropna()

            if len(combined) < self.n_components * 5:
                return None
            return combined.values
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, symbol: str, interval: str, df: pd.DataFrame) -> bool:
        """Fit the GMM on historical data for a specific symbol/interval pair."""
        features = self._extract_features(df)
        if features is None:
            return False
        key = (symbol, interval)
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(scaled)
            if not hasattr(gmm, "means_"):
                raise ValueError("GMM did not converge (means_ missing)")
            self._scalers[key] = scaler
            self._gmms[key] = gmm
            self._fitted_keys.add(key)
            self._assign_regime_labels(key, scaler, gmm, features)
            logger.debug(f"RegimeAgent fitted on {symbol}/{interval} ({len(features)} samples)")
            return True
        except Exception as e:
            logger.warning(f"GMM fit error [{symbol}/{interval}]: {e}")
            return False

    def _assign_regime_labels(self, key: tuple, scaler: StandardScaler,
                               gmm: GaussianMixture, features: np.ndarray) -> None:
        """Assign human-readable labels to GMM components based on mean ADX & BB/KC ratio."""
        if not hasattr(gmm, "means_"):
            return
        means = scaler.inverse_transform(gmm.means_)
        # means columns: adx, bb_kc_ratio, zscore, rsi, vol_ratio, ema_slope
        adx_means = means[:, 0]
        bb_kc_means = means[:, 1]
        vol_means = means[:, 4]

        # Rank components
        regime_labels: Dict[int, str] = {}
        sorted_by_adx = np.argsort(adx_means)[::-1]
        for rank, idx in enumerate(sorted_by_adx):
            if rank == 0:
                # Highest ADX → trending
                if bb_kc_means[idx] > 1.0:
                    regime_labels[idx] = "volatile"
                else:
                    regime_labels[idx] = "trending"
            elif bb_kc_means[idx] < 0.7:
                regime_labels[idx] = "ranging"
            else:
                regime_labels[idx] = "volatile"

        # Fill any unlabelled
        used = set(regime_labels.values())
        all_regimes = set(self.REGIMES)
        missing = list(all_regimes - used)
        for idx in range(self.n_components):
            if idx not in regime_labels and missing:
                regime_labels[idx] = missing.pop()

        self._regime_labels_map[key] = regime_labels

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_regime_probs(self, symbol: str, interval: str, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Return regime probability distribution for the latest observation."""
        key = (symbol, interval)
        if key not in self._fitted_keys:
            return None
        gmm = self._gmms.get(key)
        scaler = self._scalers.get(key)
        regime_labels = self._regime_labels_map.get(key, {})
        if gmm is None or scaler is None:
            return None
        features = self._extract_features(df)
        if features is None or len(features) == 0:
            return None
        try:
            last = features[-1:, :]
            scaled = scaler.transform(last)
            probs = gmm.predict_proba(scaled)[0]
            result = {}
            for i, p in enumerate(probs):
                label = regime_labels.get(i, f"regime_{i}")
                result[label] = result.get(label, 0.0) + float(p)
            return result
        except Exception as e:
            logger.debug(f"get_regime_probs error [{symbol}/{interval}]: {e}")
            return None

    def current_regime(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        """Return the most probable regime label."""
        probs = self.get_regime_probs(symbol, interval, df)
        if probs is None:
            return "unknown"
        return max(probs, key=probs.get)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df: pd.DataFrame) -> Optional[AgentResult]:
        if df is None or len(df) < 60:
            return None

        key = (symbol, interval)
        # Fit on-the-fly if not yet fitted for this symbol/interval
        if key not in self._fitted_keys:
            self.fit(symbol, interval, df)
            if key not in self._fitted_keys:
                return AgentResult(
                    agent_name=self.name,
                    symbol=symbol,
                    interval=interval,
                    score=0.5,
                    direction="neutral",
                    confidence=0.0,
                    details=["GMM not fitted — insufficient data"],
                )

        probs = self.get_regime_probs(symbol, interval, df)
        if probs is None:
            return None

        trending_prob = probs.get("trending", 0.0)
        ranging_prob = probs.get("ranging", 0.0)
        volatile_prob = probs.get("volatile", 0.0)
        regime = max(probs, key=probs.get)
        regime_prob = probs[regime]

        # Score: trending regime is most favourable for directional trades
        if regime == "trending":
            score = 0.7 + 0.3 * trending_prob
        elif regime == "volatile":
            score = 0.4 + 0.2 * volatile_prob
        else:  # ranging
            score = 0.3 + 0.2 * ranging_prob

        # Determine direction hint from ADX +DI/-DI
        try:
            adx_s, di_plus, di_minus = adx(df, 14)
            direction = "long" if di_plus.iloc[-1] > di_minus.iloc[-1] else "short"
        except Exception:
            direction = "neutral"

        details = [
            f"regime={regime}({regime_prob:.0%})",
            f"trending={trending_prob:.0%}",
            f"ranging={ranging_prob:.0%}",
            f"volatile={volatile_prob:.0%}",
        ]

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=float(np.clip(score, 0.0, 1.0)),
            direction=direction,
            confidence=regime_prob,
            details=details,
            metadata={"regime_probs": probs, "regime": regime},
        )
