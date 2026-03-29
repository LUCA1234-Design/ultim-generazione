"""
AI Service for V17.

Wraps calls to the local LM Studio AI (scout + analyst) with:
- Timeout handling
- Circuit breaker pattern (auto-disables after N consecutive failures)
- Graceful fallback to rule-based analysis via ai_fallback.py
- health_check() endpoint
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

from config.settings import (
    AI_URL_SCOUT,
    AI_MODEL_SCOUT,
    AI_URL_ANALYST,
    AI_MODEL_ANALYST,
    AI_TIMEOUT,
    AI_CALL_COOLDOWN,
)
from services.ai_fallback import generate_fallback_analysis

logger = logging.getLogger("AIService")

# Default circuit-breaker threshold
_DEFAULT_FAILURE_THRESHOLD = 3


class CircuitBreaker:
    """Simple circuit breaker with open/half-open/closed states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
                 cooldown_seconds: float = float(AI_CALL_COOLDOWN)):
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._consecutive_failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = self.CLOSED

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._cooldown:
                    self._state = self.HALF_OPEN
                    logger.info("CircuitBreaker → HALF_OPEN (cooldown expired, probing AI)")
        return self._state

    def is_available(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        self._consecutive_failures = 0
        if self._state != self.CLOSED:
            logger.info("CircuitBreaker → CLOSED (AI recovered)")
        self._state = self.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self._failure_threshold:
            if self._state != self.OPEN:
                logger.warning(
                    f"CircuitBreaker → OPEN after {self._consecutive_failures} "
                    f"consecutive failures. Cooldown: {self._cooldown}s"
                )
            self._state = self.OPEN

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "cooldown_seconds": self._cooldown,
            "seconds_until_retry": max(
                0,
                self._cooldown - (time.time() - (self._last_failure_time or time.time()))
            ) if self._state == self.OPEN else 0,
        }


class AIService:
    """
    High-level wrapper for LM Studio AI calls.

    Usage
    -----
    service = AIService()

    # Scout (fast model)
    resp = service.call_scout(messages=[{"role": "user", "content": "..."}])

    # Analyst (advanced model)
    resp = service.call_analyst(messages=[{"role": "user", "content": "..."}])

    # Fallback-aware call
    resp = service.call_with_fallback("scout", messages, symbol="BTCUSDT", interval="1h")
    """

    def __init__(
        self,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: Optional[float] = None,
    ):
        cooldown = cooldown_seconds if cooldown_seconds is not None else float(AI_CALL_COOLDOWN)
        self._cb_scout = CircuitBreaker(failure_threshold, cooldown)
        self._cb_analyst = CircuitBreaker(failure_threshold, cooldown)
        self._total_calls = 0
        self._total_failures = 0
        self._total_fallbacks = 0

    # ------------------------------------------------------------------
    # Internal HTTP call
    # ------------------------------------------------------------------

    def _post(self, url: str, model: str, messages: list,
              circuit_breaker: CircuitBreaker) -> Dict[str, Any]:
        """
        POST a chat-completion request.
        Returns the raw API response dict.
        Raises on HTTP / timeout errors.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
        }
        self._total_calls += 1
        try:
            resp = requests.post(url, json=payload, timeout=AI_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            circuit_breaker.record_success()
            return data
        except requests.exceptions.Timeout:
            self._total_failures += 1
            circuit_breaker.record_failure()
            logger.warning(f"AI call timed out after {AI_TIMEOUT}s — {url}")
            raise
        except requests.exceptions.ConnectionError:
            self._total_failures += 1
            circuit_breaker.record_failure()
            logger.warning(f"AI service unreachable — {url}")
            raise
        except requests.exceptions.HTTPError as exc:
            self._total_failures += 1
            circuit_breaker.record_failure()
            logger.error(f"AI HTTP error {exc.response.status_code} — {url}: {exc}")
            raise
        except Exception as exc:
            self._total_failures += 1
            circuit_breaker.record_failure()
            logger.error(f"AI unexpected error — {url}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Public call methods
    # ------------------------------------------------------------------

    def call_scout(self, messages: list) -> Optional[Dict[str, Any]]:
        """
        Call the scout (fast) model.
        Returns the API response or None if circuit breaker is open.
        """
        if not self._cb_scout.is_available():
            logger.debug("Scout circuit breaker OPEN — skipping AI call")
            return None
        try:
            return self._post(AI_URL_SCOUT, AI_MODEL_SCOUT, messages, self._cb_scout)
        except Exception:
            return None

    def call_analyst(self, messages: list) -> Optional[Dict[str, Any]]:
        """
        Call the analyst (advanced) model.
        Returns the API response or None if circuit breaker is open.
        """
        if not self._cb_analyst.is_available():
            logger.debug("Analyst circuit breaker OPEN — skipping AI call")
            return None
        try:
            return self._post(AI_URL_ANALYST, AI_MODEL_ANALYST, messages, self._cb_analyst)
        except Exception:
            return None

    def call_with_fallback(
        self,
        model_type: str,
        messages: list,
        symbol: str = "UNKNOWN",
        interval: str = "1h",
        df=None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call AI; if unavailable return a rule-based fallback response.

        Parameters
        ----------
        model_type : "scout" or "analyst"
        messages   : List of chat messages
        symbol     : Trading pair (for fallback labelling)
        interval   : Candle interval (for fallback labelling)
        df         : Optional DataFrame for indicator extraction in fallback
        indicators : Optional pre-computed indicator dict for fallback

        Returns
        -------
        dict — either raw AI response or fallback dict with ``degraded=True``
        """
        caller = self.call_scout if model_type == "scout" else self.call_analyst
        raw = caller(messages)
        if raw is not None:
            # Extract text content from the standard OpenAI-compatible response
            try:
                text = raw["choices"][0]["message"]["content"]
                return {"analysis": text, "degraded": False, "source": model_type}
            except (KeyError, IndexError, TypeError) as exc:
                logger.warning(f"Unexpected AI response format: {exc}")
                # Fall through to fallback

        # AI unavailable or bad response → fallback
        self._total_fallbacks += 1
        logger.info(
            f"Using fallback analysis for {symbol}/{interval} "
            f"(fallbacks so far: {self._total_fallbacks})"
        )
        return generate_fallback_analysis(symbol, interval, df=df, indicators=indicators)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Report current AI service status.

        Returns
        -------
        dict with scout/analyst circuit breaker states and overall metrics.
        """
        return {
            "scout": {
                "url": AI_URL_SCOUT,
                "model": AI_MODEL_SCOUT,
                "circuit_breaker": self._cb_scout.get_status(),
            },
            "analyst": {
                "url": AI_URL_ANALYST,
                "model": AI_MODEL_ANALYST,
                "circuit_breaker": self._cb_analyst.get_status(),
            },
            "metrics": {
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_fallbacks": self._total_fallbacks,
                "failure_rate": (
                    self._total_failures / self._total_calls
                    if self._total_calls > 0 else 0.0
                ),
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_service: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Return (or create) the module-level AIService singleton."""
    global _default_service
    if _default_service is None:
        _default_service = AIService()
    return _default_service
