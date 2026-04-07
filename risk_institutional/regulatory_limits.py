"""
risk_institutional/regulatory_limits.py — Regulatory Position Limits.

Enforces institutional-style position limits:
  - Max notional per position (relative to balance)
  - Max leverage per market regime
  - Absolute and relative position size limits

These limits act as hard constraints before any position is opened.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger("RegulatoryLimits")

# Max leverage by regime
_REGIME_LEVERAGE: Dict[str, float] = {
    "trending": 10.0,
    "ranging": 5.0,
    "volatile": 3.0,
}

# Max position size as fraction of balance
_MAX_POSITION_PCT = 0.25      # 25% of balance max per position
_MAX_NOTIONAL_USD = 100_000   # absolute cap

# Min position size (avoid dust trades)
_MIN_POSITION_USD = 10.0


def check_position_limits(
    position: Dict,
    balance: float,
    regime: str = "ranging",
) -> Dict:
    """
    Check if a proposed position complies with all regulatory limits.

    Parameters
    ----------
    position : dict with 'notional', 'leverage' (optional), 'symbol'
    balance  : current account balance
    regime   : current market regime

    Returns
    -------
    dict with 'allowed', 'violations', 'adjusted_notional'
    """
    violations = []
    notional = abs(float(position.get("notional", 0)))
    leverage = float(position.get("leverage", 1.0))
    symbol = str(position.get("symbol", "?"))

    # Check regime-based leverage limit
    max_lev = _REGIME_LEVERAGE.get(regime.lower(), 5.0)
    if leverage > max_lev:
        violations.append(
            f"Leverage {leverage:.1f}x exceeds {regime} regime limit {max_lev:.1f}x"
        )

    # Check notional as % of balance
    if balance > 0:
        notional_pct = notional / balance
        if notional_pct > _MAX_POSITION_PCT:
            violations.append(
                f"Notional {notional_pct:.1%} exceeds max {_MAX_POSITION_PCT:.1%} of balance"
            )

    # Absolute notional cap
    if notional > _MAX_NOTIONAL_USD:
        violations.append(
            f"Notional ${notional:,.0f} exceeds absolute cap ${_MAX_NOTIONAL_USD:,.0f}"
        )

    # Minimum size
    if 0 < notional < _MIN_POSITION_USD:
        violations.append(f"Notional ${notional:.2f} below minimum ${_MIN_POSITION_USD:.2f}")

    # Compute adjusted notional (maximum allowed)
    max_notional = min(
        balance * _MAX_POSITION_PCT,
        _MAX_NOTIONAL_USD,
    ) if balance > 0 else _MAX_NOTIONAL_USD

    adjusted = min(notional, max_notional)

    return {
        "allowed": len(violations) == 0,
        "violations": violations,
        "notional": notional,
        "adjusted_notional": adjusted,
        "max_allowed_notional": max_notional,
    }


def max_allowed_size(
    symbol: str,
    balance: float,
    regime: str = "ranging",
) -> float:
    """
    Return the maximum allowed notional for a new position.

    Parameters
    ----------
    symbol  : trading pair (not currently used but reserved for per-symbol limits)
    balance : account balance
    regime  : current market regime

    Returns
    -------
    float — maximum notional value in USDT
    """
    return min(
        balance * _MAX_POSITION_PCT,
        _MAX_NOTIONAL_USD,
    )


def get_regime_leverage(regime: str) -> float:
    """Return the maximum allowed leverage for the given regime."""
    return _REGIME_LEVERAGE.get(regime.lower(), 5.0)


def get_limits_summary() -> Dict:
    """Return a summary of all current regulatory limits."""
    return {
        "max_position_pct": _MAX_POSITION_PCT,
        "max_notional_usd": _MAX_NOTIONAL_USD,
        "min_position_usd": _MIN_POSITION_USD,
        "regime_leverage": dict(_REGIME_LEVERAGE),
    }
