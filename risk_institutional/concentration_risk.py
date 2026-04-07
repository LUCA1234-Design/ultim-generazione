"""
risk_institutional/concentration_risk.py — Concentration Risk Analysis.

Implements:

1. **Herfindahl-Hirschman Index (HHI)**
   - HHI = Σ(market_share_i²) for all positions
   - HHI range: [0, 1] — 0 = perfect diversification, 1 = all in one asset
   - Institutional limit: HHI < 0.25

2. **Correlation Exposure**
   - Limits on total exposure to highly correlated assets
   - Prevents over-concentration in correlated risk

3. **Sector/Type Exposure**
   - Classifies crypto assets by type (L1, DeFi, Meme, Stablecoin, etc.)
   - Limits exposure per sector

Integration: used by kill_switch.py (Level 4 — correlation check) and
by risk_agent for position sizing decisions.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("ConcentrationRisk")

# Exposure limits
_MAX_HHI = 0.40           # max Herfindahl index
_MAX_CORRELATION_EXPOSURE = 0.80  # max fraction in correlated assets
_MAX_SECTOR_EXPOSURE = 0.60      # max fraction in single sector

# Crypto sector classification (simplified)
_SECTOR_MAP: Dict[str, str] = {
    "BTC": "L1", "ETH": "L1", "BNB": "L1", "SOL": "L1", "ADA": "L1",
    "AVAX": "L1", "DOT": "L1", "ATOM": "L1", "NEAR": "L1", "FTM": "L1",
    "MATIC": "L2", "ARB": "L2", "OP": "L2",
    "UNI": "DeFi", "AAVE": "DeFi", "COMP": "DeFi", "SNX": "DeFi",
    "DOGE": "Meme", "SHIB": "Meme", "PEPE": "Meme", "FLOKI": "Meme",
    "USDT": "Stablecoin", "USDC": "Stablecoin", "BUSD": "Stablecoin",
    "LINK": "Oracle", "BAND": "Oracle",
    "XRP": "Payments", "XLM": "Payments", "XMR": "Privacy",
}


def compute_hhi(positions: List[Dict]) -> float:
    """
    Compute the Herfindahl-Hirschman Index for a portfolio.

    Parameters
    ----------
    positions : list of dicts with 'symbol' and 'notional' keys
                (notional = position size × price)

    Returns
    -------
    float — HHI in [0, 1], or 0.0 if no positions.
    """
    if not positions:
        return 0.0

    notionals = np.array([
        abs(float(p.get("notional", 0)))
        for p in positions
    ])
    total = notionals.sum()

    if total < 1e-8:
        return 0.0

    shares = notionals / total
    hhi = float(np.sum(shares ** 2))
    return hhi


def correlation_exposure(
    positions: List[Dict],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    correlation_threshold: float = 0.70,
) -> Dict:
    """
    Compute exposure to highly correlated asset groups.

    Parameters
    ----------
    positions            : list of position dicts
    correlation_matrix   : {symbol_a: {symbol_b: corr}} nested dict
    correlation_threshold: minimum correlation to be considered "correlated"

    Returns
    -------
    dict with 'max_correlated_exposure', 'correlated_groups', 'exceeds_limit'
    """
    if not positions:
        return {"max_correlated_exposure": 0.0, "correlated_groups": [], "exceeds_limit": False}

    symbols = [str(p.get("symbol", "")).replace("USDT", "") for p in positions]
    notionals = np.array([abs(float(p.get("notional", 1.0))) for p in positions])
    total = notionals.sum()

    if total < 1e-8:
        return {"max_correlated_exposure": 0.0, "correlated_groups": [], "exceeds_limit": False}

    # If no correlation matrix: assume BTC-correlated groups
    if correlation_matrix is None:
        # Default: most crypto assets are correlated with BTC
        correlated_exposure = float(notionals.sum() / total)
        return {
            "max_correlated_exposure": correlated_exposure,
            "correlated_groups": [symbols],
            "exceeds_limit": correlated_exposure > _MAX_CORRELATION_EXPOSURE,
        }

    # Build correlated groups using union-find
    n = len(symbols)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, sym_a in enumerate(symbols):
        for j, sym_b in enumerate(symbols):
            if i >= j:
                continue
            corr = (
                correlation_matrix.get(sym_a, {}).get(sym_b, 0.5)
                if correlation_matrix else 0.5
            )
            if abs(corr) >= correlation_threshold:
                union(i, j)

    # Compute group exposures
    groups: Dict[int, float] = {}
    for i in range(n):
        g = find(i)
        groups[g] = groups.get(g, 0.0) + notionals[i]

    max_group_exposure = float(max(groups.values())) / total if groups else 0.0
    correlated_groups = [
        [symbols[i] for i in range(n) if find(i) == g]
        for g in set(find(i) for i in range(n))
    ]

    return {
        "max_correlated_exposure": max_group_exposure,
        "correlated_groups": correlated_groups,
        "exceeds_limit": max_group_exposure > _MAX_CORRELATION_EXPOSURE,
    }


def check_limits(positions: List[Dict]) -> Dict:
    """
    Check all concentration risk limits.

    Parameters
    ----------
    positions : list of position dicts with 'symbol', 'notional' keys

    Returns
    -------
    dict with 'hhi', 'sector_exposures', 'violations', 'all_clear'
    """
    violations = []

    # HHI check
    hhi = compute_hhi(positions)
    if hhi > _MAX_HHI:
        violations.append(f"HHI={hhi:.3f} > limit={_MAX_HHI:.3f}")

    # Sector exposure check
    sector_exp = _compute_sector_exposure(positions)
    for sector, exposure in sector_exp.items():
        if exposure > _MAX_SECTOR_EXPOSURE:
            violations.append(f"Sector {sector}: {exposure:.1%} > limit={_MAX_SECTOR_EXPOSURE:.1%}")

    # Correlation exposure (simplified without real correlation matrix)
    corr_result = correlation_exposure(positions)
    max_corr_exp = corr_result["max_correlated_exposure"]
    if max_corr_exp > _MAX_CORRELATION_EXPOSURE:
        violations.append(f"Correlation exposure {max_corr_exp:.1%} > limit")

    return {
        "hhi": hhi,
        "sector_exposures": sector_exp,
        "max_correlated_exposure": max_corr_exp,
        "violations": violations,
        "all_clear": len(violations) == 0,
    }


def get_exposure_report(positions: List[Dict]) -> Dict:
    """Full exposure report for all positions."""
    return {
        **check_limits(positions),
        "n_positions": len(positions),
        "total_notional": sum(abs(float(p.get("notional", 0))) for p in positions),
    }


def _compute_sector_exposure(positions: List[Dict]) -> Dict[str, float]:
    """Compute sector exposure fractions."""
    if not positions:
        return {}

    total = sum(abs(float(p.get("notional", 0))) for p in positions)
    if total < 1e-8:
        return {}

    sector_totals: Dict[str, float] = {}
    for p in positions:
        symbol = str(p.get("symbol", "")).replace("USDT", "")
        sector = _SECTOR_MAP.get(symbol, "Other")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(float(p.get("notional", 0)))

    return {s: v / total for s, v in sector_totals.items()}
