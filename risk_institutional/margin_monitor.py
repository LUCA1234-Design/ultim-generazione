"""
risk_institutional/margin_monitor.py — Margin and Liquidation Monitor.

Tracks:
  - Margin utilisation (used margin / available margin)
  - Distance from liquidation price (as % buffer)
  - Funding rate cost accumulation
  - Margin health score

Integration: Called before each new trade in ExecutionEngine.
Should also be called periodically in the main loop to detect
approaching liquidation levels.
"""
import logging
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("MarginMonitor")

# Safety thresholds
_MARGIN_WARNING_PCT = 0.70    # 70% utilisation = warning
_MARGIN_CRITICAL_PCT = 0.85   # 85% utilisation = critical
_LIQ_DISTANCE_MIN = 0.05      # 5% buffer before liquidation = danger


class MarginMonitor:
    """Tracks margin utilisation and liquidation risk across positions."""

    def __init__(self, leverage: float = 10.0):
        self.leverage = leverage
        self._lock = threading.Lock()
        self._funding_costs: List[float] = []  # historical funding costs

    # ------------------------------------------------------------------

    def compute_margin_usage(
        self,
        positions: List[Dict],
        balance: float,
    ) -> Dict:
        """
        Compute margin utilisation for the current portfolio.

        Parameters
        ----------
        positions : list of dicts with 'notional', 'margin' (optional)
        balance   : available portfolio balance

        Returns
        -------
        dict with 'used_margin', 'free_margin', 'utilisation_pct', 'status'
        """
        if balance <= 0:
            return {"error": "Invalid balance"}

        used_margin = sum(
            float(p.get("margin", abs(p.get("notional", 0)) / self.leverage))
            for p in positions
        )
        free_margin = max(0.0, balance - used_margin)
        utilisation = used_margin / balance

        if utilisation >= _MARGIN_CRITICAL_PCT:
            status = "CRITICAL"
        elif utilisation >= _MARGIN_WARNING_PCT:
            status = "WARNING"
        else:
            status = "NORMAL"

        return {
            "used_margin": used_margin,
            "free_margin": free_margin,
            "utilisation_pct": utilisation,
            "status": status,
            "can_open_new": utilisation < _MARGIN_WARNING_PCT,
        }

    def liquidation_distance(
        self,
        positions: List[Dict],
        balance: float,
    ) -> Dict:
        """
        Compute distance from liquidation for each position.

        Simplified model:
          Liquidation price (long) = entry × (1 - 1/leverage + margin_ratio)
          Liquidation price (short) = entry × (1 + 1/leverage - margin_ratio)

        Parameters
        ----------
        positions : list of dicts with 'entry_price', 'current_price',
                    'side' ('long'/'short'), 'notional'
        balance   : current portfolio balance

        Returns
        -------
        dict with per-position liquidation distances and min distance
        """
        if not positions:
            return {"min_liq_distance": 1.0, "positions": []}

        results = []
        for pos in positions:
            entry = float(pos.get("entry_price", 0))
            current = float(pos.get("current_price", entry or 1))
            side = str(pos.get("side", "long")).lower()
            notional = abs(float(pos.get("notional", 0)))

            if entry <= 0 or notional <= 0:
                continue

            margin = notional / self.leverage
            maintenance_rate = 0.005  # 0.5% maintenance margin

            if side == "long":
                liq_price = entry * (1 - 1 / self.leverage + maintenance_rate)
                distance = (current - liq_price) / current if current > 0 else 1.0
            else:
                liq_price = entry * (1 + 1 / self.leverage - maintenance_rate)
                distance = (liq_price - current) / current if current > 0 else 1.0

            distance = max(0.0, float(distance))
            is_dangerous = distance < _LIQ_DISTANCE_MIN

            results.append({
                "symbol": pos.get("symbol", "?"),
                "side": side,
                "entry_price": entry,
                "current_price": current,
                "liquidation_price": liq_price,
                "distance_pct": distance,
                "is_dangerous": is_dangerous,
            })

        min_distance = min((r["distance_pct"] for r in results), default=1.0)

        return {
            "min_liq_distance": min_distance,
            "positions": results,
            "any_dangerous": any(r["is_dangerous"] for r in results),
        }

    def funding_rate_cost(
        self,
        positions: List[Dict],
        rates: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Estimate cumulative funding rate cost for all positions.

        Parameters
        ----------
        positions : list of position dicts with 'symbol', 'notional', 'side'
        rates     : dict mapping symbol → current funding rate (per 8h period)
                    If None, uses a typical 0.01% as default.

        Returns
        -------
        dict with 'total_cost_per_8h', 'annual_cost', 'positions'
        """
        rates = rates or {}
        pos_costs = []
        total_cost = 0.0

        for pos in positions:
            symbol = str(pos.get("symbol", "")).replace("USDT", "")
            notional = abs(float(pos.get("notional", 0)))
            side = str(pos.get("side", "long")).lower()

            rate = float(rates.get(symbol, rates.get(pos.get("symbol", ""), 0.0001)))

            # Funding cost: long pays if rate > 0, short receives (and vice versa)
            if side == "long":
                cost = notional * rate
            else:
                cost = -notional * rate  # short receives funding when rate > 0

            pos_costs.append({
                "symbol": pos.get("symbol", "?"),
                "funding_rate": rate,
                "cost_per_8h": cost,
            })
            total_cost += cost

        with self._lock:
            self._funding_costs.append(total_cost)

        return {
            "total_cost_per_8h": total_cost,
            "annual_cost": total_cost * 3 * 365,  # 3 periods per day × 365
            "positions": pos_costs,
        }

    def get_margin_report(
        self,
        positions: List[Dict] = None,
        balance: float = 0,
        rates: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Comprehensive margin health report.

        Returns combined margin usage, liquidation, and funding data.
        """
        positions = positions or []
        margin = self.compute_margin_usage(positions, balance)
        liq = self.liquidation_distance(positions, balance)
        funding = self.funding_rate_cost(positions, rates)

        # Compute health score [0, 1]
        utilisation = margin.get("utilisation_pct", 0)
        min_dist = liq.get("min_liq_distance", 1.0)

        health = float(
            (1 - utilisation) * 0.5 +
            min(min_dist / 0.20, 1.0) * 0.5  # 20% distance = full score
        )

        return {
            "margin": margin,
            "liquidation": liq,
            "funding": funding,
            "health_score": health,
            "alert": margin["status"] != "NORMAL" or liq.get("any_dangerous", False),
        }
