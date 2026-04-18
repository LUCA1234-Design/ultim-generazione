"""
risk_institutional/ — Institutional-Grade Risk Engine for V18.

Provides:
  - Value-at-Risk (historical, parametric, Monte Carlo) (var_engine)
  - Conditional VaR and stress testing (cvar_stress)
  - Concentration risk / HHI (concentration_risk)
  - Multi-level circuit breaker kill switch (kill_switch)
  - Margin and liquidation monitor (margin_monitor)
  - Regulatory position size limits (regulatory_limits)

All modules are thread-safe and designed to run continuously as
risk guards alongside the V17 trading pipeline.
"""

from risk_institutional.institutional_risk_manager import InstitutionalRiskManager

__all__ = ["InstitutionalRiskManager"]
