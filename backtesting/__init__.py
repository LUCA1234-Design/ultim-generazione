"""
backtesting/ — Backtesting and Validation Framework for V18.

Provides:
  - Walk-forward historical replay (historical_replay)
  - Monte Carlo strategy validation (monte_carlo_validator)
  - Regime-segmented backtesting (regime_backtest)
  - Live A/B strategy testing (ab_tester)

Results feed directly into the EvolutionEngine (Loop #15) so
that only robustly validated strategies are promoted to live use.
"""
