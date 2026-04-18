# Bot Improvements

1. **Multi-source data fusion (top signals)**
   - Add on-chain metrics (exchange inflow/outflow, whale wallets).
   - Add macro/sentiment features (DXY, rates, crypto fear-greed, funding/OI term structure).
   - Use robust feature-quality gating before feeding agents.

2. **Advanced indicators and microstructure**
   - Integrate order book imbalance/absorption and liquidation clusters.
   - Add volatility regime features (realized vol percentile, vol-of-vol).
   - Use adaptive indicator windows by regime to reduce lag in fast markets.

3. **Ensemble decision architecture**
   - Train a meta-ensemble that combines agent outputs by regime/symbol/TF.
   - Use confidence calibration (Platt/Isotonic) for final probabilities.
   - Add disagreement-aware throttling (skip low-consensus environments).

4. **Learning reliability and persistence**
   - Persist learning state periodically and on shutdown (weights, model state, buffers).
   - Restore the full runtime learning state on startup to avoid cold restarts.
   - Track and audit learning drift with automatic re-calibration triggers.

5. **Execution quality**
   - Add slippage-aware entry filter (skip if estimated impact too high).
   - Optimize sizing via risk-budget + RL hint with hard guardrails.
   - Add post-trade attribution to improve which signals get promoted.
