"""
Evolution Engine for V18 — Central Brain (Extended).

Closes all feedback loops (V17 + new V18 additions):

  V17 loops (unchanged):
  Loop #1: MetaAgent → DecisionFusion weight updates
  Loop #2: PerformanceTracker → RiskAgent win rates
  Loop #3: Auto-tunes FUSION_THRESHOLD via optimal_params DB
  Loop #4: Pattern threshold adaptation
  Loop #5: Strategy evolution (delegated to StrategyEvolver)
  Loop #6: MetaAgent state persistence (save/load)
  Loop #7: Confluence TF weight learning (delegated to ConfluenceAdapter)

  V18 new loops:
  Loop #8: RL → Execution (PPO policy informs sizing/timing)
  Loop #9: HMM → RegimeAgent (second opinion on regime transitions)
  Loop #10: Bayesian → All Agents (prior updating from closed trades)
  Loop #11: Meta-Learning → Hyperparameters (MAML adaptation)
  Loop #12: Concept Drift → Re-training (triggers GMM/HMM re-fit)
  Loop #13: VaR/CVaR → Kill Switch (continuous risk monitoring)
  Loop #14: Contrarian → Fusion (balance consensus via devil's advocate)
  Loop #15: Backtest Validator → Strategy Evolver (walk-forward gating)

Usage in main.py:
    engine = EvolutionEngine(meta, fusion, risk, strategy, confluence, tracker, pattern)
    engine.startup()                                # on boot
    engine.on_trade_close(closed_pos, ctx)          # in position monitor
    engine.tick()                                   # every 30 min in main loop
    engine.shutdown()                               # on Ctrl+C
"""
import logging
import os
import threading
import time
import json
from typing import Any, Dict, Optional

import numpy as np

from memory import experience_db
from evolution.strategy_evolver import StrategyEvolver
from evolution.confluence_adapter import ConfluenceAdapter

# ---- V18 optional imports (graceful degradation) ----------------------------
try:
    from rl.ppo_agent import PPOAgent as _PPOAgent
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False

try:
    from quant.hmm_regime import HMMRegimeDetector as _GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

try:
    from quant.bayesian_inference import BayesianOnlineLearner as _BayesianEngine
    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

try:
    from metalearning.maml_adapter import MAMLAdapter as _MAMLAdapter
    _MAML_AVAILABLE = True
except ImportError:
    _MAML_AVAILABLE = False

try:
    from metalearning.concept_drift_detector import ConceptDriftDetector as _DriftDetector
    _DRIFT_AVAILABLE = True
except ImportError:
    _DRIFT_AVAILABLE = False

try:
    from metalearning.learning_rate_scheduler import LearningRateSchedulerManager as _LRManager
    _LR_AVAILABLE = True
except ImportError:
    _LR_AVAILABLE = False

try:
    from metalearning.hyperopt_engine import TPEHyperoptEngine as _HyperoptEngine
    _HYPEROPT_AVAILABLE = True
except ImportError:
    _HYPEROPT_AVAILABLE = False

try:
    from risk_institutional.kill_switch import KillSwitch as _KillSwitch
    _KILL_SWITCH_AVAILABLE = True
except ImportError:
    _KILL_SWITCH_AVAILABLE = False

try:
    from risk_institutional.var_engine import compute_all as _compute_var
    _VAR_AVAILABLE = True
except ImportError:
    _VAR_AVAILABLE = False

try:
    from backtesting.monte_carlo_validator import validate_strategy as _validate_strategy
    from backtesting.monte_carlo_validator import is_strategy_robust as _is_strategy_robust
    _BACKTEST_AVAILABLE = True
except ImportError:
    _BACKTEST_AVAILABLE = False

try:
    from coordination.message_bus import get_message_bus, TOPIC_DRIFT_DETECTED, TOPIC_RISK_ALERT
    _MSG_BUS_AVAILABLE = True
except ImportError:
    _MSG_BUS_AVAILABLE = False

logger = logging.getLogger("EvolutionEngine")

# Tuning constants
_TUNE_INTERVAL_SEC = 1800   # 30 min between auto-tune runs
_SAVE_INTERVAL_SEC = 900    # 15 min between state saves
_V18_TICK_INTERVAL_SEC = 300  # 5 min for V18 loops
_MIN_COMPLETED = 10         # minimum completed trades before tuning
_THRESHOLD_STEP_UP = 0.02   # raise threshold by this when win-rate is too low
_THRESHOLD_STEP_DOWN = 0.01 # lower threshold by this when win-rate is excellent
_THRESHOLD_LOW_WR = 0.50    # win-rate below this triggers a raise
_THRESHOLD_HIGH_WR = 0.65   # win-rate above this triggers a lower
_THRESHOLD_MIN = 0.45
_THRESHOLD_MAX = 0.85
_DRAWDOWN_WARN = 0.08       # 8% drawdown → raise threshold
_DRAWDOWN_CRITICAL = 0.15   # 15% → safe mode

# V18 constants
_DRIFT_RETRAIN_COOLDOWN = 3600  # minimum seconds between re-trains
_BACKTEST_VALIDATION_INTERVAL = 7200  # 2h between full backtest runs
_BACKTEST_MIN_TRADES = 20    # minimum trades for backtest validation
_PPO_MODEL_PATH = os.getenv("PPO_RUNTIME_STATE_PATH", "models/ppo_runtime_state.npz")
_RUNTIME_STATE_PATH = os.getenv("EVOLUTION_RUNTIME_STATE_PATH", "models/evolution_runtime_state.json")


class EvolutionEngine:
    """Central orchestrator that wires all V17+V18 feedback loops."""

    def __init__(self,
        meta_agent,
        fusion,
        risk_agent,
        strategy_agent,
        confluence_agent,
        tracker,
        pattern_agent=None,
        regime_agent=None,
        # V18 optional components
        hmm_model=None,
        ppo_agent=None,
        bayesian_engine=None,
        maml_adapter=None,
        kill_switch=None,
    ):
        self._meta = meta_agent
        self._fusion = fusion
        self._risk = risk_agent
        self._tracker = tracker
        self._pattern = pattern_agent
        self._regime_agent = regime_agent
        if self._regime_agent is None and self._pattern is not None:
            self._regime_agent = getattr(self._pattern, "_regime_agent", None)

        # Sub-engines for loop #5 and #7
        self._strategy_evolver = StrategyEvolver(strategy_agent)
        self._confluence_adapter = ConfluenceAdapter(confluence_agent)

        self._last_tune: float = 0.0
        self._last_save: float = 0.0
        self._last_v18_tick: float = 0.0
        self._last_drift_retrain: float = 0.0
        self._last_backtest_validation: float = 0.0
        self._last_hmm_fit: float = 0.0   # V18: track last HMM re-fit time

        # Thread safety lock
        self._lock = threading.Lock()

        # ---- V18 optional components ----
        # Loop #9: HMM
        self._hmm = hmm_model
        if _HMM_AVAILABLE and self._hmm is None:
            try:
                self._hmm = _GaussianHMM()
            except Exception:
                pass

        # Loop #8: PPO RL agent
        self._ppo = ppo_agent
        if _RL_AVAILABLE and self._ppo is None:
            try:
                self._ppo = _PPOAgent()
            except Exception:
                pass
        if self._ppo is not None:
            try:
                pretrained_path = "models/ppo_pretrained.pt"
                if os.path.exists(pretrained_path):
                    loaded = self._ppo.load_pretrained(pretrained_path)
                    if loaded:
                        logger.info("🎓 PPO: loaded pretrained model from offline training")
            except Exception:
                pass

        # Loop #10: Bayesian engine
        self._bayesian = bayesian_engine
        if _BAYESIAN_AVAILABLE and self._bayesian is None:
            try:
                self._bayesian = _BayesianEngine()
            except Exception:
                pass

        # Loop #11: MAML adapter
        self._maml = maml_adapter
        if _MAML_AVAILABLE and self._maml is None:
            try:
                self._maml = _MAMLAdapter()
            except Exception:
                pass

        # Loop #12: Concept drift detector
        self._drift_detector = None
        if _DRIFT_AVAILABLE:
            try:
                self._drift_detector = _DriftDetector()
            except Exception:
                pass

        # LR scheduler
        self._lr_scheduler = None
        if _LR_AVAILABLE:
            try:
                self._lr_scheduler = _LRManager()
            except Exception:
                pass

        # Loop #13: Kill switch
        self._kill_switch = kill_switch
        if _KILL_SWITCH_AVAILABLE and self._kill_switch is None:
            try:
                self._kill_switch = _KillSwitch()
            except Exception:
                pass

        # Hyperopt engine (Loop #11)
        self._hyperopt = None
        if _HYPEROPT_AVAILABLE:
            try:
                self._hyperopt = _HyperoptEngine()
            except Exception:
                pass
        self._last_hyperopt_params: Optional[dict] = None

        # Closed trades buffer for backtest validation (Loop #15)
        self._closed_trades_buffer: list = []
        # PPO experience buffer for RL training
        self._ppo_experience_buffer: list = []

        # Checkpoint & rollback mechanism: salva lo stato dei parametri prima di ogni tick
        self._last_checkpoint: Optional[dict] = None
        self._checkpoint_win_rate: float = 0.5

        v18_modules = sum([
            self._hmm is not None,
            self._ppo is not None,
            self._bayesian is not None,
            self._maml is not None,
            self._drift_detector is not None,
            self._kill_switch is not None,
        ])
        logger.info(f"🚀 EvolutionEngine V18: {v18_modules}/6 V18 modules active")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Restore previous learning state — call once on bot boot."""

        # Loop #6: restore MetaAgent weights from disk
        try:
            loaded = self._meta.load_state()
            if loaded:
                logger.info("🧠 EvolutionEngine: MetaAgent state restored from disk")
            else:
                logger.info("🧠 EvolutionEngine: no saved MetaAgent state, starting fresh")
        except Exception as exc:
            logger.error(f"EvolutionEngine.startup load_state error: {exc}")

        # Loop #3: restore auto-tuned fusion threshold from DB
        try:
            saved_threshold = experience_db.get_param("fusion_threshold")
            if saved_threshold is not None:
                clamped = float(np.clip(float(saved_threshold), _THRESHOLD_MIN, _THRESHOLD_MAX))
                self._fusion._threshold = clamped
                logger.info(f"🔧 EvolutionEngine: restored fusion_threshold={{clamped:.3f}}")
        except Exception as exc:
            logger.error(f"EvolutionEngine.startup restore_threshold error: {exc}")

        # Loop #7: restore confluence TF performance counters from DB
        try:
            tf_data = experience_db.get_param("confluence_tf_performance")
            if tf_data and isinstance(tf_data, dict):
                self._confluence_adapter.load_state(tf_data)
                logger.info("🌊 EvolutionEngine: confluence TF performance data restored")
        except Exception as exc:
            logger.debug(f"EvolutionEngine.startup TF data restore error: {exc}")

        # Loop #5: restore strategy evolver trade count
        try:
            saved_count = experience_db.get_param("strategy_evolver_trade_count")
            if saved_count is not None:
                self._strategy_evolver.trade_count = int(saved_count)
                logger.info(
                    f"🧬 EvolutionEngine: restored strategy_evolver trade_count="
                    f"{{self._strategy_evolver.trade_count}}"
                )
        except Exception as exc:
            logger.debug(f"EvolutionEngine.startup strategy trade_count restore error: {exc}")

        # Loop #9: Fit HMM on BTCUSDT 1h data (initial fit on startup)
        try:
            if self._hmm is not None:
                from data import data_store
                df_btc = data_store.get_df("BTCUSDT", "1h")
                if df_btc is not None and len(df_btc) >= 100:
                    fitted = self._hmm.fit(df_btc)
                    if fitted:
                        self._last_hmm_fit = time.time()
                        logger.info("🔮 EvolutionEngine: HMM fitted on BTCUSDT 1h data")
                    else:
                        logger.debug("EvolutionEngine: HMM fit returned False (insufficient data at startup)")
        except Exception as exc:
            logger.debug(f"EvolutionEngine HMM fit error: {exc}")

        # Restore runtime learning state (buffers + PPO weights)
        self._load_runtime_state()

    def on_trade_close(
        self,
        closed_position: Any,
        decision_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify the evolution engine that a position just closed.

        Parameters
        ----------
        closed_position : Position object with at minimum ``decision_id``,
                          ``strategy``, and ``pnl`` attributes.
        decision_ctx    : the dict stored in decision_context[decision_id],
                          including ``agent_results``.
        """
        with self._lock:
            was_profitable = (getattr(closed_position, "pnl", None) or 0.0) > 0
            pnl_value = float(getattr(closed_position, "pnl", 0.0) or 0.0)
            ctx = decision_ctx or {}

            # Loop #5: strategy evolution (accumulate + possibly prune/mutate)
            try:
                strategy_name = getattr(closed_position, "strategy", None) or ""
                self._strategy_evolver.record_trade(strategy_name, was_profitable)
            except Exception as exc:
                logger.error(f"EvolutionEngine strategy_evolver error: {exc}")

            # Loop #7: confluence TF tracking
            try:
                agent_results = ctx.get("agent_results", {})
                confluence_result = agent_results.get("confluence")
                if confluence_result and hasattr(confluence_result, "metadata"):
                    tf_scores: Dict[str, float] = confluence_result.metadata.get("tf_scores", {})
                    self._confluence_adapter.record_trade(tf_scores, was_profitable)
            except Exception as exc:
                logger.debug(f"EvolutionEngine confluence_adapter error: {exc}")

            # Loop #4: pattern threshold adaptation
            try:
                if self._pattern is not None:
                    agent_results = ctx.get("agent_results", {})
                    pattern_result = agent_results.get("pattern")
                    interval = ctx.get("interval", "1h")
                    self._pattern.update_threshold(interval, was_profitable)
                    if pattern_result and hasattr(pattern_result, "details"):
                        raw_details = list(pattern_result.details)
                        pattern_tags = [
                            d.split("(")[0] for d in raw_details if d
                        ]
                        if pattern_tags and hasattr(self._pattern, "record_pattern_outcome"):
                            self._pattern.record_pattern_outcome(pattern_tags, was_profitable)
            except Exception as exc:
                logger.error(f"EvolutionEngine pattern_threshold error: {exc}")

            # ---- V18 on_trade_close hooks --------------------------------

            # Loop #10: Bayesian prior update
            try:
                if self._bayesian is not None:
                    self._bayesian.update_prior({
                        "win": was_profitable,
                        "pnl": pnl_value,
                    })
            except Exception as exc:
                logger.debug(f"EvolutionEngine bayesian_update error: {exc}")

            # Loop #11: MAML meta-update
            try:
                if self._maml is not None:
                    regime = ctx.get("regime", "ranging")
                    agent_results = ctx.get("agent_results", {})
                    agent_scores = {}
                    for name, res in agent_results.items():
                        if hasattr(res, "score"):
                            agent_scores[name] = float(res.score)
                        elif isinstance(res, dict):
                            agent_scores[name] = float(res.get("score", 0.5))

                    self._maml.meta_update(
                        regime=regime,
                        agent_weights=agent_scores,
                        outcomes=[{
                            "win": was_profitable,
                            "pnl": pnl_value,
                            "agents_used": agent_scores,
                        }],
                    )
            except Exception as exc:
                logger.debug(f"EvolutionEngine maml_update error: {exc}")

            # Loop #12: Concept drift update
            try:
                if self._drift_detector is not None:
                    drift_detected = self._drift_detector.update(
                        pnl_value, is_win=was_profitable
                    )
                    if drift_detected:
                        logger.warning("⚠️ Concept drift detected — triggering re-calibration")
                        now = time.time()
                        if now - self._last_drift_retrain > _DRIFT_RETRAIN_COOLDOWN:
                            self._on_concept_drift()
                            self._last_drift_retrain = now
            except Exception as exc:
                logger.debug(f"EvolutionEngine drift_detector error: {exc}")

            # Loop #15: Buffer trades for backtest validation
            try:
                self._closed_trades_buffer.append({
                    "pnl": pnl_value,
                    "win": was_profitable,
                    "strategy": getattr(closed_position, "strategy", None),
                    "regime": ctx.get("regime", "unknown"),
                    "symbol": getattr(closed_position, "symbol", ""),
                    "interval": getattr(closed_position, "interval", ""),
                    "direction": getattr(closed_position, "direction", ""),
                })
                # Keep last 500 trades
                if len(self._closed_trades_buffer) > 500:
                    self._closed_trades_buffer = self._closed_trades_buffer[-500:]
            except Exception as exc:
                logger.debug(f"EvolutionEngine trade_buffer error: {exc}")

            # Loop #8: PPO experience collection
            try:
                if self._ppo is not None:
                    symbol = str(getattr(closed_position, "symbol", "") or "UNKNOWN")
                    from config.settings import RL_N_FEATURES  # state vector dimension = 12
                    state = np.zeros(RL_N_FEATURES)
                    state[0] = float(np.clip(pnl_value / 100.0, -1.0, 1.0))
                    state[1] = 1.0 if was_profitable else 0.0
                    n_closed = len(self._closed_trades_buffer)
                    state[2] = float(np.clip(n_closed / 200.0, 0.0, 1.0))
                    state[3] = float(np.clip(ctx.get("fusion_score", 0.5), 0.0, 1.0))
                    regime = ctx.get("regime", "unknown")
                    state[4] = 1.0 if regime == "trending" else 0.0
                    state[5] = 1.0 if regime == "ranging" else 0.0
                    state[6] = 1.0 if regime == "volatile" else 0.0
                    direction = str(getattr(closed_position, "direction", "") or "")
                    state[7] = 1.0 if direction == "long" else (-1.0 if direction == "short" else 0.0)
                    agent_dirs = ctx.get("agent_directions", {})
                    if agent_dirs:
                        agree = sum(1 for d in agent_dirs.values() if d == direction)
                        state[8] = float(agree) / float(len(agent_dirs))
                    agent_scores_ctx = ctx.get("agent_scores", {})
                    if agent_scores_ctx:
                        state[9] = float(np.clip(np.mean(list(agent_scores_ctx.values())), 0.0, 1.0))
                    open_time = float(getattr(closed_position, "open_time", 0.0) or 0.0)
                    close_time = float(getattr(closed_position, "close_time", 0.0) or 0.0)
                    hold_sec = max(close_time - open_time, 0.0)
                    state[10] = float(np.clip(hold_sec / 86400.0, 0.0, 1.0))
                    if self._closed_trades_buffer:
                        wins = sum(1 for t in self._closed_trades_buffer if t.get("win"))
                        state[11] = float(wins) / float(len(self._closed_trades_buffer))

                    # Use PPO policy to select actual action and get real log_prob
                    # PPO actions: 0=hold, 1-3=short(small/med/large), 4-7=long(small/med/large)
                    action, log_prob = self._ppo.select_action(state)
                    reward = float(np.clip(pnl_value / 10.0, -1.0, 1.0))  # normalised reward
                    done = 1.0   # each trade is a terminal episode step
                    value = self._ppo.get_value(state)
                    trade_reward = 1.0 if was_profitable else -1.0

                    self._ppo_experience_buffer.append({
                        "symbol": symbol,
                        "trade_reward": trade_reward,
                        "pnl": pnl_value,
                        "state": state,
                        "action": action,
                        "log_prob": log_prob,
                        "reward": reward,
                        "value": value,
                        "done": done,
                        "next_state": state.copy(),
                    })
                    # Loop #8: PPO experience update
                    if len(self._ppo_experience_buffer) >= 10:
                        if hasattr(self._ppo, "train_step"):
                            self._ppo.train_step([
                                {
                                    "symbol": e.get("symbol", ""),
                                    "reward": e.get("trade_reward", -1.0),
                                    "pnl": e.get("pnl", 0.0),
                                }
                                for e in self._ppo_experience_buffer[-10:]
                            ])
                        self._ppo_experience_buffer = self._ppo_experience_buffer[-50:]
            except Exception as exc:
                logger.debug(f"Loop#8 RL experience error: {exc}")

    def get_rl_size_hint(self, symbol: str, interval: str, df, direction: str) -> float:
        """
        Returns a size multiplier [0.5, 1.5] suggested by the PPO agent.
        Falls back to 1.0 if PPO is not available or fails.
        """
        if self._ppo is None:
            return 1.0
        try:
            from config.settings import RL_SIZE_MULT_MIN, RL_SIZE_MULT_MAX
            closes = df["close"].values[-11:]
            if len(closes) < 11:
                return 1.0
            returns = np.diff(closes) / closes[:-1]
            direction_enc = 1.0 if direction == "long" else -1.0
            state = np.append(returns[-9:], direction_enc).astype(np.float32)
            if hasattr(self._ppo, "act"):
                action = self._ppo.act(state)
            elif hasattr(self._ppo, "suggest"):
                action = self._ppo.suggest(state)
            elif hasattr(self._ppo, "best_action"):
                action = self._ppo.best_action(state)
            elif hasattr(self._ppo, "select_action"):
                action, _ = self._ppo.select_action(state)
            else:
                return 1.0
            size_map = {0: 0.6, 1: 1.0, 2: 1.4}
            multiplier = float(size_map.get(int(action), 1.0))
            multiplier = float(np.clip(multiplier, RL_SIZE_MULT_MIN, RL_SIZE_MULT_MAX))
            logger.debug(
                f"🎓 PPO size hint [{symbol}/{interval} {direction}]: action={action} → {multiplier:.1f}x"
            )
            return multiplier
        except Exception as _e:
            logger.debug(f"🎓 PPO size hint error: {_e}")
            return 1.0

    def tick(self) -> None:
        """Periodic evolution step — call every ~30 minutes from main loop.

        Designed to be non-blocking: all heavy operations use cached DB data.
        """
        with self._lock:
            now = time.time()

            # Salva checkpoint PRIMA di fare modifiche ai parametri
            try:
                self._last_checkpoint = self._save_checkpoint()
            except Exception as exc:
                logger.debug(f"EvolutionEngine checkpoint save error: {exc}")

            # Loop #1: push updated agent weights to DecisionFusion
            try:
                weight_map = self._meta.adjust_weights()
                if weight_map:
                    self._fusion.update_weights(weight_map)
                    logger.info(f"🎚️ EvolutionEngine: agent weights updated → {weight_map}")
            except Exception as exc:
                logger.error(f"EvolutionEngine weight_update error: {exc}")

            # Loop #2: push real win rates into RiskAgent
            try:
                self._tracker.update_risk_agent_win_rates(self._risk)
            except Exception as exc:
                logger.error(f"EvolutionEngine win_rate_update error: {exc}")

            # Loop #3: auto-tune FUSION_THRESHOLD
            if now - self._last_tune >= _TUNE_INTERVAL_SEC:
                self._auto_tune_params()
                self._last_tune = now

            # Loop #6: periodically persist MetaAgent state
            if now - self._last_save >= _SAVE_INTERVAL_SEC:
                self._save_state()
                self._last_save = now

            # Loop #7: adapt confluence TF weights
            try:
                self._confluence_adapter.maybe_adapt()
            except Exception as exc:
                logger.error(f"EvolutionEngine confluence_adapt error: {exc}")

            # Drawdown circuit breaker (V17)
            self._check_drawdown()

            # Dopo le modifiche: controlla il win rate e fai rollback se degradato
            try:
                self._maybe_rollback()
            except Exception as exc:
                logger.debug(f"EvolutionEngine rollback check error: {exc}")

            # Loop #9: Re-fit HMM every 2 hours with updated data
            if self._hmm is not None and (now - self._last_hmm_fit) >= 7200:
                try:
                    from data import data_store
                    df_btc = data_store.get_df("BTCUSDT", "1h")
                    if df_btc is not None and len(df_btc) >= 100:
                        fitted = self._hmm.fit(df_btc)
                        if fitted:
                            self._last_hmm_fit = now
                            logger.info("🔮 HMM re-fitted on updated BTCUSDT data")
                except Exception as exc:
                    logger.debug(f"HMM re-fit error: {exc}")

            # Loop #9: HMM second opinion → RegimeAgent
            if self._hmm is not None and self._regime_agent is not None:
                try:
                    hmm_regime = None
                    if hasattr(self._hmm, "current_regime"):
                        hmm_regime = self._hmm.current_regime()
                    elif hasattr(self._hmm, "get_regime"):
                        hmm_regime = self._hmm.get_regime()
                    elif hasattr(self._hmm, "predict_regime"):
                        from data import data_store
                        _df_btc = data_store.get_df("BTCUSDT", "1h")
                        if _df_btc is not None and len(_df_btc) >= 50:
                            hmm_regime = self._hmm.predict_regime(_df_btc)
                    if hmm_regime and hasattr(self._regime_agent, "set_hmm_prior"):
                        self._regime_agent.set_hmm_prior(hmm_regime)
                        logger.debug(f"🔎 HMM regime prior → RegimeAgent: {hmm_regime}")
                except Exception as _hmm_err:
                    logger.debug(f"🔎 HMM→RegimeAgent propagation error: {_hmm_err}")

            # Loop #10: Bayesian win rate → RiskAgent
            if self._bayesian is not None and self._risk is not None:
                try:
                    global_wr = None
                    if hasattr(self._bayesian, "get_win_rate"):
                        global_wr = self._bayesian.get_win_rate()
                    elif hasattr(self._bayesian, "get_posterior"):
                        posterior = self._bayesian.get_posterior() or {}
                        wr_obj = posterior.get("win_rate")
                        if isinstance(wr_obj, dict):
                            global_wr = wr_obj.get("mean")
                        if global_wr is None:
                            global_wr = posterior.get("win_rate_bayes")
                    if global_wr is not None:
                        global_wr = float(global_wr)
                    if global_wr is not None and 0.0 < global_wr < 1.0:
                        self._risk.set_win_rate("global", global_wr)
                        logger.debug(f"⚖️ Bayesian global win rate → RiskAgent: {global_wr:.3f}")
                except Exception as _bay_err:
                    logger.debug(f"⚖️ Bayesian→RiskAgent propagation error: {_bay_err}")

            # ---- V18 periodic loops (every 5 min) -------------------------
            if now - self._last_v18_tick >= _V18_TICK_INTERVAL_SEC:
                self._run_v18_loops()
                self._last_v18_tick = now

            # Loop #15: Periodic backtest validation
            if (now - self._last_backtest_validation >= _BACKTEST_VALIDATION_INTERVAL and
                    len(self._closed_trades_buffer) >= _BACKTEST_MIN_TRADES):
                self._run_backtest_validation()
                self._last_backtest_validation = now

    def _save_checkpoint(self) -> dict:
        """Cattura lo stato corrente dei parametri principali per il rollback."""
        checkpoint = {
            "fusion_threshold": getattr(self._fusion, "_threshold", 0.5),
            "agent_weights": dict(getattr(self._fusion, "_weights", {})),
            "tf_weights": dict(getattr(self._confluence_adapter._confluence, "_tf_weights", {})),
            "timestamp": time.time(),
        }
        # Calcola il win rate corrente dal buffer chiuso
        if self._closed_trades_buffer:
            recent = self._closed_trades_buffer[-20:]
            self._checkpoint_win_rate = sum(1 for t in recent if t.get("win")) / len(recent)
        return checkpoint

    def _restore_checkpoint(self, checkpoint: dict) -> None:
        """Ripristina i parametri salvati nel checkpoint."""
        if checkpoint is None:
            return
        try:
            thresh = checkpoint.get("fusion_threshold")
            if thresh is not None:
                self._fusion._threshold = float(thresh)
        except Exception:
            pass
        try:
            weights = checkpoint.get("agent_weights")
            if weights:
                self._fusion.update_weights(weights)
        except Exception:
            pass
        try:
            tf_weights = checkpoint.get("tf_weights")
            if tf_weights and hasattr(self._confluence_adapter, "_confluence"):
                self._confluence_adapter._confluence.update_tf_weights(tf_weights)
        except Exception:
            pass

    def _maybe_rollback(self) -> None:
        """Controlla il win rate recente e ripristina il checkpoint se è calato >10%."""
        if self._last_checkpoint is None or len(self._closed_trades_buffer) < 10:
            return
        recent = self._closed_trades_buffer[-20:]
        current_wr = sum(1 for t in recent if t.get("win")) / len(recent)
        rollback_threshold = self._checkpoint_win_rate - 0.10
        if current_wr < rollback_threshold:
            logger.warning(
                f"⚠️ EvolutionEngine rollback: win_rate calato da "
                f"{self._checkpoint_win_rate:.2f} a {current_wr:.2f} — "
                f"ripristino parametri checkpoint"
            )
            self._restore_checkpoint(self._last_checkpoint)

    def _run_v18_loops(self) -> None:
        """Execute all V18 periodic feedback loops."""
        # Loop #8: RL policy — trigger training update if enough experience
        try:
            if self._ppo is not None and len(self._ppo_experience_buffer) >= 50:
                # Build a single trajectory from buffered experiences
                buf = self._ppo_experience_buffer[-50:]
                trajectory = {
                    "states": np.vstack([e["state"] for e in buf]),
                    "actions": np.array([e["action"] for e in buf]),
                    "log_probs": np.array([e["log_prob"] for e in buf]),
                    "rewards": np.array([e["reward"] for e in buf]),
                    "values": np.array([e["value"] for e in buf]),
                    "dones": np.array([e["done"] for e in buf]),
                }
                stats = self._ppo.update([trajectory])
                if stats:
                    logger.info(
                        f"🤖 Loop#8 PPO update: actor_loss={stats.get('actor_loss', 0):.4f}, "
                        f"entropy={stats.get('entropy', 0):.4f}, "
                        f"n_samples={stats.get('n_samples', 0)}"
                    )
        except Exception as exc:
            logger.debug(f"Loop#8 RL error: {exc}")

        # Loop #9: HMM — log regime probabilities
        try:
            if self._hmm is not None and self._hmm._is_fitted:
                from data import data_store
                df_btc = data_store.get_df("BTCUSDT", "1h")
                if df_btc is not None and len(df_btc) >= 50:
                    regime_probs = self._hmm.get_regime_probs(df_btc)
                    if regime_probs:
                        current_regime = self._hmm.predict_regime(df_btc)
                        logger.info(
                            f"🔮 Loop#9 HMM regime={current_regime}, "
                            f"probs={{{', '.join(f'{k}:{v:.2f}' for k, v in regime_probs.items())}}}"
                        )
        except Exception as exc:
            logger.debug(f"Loop#9 HMM error: {exc}")

        # Loop #10: Bayesian — log current estimates
        try:
            if self._bayesian is not None:
                posterior = self._bayesian.get_posterior()
                wr = float(posterior.get("win_rate_bayes", posterior.get("win_rate", {}).get("mean", 0.5)))
                logger.debug(f"EvolutionEngine Loop#10 Bayesian win_rate_est={wr:.3f}")
        except Exception as exc:
            logger.debug(f"Loop#10 Bayesian error: {exc}")

        # Loop #11: MAML / Hyperopt — suggest and check parameter updates
        try:
            if self._hyperopt is not None:
                best_params = self._hyperopt.get_best_params()
                if best_params:
                    # Apply best fusion_threshold if better than current
                    best_thresh = best_params.get("fusion_threshold")
                    if best_thresh is not None:
                        current = self._fusion._threshold
                        if abs(best_thresh - current) > 0.02:
                            logger.info(
                                f"EvolutionEngine Loop#11 Hyperopt: threshold update "
                                f"{current:.3f} → {best_thresh:.3f}"
                            )
        except Exception as exc:
            logger.debug(f"Loop#11 Hyperopt error: {exc}")

        # Loop #13: VaR/CVaR → Kill Switch monitoring
        try:
            if self._kill_switch is not None and _VAR_AVAILABLE:
                # Compute VaR from recent trade returns
                recent = self._closed_trades_buffer[-50:] if self._closed_trades_buffer else []
                if len(recent) >= 10:
                    returns = np.array([t.get("pnl", 0) for t in recent])
                    var_result = _compute_var(returns)
                    var_99 = float(var_result.get("historical_99", 0.0))
                    if var_99 > 0.05:
                        logger.warning(f"Loop#13 VaR-99={var_99:.3f} exceeds threshold")
                        if _MSG_BUS_AVAILABLE:
                            get_message_bus().publish(
                                TOPIC_RISK_ALERT,
                                {"type": "var_alert", "var_99": var_99}
                            )
        except Exception as exc:
            logger.debug(f"Loop#13 VaR error: {exc}")

    def _on_concept_drift(self) -> None:
        """Handler for concept drift detection (Loop #12)."""
        logger.warning("🔄 Loop#12: Concept drift — re-calibrating models")

        # Reset LR scheduler for warm restarts
        try:
            if self._lr_scheduler is not None:
                self._lr_scheduler.reset()
        except Exception as exc:
            logger.debug(f"_on_concept_drift LR reset error: {exc}")

        # Reset drift detector
        try:
            if self._drift_detector is not None:
                self._drift_detector.reset()
        except Exception as exc:
            logger.debug(f"_on_concept_drift drift_detector reset error: {exc}")

        # Publish drift event on message bus
        try:
            if _MSG_BUS_AVAILABLE:
                get_message_bus().publish(
                    TOPIC_DRIFT_DETECTED,
                    {"timestamp": time.time(), "source": "evolution_engine"}
                )
        except Exception as exc:
            logger.debug(f"_on_concept_drift message_bus error: {exc}")

    def _run_backtest_validation(self) -> None:
        """Loop #15: Validate strategy robustness via Monte Carlo (background)."""
        if not _BACKTEST_AVAILABLE:
            return
        try:
            trades = list(self._closed_trades_buffer)
            result = _validate_strategy(trades, n_sims=500)
            is_robust = _is_strategy_robust(result)
            logger.info(
                f"EvolutionEngine Loop#15 Backtest validation: "
                f"robust={is_robust}, "
                f"sharpe_mean={result.get('sharpe', {}).get('mean', 0):.3f}, "
                f"n_trades={len(trades)}"
            )
            # Store validation result
            experience_db.save_param(
                "backtest_validation_result",
                {"robust": is_robust, "n_trades": len(trades), "timestamp": time.time()},
                "periodic",
            )
        except Exception as exc:
            logger.debug(f"Loop#15 backtest_validation error: {exc}")

    def shutdown(self) -> None:
        """Persist all state — call on graceful shutdown."""
        try:
            self._meta.save_state()
        except Exception as exc:
            logger.error(f"EvolutionEngine.shutdown meta.save_state error: {exc}")
        try:
            experience_db.save_param(
                "fusion_threshold", self._fusion._threshold, "shutdown"
            )
            experience_db.save_param(
                "confluence_tf_performance",
                self._confluence_adapter.dump_state(),
                "shutdown",
            )
            experience_db.save_param(
                "strategy_evolver_trade_count",
                self._strategy_evolver.trade_count,
                "shutdown",
            )
        except Exception as exc:
            logger.error(f"EvolutionEngine.shutdown db save error: {exc}")

        self._save_runtime_state()
        logger.info("💾 EvolutionEngine: state saved on shutdown")

    def get_report(self) -> Dict[str, Any]:
        """Return a human-readable summary of the current evolution state."""
        report = {
            "fusion_threshold": self._fusion._threshold,
            "strategy_trade_count": self._strategy_evolver.trade_count,
            "tf_performance": self._confluence_adapter.get_performance_summary(),
        }

        # V18 additions
        try:
            if self._bayesian is not None:
                post = self._bayesian.get_posterior()
                wr = float(post.get("win_rate_bayes", post.get("win_rate", {}).get("mean", None) or 0.0))
                report["bayesian_win_rate"] = wr if wr != 0.0 else None
        except Exception:
            pass

        try:
            if self._hmm is not None:
                report["hmm_fitted"] = self._hmm._is_fitted
        except Exception:
            pass

        try:
            if self._kill_switch is not None:
                status = self._kill_switch.get_full_status()
                report["kill_switch"] = {
                    "is_killed": status.get("is_killed", False),
                    "is_safe_mode": status.get("is_safe_mode", False),
                }
        except Exception:
            pass

        report["v18_loops_active"] = sum([
            self._hmm is not None,
            self._ppo is not None,
            self._bayesian is not None,
            self._maml is not None,
            self._drift_detector is not None,
            self._kill_switch is not None,
        ])

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_drawdown(self) -> None:
        """Monitor drawdown and activate circuit breaker if needed."""
        try:
            stats = self._tracker.get_summary()
            max_dd = abs(stats.get("max_drawdown", 0.0))
            current_threshold = self._fusion._threshold

            if max_dd >= _DRAWDOWN_CRITICAL:
                # Safe mode: only very high quality signals
                safe_threshold = float(np.clip(current_threshold + 0.15, 0.50, 0.95))
                if self._fusion._threshold < safe_threshold:
                    self._fusion._threshold = safe_threshold
                    logger.warning(
                        f"🔴 DRAWDOWN CRITICAL ({{max_dd:.1%}}) → safe mode, "
                        f"threshold={{safe_threshold:.3f}}"
                    )
            elif max_dd >= _DRAWDOWN_WARN:
                warn_threshold = float(np.clip(current_threshold + 0.05, 0.30, 0.90))
                if self._fusion._threshold < warn_threshold:
                    self._fusion._threshold = warn_threshold
                    logger.warning(
                        f"🟡 DRAWDOWN WARNING ({{max_dd:.1%}}) → threshold raised to "
                        f"{{warn_threshold:.3f}}"
                    )
        except Exception as exc:
            logger.debug(f"_check_drawdown error: {exc}")

    def _auto_tune_params(self) -> None:
        """Adjust FUSION_THRESHOLD based on recent completed trade outcomes."""
        try:
            recent = experience_db.get_recent_decisions(limit=50)
            completed = [
                d for d in recent
                if d.get("outcome") is not None and d.get("pnl") is not None
            ]
            if len(completed) < _MIN_COMPLETED:
                return

            wins = sum(1 for d in completed if (d.get("pnl") or 0) > 0)
            win_rate = wins / len(completed)
            current = self._fusion._threshold
            # Default: no change
            new_threshold = current

            if win_rate < _THRESHOLD_LOW_WR:
                new_threshold = float(np.clip(current + _THRESHOLD_STEP_UP,
                                              _THRESHOLD_MIN, _THRESHOLD_MAX))
            elif win_rate > _THRESHOLD_HIGH_WR:
                new_threshold = float(np.clip(current - _THRESHOLD_STEP_DOWN,
                                              _THRESHOLD_MIN, _THRESHOLD_MAX))

            if new_threshold != current:
                logger.info(
                    f"🔧 Auto-tune: win_rate={{win_rate:.1%}} → "
                    f"threshold {{current:.3f}} → {{new_threshold:.3f}}"
                )

            self._fusion._threshold = new_threshold
            experience_db.save_param("fusion_threshold", new_threshold, "auto_tune")
            if self._hyperopt is not None:
                try:
                    if self._closed_trades_buffer:
                        recent50 = self._closed_trades_buffer[-50:]
                        hyp_score = sum(1 for t in recent50 if t.get("win")) / len(recent50)
                    else:
                        hyp_score = 0.5
                    if self._last_hyperopt_params:
                        self._hyperopt.report_result(self._last_hyperopt_params, hyp_score)
                    suggested = self._hyperopt.suggest_params()
                    self._last_hyperopt_params = suggested
                    if suggested and "fusion_threshold" in suggested:
                        new_thr = float(np.clip(suggested["fusion_threshold"], _THRESHOLD_MIN, _THRESHOLD_MAX))
                        self._fusion._threshold = new_thr
                        logger.info(f"Loop#11 Hyperopt: fusion_threshold -> {new_thr:.3f} (score={hyp_score:.2f})")
                except Exception as exc:
                    logger.debug(f"Loop#11 _auto_tune_params hyperopt error: {exc}")
        except Exception as exc:
            logger.error(f"_auto_tune_params error: {exc}")

    def _save_state(self) -> None:
        """Persist MetaAgent state and current tuned parameters."""
        try:
            self._meta.save_state()
        except Exception as exc:
            logger.error(f"_save_state meta.save_state error: {exc}")
        try:
            experience_db.save_param(
                "fusion_threshold", self._fusion._threshold, "periodic"
            )
            experience_db.save_param(
                "confluence_tf_performance",
                self._confluence_adapter.dump_state(),
                "periodic",
            )
            experience_db.save_param(
                "strategy_evolver_trade_count",
                self._strategy_evolver.trade_count,
                "periodic",
            )
        except Exception as exc:
            logger.error(f"_save_state db save error: {exc}")

        self._save_runtime_state()
        logger.info("💾 EvolutionEngine: periodic state save complete")

    def _save_runtime_state(self) -> None:
        """Persist runtime learning state not covered by DB params."""
        try:
            os.makedirs(os.path.dirname(_RUNTIME_STATE_PATH) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(_PPO_MODEL_PATH) or ".", exist_ok=True)
            payload = {
                "saved_at": time.time(),
                "closed_trades_buffer": self._closed_trades_buffer[-500:],
                "last_hyperopt_params": self._last_hyperopt_params,
                "checkpoint_win_rate": self._checkpoint_win_rate,
                "ppo_experience_buffer": [
                    {
                        "symbol": str(e.get("symbol", "")),
                        "trade_reward": float(e.get("trade_reward", 0.0)),
                        "pnl": float(e.get("pnl", 0.0)),
                        "state": np.asarray(e.get("state", np.zeros(12))).tolist(),
                        "action": int(e.get("action", 0)),
                        "log_prob": float(e.get("log_prob", 0.0)),
                        "reward": float(e.get("reward", 0.0)),
                        "value": float(e.get("value", 0.0)),
                        "done": float(e.get("done", 1.0)),
                        "next_state": np.asarray(e.get("next_state", np.zeros(12))).tolist(),
                    }
                    for e in self._ppo_experience_buffer[-200:]
                ],
            }
            with open(_RUNTIME_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            if self._ppo is not None and hasattr(self._ppo, "save"):
                self._ppo.save(_PPO_MODEL_PATH)
        except Exception as exc:
            logger.error(f"_save_runtime_state error: {exc}")

    def _load_runtime_state(self) -> None:
        """Restore runtime learning state from disk if available."""
        try:
            if self._ppo is not None and hasattr(self._ppo, "load") and os.path.exists(_PPO_MODEL_PATH):
                if self._ppo.load(_PPO_MODEL_PATH):
                    logger.info("🎓 EvolutionEngine: restored PPO runtime model")
        except Exception as exc:
            logger.debug(f"_load_runtime_state ppo load error: {exc}")

        try:
            if not os.path.exists(_RUNTIME_STATE_PATH):
                return
            with open(_RUNTIME_STATE_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            loaded_closed = payload.get("closed_trades_buffer", [])
            if isinstance(loaded_closed, list):
                self._closed_trades_buffer = loaded_closed[-500:]

            self._last_hyperopt_params = payload.get("last_hyperopt_params")
            self._checkpoint_win_rate = float(payload.get("checkpoint_win_rate", self._checkpoint_win_rate))

            loaded_exp = payload.get("ppo_experience_buffer", [])
            rebuilt = []
            if isinstance(loaded_exp, list):
                for e in loaded_exp[-200:]:
                    try:
                        rebuilt.append(
                            {
                                "symbol": str(e.get("symbol", "")),
                                "trade_reward": float(e.get("trade_reward", 0.0)),
                                "pnl": float(e.get("pnl", 0.0)),
                                "state": np.asarray(e.get("state", []), dtype=float),
                                "action": int(e.get("action", 0)),
                                "log_prob": float(e.get("log_prob", 0.0)),
                                "reward": float(e.get("reward", 0.0)),
                                "value": float(e.get("value", 0.0)),
                                "done": float(e.get("done", 1.0)),
                                "next_state": np.asarray(e.get("next_state", []), dtype=float),
                            }
                        )
                    except Exception:
                        continue
            self._ppo_experience_buffer = rebuilt
            logger.info(
                f"💾 EvolutionEngine: runtime state restored "
                f"(closed={len(self._closed_trades_buffer)}, ppo_exp={len(self._ppo_experience_buffer)})"
            )
        except Exception as exc:
            logger.debug(f"_load_runtime_state error: {exc}")
