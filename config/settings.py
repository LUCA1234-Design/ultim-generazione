"""
V17 Agentic AI Trading System — Configuration
All settings in V16 style: os.getenv with hardcoded fallbacks.
"""
import os

# ============================================================
# API CREDENTIALS
# ============================================================

API_KEY = os.getenv("BINANCE_API_KEY", "v5lsKf3Ajri6DXZPkUuD8zMWCHN861vMk3fTTrDA19UnOZtKvabmJHH6x3DkpumZ")
API_SECRET = os.getenv("BINANCE_API_SECRET", "XW0MnFlgNg40v8EvIuJQSAyo9hxWseXzKKPsnj1IrhqpAAyRsZyqBNmff7ZgMI")
TELEGRAM_TOKEN = "8436199553:AAEJAYyl3HCbeg3hzT1m9DhYIo_WniLjyVI"
TELEGRAM_CHAT_ID = "675648539"

# ============================================================
# AI SETTINGS
# ============================================================

AI_ENABLED = True
AI_SYNC_ON_SIGNAL = True
AI_SECTION_IN_MSG = True
AI_URL_SCOUT = "http://127.0.0.1:1234/v1/chat/completions"
AI_MODEL_SCOUT = "qwen2.5-1.5b-instruct"
AI_URL_ANALYST = "http://127.0.0.1:1234/v1/chat/completions"
AI_MODEL_ANALYST = "qwen2.5-coder-7b-instruct"
AI_TIMEOUT = 45
AI_CALL_COOLDOWN = 300

# ============================================================
# TRADING ENGINE
# ============================================================

PAPER_TRADING = True           # Paper trading ON by default
ACCOUNT_BALANCE = 1000.0
THRESHOLD_BASE = 0.35
MAX_OPEN_POSITIONS = 3
LEVERAGE = 10

# ============================================================
# RISK GUARDS / QUALITY GATES
# ============================================================

MAX_DAILY_LOSS_USDT = 50.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_CONSECUTIVE_LOSSES = 3

MIN_FUSION_SCORE = 0.50          # Raised from 0.25: sniper calibration — only high-confidence signals
MIN_AGENT_CONFIRMATIONS = 4      # Raised from 3: require more agent consensus
NON_OPTIMAL_HOUR_PENALTY = 0.05  # Raised from 0.01: significant penalty outside optimal hours
MIN_RR = 1.60                    # Fixed from 1.80: base R/R (SL=1.5×ATR, TP1=2.5×ATR) gives 1.667 — must be below base

WS_STALE_TIMEOUT = 60
WS_HEALTH_LOG_INTERVAL = 120
WS_MAX_FAIL_COUNT_ALERT = 20

# ============================================================
# HIDDEN GEMS (HG)
# ============================================================

HG_ENABLED = True
HG_MONITOR_ALL = True
HG_TF = ["1h", "15m"]
HG_TF_SECONDS = {"1h": 3600, "15m": 900}
HG_COOLDOWN = 180
HG_RVOL_PARTIAL_MIN = 2.0
HG_RVOL_BAR_MIN = 1.3
HG_SQUEEZE_MIN_BARS = 10
HG_NR7_RVOL_MIN = 1.2
HG_RS_SLOPE_MIN = 0.0014
HG_LOOKBACK_RS = 48
HG_LOOKBACK_HIGH = 20
HG_SQZ_ON = True
HG_NR7_ON = True
HG_RS_ON = True
HG_MIN_QUOTE_VOL = 70000
HG_QVOL_LOOKBACK = 20
HG_CFG = {
    "1h": {"rvol_partial_min": 1.2, "rvol_bar_min": 1.2, "min_score": 0.55, "cooldown": HG_COOLDOWN},
    "15m": {"rvol_partial_min": 1.8, "rvol_bar_min": 1.3, "min_score": 0.65, "cooldown": 300},
}

# ============================================================
# SIGNAL MANAGEMENT
# ============================================================

SIGNAL_COOLDOWN = 1800
SIGNAL_COOLDOWN_BY_TF = {"15m": 900, "1h": 3600, "4h": 7200}
DIVERGENCE_MAX_AGE_HOURS = 4
DIVERGENCE_MAX_AGE_CANDLES = 3
DIVERGENCE_MAX_AGE_BY_TF = {"15m": 2, "1h": 2, "4h": 1}

# ============================================================
# BREAKOUT RULES
# ============================================================

BREAKOUT_RULES = {
    "1h": {"vol_min": 0.6, "break_mult": 1.001, "min_closes": 1, "atr_mult": 0.08},
    "15m": {"vol_min": 0.6, "break_mult": 1.0004, "min_closes": 1, "atr_mult": 0.05},
}

# ============================================================
# TIME FILTERS
# ============================================================

ORARI_VIETATI_UTC = [3, 4]  # Only the 2 deadest hours (3:00-4:59 UTC = 5:00-6:59 AM Italy)
ORARI_MIGLIORI_UTC = list(range(7, 17)) + list(range(19, 24)) + [0, 1]  # Expanded, added 0, 1, 7

# ============================================================
# DATA SETTINGS
# ============================================================

HISTORICAL_LIMIT = 500
SYMBOLS_LIMIT = 120
WS_GROUP_SIZE = 40
WS_RECONNECT_DELAY_BASE = 5
WS_MAX_RECONNECT_DELAY = 60
POLL_CLOSED_ENABLE = True
POLL_CLOSED_INTERVAL = 60
STARTUP_TIMEOUT = 25
TELEGRAM_RATE_LIMIT = 3
TELEGRAM_TEST_ON_START = True

# ============================================================
# DATABASE
# ============================================================

DB_PATH = "v17_experience.db"

# ============================================================
# REGIME AGENT
# ============================================================

REGIME_N_COMPONENTS = 3       # Number of Gaussian mixture components
REGIME_NAMES = ["trending", "ranging", "volatile"]
REGIME_LOOKBACK = 100

# ============================================================
# META AGENT
# ============================================================

META_EVAL_WINDOW = 50         # Number of decisions to evaluate per agent
META_MIN_SAMPLES = 10         # Minimum samples before adjusting weights
META_WEIGHT_DECAY = 0.95      # Exponential decay for old samples

# ============================================================
# DECISION FUSION
# ============================================================

FUSION_THRESHOLD_DEFAULT = 0.55  # Raised from 0.30: sniper calibration — fewer but higher-quality signals
FUSION_AGENT_WEIGHTS = {
    "regime": 0.15,
    "pattern": 0.25,       # slightly reduced to accommodate V18 agents
    "confluence": 0.25,    # slightly reduced to accommodate V18 agents
    "risk": 0.12,          # slightly reduced to accommodate V18 agents
    "strategy": 0.05,      # reduced — mutated strategies are noisy
    "meta": 0.05,          # unchanged
    # V18 agents (active only when available)
    "orderflow": 0.05,     # order flow / microstructure signal
    "sentiment": 0.03,     # funding rate / OI / L-S ratio proxy
    "correlation": 0.03,   # cross-asset correlation & divergence
    "contrarian": 0.02,    # devil's advocate — penalises groupthink
}

# ============================================================
# HEARTBEAT
# ============================================================

HEARTBEAT_INTERVAL = 7200       # Every 2 hours (in seconds)
HEARTBEAT_ENABLED = True

# ============================================================
# PERFORMANCE TRACKER
# ============================================================

PERF_TP1_MULT = 2.0           # ATR multiplier for TP1 evaluation
PERF_SL_MULT = 2.0            # ATR multiplier for SL evaluation
PERF_LOOKBACK_HOURS = 24      # How far back to evaluate outcomes

# ============================================================
# TRAINING MODE
# Enable reduced thresholds to accumulate trades rapidly so
# the 7 feedback loops receive enough data to activate.
# Once TRAINING_TARGET_TRADES completed trades are recorded
# in the DB the system auto-switches to Sniper Mode.
# ============================================================

TRAINING_MODE = True                # Enable reduced thresholds for fast learning
TRAINING_TARGET_TRADES = 200        # Completed trades required before switching to Sniper Mode

# Training Mode overrides (lower thresholds to generate more signals)
TRAINING_FUSION_THRESHOLD = 0.35
TRAINING_MIN_FUSION_SCORE = 0.30
TRAINING_MIN_AGENT_CONFIRMATIONS = 3
TRAINING_MIN_RR = 1.20
TRAINING_NON_OPTIMAL_HOUR_PENALTY = 0.02
TRAINING_SIGNAL_COOLDOWN_BY_TF = {"15m": 600, "1h": 1800, "4h": 3600}
TRAINING_MAX_OPEN_POSITIONS = 5

# Sniper Mode values (restored after training completes)
SNIPER_FUSION_THRESHOLD = 0.55
SNIPER_MIN_FUSION_SCORE = 0.50
SNIPER_MIN_AGENT_CONFIRMATIONS = 4
SNIPER_MIN_RR = 1.60
SNIPER_NON_OPTIMAL_HOUR_PENALTY = 0.05
SNIPER_SIGNAL_COOLDOWN_BY_TF = {"15m": 900, "1h": 3600, "4h": 7200}
SNIPER_MAX_OPEN_POSITIONS = 3

# Apply training overrides automatically when TRAINING_MODE is active
if TRAINING_MODE:
    FUSION_THRESHOLD_DEFAULT = TRAINING_FUSION_THRESHOLD
    MIN_FUSION_SCORE = TRAINING_MIN_FUSION_SCORE
    MIN_AGENT_CONFIRMATIONS = TRAINING_MIN_AGENT_CONFIRMATIONS
    MIN_RR = TRAINING_MIN_RR
    NON_OPTIMAL_HOUR_PENALTY = TRAINING_NON_OPTIMAL_HOUR_PENALTY
    SIGNAL_COOLDOWN_BY_TF = TRAINING_SIGNAL_COOLDOWN_BY_TF
    MAX_OPEN_POSITIONS = TRAINING_MAX_OPEN_POSITIONS

# ============================================================
# V18 — REINFORCEMENT LEARNING
# ============================================================

RL_LEARNING_RATE = 3e-4         # PPO actor-critic learning rate
RL_GAMMA = 0.99                 # discount factor
RL_GAE_LAMBDA = 0.95            # GAE lambda for advantage estimation
RL_CLIP_EPSILON = 0.2           # PPO clipping epsilon
RL_BATCH_SIZE = 64              # mini-batch size for PPO updates
RL_REPLAY_BUFFER_SIZE = 10_000  # prioritized replay buffer capacity
RL_N_FEATURES = 12              # state space dimension
RL_N_ACTIONS = 8                # action space size
RL_PPO_EPOCHS = 4               # optimisation epochs per PPO update
RL_TRAIN_OFFLINE = True         # train offline on historical sim episodes
RL_INFERENCE_ENABLED = True     # use PPO for live inference suggestions

# ============================================================
# V18 — META-LEARNING
# ============================================================

MAML_LEARNING_RATE = 0.01       # outer loop meta-learning rate
MAML_ADAPT_LEARNING_RATE = 0.1  # inner loop adaptation rate
MAML_N_SHOTS = 10               # few-shot adaptation: trades per regime
HYPEROPT_BUDGET = 200           # maximum hyperparameter evaluations
HYPEROPT_GAMMA = 0.25           # TPE good/bad fraction
DRIFT_DETECTION_THRESHOLD = 50.0  # Page-Hinkley threshold
DRIFT_RETRAIN_COOLDOWN = 3600   # min seconds between re-trains
LR_SCHEDULER_T0 = 50            # cosine annealing initial period
LR_SCHEDULER_T_MULT = 2         # restart period multiplier

# ============================================================
# V18 — RISK INSTITUTIONAL
# ============================================================

VAR_CONFIDENCE_LEVELS = (0.95, 0.99)
VAR_MONTE_CARLO_SIMS = 10_000
CVAR_CONFIDENCE = 0.95
MAX_HHI = 0.40
MAX_CORRELATION_EXPOSURE = 0.80

KILL_L1_POSITION_LOSS_PCT = 0.05
KILL_L2_DAILY_LOSS_PCT = 0.03
KILL_L3_DRAWDOWN_PCT = 0.12
KILL_L4_CORRELATION = 0.85
KILL_L5_VOL_MULTIPLIER = 4.0

MARGIN_WARNING_PCT = 0.70
MARGIN_CRITICAL_PCT = 0.85
MARGIN_LIQ_DISTANCE_MIN = 0.05

# ============================================================
# V18 — COORDINATION
# ============================================================

CONSENSUS_ROUNDS = 3
CONSENSUS_CONTRARIAN_THRESHOLD = 0.70
CONSENSUS_CONTRARIAN_PENALTY = 0.30
SUPERVISOR_ERROR_THRESHOLD = 0.30
SUPERVISOR_RECOVERY_COOLDOWN = 300
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3

# ============================================================
# V18 — BACKTESTING
# ============================================================

BACKTEST_N_SIMS = 1000
BACKTEST_CONFIDENCE = 0.95
WALK_FORWARD_FOLDS = 5
WALK_FORWARD_TRAIN_PCT = 0.70
BACKTEST_MIN_SHARPE = 0.5
BACKTEST_MIN_WIN_RATE = 0.45
BACKTEST_MAX_DRAWDOWN = 0.20
BACKTEST_VALIDATION_INTERVAL = 7200
BACKTEST_MIN_TRADES = 20
