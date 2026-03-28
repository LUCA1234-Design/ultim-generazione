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
MAX_OPEN_POSITIONS = 5
LEVERAGE = 10

# ============================================================
# RISK GUARDS / QUALITY GATES
# ============================================================

MAX_DAILY_LOSS_USDT = 50.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_CONSECUTIVE_LOSSES = 3

MIN_FUSION_SCORE = 0.55          # Lowered from 0.70: RegimeAgent ranging scores ~0.45-0.65 were pulling the average below threshold
MIN_AGENT_CONFIRMATIONS = 3      # Lowered from 4: MetaAgent/StrategyAgent can fail, 3 is enough to confirm a signal
NON_OPTIMAL_HOUR_PENALTY = 0.02  # Extra fusion score required outside optimal trading hours (was 0.05)
MIN_RR = 1.50

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

SIGNAL_COOLDOWN = 600
SIGNAL_COOLDOWN_BY_TF = {"15m": 300, "1h": 600, "4h": 3600}
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

ORARI_VIETATI_UTC = list(range(2, 6))
ORARI_MIGLIORI_UTC = list(range(8, 16)) + list(range(20, 24))

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
TELEGRAM_TEST_ON_START = False

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

FUSION_THRESHOLD_DEFAULT = 0.65
FUSION_AGENT_WEIGHTS = {
    "regime": 0.20,
    "pattern": 0.25,
    "confluence": 0.25,
    "risk": 0.15,
    "strategy": 0.15,
}

# ============================================================
# PERFORMANCE TRACKER
# ============================================================

PERF_TP1_MULT = 2.0           # ATR multiplier for TP1 evaluation
PERF_SL_MULT = 2.0            # ATR multiplier for SL evaluation
PERF_LOOKBACK_HOURS = 24      # How far back to evaluate outcomes
