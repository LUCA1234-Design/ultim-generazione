# ============================================================
# 🤖 BOT V16 "CECCHINO ISTITUZIONALE" — SMART MONEY EDITION
# ============================================================
import time, json, requests, datetime, numpy as np, pandas as pd
import threading, queue, traceback, websocket, ssl, logging, gc, os, csv, random, sys, math, sqlite3
from io import BytesIO
from threading import Lock
from scipy.stats import zscore
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from binance.client import Client

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("BINANCE_API_KEY", "v5lsKf3Ajri6DXZPkUuD8zMWCHN861vMk3fTTrDA19UnOZtKvabmJHH6x3DkpumZ")
API_SECRET = os.getenv("BINANCE_API_SECRET", "XW0MnFlgNg40v8EvIuJQSAyo9hxWseXzKKPsnj1IrhqpAAyRsZyqBNmff7ZgMI")
TELEGRAM_TOKEN = "8436199553:AAEJAYyl3HCbeg3hzT1m9DhYIo_WniLjyVI"
TELEGRAM_CHAT_ID = "675648539"
AI_ENABLED = True
AI_SYNC_ON_SIGNAL = True
AI_SECTION_IN_MSG = True
AI_URL_SCOUT = "http://127.0.0.1:1234/v1/chat/completions"
AI_MODEL_SCOUT = "qwen2.5-1.5b-instruct"
AI_URL_ANALYST = "http://127.0.0.1:1234/v1/chat/completions"
AI_MODEL_ANALYST = "qwen2.5-coder-7b-instruct"
AI_TIMEOUT = 45
THRESHOLD_BASE = 0.35
ACCOUNT_BALANCE = 1000.0
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
    "1h": {"rvol_partial_min": 1.4, "rvol_bar_min": 1.3, "min_score": 0.65, "cooldown": HG_COOLDOWN},
    "15m": {"rvol_partial_min": 2.3, "rvol_bar_min": 1.5, "min_score": 0.78, "cooldown": 300},
}
SIGNAL_COOLDOWN = 600
SIGNAL_COOLDOWN_BY_TF = {"15m": 300, "1h": 600, "4h": 3600}
DIVERGENCE_MAX_AGE_HOURS = 4
DIVERGENCE_MAX_AGE_CANDLES = 3
DIVERGENCE_MAX_AGE_BY_TF = {"15m": 2, "1h": 2, "4h": 1}
BREAKOUT_RULES = {
    "1h": {"vol_min": 0.6, "break_mult": 1.001, "min_closes": 1, "atr_mult": 0.08},
    "15m": {"vol_min": 0.6, "break_mult": 1.0004, "min_closes": 1, "atr_mult": 0.05},
}
ORARI_VIETATI_UTC = list(range(2, 6))
ORARI_MIGLIORI_UTC = list(range(8, 16)) + list(range(20, 24))

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_v16.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("BotV16")

# ============================================================
# GLOBAL STATE
# ============================================================

client = None
symbols_whitelist = []
symbols_hg_all = []
historical_data = {}          # {symbol: {interval: DataFrame}}
realtime_data = {}            # {symbol: {interval: DataFrame}}
WS_HEALTH = {}                # {name: {alive, last_msg}}
LAST_KLINE_TIME = {}          # {symbol_interval: open_time_ms}
last_signal_time = {}         # {symbol_interval: timestamp}
last_hg_signal_time = {}      # {symbol: timestamp}
last_ai_call_per_symbol = {}  # {symbol: timestamp}
last_message_time = {}        # {chat_id: timestamp}
data_lock = Lock()
signal_lock = Lock()
ai_lock = Lock()
db_conn = None
DB_PATH = "bot_v16.db"

# ============================================================
# DATABASE
# ============================================================

def init_database():
    global db_conn
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = db_conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, symbol TEXT, interval TEXT, direction TEXT,
        pattern TEXT, score REAL, entry REAL, sl REAL, tp1 REAL, tp2 REAL,
        rr REAL, kelly REAL, ai_verdict TEXT, msg TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS state (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS logbook (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, symbol TEXT, interval TEXT, event TEXT, detail TEXT
    )""")
    db_conn.commit()
    logger.info("✅ Database inizializzato")

def db_save_last_signal_time():
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                  ("last_signal_time", json.dumps(last_signal_time)))
        db_conn.commit()
    except Exception as e:
        logger.error(f"DB save last_signal_time: {e}")

def db_load_last_signal_time():
    global last_signal_time
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("SELECT value FROM state WHERE key=?", ("last_signal_time",))
        row = c.fetchone()
        if row:
            last_signal_time = json.loads(row[0])
    except Exception as e:
        logger.error(f"DB load last_signal_time: {e}")

def db_save_last_ai_call_per_symbol():
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                  ("last_ai_call", json.dumps(last_ai_call_per_symbol)))
        db_conn.commit()
    except Exception as e:
        logger.error(f"DB save last_ai_call: {e}")

def db_load_last_ai_call_per_symbol():
    global last_ai_call_per_symbol
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("SELECT value FROM state WHERE key=?", ("last_ai_call",))
        row = c.fetchone()
        if row:
            last_ai_call_per_symbol = json.loads(row[0])
    except Exception as e:
        logger.error(f"DB load last_ai_call: {e}")

def db_save_ws_health():
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                  ("ws_health", json.dumps({k: {"alive": v.get("alive", False)} for k, v in WS_HEALTH.items()})))
        db_conn.commit()
    except Exception as e:
        logger.error(f"DB save ws_health: {e}")

def db_load_ws_health():
    pass  # WS health is re-established at startup

def db_save_last_message_time():
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                  ("last_message_time", json.dumps(last_message_time)))
        db_conn.commit()
    except Exception as e:
        logger.error(f"DB save last_message_time: {e}")

def db_load_last_message_time():
    global last_message_time
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("SELECT value FROM state WHERE key=?", ("last_message_time",))
        row = c.fetchone()
        if row:
            last_message_time = json.loads(row[0])
    except Exception as e:
        logger.error(f"DB load last_message_time: {e}")

def db_log_signal(symbol, interval, direction, pattern, score, entry, sl, tp1, tp2, rr, kelly, ai_verdict, msg):
    if db_conn is None:
        return
    try:
        c = db_conn.cursor()
        c.execute("""INSERT INTO signals
            (ts, symbol, interval, direction, pattern, score, entry, sl, tp1, tp2, rr, kelly, ai_verdict, msg)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), symbol, interval, direction, pattern, score, entry, sl, tp1, tp2, rr, kelly, ai_verdict, msg[:500]))
        db_conn.commit()
    except Exception as e:
        logger.error(f"DB log signal: {e}")

# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_RATE_LIMIT = 3  # seconds between messages

def send_telegram(msg, parse_mode="Markdown", chat_id=None):
    cid = chat_id or TELEGRAM_CHAT_ID
    now = time.time()
    last = last_message_time.get(cid, 0)
    if now - last < TELEGRAM_RATE_LIMIT:
        time.sleep(TELEGRAM_RATE_LIMIT - (now - last))
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": cid, "text": msg, "parse_mode": parse_mode}
        resp = requests.post(url, json=payload, timeout=10)
        last_message_time[cid] = time.time()
        return resp.json()
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return None

def send_telegram_photo(buf, caption="", chat_id=None):
    cid = chat_id or TELEGRAM_CHAT_ID
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", buf, "image/png")}
        data = {"chat_id": cid, "caption": caption, "parse_mode": "Markdown"}
        resp = requests.post(url, files=files, data=data, timeout=20)
        return resp.json()
    except Exception as e:
        logger.error(f"Telegram photo error: {e}")
        return None

def test_telegram():
    result = send_telegram("🔧 Test connessione Telegram Bot V16")
    if result and result.get("ok"):
        logger.info("✅ Telegram OK")
    else:
        logger.warning(f"⚠️ Telegram test: {result}")

# ============================================================
# INDICATORS
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_obv(df):
    close = df["close"]
    volume = df["volume"]
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

def calc_bollinger(series, period=20, std_dev=2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower

def calc_keltner(df, period=20, atr_mult=1.5):
    ema = df["close"].ewm(span=period, min_periods=period).mean()
    atr = calc_atr(df, period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, ema, lower

def calc_adx(df, period=14):
    h = df["high"]
    l = df["low"]
    c = df["close"]
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    dm_plus = np.where((h - prev_h) > (prev_l - l), np.maximum(h - prev_h, 0), 0)
    dm_minus = np.where((prev_l - l) > (h - prev_h), np.maximum(prev_l - l, 0), 0)
    dm_plus = pd.Series(dm_plus, index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, di_plus, di_minus

def calc_zscore(series, period=20):
    rolling_mean = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)

def calc_cvd(df):
    """Cumulative Volume Delta (buy volume - sell volume)"""
    close = df["close"]
    open_ = df["open"]
    volume = df["volume"]
    buy_pct = (close - open_) / (df["high"] - df["low"] + 1e-10)
    buy_pct = buy_pct.clip(0, 1)
    buy_vol = buy_pct * volume
    sell_vol = (1 - buy_pct) * volume
    delta = buy_vol - sell_vol
    cvd = delta.cumsum()
    return cvd, delta

def calc_volume_delta(df):
    """Volume delta per bar"""
    _, delta = calc_cvd(df)
    return delta

def detect_liquidity_sweep(df, lookback=20):
    """Detect liquidity sweep: price wicks beyond recent high/low then reverses"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    sweeps = pd.Series(0, index=df.index)
    for i in range(lookback, len(df)):
        recent_high = high.iloc[i-lookback:i].max()
        recent_low = low.iloc[i-lookback:i].min()
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        o = open_.iloc[i]
        if h > recent_high and c < recent_high:
            sweeps.iloc[i] = -1  # Bearish sweep
        elif l < recent_low and c > recent_low:
            sweeps.iloc[i] = 1   # Bullish sweep
    return sweeps

# ============================================================
# HISTORICAL DATA MANAGEMENT
# ============================================================

HISTORICAL_LIMIT = 500

def fetch_klines(symbol, interval, limit=HISTORICAL_LIMIT):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        rows = []
        for k in klines:
            rows.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_vol": float(k[10]),
            })
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_vol"]]
        return df
    except Exception as e:
        logger.error(f"fetch_klines {symbol} {interval}: {e}")
        return None

def init_historical_for_symbol(symbol):
    with data_lock:
        if symbol not in historical_data:
            historical_data[symbol] = {}
        if symbol not in realtime_data:
            realtime_data[symbol] = {}
    for interval in ["15m", "1h", "4h"]:
        try:
            df = fetch_klines(symbol, interval)
            if df is not None and not df.empty:
                with data_lock:
                    historical_data[symbol][interval] = df
                    realtime_data[symbol][interval] = df.copy()
            time.sleep(0.05)
        except Exception as e:
            logger.debug(f"init_historical {symbol} {interval}: {e}")

def get_df(symbol, interval):
    with data_lock:
        rt = realtime_data.get(symbol, {}).get(interval)
        if rt is not None and not rt.empty:
            return rt.copy()
        hist = historical_data.get(symbol, {}).get(interval)
        if hist is not None and not hist.empty:
            return hist.copy()
    return None

def update_realtime(symbol, interval, kline):
    try:
        open_time_ms = int(kline["t"])
        row = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline.get("q", 0)),
            "trades": int(kline.get("n", 0)),
            "taker_buy_vol": float(kline.get("V", 0)),
        }
        idx = pd.to_datetime(open_time_ms, unit="ms", utc=True)
        with data_lock:
            if symbol not in realtime_data:
                realtime_data[symbol] = {}
            df = realtime_data[symbol].get(interval)
            if df is None or df.empty:
                df = historical_data.get(symbol, {}).get(interval)
                if df is None:
                    return
                df = df.copy()
            new_row = pd.DataFrame([row], index=[idx])
            if idx in df.index:
                df.loc[idx] = row
            else:
                df = pd.concat([df, new_row])
            if len(df) > HISTORICAL_LIMIT:
                df = df.iloc[-HISTORICAL_LIMIT:]
            realtime_data[symbol][interval] = df
    except Exception as e:
        logger.debug(f"update_realtime {symbol} {interval}: {e}")

# ============================================================
# PATTERN DETECTORS
# ============================================================

def detect_squeeze(df, min_bars=10):
    """Bollinger Bands inside Keltner Channels — squeeze"""
    if len(df) < 30:
        return False, 0
    bb_upper, bb_mid, bb_lower = calc_bollinger(df["close"], 20, 2.0)
    kc_upper, kc_mid, kc_lower = calc_keltner(df, 20, 1.5)
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    n_squeeze = squeeze.iloc[-min_bars:].sum()
    in_squeeze = squeeze.iloc[-1]
    momentum = df["close"].iloc[-1] - df["close"].iloc[-5]
    return bool(in_squeeze), int(n_squeeze)

def detect_nr7(df):
    """Narrowest Range 7 — lowest range of last 7 bars"""
    if len(df) < 7:
        return False
    ranges = (df["high"] - df["low"]).iloc[-7:]
    return float(ranges.iloc[-1]) == float(ranges.min())

def detect_rs_leader(df_sym, df_btc, lookback=48, slope_min=0.0014):
    """Relative Strength vs BTC — symbol outperforming"""
    if df_sym is None or df_btc is None:
        return False, 0.0
    if len(df_sym) < lookback or len(df_btc) < lookback:
        return False, 0.0
    try:
        sym_ret = df_sym["close"].iloc[-lookback:].pct_change().fillna(0)
        btc_ret = df_btc["close"].iloc[-lookback:].pct_change().fillna(0)
        rs = (1 + sym_ret).cumprod() / (1 + btc_ret).cumprod()
        rs = rs.dropna()
        if len(rs) < 2:
            return False, 0.0
        x = np.arange(len(rs))
        slope = np.polyfit(x, rs.values, 1)[0]
        return slope > slope_min, float(slope)
    except Exception:
        return False, 0.0

def detect_rsi_divergence(df, rsi_period=14, lookback=30):
    """Bullish/bearish RSI divergence detection"""
    if len(df) < lookback + rsi_period:
        return None, None
    try:
        rsi = calc_rsi(df["close"], rsi_period)
        price_lows = argrelextrema(df["low"].values, np.less, order=3)[0]
        rsi_lows = argrelextrema(rsi.values, np.less, order=3)[0]
        price_highs = argrelextrema(df["high"].values, np.greater, order=3)[0]
        rsi_highs = argrelextrema(rsi.values, np.greater, order=3)[0]

        # Bullish divergence: price lower low, RSI higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            p1, p2 = price_lows[-2], price_lows[-1]
            r1, r2 = rsi_lows[-2], rsi_lows[-1]
            if (df["low"].iloc[p2] < df["low"].iloc[p1] and
                    rsi.iloc[r2] > rsi.iloc[r1] and
                    abs(p2 - r2) <= 3):
                age_candles = len(df) - 1 - p2
                return "bullish", age_candles

        # Bearish divergence: price higher high, RSI lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            p1, p2 = price_highs[-2], price_highs[-1]
            r1, r2 = rsi_highs[-2], rsi_highs[-1]
            if (df["high"].iloc[p2] > df["high"].iloc[p1] and
                    rsi.iloc[r2] < rsi.iloc[r1] and
                    abs(p2 - r2) <= 3):
                age_candles = len(df) - 1 - p2
                return "bearish", age_candles

        return None, None
    except Exception:
        return None, None

def detect_hammer(df):
    """Bullish hammer / bearish shooting star"""
    if len(df) < 3:
        return None
    c = df.iloc[-1]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    total_range = c["high"] - c["low"]
    if total_range < 1e-10:
        return None
    if lower_wick > 2 * body and upper_wick < body and body / total_range < 0.4:
        return "hammer_bullish"
    if upper_wick > 2 * body and lower_wick < body and body / total_range < 0.4:
        return "shooting_star_bearish"
    return None

# ============================================================
# SIGNAL SCORING & CONFLUENCE
# ============================================================

def calc_score(symbol, interval, df):
    """Calculate confluence score (0.0 - 1.0) for a symbol/interval"""
    score = 0.0
    details = []

    if df is None or len(df) < 50:
        return 0.0, []

    close = df["close"]
    rsi = calc_rsi(close)
    atr = calc_atr(df)
    macd_line, macd_sig, macd_hist = calc_macd(close)
    adx, di_plus, di_minus = calc_adx(df)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    obv = calc_obv(df)
    _, delta = calc_cvd(df)
    sweeps = detect_liquidity_sweep(df)
    zscore_val = calc_zscore(close).iloc[-1]

    last_rsi = rsi.iloc[-1]
    last_adx = adx.iloc[-1]
    last_macd_hist = macd_hist.iloc[-1]
    last_close = close.iloc[-1]
    last_bb_lower = bb_lower.iloc[-1]
    last_bb_upper = bb_upper.iloc[-1]
    last_sweep = sweeps.iloc[-1]
    last_obv_slope = obv.iloc[-1] - obv.iloc[-5]
    last_delta = delta.iloc[-1]

    # RSI oversold/overbought
    if last_rsi < 35:
        score += 0.15
        details.append(f"RSI oversold {last_rsi:.1f}")
    elif last_rsi > 65:
        score += 0.10
        details.append(f"RSI overbought {last_rsi:.1f}")

    # MACD
    if last_macd_hist > 0 and macd_hist.iloc[-2] < 0:
        score += 0.15
        details.append("MACD cross up")
    elif last_macd_hist < 0 and macd_hist.iloc[-2] > 0:
        score += 0.10
        details.append("MACD cross down")

    # ADX trend strength
    if last_adx > 25:
        score += 0.10
        details.append(f"ADX trend {last_adx:.1f}")

    # BB squeeze bounce
    if last_close <= last_bb_lower * 1.005:
        score += 0.15
        details.append("BB lower touch")
    elif last_close >= last_bb_upper * 0.995:
        score += 0.10
        details.append("BB upper touch")

    # Liquidity sweep
    if last_sweep == 1:
        score += 0.20
        details.append("Bullish sweep")
    elif last_sweep == -1:
        score += 0.15
        details.append("Bearish sweep")

    # Volume delta
    if last_delta > 0 and last_obv_slope > 0:
        score += 0.10
        details.append("Positive delta+OBV")
    elif last_delta < 0 and last_obv_slope < 0:
        score += 0.08
        details.append("Negative delta+OBV")

    # Z-score extremes
    if abs(zscore_val) > 2.0:
        score += 0.10
        details.append(f"Z-score {zscore_val:.2f}")

    # Squeeze
    squeeze_active, squeeze_bars = detect_squeeze(df, HG_SQUEEZE_MIN_BARS)
    if squeeze_active and squeeze_bars >= HG_SQUEEZE_MIN_BARS:
        score += 0.15
        details.append(f"Squeeze {squeeze_bars}b")

    # NR7
    if detect_nr7(df):
        score += 0.10
        details.append("NR7")

    # Hammer
    hammer = detect_hammer(df)
    if hammer:
        score += 0.10
        details.append(hammer)

    return min(score, 1.0), details

def is_in_forbidden_hours():
    hour = datetime.datetime.utcnow().hour
    return hour in ORARI_VIETATI_UTC

def is_signal_cooled_down(symbol, interval):
    key = f"{symbol}_{interval}"
    cooldown = SIGNAL_COOLDOWN_BY_TF.get(interval, SIGNAL_COOLDOWN)
    last = last_signal_time.get(key, 0)
    return (time.time() - last) >= cooldown

def mark_signal_sent(symbol, interval):
    key = f"{symbol}_{interval}"
    last_signal_time[key] = time.time()

# ============================================================
# KELLY CRITERION & RISK
# ============================================================

def kelly_fraction(win_rate=0.55, rr=2.0):
    if rr <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.01
    q = 1 - win_rate
    k = (win_rate * rr - q) / rr
    return max(0.005, min(k * 0.5, 0.05))  # Half-Kelly, capped at 5%

def calc_position_size(entry, sl, balance=ACCOUNT_BALANCE, win_rate=0.55):
    risk_per_unit = abs(entry - sl)
    if risk_per_unit < 1e-10:
        return 0
    rr = 2.0
    k = kelly_fraction(win_rate, rr)
    risk_amount = balance * k
    size = risk_amount / risk_per_unit
    return round(size, 4)

def calc_levels(df, direction="long"):
    atr = calc_atr(df).iloc[-1]
    close = df["close"].iloc[-1]
    if direction == "long":
        sl = close - 2.0 * atr
        tp1 = close + 2.0 * atr
        tp2 = close + 4.0 * atr
    else:
        sl = close + 2.0 * atr
        tp1 = close - 2.0 * atr
        tp2 = close - 4.0 * atr
    rr = abs(tp1 - close) / max(abs(close - sl), 1e-10)
    return sl, tp1, tp2, rr

# ============================================================
# AI INTEGRATION
# ============================================================

AI_CALL_COOLDOWN = 300  # seconds

def can_call_ai(symbol):
    last = last_ai_call_per_symbol.get(symbol, 0)
    return (time.time() - last) >= AI_CALL_COOLDOWN

def call_ai(prompt, url=None, model=None, timeout=None):
    u = url or AI_URL_ANALYST
    m = model or AI_MODEL_ANALYST
    t = timeout or AI_TIMEOUT
    try:
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 200,
        }
        resp = requests.post(u, json=payload, timeout=t)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"AI call failed: {e}")
        return "NEUTRAL"

def get_ai_verdict(symbol, interval, direction, score, details, df):
    if not AI_ENABLED or not can_call_ai(symbol):
        return "SKIP"
    try:
        close = df["close"].iloc[-1]
        rsi_val = calc_rsi(df["close"]).iloc[-1]
        adx_val = calc_adx(df)[0].iloc[-1]
        prompt = (
            f"You are a professional crypto futures trader. Analyze this signal:\n"
            f"Symbol: {symbol}, Timeframe: {interval}, Direction: {direction}\n"
            f"Price: {close:.4f}, Score: {score:.2f}, RSI: {rsi_val:.1f}, ADX: {adx_val:.1f}\n"
            f"Patterns: {', '.join(details[:5])}\n"
            f"Answer with only: APPROVE, REJECT, or NEUTRAL"
        )
        with ai_lock:
            verdict = call_ai(prompt)
            last_ai_call_per_symbol[symbol] = time.time()
        if "APPROVE" in verdict.upper():
            return "APPROVE"
        elif "REJECT" in verdict.upper():
            return "REJECT"
        return "NEUTRAL"
    except Exception as e:
        logger.debug(f"AI verdict error: {e}")
        return "NEUTRAL"

# ============================================================
# MULTI-TIMEFRAME CONFLUENCE ("MURO DI BERLINO")
# ============================================================

def check_mtf_confluence(symbol, primary_tf, direction):
    """Check alignment across multiple timeframes"""
    tf_order = {"15m": 0, "1h": 1, "4h": 2}
    primary_idx = tf_order.get(primary_tf, 1)
    aligned = 0
    total = 0
    for tf, idx in tf_order.items():
        if idx <= primary_idx:
            continue
        df = get_df(symbol, tf)
        if df is None or len(df) < 50:
            continue
        total += 1
        score, _ = calc_score(symbol, tf, df)
        rsi = calc_rsi(df["close"]).iloc[-1]
        if direction == "long" and (score > 0.4 or rsi < 50):
            aligned += 1
        elif direction == "short" and (score > 0.4 or rsi > 50):
            aligned += 1
    return aligned, total

# ============================================================
# SIGNAL PROCESSING
# ============================================================

def process_closed_candle(symbol, interval, kline):
    """Main signal processing on candle close"""
    try:
        if is_in_forbidden_hours():
            return

        df = get_df(symbol, interval)
        if df is None or len(df) < 50:
            return

        score, details = calc_score(symbol, interval, df)

        threshold = THRESHOLD_BASE
        if interval == "15m":
            threshold = 0.45
        elif interval == "1h":
            threshold = 0.40
        elif interval == "4h":
            threshold = 0.35

        if score < threshold:
            return

        if not is_signal_cooled_down(symbol, interval):
            return

        # Determine direction
        rsi = calc_rsi(df["close"]).iloc[-1]
        _, delta = calc_cvd(df)
        last_delta = delta.iloc[-1]

        direction = "long" if (rsi < 50 and last_delta > 0) else "short"

        # MTF confluence check
        aligned, total = check_mtf_confluence(symbol, interval, direction)
        if total > 0 and aligned == 0:
            return  # No higher TF alignment

        # RSI divergence check
        div_type, div_age = detect_rsi_divergence(df)
        max_age = DIVERGENCE_MAX_AGE_BY_TF.get(interval, DIVERGENCE_MAX_AGE_CANDLES)

        # AI verdict
        ai_verdict = "SKIP"
        if AI_SYNC_ON_SIGNAL:
            ai_verdict = get_ai_verdict(symbol, interval, direction, score, details, df)
            if ai_verdict == "REJECT":
                logger.info(f"🤖 AI REJECTED {symbol} {interval}")
                return

        # Calculate levels
        sl, tp1, tp2, rr = calc_levels(df, direction)
        entry = df["close"].iloc[-1]
        kelly = kelly_fraction(0.55, rr)
        size = calc_position_size(entry, sl)

        # Build message
        dir_emoji = "📈" if direction == "long" else "📉"
        hour = datetime.datetime.utcnow().hour
        time_quality = "🟢" if hour in ORARI_MIGLIORI_UTC else "🟡"

        msg = (
            f"{dir_emoji} *{symbol}* [{interval}] — *{direction.upper()}*\n\n"
            f"💰 Entry: `{entry:.4f}`\n"
            f"🛑 SL: `{sl:.4f}`\n"
            f"🎯 TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
            f"⚖️ R/R: `{rr:.2f}x` | Kelly: `{kelly*100:.1f}%`\n\n"
            f"📊 Score: `{score:.2f}` | ADX: `{calc_adx(df)[0].iloc[-1]:.1f}`\n"
            f"📐 RSI: `{rsi:.1f}` | Delta: `{'↑' if last_delta > 0 else '↓'}`\n"
        )

        if div_type and div_age is not None and div_age <= max_age:
            msg += f"🔀 Divergenza RSI {div_type} ({div_age}c)\n"

        msg += f"\n🧩 Pattern: {', '.join(details[:5])}\n"
        msg += f"{time_quality} Orario UTC: {hour:02d}:xx\n"

        if AI_SECTION_IN_MSG and ai_verdict != "SKIP":
            msg += f"\n🤖 AI: {ai_verdict}"

        with signal_lock:
            mark_signal_sent(symbol, interval)
            send_telegram(msg)
            db_log_signal(symbol, interval, direction, ",".join(details[:5]),
                          score, entry, sl, tp1, tp2, rr, kelly, ai_verdict, msg)

        logger.info(f"✅ SIGNAL {symbol} {interval} {direction.upper()} score={score:.2f}")

    except Exception as e:
        logger.error(f"process_closed_candle {symbol} {interval}: {e}")
        logger.debug(traceback.format_exc())

# ============================================================
# HIDDEN GEMS (HG) SCANNER
# ============================================================

def scan_hg_symbol(symbol, tf):
    """Scan a symbol for Hidden Gem setups"""
    try:
        cfg = HG_CFG.get(tf, {})
        min_score = cfg.get("min_score", 0.65)
        cooldown = cfg.get("cooldown", HG_COOLDOWN)

        last = last_hg_signal_time.get(f"{symbol}_{tf}", 0)
        if (time.time() - last) < cooldown:
            return

        df = get_df(symbol, tf)
        if df is None or len(df) < 50:
            return

        # Check quote volume
        if "quote_volume" in df.columns:
            avg_qvol = df["quote_volume"].iloc[-HG_QVOL_LOOKBACK:].mean()
            if avg_qvol < HG_MIN_QUOTE_VOL:
                return

        score, details = calc_score(symbol, tf, df)
        if score < min_score:
            return

        # RVOL check
        avg_vol = df["volume"].iloc[-20:-1].mean()
        last_vol = df["volume"].iloc[-1]
        rvol = last_vol / avg_vol if avg_vol > 0 else 0
        rvol_min = cfg.get("rvol_bar_min", HG_RVOL_BAR_MIN)
        if rvol < rvol_min:
            return

        last_hg_signal_time[f"{symbol}_{tf}"] = time.time()
        rsi = calc_rsi(df["close"]).iloc[-1]
        close = df["close"].iloc[-1]

        msg = (
            f"💎 *HG: {symbol}* [{tf}]\n\n"
            f"📊 Score: `{score:.2f}` | RVOL: `{rvol:.1f}x`\n"
            f"📐 RSI: `{rsi:.1f}` | Price: `{close:.4f}`\n"
            f"🧩 {', '.join(details[:4])}\n"
        )
        send_telegram(msg)
        logger.info(f"💎 HG {symbol} {tf} score={score:.2f} rvol={rvol:.1f}x")

    except Exception as e:
        logger.debug(f"scan_hg {symbol} {tf}: {e}")

def scan_hg_all():
    """Scan all HG symbols periodically"""
    while True:
        try:
            for tf in HG_TF:
                syms = list(symbols_hg_all) if HG_MONITOR_ALL else list(symbols_whitelist)
                for sym in syms:
                    if HG_ENABLED:
                        scan_hg_symbol(sym, tf)
                    time.sleep(0.02)
        except Exception as e:
            logger.error(f"scan_hg_all: {e}")
        time.sleep(60)

# ============================================================
# DIVERGENCE SCAN ON STARTUP
# ============================================================

def scan_divergences_on_startup():
    logger.info("🔍 Scansione divergenze all'avvio...")
    found = 0
    for symbol in symbols_whitelist[:30]:
        for interval in ["1h", "4h"]:
            df = get_df(symbol, interval)
            if df is None or len(df) < 50:
                continue
            div_type, div_age = detect_rsi_divergence(df)
            max_age = DIVERGENCE_MAX_AGE_BY_TF.get(interval, DIVERGENCE_MAX_AGE_CANDLES)
            if div_type and div_age is not None and div_age <= max_age:
                found += 1
                logger.info(f"  📐 Divergenza {div_type} su {symbol} {interval} ({div_age}c)")
    logger.info(f"✅ Scansione avvio completata: {found} divergenze trovate")

# ============================================================
# WEBSOCKET MANAGEMENT
# ============================================================

WS_RECONNECT_DELAY_BASE = 5
WS_MAX_RECONNECT_DELAY = 60

def build_stream_url_v4(symbols_group, tf):
    streams = [f"{s.lower()}@kline_{tf}" for s in symbols_group]
    return "wss://fstream.binance.com/stream?streams=" + "/".join(streams)

def split_symbols_v4(tf, group_size=10, symbols=None):
    use_list = symbols if symbols is not None else symbols_whitelist
    groups = []
    current = []
    for sym in use_list:
        current.append(sym)
        if len(current) >= group_size:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups

def on_ws_message(ws_name, message):
    try:
        data = json.loads(message)
        stream_data = data.get("data", data)
        kline = stream_data.get("k")
        if not kline:
            return
        symbol = kline["s"]
        interval = kline["i"]
        is_closed = kline.get("x", False)
        key = f"{symbol}_{interval}"

        WS_HEALTH[ws_name] = {"alive": True, "last_msg": time.time()}
        update_realtime(symbol, interval, kline)

        if is_closed:
            open_time = int(kline["t"])
            if LAST_KLINE_TIME.get(key) != open_time:
                LAST_KLINE_TIME[key] = open_time
                if interval in ("15m", "1h", "4h"):
                    threading.Thread(
                        target=process_closed_candle,
                        args=(symbol, interval, kline),
                        daemon=True
                    ).start()
    except Exception as e:
        logger.debug(f"WS msg error {ws_name}: {e}")

def create_websocket_thread(ws_name, url, retry_count=0):
    def run():
        delay = min(WS_RECONNECT_DELAY_BASE * (2 ** retry_count), WS_MAX_RECONNECT_DELAY)
        retries = 0
        while True:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=lambda ws, msg: on_ws_message(ws_name, msg),
                    on_error=lambda ws, err: logger.warning(f"WS {ws_name} error: {err}"),
                    on_close=lambda ws, c, m: logger.info(f"WS {ws_name} closed"),
                    on_open=lambda ws: logger.info(f"WS {ws_name} connected"),
                )
                WS_HEALTH[ws_name] = {"alive": True, "last_msg": time.time()}
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"WS {ws_name} exception: {e}")
            WS_HEALTH[ws_name] = {"alive": False, "last_msg": time.time()}
            retries += 1
            wait = min(WS_RECONNECT_DELAY_BASE * (2 ** retries), WS_MAX_RECONNECT_DELAY)
            logger.info(f"WS {ws_name} reconnecting in {wait}s...")
            time.sleep(wait)
    return run

def start_multi_websocket_v4():
    logger.info("🌐 Avvio WebSocket multi-stream...")
    threads_created = 0
    for tf in ["15m", "1h", "4h"]:
        groups = split_symbols_v4(tf, group_size=10)
        for i, group in enumerate(groups):
            ws_name = f"WS_{tf}_{i}"
            url = build_stream_url_v4(group, tf)
            runner = create_websocket_thread(ws_name, url)
            t = threading.Thread(target=runner, daemon=True, name=ws_name)
            t.start()
            threads_created += 1
            time.sleep(0.1)
    logger.info(f"✅ {threads_created} WebSocket thread avviati")

# ============================================================
# FALLBACK REST — CHIUSURA CANDELE
# ============================================================

POLL_CLOSED_ENABLE = True
POLL_CLOSED_INTERVAL = 60

def _get_tf_seconds(interval_str):
    if interval_str == "15m":
        return 900
    if interval_str == "1h":
        return 3600
    if interval_str == "4h":
        return 14400
    return 3600

def _poll_closed_candles(interval: str, symbols_list):
    try:
        tf_seconds = _get_tf_seconds(interval)
        current_time = time.time()
        for symbol in symbols_list:
            try:
                df = historical_data.get(symbol, {}).get(interval)
                if df is not None and not df.empty:
                    last_candle_time = df.index[-1].timestamp()
                    if (current_time - last_candle_time) < (tf_seconds + 60):
                        continue
                kl = client.futures_klines(symbol=symbol, interval=interval, limit=2)
                if not kl or len(kl) < 2:
                    continue
                last = kl[-1]
                close_time = int(last[6])
                if close_time > int(current_time * 1000):
                    continue
                open_time = int(last[0])
                key = f"{symbol}_{interval}"
                if LAST_KLINE_TIME.get(key) == open_time:
                    continue
                k = {
                    "t": open_time, "o": last[1], "h": last[2], "l": last[3],
                    "c": last[4], "v": last[5], "V": last[10], "x": True,
                    "s": symbol, "i": interval,
                }
                LAST_KLINE_TIME[key] = open_time
                update_realtime(symbol, interval, k)
                if interval in ("1h", "15m", "4h"):
                    process_closed_candle(symbol, interval, k)
                time.sleep(0.1)
            except Exception as e:
                logger.debug(f"[POLL-{interval}] {symbol} Fallito: {e}")
    except Exception as e:
        logger.error(f"[POLL-{interval}] Errore critico nel loop: {e}")

def poll_closed_candles_all():
    time.sleep(30)
    while True:
        try:
            _poll_closed_candles("15m", symbols_whitelist)
            _poll_closed_candles("1h", symbols_whitelist)
            _poll_closed_candles("4h", symbols_whitelist)
        except Exception as e:
            logger.error(f"[POLL-MASTER] Errore: {e}")
        time.sleep(POLL_CLOSED_INTERVAL)

# ============================================================
# STARTUP HEALTH CHECK
# ============================================================

STARTUP_TIMEOUT = 25

def startup_health_check(timeout=STARTUP_TIMEOUT):
    start = time.time()
    logger.info("🔍 Controllo salute iniziale WebSocket...")
    while time.time() - start < timeout:
        if not WS_HEALTH:
            time.sleep(1)
            continue
        now = time.time()
        all_ok = True
        for name, data in WS_HEALTH.items():
            last = data.get("last_msg", 0)
            alive = data.get("alive", False)
            if (not alive) or (now - last > 10):
                all_ok = False
        if all_ok:
            logger.info("✅ Tutti i WS attivi")
            return True
        time.sleep(1)
    logger.warning("⚠️ Alcuni WS non rispondono")
    return False

# ============================================================
# GESTIONE SIMBOLI & UNIVERSO
# ============================================================

def load_top_symbols(limit=120):
    global symbols_whitelist
    try:
        info = client.futures_exchange_info()
        tickers = client.futures_ticker()
        qvol_map = {}
        for t in tickers:
            sym = t.get("symbol")
            if sym:
                qvol_map[sym] = float(t.get("quoteVolume", 0.0))
        valid = []
        for s in info["symbols"]:
            if (s.get("contractType") == "PERPETUAL" and
                    s.get("quoteAsset") == "USDT" and
                    s.get("status") == "TRADING"):
                sym = s.get("symbol")
                if sym and qvol_map.get(sym, 0.0) > 0:
                    valid.append(sym)
        valid.sort(key=lambda sym: qvol_map.get(sym, 0.0), reverse=True)
        symbols_whitelist = valid[:limit]
        logger.info(f"✅ Divergenze: caricati {len(symbols_whitelist)} simboli top USDT-M")
    except Exception as e:
        logger.error(f"Errore load_top_symbols: {e}")
        symbols_whitelist = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def load_all_futures_symbols():
    global symbols_hg_all
    try:
        info = client.futures_exchange_info()
        valid = [
            s["symbol"] for s in info["symbols"]
            if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        ]
        symbols_hg_all = sorted(valid)
        logger.info(f"💎 HG: Caricati {len(symbols_hg_all)} simboli")
    except Exception:
        symbols_hg_all = symbols_whitelist[:]

def filter_hg_symbols_by_liquidity(min_quote_usdt=70000):
    global symbols_hg_all
    try:
        tickers = client.futures_ticker()
        qvol = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in tickers if "symbol" in t}
        symbols_hg_all = [s for s in symbols_hg_all if qvol.get(s, 0) >= min_quote_usdt]
        logger.info(f"💎 HG: Filtro liquidità > {min_quote_usdt} -> {len(symbols_hg_all)} simboli rimasti")
    except Exception:
        pass

def get_symbols_for_tf(tf):
    if tf == "1h" and HG_ENABLED and HG_MONITOR_ALL:
        return sorted(set(symbols_whitelist + symbols_hg_all))
    return symbols_whitelist

# ============================================================
# MAIN
# ============================================================

TELEGRAM_TEST_ON_START = False

def load_universes():
    load_top_symbols(120)
    load_all_futures_symbols()
    filter_hg_symbols_by_liquidity(HG_MIN_QUOTE_VOL)

def main():
    logger.info("=" * 60)
    logger.info("🎯 BOT V16 CECCHINO ISTITUZIONALE — SMART MONEY EDITION")
    logger.info("=" * 60)
    logger.info("🛡️ MODULI ATTIVI:")
    logger.info("   - Controllo Multi-Timeframe (Muro di Berlino): ON")
    logger.info("   - Analisi Volume Delta (Mani Forti): ON")
    logger.info("   - Rilevamento Caccia Agli Stop (Sweep): ON")
    logger.info("   - AI Paranoica (Veto Estremo): ON")
    logger.info("=" * 60)

    if TELEGRAM_TEST_ON_START:
        test_telegram()

    try:
        gc.set_threshold(700, 10, 10)

        init_database()
        db_load_last_signal_time()
        db_load_last_ai_call_per_symbol()
        db_load_ws_health()
        db_load_last_message_time()

        global client
        client = Client(API_KEY, API_SECRET)
        client.API_URL = "https://fapi.binance.com"

        load_universes()

        if not symbols_whitelist:
            raise ValueError("❌ Nessun simbolo caricato per divergenze!")
        if not symbols_hg_all:
            logger.warning("⚠️ Nessun universo Hidden Gems — uso fallback top list")

        logger.info(f"📥 Caricamento storico per {len(symbols_whitelist)} simboli (divergenze)...")
        for idx, sym in enumerate(symbols_whitelist, 1):
            init_historical_for_symbol(sym)
            if idx % 20 == 0 or idx == len(symbols_whitelist):
                logger.info(f"📊 Progresso divergenze: {idx}/{len(symbols_whitelist)} completati")

        logger.info(f"📥 Caricamento storico per {len(symbols_hg_all)} simboli (Hidden Gems)...")
        for idx, sym in enumerate(symbols_hg_all, 1):
            init_historical_for_symbol(sym)
            if idx % 20 == 0 or idx == len(symbols_hg_all):
                logger.info(f"📊 Progresso HG: {idx}/{len(symbols_hg_all)} completati")

        scan_divergences_on_startup()
        start_multi_websocket_v4()

        if HG_ENABLED:
            threading.Thread(target=scan_hg_all, daemon=True, name="HG-SCANNER").start()

        if POLL_CLOSED_ENABLE:
            logger.info("[POLL-CLOSED] Avvio polling REST candele chiuse...")
            threading.Thread(target=poll_closed_candles_all, daemon=True, name="POLL-CLOSED").start()

        ok = startup_health_check(timeout=STARTUP_TIMEOUT)
        if ok:
            logger.info("✅ Tutti i WS attivi e ricevono dati")
        else:
            logger.warning("⚠️ Alcuni WS non rispondono")

        send_telegram(
            "🎯 *Bot V16 Cecchino Istituzionale Avviato*\n\n"
            f"✅ Divergenze: {len(symbols_whitelist)} simboli\n"
            f"💎 Perle (HG): {len(symbols_hg_all)} simboli\n"
            "🧠 Moduli Quant: Allineamento MTF & Volume Delta ON\n"
            "⏰ In attesa del setup perfetto..."
        )

        logger.info("=" * 60)
        logger.info("🚀 BOT OPERATIVO E IN ASCOLTO — Premi Ctrl+C per terminare")
        logger.info("=" * 60)

        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("⏹️ INTERRUZIONE MANUALE (Ctrl+C)")
        logger.info("=" * 60)
        try:
            db_save_last_signal_time()
            db_save_last_ai_call_per_symbol()
            db_save_ws_health()
            db_save_last_message_time()
            logger.info("💾 Stato salvato su database")
        except Exception as e:
            logger.error(f"Errore salvataggio finale: {e}")

        send_telegram("⏹️ Bot V16 terminato")
        logger.info("👋 Bot terminato correttamente")
        sys.exit(0)

    except Exception as e:
        logger.critical("=" * 60)
        logger.critical(f"❌ ERRORE FATALE: {e}")
        logger.critical("=" * 60)
        traceback.print_exc()
        try:
            send_telegram(f"🔴 ERRORE FATALE\n\n{str(e)[:200]}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
