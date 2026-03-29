"""
Shared fixtures for V17 agent unit tests.
"""
import time
import numpy as np
import pandas as pd
import pytest

from agents.base_agent import AgentResult


# ---------------------------------------------------------------------------
# DataFrame factory
# ---------------------------------------------------------------------------

def make_ohlcv_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a realistic synthetic OHLCV + indicator DataFrame."""
    rng = np.random.default_rng(seed)
    close = 30_000.0 + np.cumsum(rng.normal(0, 150, n))
    close = np.clip(close, 1.0, None)
    spread = rng.uniform(0.005, 0.015, n) * close
    high = close + spread * rng.uniform(0.3, 1.0, n)
    low = close - spread * rng.uniform(0.3, 1.0, n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.uniform(50, 500, n) * 1_000

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )

    # Common indicators (precomputed approximations)
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi"] = df["rsi"].fillna(50)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # EMAs / SMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """200-row OHLCV DataFrame with common indicator columns."""
    return make_ohlcv_df(200)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Intentionally tiny DataFrame to test edge cases."""
    return make_ohlcv_df(10)


@pytest.fixture
def bullish_df() -> pd.DataFrame:
    """Strongly uptrending DataFrame."""
    rng = np.random.default_rng(1)
    n = 200
    close = 20_000.0 + np.cumsum(np.abs(rng.normal(200, 50, n)))
    spread = 0.005 * close
    high = close + spread
    low = close - spread * 0.3
    open_ = close - spread * 0.1
    volume = rng.uniform(100, 600, n) * 1_000
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(70)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    return df.reset_index(drop=True)


@pytest.fixture
def bearish_df() -> pd.DataFrame:
    """Strongly downtrending DataFrame."""
    rng = np.random.default_rng(2)
    n = 200
    close = 50_000.0 - np.cumsum(np.abs(rng.normal(200, 50, n)))
    close = np.clip(close, 1.0, None)
    spread = 0.005 * close
    high = close + spread * 0.3
    low = close - spread
    open_ = close + spread * 0.1
    volume = rng.uniform(100, 600, n) * 1_000
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(30)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    return df.reset_index(drop=True)


def make_agent_result(
    agent_name: str = "test_agent",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    score: float = 0.6,
    direction: str = "long",
    confidence: float = 0.7,
) -> AgentResult:
    """Factory for AgentResult objects."""
    return AgentResult(
        agent_name=agent_name,
        symbol=symbol,
        interval=interval,
        score=score,
        direction=direction,
        confidence=confidence,
        details=["test"],
        metadata={},
    )


@pytest.fixture
def agent_result_factory():
    return make_agent_result


@pytest.fixture
def mock_agent_results():
    """Dict of realistic agent results for fusion tests."""
    return {
        "pattern": make_agent_result("pattern", score=0.65, direction="long", confidence=0.7),
        "regime": make_agent_result("regime", score=0.60, direction="long", confidence=0.6),
        "confluence": make_agent_result("confluence", score=0.70, direction="long", confidence=0.8),
        "risk": make_agent_result("risk", score=0.55, direction="long", confidence=0.5),
        "strategy": make_agent_result("strategy", score=0.62, direction="long", confidence=0.65),
    }
