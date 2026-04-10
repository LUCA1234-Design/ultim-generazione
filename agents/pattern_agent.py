"""
Pattern Agent for V17.
Preserves ALL V16 detectors with auto-calibrating thresholds:
- Squeeze (immediate & validated)
- NR7 (Narrowest Range 7)
- RS Leader (Relative Strength vs BTC)
- Hammer / Shooting Star
- RSI Divergence (bullish & bearish)
- Breakout detector
"""
import logging
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Optional, Dict, Tuple, Any

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import (
    rsi, atr, bollinger_bands, keltner_channels, adx,
    squeeze_intensity, volume_ratio,
)
from config.settings import (
    HG_SQUEEZE_MIN_BARS, HG_RS_SLOPE_MIN, HG_LOOKBACK_RS,
    DIVERGENCE_MAX_AGE_BY_TF, BREAKOUT_RULES,
)

logger = logging.getLogger("PatternAgent")


class PatternAgent(BaseAgent):
    """Detects all V16 patterns with auto-calibrating thresholds."""

    # ----------------------------------------------------------------
    # Calibration state (per symbol/interval)
    # ----------------------------------------------------------------
    _THRESHOLD_DEFAULTS = {
        "15m": 0.45,
        "1h":  0.40,
        "4h":  0.35,
    }

    def __init__(self):
        super().__init__("pattern", initial_weight=0.30)
        self._thresholds: Dict[str, float] = {}  # key -> adaptive threshold
        self._hit_history: Dict[str, list] = {}  # key -> list of bool (hit/miss)
        # Per-pattern win rate tracking
        self._pattern_wins: Dict[str, int] = {}
        self._pattern_total: Dict[str, int] = {}

    def _get_threshold(self, interval: str) -> float:
        default = self._THRESHOLD_DEFAULTS.get(interval, 0.40)
        return self._thresholds.get(interval, default)

    def update_threshold(self, interval: str, was_correct: bool) -> None:
        """Auto-calibrate threshold based on outcomes."""
        key = interval
        history = self._hit_history.setdefault(key, [])
        history.append(was_correct)
        if len(history) > 50:
            history.pop(0)
        if len(history) >= 20:
            hit_rate = sum(history) / len(history)
            current = self._thresholds.get(key, self._THRESHOLD_DEFAULTS.get(interval, 0.40))
            if hit_rate < 0.40:
                # Too many false positives → raise threshold
                self._thresholds[key] = min(current + 0.02, 0.85)
            elif hit_rate > 0.65:
                # Good performance → lower threshold slightly
                self._thresholds[key] = max(current - 0.01, 0.20)

    def record_pattern_outcome(self, patterns: list, was_correct: bool) -> None:
        """Record outcome for each pattern that contributed to the signal."""
        for pattern in patterns:
            self._pattern_total[pattern] = self._pattern_total.get(pattern, 0) + 1
            if was_correct:
                self._pattern_wins[pattern] = self._pattern_wins.get(pattern, 0) + 1

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Return win rate stats per pattern.

        The ``wr`` field is ``0.0`` when no trades have been recorded for a pattern.
        """
        stats = {}
        for pattern, total in self._pattern_total.items():
            wins = self._pattern_wins.get(pattern, 0)
            stats[pattern] = {
                "wins": wins,
                "total": total,
                "wr": wins / total if total > 0 else 0.0,
            }
        return stats

    # ----------------------------------------------------------------
    # Pattern Detectors
    # ----------------------------------------------------------------

    def detect_squeeze(self, df: pd.DataFrame, min_bars: int = HG_SQUEEZE_MIN_BARS) -> Tuple[bool, int]:
        """Bollinger inside Keltner = squeeze.  Returns (active, n_squeeze_bars)."""
        if len(df) < 30:
            return False, 0
        sq = squeeze_intensity(df, 20, 2.0, 20, 1.5)
        n_squeeze = int(sq.iloc[-min_bars:].sum())
        active = bool(sq.iloc[-1])
        return active, n_squeeze

    def detect_squeeze_validated(self, df: pd.DataFrame, min_bars: int = HG_SQUEEZE_MIN_BARS) -> Tuple[bool, int]:
        """Validated squeeze: squeeze recently ended (momentum breakout setup)."""
        if len(df) < 30:
            return False, 0
        sq = squeeze_intensity(df, 20, 2.0, 20, 1.5)
        # Squeeze ended on last bar
        if sq.iloc[-1] == 1:
            return False, 0
        n_squeeze = int(sq.iloc[-min_bars - 1:-1].sum())
        was_squeezing = n_squeeze >= min_bars
        return was_squeezing, n_squeeze

    def detect_nr7(self, df: pd.DataFrame) -> bool:
        """Narrowest Range of last 7 bars."""
        if len(df) < 7:
            return False
        ranges = (df["high"] - df["low"]).iloc[-7:]
        return float(ranges.iloc[-1]) == float(ranges.min())

    def detect_rs_leader(self, df_sym: pd.DataFrame, df_btc: pd.DataFrame,
                         lookback: int = HG_LOOKBACK_RS,
                         slope_min: float = HG_RS_SLOPE_MIN) -> Tuple[bool, float]:
        """Relative strength vs BTC.  Returns (is_leader, slope)."""
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
            slope = float(np.polyfit(x, rs.values, 1)[0])
            return slope > slope_min, slope
        except Exception:
            return False, 0.0

    def detect_rsi_divergence(self, df: pd.DataFrame,
                               rsi_period: int = 14,
                               lookback: int = 30) -> Tuple[Optional[str], Optional[int]]:
        """RSI divergence detection.  Returns (div_type, age_candles) or (None, None)."""
        if len(df) < lookback + rsi_period:
            return None, None
        try:
            rsi_series = rsi(df["close"], rsi_period)
            price_lows = argrelextrema(df["low"].values, np.less, order=3)[0]
            price_highs = argrelextrema(df["high"].values, np.greater, order=3)[0]
            rsi_lows = argrelextrema(rsi_series.values, np.less, order=3)[0]
            rsi_highs = argrelextrema(rsi_series.values, np.greater, order=3)[0]

            # Bullish divergence
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                r1, r2 = rsi_lows[-2], rsi_lows[-1]
                if (df["low"].iloc[p2] < df["low"].iloc[p1] and
                        rsi_series.iloc[r2] > rsi_series.iloc[r1] and
                        abs(p2 - r2) <= 3):
                    return "bullish", len(df) - 1 - p2

            # Bearish divergence
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                r1, r2 = rsi_highs[-2], rsi_highs[-1]
                if (df["high"].iloc[p2] > df["high"].iloc[p1] and
                        rsi_series.iloc[r2] < rsi_series.iloc[r1] and
                        abs(p2 - r2) <= 3):
                    return "bearish", len(df) - 1 - p2

            return None, None
        except Exception:
            return None, None

    def detect_hammer(self, df: pd.DataFrame) -> Optional[str]:
        """Hammer or Shooting Star on the last candle."""
        if len(df) < 3:
            return None
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]
        total_range = c["high"] - c["low"]
        if total_range < 1e-10:
            return None
        body_ratio = body / total_range
        if lower_wick > 2 * body and upper_wick < body and body_ratio < 0.4:
            return "hammer_bullish"
        if upper_wick > 2 * body and lower_wick < body and body_ratio < 0.4:
            return "shooting_star_bearish"
        return None

    def detect_breakout(self, df: pd.DataFrame, interval: str) -> Optional[str]:
        """Volume-confirmed breakout above/below recent range."""
        rules = BREAKOUT_RULES.get(interval)
        if rules is None or len(df) < 20:
            return None
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            vol = df["volume"]
            avg_vol = vol.iloc[-21:-1].mean()
            last_vol = vol.iloc[-1]
            rvol = last_vol / avg_vol if avg_vol > 0 else 0
            if rvol < rules["vol_min"]:
                return None
            recent_high = high.iloc[-21:-1].max()
            recent_low = low.iloc[-21:-1].min()
            c = close.iloc[-1]
            if c > recent_high * rules["break_mult"]:
                return "breakout_long"
            if c < recent_low * (2 - rules["break_mult"]):
                return "breakout_short"
            return None
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Market Structure Detection
    # ----------------------------------------------------------------

    def detect_market_structure(self, df: pd.DataFrame, lookback: int = 20) -> str:
        """Detect Higher High/Higher Low (uptrend) or Lower High/Lower Low (downtrend).

        Returns: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(df) < lookback + 5:
            return "sideways"
        try:
            highs = df["high"].iloc[-lookback:].values
            lows = df["low"].iloc[-lookback:].values
            # Divide in 3 segmenti e confronta
            seg = lookback // 3
            h1 = highs[:seg].max()
            h2 = highs[seg:2*seg].max()
            h3 = highs[2*seg:].max()
            l1 = lows[:seg].min()
            l2 = lows[seg:2*seg].min()
            l3 = lows[2*seg:].min()

            hh = h3 > h2 > h1  # Higher Highs
            hl = l3 > l2 > l1  # Higher Lows
            lh = h3 < h2 < h1  # Lower Highs
            ll = l3 < l2 < l1  # Lower Lows

            if hh and hl:
                return "uptrend"
            elif lh and ll:
                return "downtrend"
            elif hh or hl:
                return "weak_uptrend"
            elif lh or ll:
                return "weak_downtrend"
            else:
                return "sideways"
        except Exception:
            return "sideways"

    # ----------------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------------

    def _score_patterns(self, symbol: str, interval: str, df: pd.DataFrame,
                        df_btc: Optional[pd.DataFrame] = None) -> Tuple[float, str, list]:
        """Return (score, direction, details)."""
        score = 0.0
        details = []

        rsi_val = rsi(df["close"], 14).iloc[-1]
        adx_val, di_p, di_m = adx(df, 14)
        last_adx = adx_val.iloc[-1]
        last_di_p = di_p.iloc[-1]
        last_di_m = di_m.iloc[-1]
        vol_r = volume_ratio(df, 20).iloc[-1]

        # ---- BASE SCORE: general market conditions (ensures non-zero score) ----
        di_spread = abs(float(last_di_p) - float(last_di_m))
        if di_spread > 5:
            score += 0.10
            details.append(f"DI_spread({di_spread:.1f})")

        if float(vol_r) >= 1.0:
            score += 0.05
            details.append(f"vol_ok({float(vol_r):.1f}x)")

        if float(rsi_val) < 40 or float(rsi_val) > 60:
            score += 0.05
            details.append(f"RSI_active({float(rsi_val):.0f})")

        # Squeeze (immediate)
        sq_active, sq_bars = self.detect_squeeze(df)
        if sq_active and sq_bars >= HG_SQUEEZE_MIN_BARS:
            score += 0.20
            details.append(f"squeeze_active({sq_bars}b)")

        # Squeeze (validated — breakout after squeeze)
        sq_val, sq_val_bars = self.detect_squeeze_validated(df)
        if sq_val:
            score += 0.25
            details.append(f"squeeze_breakout({sq_val_bars}b)")

        # NR7
        if self.detect_nr7(df):
            score += 0.10
            details.append("NR7")
            # NR7 needs volume confirmation
            if vol_r >= 1.2:
                score += 0.05
                details.append(f"NR7+vol({vol_r:.1f}x)")

        # RS Leader
        if df_btc is not None:
            is_leader, rs_slope = self.detect_rs_leader(df, df_btc)
            if is_leader:
                score += 0.15
                details.append(f"RS_leader({rs_slope:.4f})")

        # RSI Divergence
        div_type, div_age = self.detect_rsi_divergence(df)
        max_age = DIVERGENCE_MAX_AGE_BY_TF.get(interval, 3)
        if div_type and div_age is not None and div_age <= max_age:
            score += 0.20
            details.append(f"rsi_div_{div_type}({div_age}c)")

        # Hammer / Shooting Star
        hammer = self.detect_hammer(df)
        if hammer:
            score += 0.10
            details.append(hammer)

        # Breakout
        bo = self.detect_breakout(df, interval)
        if bo:
            score += 0.15
            details.append(bo)

        # ADX trend strength bonus
        if last_adx > 25:
            score += 0.10
            details.append(f"ADX({last_adx:.1f})")

        # === VOTO PESATO DIREZIONE CECCHINO ===
        long_votes = 0.0
        short_votes = 0.0

        # RSI (peso 1.0)
        if rsi_val < 45:
            long_votes += 1.0
        elif rsi_val > 55:
            short_votes += 1.0

        # DI+ vs DI- (peso 1.5 — più affidabile)
        di_spread_val = float(last_di_p) - float(last_di_m)
        if di_spread_val > 3:
            long_votes += 1.5
        elif di_spread_val < -3:
            short_votes += 1.5

        # Divergenza RSI (peso 2.5 — segnale molto forte)
        if div_type == "bullish":
            long_votes += 2.5
        elif div_type == "bearish":
            short_votes += 2.5

        # Breakout confermato (peso 2.0)
        if bo == "breakout_long":
            long_votes += 2.0
        elif bo == "breakout_short":
            short_votes += 2.0

        # Hammer / Shooting Star (peso 1.5)
        if hammer == "hammer_bullish":
            long_votes += 1.5
        elif hammer == "shooting_star_bearish":
            short_votes += 1.5

        # Squeeze breakout: usa la candela corrente per determinare direzione
        if sq_val:
            last_close_val = float(df["close"].iloc[-1])
            prev_close_val = float(df["close"].iloc[-2]) if len(df) > 2 else last_close_val
            if last_close_val > prev_close_val:
                long_votes += 1.0
            else:
                short_votes += 1.0

        # Volume delta (CVD) — importare da smart_money se disponibile
        try:
            from indicators.smart_money import cumulative_volume_delta
            _, delta_series = cumulative_volume_delta(df)
            recent_delta = float(delta_series.iloc[-3:].sum())
            if recent_delta > 0:
                long_votes += 0.8
            elif recent_delta < 0:
                short_votes += 0.8
        except Exception:
            pass

        # EMA slope (peso 1.0)
        try:
            from indicators.technical import ema_slope as _ema_slope
            slope = float(_ema_slope(df["close"], 20, 5).iloc[-1])
            if slope > 0:
                long_votes += 1.0
            elif slope < 0:
                short_votes += 1.0
        except Exception:
            pass

        # Decisione finale
        if long_votes == short_votes:
            direction = "long" if rsi_val <= 50 else "short"
        elif long_votes > short_votes:
            direction = "long"
        else:
            direction = "short"

        details.append(f"dir_votes(L={long_votes:.1f}/S={short_votes:.1f})")

        # === FILTRO STRUTTURA MERCATO ===
        market_structure = self.detect_market_structure(df)
        details.append(f"structure={market_structure}")

        # Bonus se struttura allineata con direzione
        if direction == "long" and market_structure in ("uptrend", "weak_uptrend"):
            score += 0.10
            details.append("structure_aligned_long(+0.10)")
        elif direction == "short" and market_structure in ("downtrend", "weak_downtrend"):
            score += 0.10
            details.append("structure_aligned_short(+0.10)")
        # Penalità se struttura opposta alla direzione
        elif direction == "long" and market_structure == "downtrend":
            score *= 0.70
            details.append("structure_OPPOSED_long(x0.70)")
        elif direction == "short" and market_structure == "uptrend":
            score *= 0.70
            details.append("structure_OPPOSED_short(x0.70)")

        return float(np.clip(score, 0.0, 1.0)), direction, details

    # ----------------------------------------------------------------
    # BaseAgent interface
    # ----------------------------------------------------------------

    def analyse(self, symbol: str, interval: str, df, df_btc=None) -> Optional[AgentResult]:
        if df is None or len(df) < 50:
            return None

        score, direction, details = self._score_patterns(symbol, interval, df, df_btc)
        threshold = self._get_threshold(interval)

        rsi_val = rsi(df["close"], 14).iloc[-1]
        adx_val, _, _ = adx(df, 14)
        last_adx = float(adx_val.iloc[-1])

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=score,
            direction=direction,
            confidence=float(np.clip(score / max(threshold, 0.01), 0.0, 1.0)),
            details=details,
            metadata={
                "threshold": threshold,
                "rsi": float(rsi_val),
                "adx": last_adx,
            },
        )
