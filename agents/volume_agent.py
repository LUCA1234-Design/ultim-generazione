"""
Volume Agent for V19 — Analisi volumetrica avanzata.
Segnali: Delta Normalizzato, Absorption, Fair Value Gaps, BOS/CHoCH,
         Volume Imbalance Zones, VWAP position, Cumulative Taker Delta.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent, AgentResult
from indicators.technical import vwap, vwap_bands
from indicators.smart_money import (
    normalized_delta, delta_momentum, detect_absorption,
    nearest_fvg, detect_bos_choch, nearest_imbalance_zone,
    cumulative_taker_delta,
)

logger = logging.getLogger("VolumeAgent")

# Pesi segnale per il calcolo dei voti direzionali
_SIGNAL_WEIGHTS = {
    "delta_momentum": 1.5,
    "absorption": 2.0,
    "fvg": 1.8,
    "bos_choch": 2.5,
    "imbalance_zone": 1.2,
    "vwap": 1.0,
    "taker_cvd": 1.3,
}


class VolumeAgent(BaseAgent):
    """Agente di analisi volumetrica avanzata (V19).

    Combina 7 segnali volumetrici in un unico score [0.0, 1.0] con voto direzionale.
    """

    def __init__(self):
        super().__init__("volume", initial_weight=0.15)

    def analyse(self, symbol: str, interval: str, df,
                direction: str = "long") -> Optional[AgentResult]:
        """Analizza il DataFrame e ritorna un AgentResult con score volumetrico."""
        if df is None or len(df) < 50:
            return None

        score = 0.0
        long_votes = 0.0
        short_votes = 0.0
        details = []
        metadata: dict = {}

        # ------------------------------------------------------------------
        # 1. Delta Momentum
        # ------------------------------------------------------------------
        try:
            dm = delta_momentum(df, period=5)
            last_dm = float(dm.iloc[-1])
            if np.isnan(last_dm):
                last_dm = 0.0
            metadata["delta_momentum"] = last_dm
            if last_dm > 0.1:
                long_votes += _SIGNAL_WEIGHTS["delta_momentum"]
                score += 0.12
                details.append(f"delta_mom_bull({last_dm:.2f})")
            elif last_dm < -0.1:
                short_votes += _SIGNAL_WEIGHTS["delta_momentum"]
                score += 0.12
                details.append(f"delta_mom_bear({last_dm:.2f})")
        except Exception as exc:
            logger.debug(f"[{symbol}] delta_momentum error: {exc}")

        # ------------------------------------------------------------------
        # 2. Absorption Detection
        # ------------------------------------------------------------------
        try:
            absorption = detect_absorption(df)
            last_abs = int(absorption.iloc[-1])
            recent_abs_sum = int(absorption.iloc[-3:].sum())
            metadata["absorption"] = last_abs
            if last_abs == 1 or recent_abs_sum >= 2:
                long_votes += _SIGNAL_WEIGHTS["absorption"]
                score += 0.18
                details.append("absorption_bull")
            elif last_abs == -1 or recent_abs_sum <= -2:
                short_votes += _SIGNAL_WEIGHTS["absorption"]
                score += 0.18
                details.append("absorption_bear")
        except Exception as exc:
            logger.debug(f"[{symbol}] absorption error: {exc}")

        # ------------------------------------------------------------------
        # 3. Fair Value Gaps
        # ------------------------------------------------------------------
        try:
            fvg_info = nearest_fvg(df)
            if fvg_info:
                dist = fvg_info.get("distance_pct", 1.0)
                fvg_type = fvg_info.get("type", "")
                metadata["fvg"] = fvg_info
                if dist <= 0.01:
                    if fvg_type == "bullish":
                        long_votes += _SIGNAL_WEIGHTS["fvg"]
                        score += 0.15
                        details.append(f"FVG_bull({dist:.1%})")
                    elif fvg_type == "bearish":
                        short_votes += _SIGNAL_WEIGHTS["fvg"]
                        score += 0.15
                        details.append(f"FVG_bear({dist:.1%})")
                elif dist <= 0.03:
                    if fvg_type == "bullish":
                        long_votes += _SIGNAL_WEIGHTS["fvg"] * 0.5
                        score += 0.08
                        details.append(f"FVG_bull_near({dist:.1%})")
                    elif fvg_type == "bearish":
                        short_votes += _SIGNAL_WEIGHTS["fvg"] * 0.5
                        score += 0.08
                        details.append(f"FVG_bear_near({dist:.1%})")
        except Exception as exc:
            logger.debug(f"[{symbol}] fvg error: {exc}")

        # ------------------------------------------------------------------
        # 4. BOS / CHoCH
        # ------------------------------------------------------------------
        try:
            bos_s = detect_bos_choch(df)
            recent_events = list(bos_s.iloc[-3:].values)
            metadata["bos_choch"] = int(bos_s.iloc[-1])
            if any(v == 2 for v in recent_events):
                long_votes += _SIGNAL_WEIGHTS["bos_choch"]
                score += 0.22
                details.append("BOS_bull")
            elif any(v == 1 for v in recent_events):
                long_votes += _SIGNAL_WEIGHTS["bos_choch"] * 0.8
                score += 0.18
                details.append("CHoCH_bull")
            elif any(v == -2 for v in recent_events):
                short_votes += _SIGNAL_WEIGHTS["bos_choch"]
                score += 0.22
                details.append("BOS_bear")
            elif any(v == -1 for v in recent_events):
                short_votes += _SIGNAL_WEIGHTS["bos_choch"] * 0.8
                score += 0.18
                details.append("CHoCH_bear")
        except Exception as exc:
            logger.debug(f"[{symbol}] bos_choch error: {exc}")

        # ------------------------------------------------------------------
        # 5. Volume Imbalance Zone
        # ------------------------------------------------------------------
        try:
            iz = nearest_imbalance_zone(df)
            if iz:
                dist = iz.get("distance_pct", 1.0)
                iz_dir = iz.get("direction", "")
                metadata["imbalance_zone"] = {"direction": iz_dir, "distance_pct": dist}
                if dist <= 0.005:
                    if iz_dir == "support":
                        long_votes += _SIGNAL_WEIGHTS["imbalance_zone"]
                        score += 0.10
                        details.append(f"IZ_support({dist:.1%})")
                    elif iz_dir == "resistance":
                        short_votes += _SIGNAL_WEIGHTS["imbalance_zone"]
                        score += 0.10
                        details.append(f"IZ_resist({dist:.1%})")
        except Exception as exc:
            logger.debug(f"[{symbol}] imbalance_zone error: {exc}")

        # ------------------------------------------------------------------
        # 6. VWAP Position
        # ------------------------------------------------------------------
        try:
            vwap_s, vwap_up, vwap_dn = vwap_bands(df, n_std=1.0)
            last_close = float(df["close"].iloc[-1])
            last_vwap = float(vwap_s.iloc[-1])
            last_vwap_up = float(vwap_up.iloc[-1])
            last_vwap_dn = float(vwap_dn.iloc[-1])
            if last_vwap > 0:
                vwap_dist_pct = (last_close - last_vwap) / last_vwap
            else:
                vwap_dist_pct = 0.0
            metadata["vwap"] = last_vwap
            metadata["vwap_dist_pct"] = vwap_dist_pct
            # Posizione rispetto al VWAP
            if last_close > last_vwap:
                long_votes += _SIGNAL_WEIGHTS["vwap"] * min(abs(vwap_dist_pct) * 10, 1.0)
                score += 0.08
                details.append(f"above_VWAP({vwap_dist_pct:+.1%})")
            elif last_close < last_vwap:
                short_votes += _SIGNAL_WEIGHTS["vwap"] * min(abs(vwap_dist_pct) * 10, 1.0)
                score += 0.08
                details.append(f"below_VWAP({vwap_dist_pct:+.1%})")
            # Estremi delle bande
            if not np.isnan(last_vwap_dn) and last_close < last_vwap_dn:
                long_votes += _SIGNAL_WEIGHTS["vwap"] * 0.5
                score += 0.05
                details.append("below_VWAP_band(oversold)")
            elif not np.isnan(last_vwap_up) and last_close > last_vwap_up:
                short_votes += _SIGNAL_WEIGHTS["vwap"] * 0.5
                score += 0.05
                details.append("above_VWAP_band(overbought)")
        except Exception as exc:
            logger.debug(f"[{symbol}] vwap error: {exc}")

        # ------------------------------------------------------------------
        # 7. Cumulative Taker Delta slope
        # ------------------------------------------------------------------
        try:
            ctd = cumulative_taker_delta(df)
            if len(ctd) >= 5:
                ctd_slope = float(ctd.iloc[-1] - ctd.iloc[-5])
                total_vol = float(df["volume"].iloc[-5:].sum())
                norm_slope = ctd_slope / total_vol if total_vol > 0 else 0.0
                metadata["taker_cvd_slope"] = norm_slope
                if norm_slope > 0.05:
                    long_votes += _SIGNAL_WEIGHTS["taker_cvd"]
                    score += 0.10
                    details.append(f"taker_CVD_bull({norm_slope:.2f})")
                elif norm_slope < -0.05:
                    short_votes += _SIGNAL_WEIGHTS["taker_cvd"]
                    score += 0.10
                    details.append(f"taker_CVD_bear({norm_slope:.2f})")
        except Exception as exc:
            logger.debug(f"[{symbol}] taker_cvd error: {exc}")

        # ------------------------------------------------------------------
        # Direzione finale
        # ------------------------------------------------------------------
        if long_votes > short_votes:
            voted_direction = "long"
        elif short_votes > long_votes:
            voted_direction = "short"
        else:
            voted_direction = direction  # fallback alla direzione suggerita

        details.append(f"vol_votes(L={long_votes:.1f}/S={short_votes:.1f})")
        metadata["long_votes"] = long_votes
        metadata["short_votes"] = short_votes

        final_score = float(np.clip(score, 0.0, 1.0))

        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            interval=interval,
            score=final_score,
            direction=voted_direction,
            confidence=final_score,
            details=details,
            metadata=metadata,
        )
