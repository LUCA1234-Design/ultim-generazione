"""
Candlestick chart generator for signal notifications.
Uses mplfinance to create clean charts with entry, SL, TP levels.
"""
import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend, thread-safe — MUST be before any other matplotlib import
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf

logger = logging.getLogger("ChartGenerator")


def generate_signal_chart(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    n_candles: int = 60,
    rr: float = 0.0,
    kelly_pct: float = 0.0,
) -> Optional[bytes]:
    """Generate a candlestick chart with entry, SL, TP1, TP2 horizontal lines.

    Args:
        df: DataFrame with columns open, high, low, close, volume (and a DatetimeIndex)
        symbol: e.g. "BTCUSDT"
        interval: e.g. "15m"
        direction: "long" or "short"
        entry: entry price
        sl: stop loss price
        tp1: take profit 1 price
        tp2: take profit 2 price
        n_candles: number of candles to show (last N)
        rr: Risk/Reward ratio (optional, for annotations)
        kelly_pct: Kelly criterion position size % (optional, for info box)

    Returns:
        PNG image as bytes, or None on error.
    """
    try:
        # Take last n_candles
        plot_df = df.tail(n_candles).copy()

        if len(plot_df) < 10:
            return None

        # Ensure DatetimeIndex
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            if "open_time" in plot_df.columns:
                plot_df.index = pd.to_datetime(plot_df["open_time"], unit="ms")
            elif "timestamp" in plot_df.columns:
                plot_df.index = pd.to_datetime(plot_df["timestamp"], unit="ms")
            else:
                plot_df.index = pd.date_range(
                    end=pd.Timestamp.now(), periods=len(plot_df), freq="1min"
                )

        # Ensure required columns with proper capitalization for mplfinance
        plot_df = plot_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        # Make sure numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

        plot_df = plot_df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(plot_df) < 5:
            return None

        # Colors
        dir_emoji = "LONG" if direction == "long" else "SHORT"
        entry_color = "#2196F3"  # blue
        sl_color = "#F44336"     # red
        tp1_color = "#4CAF50"    # green
        tp2_color = "#00E676"    # bright green

        # Calculate % gain for TP1 and TP2
        if direction == "long":
            pct_tp1 = ((tp1 - entry) / entry) * 100 if entry > 0 else 0.0
            pct_tp2 = ((tp2 - entry) / entry) * 100 if entry > 0 else 0.0
        else:
            pct_tp1 = ((entry - tp1) / entry) * 100 if entry > 0 else 0.0
            pct_tp2 = ((entry - tp2) / entry) * 100 if entry > 0 else 0.0

        from config.settings import HIGH_MARGIN_MIN_RR, HIGH_MARGIN_BADGE
        is_high_margin = rr >= HIGH_MARGIN_MIN_RR

        # Horizontal lines for entry, SL, TP1, TP2
        hlines = dict(
            hlines=[entry, sl, tp1, tp2],
            colors=[entry_color, sl_color, tp1_color, tp2_color],
            linestyle=["--", "-", "-", "-"],
            linewidths=[1.5, 1.5, 1.2, 1.2],
        )

        # Dark style
        mc = mpf.make_marketcolors(
            up="#26A69A",
            down="#EF5350",
            edge={"up": "#26A69A", "down": "#EF5350"},
            wick={"up": "#26A69A", "down": "#EF5350"},
            volume={"up": "#26A69A", "down": "#EF5350"},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            base_mpf_style="nightclouds",
            rc={"font.size": 9},
        )

        # Title — add HIGH MARGIN badge and gold color when applicable
        title = f"{symbol} [{interval}] — {dir_emoji}"
        if is_high_margin:
            title = f"{HIGH_MARGIN_BADGE} | {title}"

        # Render to buffer
        buf = io.BytesIO()
        fig, axes = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            title=title,
            volume="Volume" in plot_df.columns,
            hlines=hlines,
            figsize=(10, 6),
            returnfig=True,
        )

        # Add legend text for the levels
        ax = axes[0]

        # Coloured profit/risk zones using axhspan
        try:
            if direction == "long":
                # Profit zone between entry and TP1
                ax.axhspan(entry, tp1, alpha=0.15, color="#4CAF50", zorder=0)
                # Extended profit zone between TP1 and TP2
                ax.axhspan(tp1, tp2, alpha=0.10, color="#00E676", zorder=0)
                # Risk zone between SL and entry
                ax.axhspan(sl, entry, alpha=0.12, color="#F44336", zorder=0)
            else:
                # Profit zone between TP1 and entry
                ax.axhspan(tp1, entry, alpha=0.15, color="#4CAF50", zorder=0)
                # Extended profit zone between TP2 and TP1
                ax.axhspan(tp2, tp1, alpha=0.10, color="#00E676", zorder=0)
                # Risk zone between entry and SL
                ax.axhspan(entry, sl, alpha=0.12, color="#F44336", zorder=0)
        except Exception as _zone_err:
            logger.debug(f"axhspan zone error: {_zone_err}")

        ax.text(
            0.02, 0.98, f"Entry: {entry:.4f}",
            transform=ax.transAxes, fontsize=9,
            color=entry_color, va="top", fontweight="bold",
        )
        ax.text(
            0.02, 0.94, f"SL: {sl:.4f}",
            transform=ax.transAxes, fontsize=9,
            color=sl_color, va="top", fontweight="bold",
        )
        ax.text(
            0.02, 0.90, f"TP1: {tp1:.4f} (+{pct_tp1:.2f}%)",
            transform=ax.transAxes, fontsize=9,
            color=tp1_color, va="top", fontweight="bold",
        )
        ax.text(
            0.02, 0.86, f"TP2: {tp2:.4f} (+{pct_tp2:.2f}%)",
            transform=ax.transAxes, fontsize=9,
            color=tp2_color, va="top", fontweight="bold",
        )

        # Info box in top-right corner: R/R, % TP1, % TP2, Kelly
        info_lines = []
        if rr > 0:
            info_lines.append(f"R/R: {rr:.2f}x")
        info_lines.append(f"TP1: +{pct_tp1:.2f}%")
        info_lines.append(f"TP2: +{pct_tp2:.2f}%")
        if kelly_pct > 0:
            info_lines.append(f"Kelly: {kelly_pct:.1f}%")
        if info_lines:
            info_text = "\n".join(info_lines)
            box_color = "#FFD700" if is_high_margin else "#FFFFFF"
            ax.text(
                0.98, 0.98, info_text,
                transform=ax.transAxes, fontsize=9,
                color="#000000", va="top", ha="right", fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=box_color,
                    alpha=0.85,
                    edgecolor="#AAAAAA",
                ),
            )

        # R/R annotation on chart
        if rr > 0:
            rr_color = "#FFD700" if is_high_margin else "#FFFFFF"
            ax.text(
                0.50, 0.02, f"R/R: {rr:.2f}x",
                transform=ax.transAxes, fontsize=10,
                color=rr_color, va="bottom", ha="center", fontweight="bold",
            )

        # Title color: gold for high margin signals
        if is_high_margin:
            try:
                for t in fig.texts:
                    t.set_color("#FFD700")
                    t.set_fontweight("bold")
            except Exception as _title_err:
                logger.debug(f"Title color error: {_title_err}")

        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)

        plt.close(fig)

        return buf.read()

    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None
