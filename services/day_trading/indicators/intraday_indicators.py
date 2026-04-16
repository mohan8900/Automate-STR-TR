"""
Intraday technical indicator engine.
Computes VWAP, ORB levels, RSI, EMA, ATR, micro S/R, and more
from 1-minute and 5-minute OHLCV DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

log = get_logger("intraday_indicators")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class ORBLevels:
    high: float
    low: float
    range_size: float
    midpoint: float
    is_valid: bool
    computed_at: datetime


@dataclass
class IntradayTechnicals:
    vwap: float
    vwap_upper: float
    vwap_lower: float
    orb: ORBLevels
    rsi_7: float
    ema_5: float
    ema_9: float
    ema_21: float
    atr: float
    relative_volume: float
    micro_supports: list[float] = field(default_factory=list)
    micro_resistances: list[float] = field(default_factory=list)
    price_vs_vwap_pct: float = 0.0


# ── Indicator Engine ─────────────────────────────────────────────────────────


class IntradayIndicatorEngine:
    """Static/class methods for computing intraday technicals."""

    @classmethod
    def compute_all(
        cls,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        orb_window: int = 15,
    ) -> IntradayTechnicals:
        """
        Master method: compute all intraday indicators and return a
        populated IntradayTechnicals instance.

        Parameters
        ----------
        df_1m : pd.DataFrame
            1-minute OHLCV DataFrame (columns: open, high, low, close, volume).
        df_5m : pd.DataFrame
            5-minute OHLCV DataFrame (same columns).
        orb_window : int
            Number of minutes for the Opening Range Breakout window.
        """
        # Use 1m data for most calculations; 5m for confirmations
        primary = df_1m if not df_1m.empty else df_5m

        # VWAP
        vwap_series = cls.compute_vwap(primary)
        vwap_val = float(vwap_series.iloc[-1]) if len(vwap_series) > 0 else 0.0
        upper, lower = cls.compute_vwap_bands(primary, vwap_series)
        vwap_upper = float(upper.iloc[-1]) if len(upper) > 0 else 0.0
        vwap_lower = float(lower.iloc[-1]) if len(lower) > 0 else 0.0

        # ORB
        orb = cls.compute_orb_levels(df_1m, window_minutes=orb_window)

        # Oscillators & moving averages (from 1m close)
        close_series = primary["close"] if not primary.empty else pd.Series(dtype=float)
        rsi_7 = cls.compute_rsi(close_series, period=7)
        ema_5 = cls.compute_ema(close_series, period=5)
        ema_9 = cls.compute_ema(close_series, period=9)
        ema_21 = cls.compute_ema(close_series, period=21)

        # ATR
        atr = cls.compute_atr(primary, period=14)

        # Relative volume
        relative_volume = cls._compute_relative_volume(primary)

        # Micro support / resistance
        supports, resistances = cls.compute_micro_support_resistance(
            primary, lookback=60
        )

        # Price vs VWAP %
        last_close = float(close_series.iloc[-1]) if len(close_series) > 0 else 0.0
        price_vs_vwap_pct = (
            ((last_close - vwap_val) / vwap_val * 100.0) if vwap_val > 0 else 0.0
        )

        return IntradayTechnicals(
            vwap=vwap_val,
            vwap_upper=vwap_upper,
            vwap_lower=vwap_lower,
            orb=orb,
            rsi_7=rsi_7,
            ema_5=ema_5,
            ema_9=ema_9,
            ema_21=ema_21,
            atr=atr,
            relative_volume=relative_volume,
            micro_supports=supports,
            micro_resistances=resistances,
            price_vs_vwap_pct=price_vs_vwap_pct,
        )

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Cumulative VWAP = cumsum(typical_price * volume) / cumsum(volume).
        Returns a Series aligned with df's index.
        """
        if df.empty or "volume" not in df.columns:
            return pd.Series(dtype=float)

        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()

        # Avoid division by zero
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return vwap.fillna(method="ffill").fillna(0.0)

    @staticmethod
    def compute_vwap_bands(
        df: pd.DataFrame,
        vwap_series: pd.Series,
        std_mult: float = 2.0,
    ) -> tuple[pd.Series, pd.Series]:
        """
        VWAP upper and lower bands using rolling standard deviation
        of (close - VWAP).
        """
        if df.empty or vwap_series.empty:
            empty = pd.Series(dtype=float)
            return empty, empty

        deviation = df["close"] - vwap_series
        # Use expanding std for intraday (cumulative since market open)
        rolling_std = deviation.expanding(min_periods=2).std().fillna(0.0)

        upper = vwap_series + std_mult * rolling_std
        lower = vwap_series - std_mult * rolling_std
        return upper, lower

    # ------------------------------------------------------------------
    # ORB (Opening Range Breakout)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_orb_levels(
        df_1m: pd.DataFrame,
        window_minutes: int = 15,
    ) -> ORBLevels:
        """
        Compute Opening Range Breakout levels from the first
        `window_minutes` of 1-minute candles.
        """
        if df_1m.empty or len(df_1m) < window_minutes:
            return ORBLevels(
                high=0.0,
                low=0.0,
                range_size=0.0,
                midpoint=0.0,
                is_valid=False,
                computed_at=datetime.now(),
            )

        orb_candles = df_1m.iloc[:window_minutes]
        orb_high = float(orb_candles["high"].max())
        orb_low = float(orb_candles["low"].min())
        range_size = orb_high - orb_low
        midpoint = (orb_high + orb_low) / 2.0

        return ORBLevels(
            high=orb_high,
            low=orb_low,
            range_size=range_size,
            midpoint=midpoint,
            is_valid=range_size > 0,
            computed_at=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Micro Support / Resistance (pivot detection)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_micro_support_resistance(
        df: pd.DataFrame,
        lookback: int = 60,
    ) -> tuple[list[float], list[float]]:
        """
        Detect micro support and resistance levels via simple pivot
        point detection on the last `lookback` candles.
        A pivot high is a candle whose high is higher than both neighbours.
        A pivot low is a candle whose low is lower than both neighbours.
        """
        if df.empty or len(df) < 3:
            return [], []

        data = df.tail(lookback)
        highs = data["high"].values
        lows = data["low"].values

        resistances: list[float] = []
        supports: list[float] = []

        for i in range(1, len(highs) - 1):
            # Pivot high -> resistance
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                resistances.append(float(highs[i]))
            # Pivot low -> support
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                supports.append(float(lows[i]))

        # Deduplicate: cluster levels within 0.1% of each other
        supports = _cluster_levels(supports)
        resistances = _cluster_levels(resistances)

        return supports, resistances

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 7) -> float:
        """Standard RSI. Returns the latest value (0-100), or 50.0 on error."""
        if series.empty or len(series) < period + 1:
            return 50.0

        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        last = rsi.iloc[-1]
        return float(last) if not np.isnan(last) else 50.0

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ema(series: pd.Series, period: int = 9) -> float:
        """Exponential Moving Average. Returns the latest value, or 0.0."""
        if series.empty or len(series) < period:
            return 0.0
        ema = series.ewm(span=period, adjust=False).mean()
        last = ema.iloc[-1]
        return float(last) if not np.isnan(last) else 0.0

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range on OHLC data. Returns latest value, or 0.0."""
        if df.empty or len(df) < period + 1:
            return 0.0

        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        last = atr.iloc[-1]
        return float(last) if not np.isnan(last) else 0.0

    # ------------------------------------------------------------------
    # Relative volume (internal helper)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_relative_volume(df: pd.DataFrame) -> float:
        """
        Ratio of recent volume to the session average.
        Uses last 5 bars vs. full session average.
        """
        if df.empty or "volume" not in df.columns or len(df) < 5:
            return 1.0

        session_avg = df["volume"].mean()
        if session_avg <= 0:
            return 1.0
        recent_avg = df["volume"].tail(5).mean()
        return float(recent_avg / session_avg)


# ── Module-level helpers ─────────────────────────────────────────────────────


def _cluster_levels(
    levels: list[float],
    tolerance_pct: float = 0.001,
) -> list[float]:
    """
    Merge price levels that are within `tolerance_pct` of each other,
    keeping the average of each cluster.
    """
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]

    for lvl in sorted_levels[1:]:
        cluster_avg = sum(clusters[-1]) / len(clusters[-1])
        if abs(lvl - cluster_avg) / cluster_avg <= tolerance_pct:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return [round(sum(c) / len(c), 2) for c in clusters]
