"""Intraday feature engineering for XGBoost model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)


class IntradayFeatureEngineer:
    """Build a flat feature matrix from 1-minute and 5-minute OHLCV candles."""

    # ------------------------------------------------------------------
    def build_features(
        self,
        df_1m: pd.DataFrame,
        df_5m: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of engineered features (one row per 1m bar).

        Parameters
        ----------
        df_1m : DataFrame with columns [open, high, low, close, volume]
        df_5m : DataFrame with columns [open, high, low, close, volume]
        include_target : if True, append a binary target and target_return column.
        """
        df = df_1m.copy()

        # ---- Price returns ----
        df["return_1m"] = df["close"].pct_change(1)
        df["return_5m"] = df["close"].pct_change(5)
        df["return_15m"] = df["close"].pct_change(15)

        # ---- VWAP distance ----
        df["vwap"] = _vwap(df)
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

        # ---- RSI ----
        df["rsi_7"] = _rsi(df["close"], period=7)
        df["rsi_14"] = _rsi(df["close"], period=14)

        # ---- EMA distances ----
        for span in (5, 9, 21):
            ema = df["close"].ewm(span=span, adjust=False).mean()
            df[f"ema_{span}_dist"] = (df["close"] - ema) / ema

        # ---- Volume ratio ----
        vol_ma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (vol_ma + 1e-10)

        # ---- ATR normalised ----
        df["atr_normalized"] = _atr(df, period=14) / df["close"]

        # ---- Candle ratios ----
        hl_range = df["high"] - df["low"] + 1e-10
        df["body_ratio"] = (df["close"] - df["open"]).abs() / hl_range
        df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / hl_range
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / hl_range

        # ---- Streaks ----
        green = (df["close"] > df["open"]).astype(int)
        red = (df["close"] < df["open"]).astype(int)
        df["green_streak"] = _streak(green)
        df["red_streak"] = _streak(red)

        # ---- Time features ----
        idx = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
        bars_since = idx - idx[0]
        df["bars_since_open"] = np.arange(len(df))
        hour_frac = idx.hour + idx.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
        df["minute_sin"] = np.sin(2 * np.pi * idx.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * idx.minute / 60)

        # ---- Spread ----
        df["spread_pct"] = (df["high"] - df["low"]) / df["close"]

        # ---- Target ----
        if include_target:
            forward_close = df["close"].shift(-5)
            df["target"] = (forward_close > df["close"]).astype(int)
            df["target_return"] = (forward_close - df["close"]) / df["close"]

        # Drop rows with NaN from rolling calculations
        df.dropna(inplace=True)

        # Keep only feature columns (and target if present)
        feature_cols = [
            "return_1m", "return_5m", "return_15m",
            "vwap_distance",
            "rsi_7", "rsi_14",
            "ema_5_dist", "ema_9_dist", "ema_21_dist",
            "volume_ratio", "atr_normalized",
            "body_ratio", "upper_shadow", "lower_shadow",
            "green_streak", "red_streak",
            "bars_since_open", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "spread_pct",
        ]
        if include_target:
            feature_cols += ["target", "target_return"]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns: {missing}")

        return df[[c for c in feature_cols if c in df.columns]]


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP across the session."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift(1)).abs()
    low_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _streak(indicator: pd.Series) -> pd.Series:
    """Running count of consecutive True (1) values; resets on 0."""
    streaks = pd.Series(0, index=indicator.index, dtype=int)
    count = 0
    for i in range(len(indicator)):
        if indicator.iloc[i] == 1:
            count += 1
        else:
            count = 0
        streaks.iloc[i] = count
    return streaks
