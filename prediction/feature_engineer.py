"""
Feature engineering for ML prediction models.
Converts raw OHLCV data + fundamentals into ML-ready feature vectors.
Uses rolling windows to avoid look-ahead bias.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

log = get_logger("feature_engineer")


class FeatureEngineer:
    """Builds feature matrices from price data for ML models."""

    # Target: next N-day return direction (1=up, 0=down)
    DEFAULT_TARGET_DAYS = 5  # Swing trading horizon

    def build_features(
        self,
        df: pd.DataFrame,
        target_days: int = DEFAULT_TARGET_DAYS,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Build feature matrix from OHLCV DataFrame.
        All features use only past data (no look-ahead bias).

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            target_days: forward return horizon for target variable
            include_target: whether to compute target column (False for live prediction)

        Returns:
            DataFrame with feature columns + optional 'target' column
        """
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars, got {len(df)}")

        features = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_ = df["open"]

        # ── Price returns at multiple horizons ────────────────────────────
        for period in [1, 2, 3, 5, 10, 20]:
            features[f"return_{period}d"] = close.pct_change(period)

        # ── Moving average features ───────────────────────────────────────
        for period in [5, 10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f"sma_{period}_dist"] = (close - sma) / sma  # Distance from MA

        # EMA cross signals
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        features["ema_9_21_cross"] = (ema9 - ema21) / close

        if len(df) >= 200:
            sma200 = close.rolling(200).mean()
            features["sma_200_dist"] = (close - sma200) / sma200
            sma50 = close.rolling(50).mean()
            features["golden_cross"] = (sma50 > sma200).astype(float)
        else:
            features["sma_200_dist"] = 0.0
            features["golden_cross"] = 0.0

        # ── RSI ───────────────────────────────────────────────────────────
        for period in [7, 14, 21]:
            features[f"rsi_{period}"] = self._rsi(close, period)

        # RSI slope (is momentum increasing or decreasing?)
        rsi14 = features["rsi_14"]
        features["rsi_14_slope"] = rsi14 - rsi14.shift(5)

        # ── MACD ──────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        features["macd_histogram"] = histogram / close
        features["macd_signal_dist"] = (macd - signal) / close
        features["macd_hist_slope"] = histogram - histogram.shift(3)

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = bb_upper - bb_lower
        features["bb_position"] = np.where(
            bb_range > 0, (close - bb_lower) / bb_range, 0.5
        )
        features["bb_width"] = np.where(bb_mid > 0, bb_range / bb_mid, 0)
        features["bb_squeeze"] = (
            features["bb_width"] < features["bb_width"].rolling(50).quantile(0.1)
        ).astype(float)

        # ── Stochastic ────────────────────────────────────────────────────
        lowest14 = low.rolling(14).min()
        highest14 = high.rolling(14).max()
        stoch_range = highest14 - lowest14 + 1e-10
        features["stoch_k"] = 100 * (close - lowest14) / stoch_range
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        # ── ATR (volatility) ──────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        features["atr_pct"] = atr14 / close
        features["atr_expanding"] = (atr14 > atr14.shift(5)).astype(float)

        # Volatility regime (z-score of recent volatility)
        vol_20 = close.pct_change().rolling(20).std()
        vol_60 = close.pct_change().rolling(60).std()
        features["vol_regime"] = np.where(
            vol_60 > 0, (vol_20 - vol_60) / vol_60, 0
        )

        # ── Volume features ───────────────────────────────────────────────
        vol_sma20 = volume.rolling(20).mean()
        features["relative_volume"] = np.where(
            vol_sma20 > 0, volume / vol_sma20, 1.0
        )
        features["volume_trend"] = (
            volume.rolling(5).mean() / volume.rolling(20).mean()
        ).fillna(1.0)

        # OBV trend
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()
        obv_ma = obv.rolling(20).mean()
        features["obv_slope"] = (obv - obv_ma) / (obv_ma.abs() + 1e-10)

        # Volume-price divergence
        features["vol_price_div"] = (
            features["relative_volume"] * np.sign(features["return_1d"])
        )

        # ── Price pattern features ────────────────────────────────────────
        # Candle body ratio
        body = (close - open_).abs()
        full_range = high - low + 1e-10
        features["body_ratio"] = body / full_range

        # Upper/lower shadow
        features["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
        features["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range

        # Higher highs / lower lows
        features["higher_high_5d"] = (high > high.shift(1).rolling(5).max()).astype(float)
        features["lower_low_5d"] = (low < low.shift(1).rolling(5).min()).astype(float)

        # Distance from 52-week high/low
        high_52w = high.rolling(252).max()
        low_52w = low.rolling(252).min()
        features["pct_from_52w_high"] = (close - high_52w) / high_52w
        features["pct_from_52w_low"] = (close - low_52w) / (low_52w + 1e-10)

        # ── Temporal features ─────────────────────────────────────────────
        features["day_of_week"] = df.index.dayofweek / 4.0  # Normalized 0-1
        features["month"] = df.index.month / 12.0

        # ── Statistical features ──────────────────────────────────────────
        # Skewness of returns (asymmetry)
        features["return_skew_20d"] = close.pct_change().rolling(20).skew()

        # Kurtosis (tail risk)
        features["return_kurt_20d"] = close.pct_change().rolling(20).kurt()

        # Auto-correlation (trending vs mean-reverting)
        features["autocorr_5d"] = close.pct_change().rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 5 else 0,
            raw=False
        )

        # ── Target variable ───────────────────────────────────────────────
        if include_target:
            future_return = close.shift(-target_days) / close - 1
            features["target"] = (future_return > 0).astype(int)
            features["target_return"] = future_return

        # Drop rows with NaN from rolling windows
        features = features.dropna()

        log.debug(
            f"Built {len(features.columns)} features, {len(features)} samples"
        )
        return features

    def add_fundamental_features(
        self,
        features: pd.DataFrame,
        fundamental_data: dict,
    ) -> pd.DataFrame:
        """Add fundamental data as static features (same value for all rows of a symbol)."""
        fund_features = {}

        def safe(key, default=0.0):
            val = fundamental_data.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        fund_features["pe_ratio"] = safe("pe_ratio", 20.0)
        fund_features["peg_ratio"] = safe("peg_ratio", 1.5)
        fund_features["debt_to_equity"] = safe("debt_to_equity", 50.0)
        fund_features["roe"] = safe("return_on_equity", 0.15)
        fund_features["operating_margin"] = safe("operating_margin", 0.1)
        fund_features["revenue_growth"] = safe("revenue_growth_yoy", 0.0)
        fund_features["fcf_yield"] = safe("fcf_yield", 0.0)
        fund_features["beta"] = safe("beta", 1.0)

        for key, val in fund_features.items():
            features[f"fund_{key}"] = val

        return features

    def add_market_context_features(
        self,
        features: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Add market-wide context features (benchmark trend, VIX)."""
        if benchmark_df is not None and not benchmark_df.empty:
            bench_close = benchmark_df["close"].reindex(features.index, method="ffill")

            # Benchmark returns
            features["bench_return_5d"] = bench_close.pct_change(5)
            features["bench_return_20d"] = bench_close.pct_change(20)

            # Benchmark vs its own SMA50
            bench_sma50 = bench_close.rolling(50).mean()
            features["bench_above_sma50"] = (bench_close > bench_sma50).astype(float)

            # Market breadth proxy: stock relative strength vs benchmark
            if "return_5d" in features.columns:
                features["relative_strength_5d"] = (
                    features["return_5d"] - features["bench_return_5d"]
                )

        if vix_df is not None and not vix_df.empty:
            vix_close = vix_df["close"].reindex(features.index, method="ffill")
            features["vix_level"] = vix_close / 100.0  # Normalize
            features["vix_change_5d"] = vix_close.pct_change(5)

        return features

    # ── Helper methods ────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50)
