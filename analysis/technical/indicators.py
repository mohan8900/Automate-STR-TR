"""
Technical analysis indicators — RSI, MACD, Bollinger Bands, ATR, MA, Volume, OBV.
All computed from a standard OHLCV DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False

from core.logger import get_logger

log = get_logger("technical")


@dataclass
class TechnicalSignals:
    symbol: str
    current_price: float
    # Moving Averages
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    price_vs_sma50_pct: float       # (price/SMA50) - 1
    price_vs_sma200_pct: float      # (price/SMA200) - 1
    golden_cross: bool              # SMA50 > SMA200
    death_cross: bool
    # Momentum
    rsi_14: float
    rsi_signal: Literal["oversold", "neutral", "overbought"]
    rsi_divergence: Literal["bullish", "bearish", "none"]
    macd_line: float
    macd_signal_line: float
    macd_histogram: float
    macd_crossover: Literal["bullish", "bearish", "none"]
    stoch_k: float                  # Stochastic %K
    stoch_d: float                  # Stochastic %D
    # Volatility
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float                 # (upper-lower)/middle — squeeze detection
    bb_position: float              # (price-lower)/(upper-lower) [0-1]
    atr_14: float
    atr_pct: float                  # ATR as % of price
    # Volume
    volume_sma_20: float
    relative_volume: float
    obv_trend: Literal["up", "down", "flat"]
    vwap: Optional[float]           # Only meaningful intraday
    # Support & Resistance (filled by SupportResistanceAnalyzer)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    risk_reward_to_resistance: Optional[float] = None
    # Computed score
    technical_score: float = 50.0


class TechnicalAnalyzer:

    def analyze(self, symbol: str, df: pd.DataFrame) -> TechnicalSignals:
        """Compute all technical signals from OHLCV DataFrame."""
        if len(df) < 50:
            return self._empty_signals(symbol, df)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        price = float(close.iloc[-1])

        # ── Moving Averages ───────────────────────────────────────────────
        sma20 = self._sma(close, 20)
        sma50 = self._sma(close, 50)
        sma200 = self._sma(close, 200) if len(df) >= 200 else sma50
        ema9 = self._ema(close, 9)
        ema21 = self._ema(close, 21)

        golden_cross = sma50 > sma200
        death_cross = sma50 < sma200

        # ── RSI ───────────────────────────────────────────────────────────
        rsi = self._rsi(close, 14)
        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"

        rsi_divergence = self._detect_rsi_divergence(close, rsi)

        # ── MACD ──────────────────────────────────────────────────────────
        macd_line, macd_sig, macd_hist = self._macd(close)
        macd_crossover = self._detect_macd_crossover(macd_line, macd_sig, close)

        # ── Stochastic ────────────────────────────────────────────────────
        stoch_k, stoch_d = self._stochastic(high, low, close)

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(close)
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_range = bb_upper - bb_lower
        bb_pos = (price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # ── ATR ───────────────────────────────────────────────────────────
        atr = self._atr(high, low, close, 14)
        atr_pct = atr / price if price > 0 else 0

        # ── Volume ────────────────────────────────────────────────────────
        vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
        rel_vol = float(volume.iloc[-1]) / vol_sma20 if vol_sma20 > 0 else 1.0
        obv_trend = self._obv_trend(close, volume)

        # ── Technical Score ───────────────────────────────────────────────
        score = self._compute_score(
            price, sma50, sma200, rsi, rsi_signal, rsi_divergence,
            macd_hist, macd_crossover, bb_pos, rel_vol, golden_cross
        )

        return TechnicalSignals(
            symbol=symbol,
            current_price=price,
            sma_20=sma20,
            sma_50=sma50,
            sma_200=sma200,
            ema_9=ema9,
            ema_21=ema21,
            price_vs_sma50_pct=(price / sma50 - 1) if sma50 > 0 else 0,
            price_vs_sma200_pct=(price / sma200 - 1) if sma200 > 0 else 0,
            golden_cross=golden_cross,
            death_cross=death_cross,
            rsi_14=round(rsi, 1),
            rsi_signal=rsi_signal,
            rsi_divergence=rsi_divergence,
            macd_line=round(macd_line, 4),
            macd_signal_line=round(macd_sig, 4),
            macd_histogram=round(macd_hist, 4),
            macd_crossover=macd_crossover,
            stoch_k=round(stoch_k, 1),
            stoch_d=round(stoch_d, 1),
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_mid, 2),
            bb_lower=round(bb_lower, 2),
            bb_width=round(bb_width, 4),
            bb_position=round(bb_pos, 3),
            atr_14=round(atr, 3),
            atr_pct=round(atr_pct, 4),
            volume_sma_20=vol_sma20,
            relative_volume=round(rel_vol, 2),
            obv_trend=obv_trend,
            vwap=None,
            technical_score=score,
        )

    # ── Indicator computations ────────────────────────────────────────────

    def _sma(self, s: pd.Series, n: int) -> float:
        if len(s) < n:
            return float(s.mean())
        return float(s.rolling(n).mean().iloc[-1])

    def _ema(self, s: pd.Series, n: int) -> float:
        if len(s) < n:
            return float(s.mean())
        return float(s.ewm(span=n, adjust=False).mean().iloc[-1])

    def _rsi(self, s: pd.Series, n: int = 14) -> float:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _macd(self, s: pd.Series) -> tuple[float, float, float]:
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

    def _stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14
    ) -> tuple[float, float]:
        lowest = low.rolling(k).min()
        highest = high.rolling(k).max()
        stoch = 100 * (close - lowest) / (highest - lowest + 1e-10)
        stoch_k = float(stoch.iloc[-1])
        stoch_d = float(stoch.rolling(3).mean().iloc[-1])
        return stoch_k, stoch_d

    def _bollinger_bands(
        self, s: pd.Series, n: int = 20, k: float = 2.0
    ) -> tuple[float, float, float]:
        mid = s.rolling(n).mean()
        std = s.rolling(n).std()
        upper = mid + k * std
        lower = mid - k * std
        return float(upper.iloc[-1]), float(mid.iloc[-1]), float(lower.iloc[-1])

    def _atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
    ) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(n).mean().iloc[-1])

    def _obv_trend(self, close: pd.Series, volume: pd.Series) -> str:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()
        obv_ma = obv.rolling(20).mean()
        if len(obv_ma.dropna()) < 2:
            return "flat"
        recent_slope = float(obv_ma.iloc[-1]) - float(obv_ma.iloc[-10])
        if recent_slope > 0:
            return "up"
        elif recent_slope < 0:
            return "down"
        return "flat"

    def _detect_rsi_divergence(self, close: pd.Series, current_rsi: float) -> str:
        """Simple divergence: price makes new high but RSI does not (bearish), vice versa."""
        if len(close) < 20:
            return "none"
        price_high_20d = float(close.tail(20).max())
        price_high_40d = float(close.tail(40).max()) if len(close) >= 40 else price_high_20d

        if float(close.iloc[-1]) >= price_high_20d * 0.99 and current_rsi < 60:
            return "bearish"  # Price at high, RSI lagging = bearish divergence
        return "none"

    def _detect_macd_crossover(
        self, macd: float, signal: float, close: pd.Series
    ) -> str:
        """Detect if a MACD crossover happened in the last 3 bars."""
        if len(close) < 35:
            return "none"
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_s = ema12 - ema26
        sig_s = macd_s.ewm(span=9, adjust=False).mean()
        hist = macd_s - sig_s

        if len(hist) < 3:
            return "none"
        if hist.iloc[-1] > 0 and hist.iloc[-3] <= 0:
            return "bullish"
        if hist.iloc[-1] < 0 and hist.iloc[-3] >= 0:
            return "bearish"
        return "none"

    def _compute_score(
        self,
        price: float,
        sma50: float,
        sma200: float,
        rsi: float,
        rsi_signal: str,
        rsi_divergence: str,
        macd_hist: float,
        macd_crossover: str,
        bb_pos: float,
        rel_vol: float,
        golden_cross: bool,
    ) -> float:
        """Composite bull score 0–100."""
        score = 50.0

        # Trend components
        if price > sma50:
            score += 10
        if price > sma200:
            score += 10
        if golden_cross:
            score += 5
        else:
            score -= 5

        # RSI
        if rsi_signal == "oversold":
            score += 8  # Potential reversal
        elif rsi_signal == "overbought":
            score -= 8
        if rsi_divergence == "bearish":
            score -= 10

        # MACD
        if macd_hist > 0:
            score += 5
        else:
            score -= 5
        if macd_crossover == "bullish":
            score += 8
        elif macd_crossover == "bearish":
            score -= 8

        # Bollinger Band position (near lower band = oversold bounce candidate)
        if bb_pos < 0.2:
            score += 5
        elif bb_pos > 0.8:
            score -= 5

        # Volume confirmation
        if rel_vol > 1.5:
            score += 5

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Wilder's ADX with +DI/-DI. Returns (adx, plus_di, minus_di) as Series."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()

        up = high.diff()
        down = -low.diff()
        plus_dm = ((up > down) & (up > 0)).astype(float) * up
        minus_dm = ((down > up) & (down > 0)).astype(float) * down

        plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx_series = dx.ewm(alpha=1 / n, adjust=False).mean()
        return adx_series, plus_di, minus_di

    def _empty_signals(self, symbol: str, df: pd.DataFrame) -> TechnicalSignals:
        price = float(df["close"].iloc[-1]) if not df.empty else 0
        return TechnicalSignals(
            symbol=symbol, current_price=price,
            sma_20=price, sma_50=price, sma_200=price,
            ema_9=price, ema_21=price,
            price_vs_sma50_pct=0, price_vs_sma200_pct=0,
            golden_cross=False, death_cross=False,
            rsi_14=50, rsi_signal="neutral", rsi_divergence="none",
            macd_line=0, macd_signal_line=0, macd_histogram=0, macd_crossover="none",
            stoch_k=50, stoch_d=50,
            bb_upper=price * 1.02, bb_middle=price, bb_lower=price * 0.98,
            bb_width=0.04, bb_position=0.5,
            atr_14=price * 0.02, atr_pct=0.02,
            volume_sma_20=0, relative_volume=1, obv_trend="flat", vwap=None,
            technical_score=50,
        )
