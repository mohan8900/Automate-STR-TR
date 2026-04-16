"""
Per-symbol regime classifier.

Classifies an individual stock into UPTREND / DOWNTREND / RANGE / CHOPPY_VOLATILE
using ADX, Bollinger band width, and realized-vol percentile against the stock's
own 1-year history. Complements the market-wide MarketRegimeClassifier: a stock
can be ranging inside a bull market, and strategies should adapt.

Output drives three actionable flags:
  - momentum_ok       : trend is strong enough to trade breakouts / continuation
  - mean_reversion_ok : range is clean enough to fade extremes
  - reduce_size       : volatility or choppiness is too high for full sizing
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from analysis.technical.indicators import TechnicalAnalyzer
from core.logger import get_logger

log = get_logger("symbol_regime")


class SymbolRegime(str, Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    CHOPPY_VOLATILE = "CHOPPY_VOLATILE"


@dataclass
class SymbolRegimeReading:
    regime: SymbolRegime
    adx: float                  # ADX(14), 0–100. >25 = trending, <20 = range
    plus_di: float              # +DI(14)
    minus_di: float             # -DI(14)
    vol_percentile: float       # Realized-vol percentile vs own 1y history, 0–100
    bb_width_percentile: float  # BB width percentile vs own 1y history, 0–100
    price_vs_sma20_pct: float
    price_vs_sma50_pct: float
    description: str
    momentum_ok: bool
    mean_reversion_ok: bool
    reduce_size: bool


class SymbolRegimeClassifier:
    # Tunables — intentionally exposed as class attrs so strategies can override.
    ADX_TREND_THRESHOLD = 25.0      # Above = trending
    ADX_RANGE_THRESHOLD = 20.0      # Below = range / no trend
    VOL_HIGH_PCTILE = 80.0          # Above = CHOPPY_VOLATILE override
    VOL_REDUCE_SIZE_PCTILE = 75.0   # Above = cut size even if not choppy

    def classify(self, symbol: str, df: pd.DataFrame) -> SymbolRegimeReading:
        if df is None or df.empty or len(df) < 50:
            return self._default(symbol)

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            price = float(close.iloc[-1])

            adx_s, plus_di_s, minus_di_s = TechnicalAnalyzer.adx(high, low, close, n=14)
            adx = _last_finite(adx_s, 0.0)
            plus_di = _last_finite(plus_di_s, 0.0)
            minus_di = _last_finite(minus_di_s, 0.0)

            sma20 = float(close.rolling(20).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            vs_sma20 = (price / sma20 - 1) if sma20 > 0 else 0.0
            vs_sma50 = (price / sma50 - 1) if sma50 > 0 else 0.0

            # Realized vol percentile vs own 1y history
            returns = close.pct_change()
            vol_series = returns.rolling(20).std() * (252 ** 0.5)
            vol_pctile = _percentile_rank(vol_series, window=252)

            # BB width percentile — squeeze detection
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_width = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid
            bb_width_pctile = _percentile_rank(bb_width, window=252)

            # Classify
            if vol_pctile >= self.VOL_HIGH_PCTILE and adx < self.ADX_TREND_THRESHOLD:
                regime = SymbolRegime.CHOPPY_VOLATILE
            elif adx >= self.ADX_TREND_THRESHOLD and plus_di > minus_di:
                regime = SymbolRegime.UPTREND
            elif adx >= self.ADX_TREND_THRESHOLD and minus_di > plus_di:
                regime = SymbolRegime.DOWNTREND
            else:
                regime = SymbolRegime.RANGE

            momentum_ok = regime in (SymbolRegime.UPTREND, SymbolRegime.DOWNTREND)
            mean_reversion_ok = (
                regime == SymbolRegime.RANGE
                and adx < self.ADX_RANGE_THRESHOLD
                and vol_pctile < self.VOL_REDUCE_SIZE_PCTILE
            )
            reduce_size = (
                regime == SymbolRegime.CHOPPY_VOLATILE
                or vol_pctile >= self.VOL_REDUCE_SIZE_PCTILE
            )

            description = _describe(regime, adx, vol_pctile, bb_width_pctile)

            return SymbolRegimeReading(
                regime=regime,
                adx=round(adx, 1),
                plus_di=round(plus_di, 1),
                minus_di=round(minus_di, 1),
                vol_percentile=round(vol_pctile, 1),
                bb_width_percentile=round(bb_width_pctile, 1),
                price_vs_sma20_pct=round(vs_sma20, 4),
                price_vs_sma50_pct=round(vs_sma50, 4),
                description=description,
                momentum_ok=momentum_ok,
                mean_reversion_ok=mean_reversion_ok,
                reduce_size=reduce_size,
            )

        except Exception as e:
            log.warning(f"Symbol regime classification failed for {symbol}: {e}")
            return self._default(symbol)

    def _default(self, symbol: str) -> SymbolRegimeReading:
        return SymbolRegimeReading(
            regime=SymbolRegime.RANGE,
            adx=0.0,
            plus_di=0.0,
            minus_di=0.0,
            vol_percentile=50.0,
            bb_width_percentile=50.0,
            price_vs_sma20_pct=0.0,
            price_vs_sma50_pct=0.0,
            description="Insufficient data — defaulting to RANGE (conservative).",
            momentum_ok=False,
            mean_reversion_ok=False,
            reduce_size=True,
        )


def _last_finite(s: pd.Series, default: float) -> float:
    if s is None or len(s) == 0:
        return default
    val = s.iloc[-1]
    if pd.isna(val):
        return default
    return float(val)


def _percentile_rank(s: pd.Series, window: int) -> float:
    """Return the percentile rank (0–100) of the latest value against the trailing window."""
    history = s.dropna().tail(window)
    if len(history) < 20:
        return 50.0
    current = history.iloc[-1]
    return float((history < current).mean()) * 100


def _describe(regime: SymbolRegime, adx: float, vol_p: float, bb_p: float) -> str:
    if regime == SymbolRegime.UPTREND:
        return f"Clean uptrend (ADX {adx:.0f}) — trade breakouts/continuation, avoid fades."
    if regime == SymbolRegime.DOWNTREND:
        return f"Clean downtrend (ADX {adx:.0f}) — no new longs, short-bias only."
    if regime == SymbolRegime.CHOPPY_VOLATILE:
        return f"Choppy + high vol (vol pctile {vol_p:.0f}) — stand aside or cut size hard."
    squeeze = " (BB squeeze)" if bb_p < 20 else ""
    return f"Range-bound (ADX {adx:.0f}){squeeze} — mean-reversion only, no breakout chasing yet."
