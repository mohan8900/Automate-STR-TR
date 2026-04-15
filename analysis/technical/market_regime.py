"""
Market Regime Classifier — determines the overall market environment.
Drives strategy gating: only open longs in BULL, only shorts in BEAR, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yfinance as yf
import pandas as pd

from config.watchlists import BENCHMARK
from core.logger import get_logger

log = get_logger("market_regime")


class MarketRegime(str, Enum):
    BULL = "BULL"           # Trend up, open longs
    BEAR = "BEAR"           # Trend down, only shorts/cash
    SIDEWAYS = "SIDEWAYS"   # Range-bound, mean reversion
    VOLATILE = "VOLATILE"   # High uncertainty, reduce all exposure


@dataclass
class RegimeReading:
    regime: MarketRegime
    benchmark_vs_sma200_pct: float
    breadth_score: float            # 0–100 (% of stocks above 50-day SMA, estimated)
    trend_strength: float           # ADX-like measure
    description: str
    new_long_allowed: bool
    new_short_allowed: bool
    position_size_multiplier: float


class MarketRegimeClassifier:

    # Strategy gating by regime
    REGIME_RULES = {
        MarketRegime.BULL: {
            "new_long": True, "new_short": False, "size_mult": 1.0,
            "description": "Bull market — full long bias, normal position sizing",
        },
        MarketRegime.SIDEWAYS: {
            "new_long": True, "new_short": True, "size_mult": 0.75,
            "description": "Sideways market — mean reversion preferred, reduced sizing",
        },
        MarketRegime.BEAR: {
            "new_long": False, "new_short": True, "size_mult": 0.60,
            "description": "Bear market — no new longs, short-focused, reduced sizing",
        },
        MarketRegime.VOLATILE: {
            "new_long": False, "new_short": False, "size_mult": 0.40,
            "description": "Volatile market — all new positions blocked, preserve capital",
        },
    }

    def __init__(self, exchange: str = "US"):
        self.exchange = exchange
        self.benchmark = BENCHMARK.get(exchange, "SPY")

    def classify(self) -> RegimeReading:
        try:
            ticker = yf.Ticker(self.benchmark)
            df = ticker.history(period="1y", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 50:
                return self._default_regime()

            df.columns = [c.lower() for c in df.columns]
            close = df["close"]
            price = float(close.iloc[-1])

            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma50
            vs_sma200 = (price / sma200) - 1

            # Trend strength: how cleanly is price above/below key MAs?
            pct_above_sma50 = float((close.tail(20) > close.tail(20).rolling(50).mean().iloc[-1]).mean())

            # 20-day return volatility (for VOLATILE regime detection)
            returns = close.pct_change().tail(20)
            rolling_vol = float(returns.std() * (252 ** 0.5))  # Annualized

            # Classify regime
            if rolling_vol > 0.30:  # Annualized vol > 30%
                regime = MarketRegime.VOLATILE
            elif price > sma200 and price > sma50:
                regime = MarketRegime.BULL
            elif price < sma200 and price < sma50:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS

            rules = self.REGIME_RULES[regime]
            return RegimeReading(
                regime=regime,
                benchmark_vs_sma200_pct=round(vs_sma200, 4),
                breadth_score=round(pct_above_sma50 * 100, 1),
                trend_strength=round(abs(vs_sma200) * 100, 1),
                description=rules["description"],
                new_long_allowed=rules["new_long"],
                new_short_allowed=rules["new_short"],
                position_size_multiplier=rules["size_mult"],
            )

        except Exception as e:
            log.warning(f"Regime classification failed: {e}")
            return self._default_regime()

    def _default_regime(self) -> RegimeReading:
        return RegimeReading(
            regime=MarketRegime.SIDEWAYS,
            benchmark_vs_sma200_pct=0,
            breadth_score=50,
            trend_strength=0,
            description="Could not determine regime — defaulting to SIDEWAYS (conservative)",
            new_long_allowed=True,
            new_short_allowed=False,
            position_size_multiplier=0.75,
        )
