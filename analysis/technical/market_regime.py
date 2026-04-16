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

from analysis.technical.indicators import TechnicalAnalyzer
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
    breadth_score: float            # 0–100 (% of watchlist above SMA50; 50 if not yet computed)
    trend_strength: float           # ADX(14) on the benchmark, 0–100 (>25 = trending)
    trend_persistence: float        # % of last 50 days benchmark closed above SMA50
    realized_vol_annualized: float  # 20-day annualized vol on benchmark
    vol_percentile: float           # Rank of current vol vs 1-year history, 0–100
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

    def classify(self, breadth_score: Optional[float] = None) -> RegimeReading:
        """
        Classify the market-wide regime from the benchmark (e.g. ^NSEI, SPY).

        breadth_score: optional 0–100 score measuring % of watchlist above SMA50.
        Passed in by the aggregator; defaults to 50 if not provided (Phase B).
        """
        try:
            ticker = yf.Ticker(self.benchmark)
            df = ticker.history(period="1y", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 50:
                return self._default_regime()

            df.columns = [c.lower() for c in df.columns]
            close = df["close"]
            high = df["high"]
            low = df["low"]
            price = float(close.iloc[-1])

            sma50_series = close.rolling(50).mean()
            sma50 = float(sma50_series.iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma50
            vs_sma200 = (price / sma200) - 1

            # Real ADX(14) on the benchmark — trend strength, not just distance from MA
            adx_series, _, _ = TechnicalAnalyzer.adx(high, low, close, n=14)
            adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0

            # Trend persistence: % of last 50 sessions benchmark closed above its SMA50
            # (Replaces the broken `pct_above_sma50` calc from the prior version.)
            valid = sma50_series.dropna().tail(50)
            if len(valid) > 0:
                aligned_close = close.loc[valid.index]
                persistence = float((aligned_close > valid).mean()) * 100
            else:
                persistence = 50.0

            # Realized vol: 20-day annualized
            returns = close.pct_change()
            rolling_vol_series = returns.rolling(20).std() * (252 ** 0.5)
            rolling_vol = float(rolling_vol_series.iloc[-1])

            # Adaptive vol: where does current vol sit vs the last year?
            vol_history = rolling_vol_series.dropna().tail(252)
            if len(vol_history) > 20:
                vol_pctile = float((vol_history < rolling_vol).mean()) * 100
            else:
                vol_pctile = 50.0

            # Classify: VOLATILE when vol is in the top quintile vs own history
            # (replaces hardcoded 30% threshold that was tuned for US markets).
            if vol_pctile >= 80:
                regime = MarketRegime.VOLATILE
            elif price > sma200 and price > sma50 and adx_val >= 20:
                regime = MarketRegime.BULL
            elif price < sma200 and price < sma50 and adx_val >= 20:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAYS

            rules = self.REGIME_RULES[regime]
            return RegimeReading(
                regime=regime,
                benchmark_vs_sma200_pct=round(vs_sma200, 4),
                breadth_score=round(breadth_score if breadth_score is not None else 50.0, 1),
                trend_strength=round(adx_val, 1),
                trend_persistence=round(persistence, 1),
                realized_vol_annualized=round(rolling_vol, 4),
                vol_percentile=round(vol_pctile, 1),
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
            trend_persistence=50,
            realized_vol_annualized=0,
            vol_percentile=50,
            description="Could not determine regime — defaulting to SIDEWAYS (conservative)",
            new_long_allowed=True,
            new_short_allowed=False,
            position_size_multiplier=0.75,
        )
