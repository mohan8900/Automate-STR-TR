"""
Market-level prediction — predicts overall market direction.
Used to gate individual stock predictions (don't go long in a bear market).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from config.watchlists import BENCHMARK
from data.price_feed import PriceFeed
from services.swing_trading.prediction.ensemble_model import EnsemblePredictor, PredictionResult
from core.logger import get_logger

log = get_logger("market_predictor")


@dataclass
class MarketPrediction:
    benchmark: str
    direction: str              # BULLISH | BEARISH | NEUTRAL
    probability: float          # Probability of up move
    confidence: float
    breadth_bullish_pct: float  # % of watchlist stocks predicted bullish
    recommended_exposure: float  # 0.0 to 1.0 — suggested portfolio allocation to equities


class MarketPredictor:
    """
    Predicts market-level direction using benchmark index analysis.
    Combines benchmark model prediction with market breadth.
    """

    def __init__(self, exchange: str = "US"):
        self.exchange = exchange
        self.benchmark_symbol = BENCHMARK.get(exchange, "SPY")
        self.predictor = EnsemblePredictor(target_days=5)
        self.price_feed = PriceFeed(exchange=exchange)

    def predict_market(self) -> MarketPrediction:
        """Predict overall market direction."""
        # Predict benchmark direction
        try:
            bench_df = self.price_feed.get_historical(
                self.benchmark_symbol, period="2y", interval="1d"
            )
            bench_prediction = self.predictor.predict(
                self.benchmark_symbol, bench_df
            )
        except Exception as e:
            log.warning(f"Market prediction failed: {e}")
            bench_prediction = PredictionResult(
                symbol=self.benchmark_symbol,
                direction="NEUTRAL",
                probability=0.5,
                confidence=0.0,
                feature_importance={},
                model_accuracy=0.5,
                prediction_horizon_days=5,
            )

        # Calculate recommended exposure based on probability
        prob = bench_prediction.probability
        if prob > 0.65:
            exposure = 1.0  # Full exposure
        elif prob > 0.55:
            exposure = 0.75
        elif prob > 0.45:
            exposure = 0.5  # Neutral - half exposure
        elif prob > 0.35:
            exposure = 0.25
        else:
            exposure = 0.1  # Minimal exposure in bearish prediction

        return MarketPrediction(
            benchmark=self.benchmark_symbol,
            direction=bench_prediction.direction,
            probability=bench_prediction.probability,
            confidence=bench_prediction.confidence,
            breadth_bullish_pct=0.0,  # Filled by caller with stock-level predictions
            recommended_exposure=exposure,
        )

    def predict_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamental_data: Optional[dict] = None,
    ) -> PredictionResult:
        """Predict individual stock direction with benchmark context."""
        benchmark_df = None
        try:
            benchmark_df = self.price_feed.get_historical(
                self.benchmark_symbol, period="2y", interval="1d"
            )
        except Exception:
            pass

        return self.predictor.predict(
            symbol=symbol,
            df=df,
            fundamental_data=fundamental_data,
            benchmark_df=benchmark_df,
        )
