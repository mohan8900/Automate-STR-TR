"""XGBoost intraday direction model with joblib persistence."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from services.day_trading.ml.feature_engineer import IntradayFeatureEngineer
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntradayPrediction:
    direction: str          # BULLISH | BEARISH | NEUTRAL
    probability: float      # 0-1
    confidence: float       # 0-1
    features_used: int


class IntradayXGBoostModel:
    """Thin wrapper around XGBClassifier for 1-minute direction prediction."""

    def __init__(
        self,
        model_path: str = "data/models/intraday_xgb.joblib",
        retrain_interval_days: int = 7,
    ) -> None:
        self.model_path = Path(model_path)
        self.retrain_interval_days = retrain_interval_days
        self._model = None
        self._feature_engineer = IntradayFeatureEngineer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> IntradayPrediction:
        """Return a directional prediction for the current bar."""
        features = self._feature_engineer.build_features(df_1m, df_5m, include_target=False)
        if features.empty:
            logger.warning("Feature DataFrame empty — returning neutral prediction")
            return self._neutral()

        last_row = features.iloc[[-1]]

        if self._model is None:
            self._load_model()

        if self._model is None:
            logger.info("No trained model available — returning neutral prediction")
            return self._neutral()

        try:
            prob = float(self._model.predict_proba(last_row)[0][1])  # P(up)
        except Exception:
            logger.exception("Prediction failed")
            return self._neutral()

        direction = _prob_to_direction(prob)
        confidence = abs(prob - 0.5) * 2.0

        return IntradayPrediction(
            direction=direction,
            probability=round(prob, 4),
            confidence=round(confidence, 4),
            features_used=last_row.shape[1],
        )

    def train(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> None:
        """Train (or retrain) the model on the supplied candle data."""
        from xgboost import XGBClassifier

        features = self._feature_engineer.build_features(df_1m, df_5m, include_target=True)
        if features.empty or "target" not in features.columns:
            logger.error("Cannot train — no usable features/target")
            return

        X = features.drop(columns=["target", "target_return"])
        y = features["target"]

        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_acc = float((clf.predict(X_val) == y_val).mean())
        logger.info(f"XGBoost trained — val accuracy: {val_acc:.4f} ({len(X_train)} train / {len(X_val)} val)")

        self._model = clf
        self._save_model()

    def should_retrain(self) -> bool:
        """True when the saved model is older than *retrain_interval_days*."""
        if not self.model_path.exists():
            return True
        age_days = (time.time() - self.model_path.stat().st_mtime) / 86400
        return age_days >= self.retrain_interval_days

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if not self.model_path.exists():
            self._model = None
            return
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception:
            logger.exception("Failed to load model")
            self._model = None

    def _save_model(self) -> None:
        try:
            import joblib
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception:
            logger.exception("Failed to save model")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neutral() -> IntradayPrediction:
        return IntradayPrediction(
            direction="NEUTRAL",
            probability=0.5,
            confidence=0.0,
            features_used=0,
        )


def _prob_to_direction(prob: float) -> str:
    if prob > 0.55:
        return "BULLISH"
    if prob < 0.45:
        return "BEARISH"
    return "NEUTRAL"
