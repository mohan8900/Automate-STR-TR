"""
Ensemble ML model for stock price direction prediction.
Combines XGBoost + Random Forest with walk-forward validation.
Outputs probability of upward movement for the next N days.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.logger import get_logger
from prediction.feature_engineer import FeatureEngineer

log = get_logger("ensemble_model")

# Model storage directory
MODEL_DIR = Path("data/models")


@dataclass
class PredictionResult:
    symbol: str
    direction: str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    probability: float      # 0.0 to 1.0 (probability of upward move)
    confidence: float       # 0.0 to 1.0 (how confident the model is)
    feature_importance: dict[str, float]  # Top contributing features
    model_accuracy: float   # Historical walk-forward accuracy
    prediction_horizon_days: int


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_backtest: float
    win_rate: float
    profit_factor: float
    total_predictions: int
    train_end_date: str


class EnsemblePredictor:
    """
    Ensemble model combining XGBoost and Random Forest.
    Uses walk-forward validation to prevent overfitting.
    """

    def __init__(
        self,
        target_days: int = 5,
        train_window_days: int = 504,  # ~2 years
        retrain_interval_days: int = 30,
    ):
        self.target_days = target_days
        self.train_window = train_window_days
        self.retrain_interval = retrain_interval_days
        self.feature_engineer = FeatureEngineer()

        self._models: dict[str, list] = {}  # symbol -> [model1, model2]
        self._metrics: dict[str, ModelMetrics] = {}
        self._feature_names: list[str] = []
        self._last_train_date: dict[str, datetime] = {}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamental_data: Optional[dict] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """
        Predict direction for a symbol. Trains model if needed.
        Returns PredictionResult with probability and confidence.
        """
        # Check if model needs training/retraining
        needs_training = self._needs_retrain(symbol)
        if needs_training:
            log.info(f"Training model for {symbol}...")
            self._train(symbol, df, fundamental_data, benchmark_df)

        # Build features for the latest bar (no target needed)
        features = self.feature_engineer.build_features(
            df, target_days=self.target_days, include_target=False
        )
        if fundamental_data:
            features = self.feature_engineer.add_fundamental_features(
                features, fundamental_data
            )
        if benchmark_df is not None:
            features = self.feature_engineer.add_market_context_features(
                features, benchmark_df
            )

        if features.empty:
            return self._neutral_prediction(symbol)

        # Use only features the model was trained on
        available_features = [f for f in self._feature_names if f in features.columns]
        if not available_features:
            return self._neutral_prediction(symbol)

        X_latest = features[available_features].iloc[[-1]]

        models = self._models.get(symbol, [])
        if not models:
            # Try loading from disk
            models = self._load_models(symbol)
            if not models:
                return self._neutral_prediction(symbol)

        # Ensemble prediction: average probabilities from all models
        probabilities = []
        for model in models:
            try:
                prob = model.predict_proba(X_latest)[:, 1][0]
                probabilities.append(prob)
            except Exception as e:
                log.warning(f"Model prediction failed for {symbol}: {e}")

        if not probabilities:
            return self._neutral_prediction(symbol)

        avg_prob = np.mean(probabilities)
        # Confidence = how much models agree (inverse of variance)
        if len(probabilities) > 1:
            variance = np.var(probabilities)
            confidence = max(0, 1 - variance * 4)  # Scale: 0 variance=1.0 confidence
        else:
            confidence = 0.5

        # Direction classification
        if avg_prob > 0.55:
            direction = "BULLISH"
        elif avg_prob < 0.45:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Feature importance (from first model, usually XGBoost)
        importance = self._get_feature_importance(models[0], available_features)

        metrics = self._metrics.get(symbol)
        model_accuracy = metrics.accuracy if metrics else 0.5

        return PredictionResult(
            symbol=symbol,
            direction=direction,
            probability=round(avg_prob, 4),
            confidence=round(confidence, 4),
            feature_importance=importance,
            model_accuracy=round(model_accuracy, 4),
            prediction_horizon_days=self.target_days,
        )

    def _train(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamental_data: Optional[dict] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train ensemble models using walk-forward validation."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            log.error("scikit-learn not installed. Run: pip install scikit-learn")
            return

        try:
            from xgboost import XGBClassifier
            has_xgboost = True
        except ImportError:
            has_xgboost = False
            log.warning("XGBoost not installed, using Random Forest only")

        # Build full feature matrix with targets
        features = self.feature_engineer.build_features(
            df, target_days=self.target_days, include_target=True
        )
        if fundamental_data:
            features = self.feature_engineer.add_fundamental_features(
                features, fundamental_data
            )
        if benchmark_df is not None:
            features = self.feature_engineer.add_market_context_features(
                features, benchmark_df
            )

        if len(features) < 100:
            log.warning(f"Insufficient data for {symbol}: {len(features)} samples")
            return

        # Separate features and target
        target_cols = ["target", "target_return"]
        feature_cols = [c for c in features.columns if c not in target_cols]
        self._feature_names = feature_cols

        X = features[feature_cols]
        y = features["target"]

        # Walk-forward split: train on first 80%, validate on last 20%
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle any remaining NaN/inf
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

        models = []

        # Model 1: XGBoost (if available)
        if has_xgboost:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            models.append(xgb)

        # Model 2: Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        models.append(rf)

        # Evaluate on validation set
        metrics = self._evaluate(models, X_val, y_val, features, split_idx)
        self._metrics[symbol] = metrics
        self._models[symbol] = models
        self._last_train_date[symbol] = datetime.now()

        # Save models to disk
        self._save_models(symbol, models)

        log.info(
            f"Model trained for {symbol}: accuracy={metrics.accuracy:.1%} "
            f"precision={metrics.precision:.1%} win_rate={metrics.win_rate:.1%} "
            f"profit_factor={metrics.profit_factor:.2f}"
        )

    def _evaluate(
        self,
        models: list,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        full_features: pd.DataFrame,
        split_idx: int,
    ) -> ModelMetrics:
        """Evaluate ensemble on validation set."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Ensemble predictions
        probas = []
        for model in models:
            probas.append(model.predict_proba(
                X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            )[:, 1])
        avg_proba = np.mean(probas, axis=0)
        y_pred = (avg_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0.5)
        recall = recall_score(y_val, y_pred, zero_division=0.5)
        f1 = f1_score(y_val, y_pred, zero_division=0.5)

        # Simulated P&L on validation period
        val_returns = full_features["target_return"].iloc[split_idx:].values
        predicted_long = avg_proba > 0.55
        strategy_returns = np.where(predicted_long, val_returns, 0)

        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        win_rate = len(wins) / max(1, len(wins) + len(losses))
        total_win = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        profit_factor = total_win / total_loss

        # Sharpe ratio
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std() + 1e-10
        sharpe = (mean_ret / std_ret) * np.sqrt(252 / self.target_days)

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            sharpe_backtest=round(sharpe, 2),
            win_rate=win_rate,
            profit_factor=round(profit_factor, 2),
            total_predictions=len(y_val),
            train_end_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _needs_retrain(self, symbol: str) -> bool:
        """Check if model needs retraining."""
        if symbol not in self._models:
            # Try loading from disk
            models = self._load_models(symbol)
            if models:
                self._models[symbol] = models
                return False
            return True

        last_train = self._last_train_date.get(symbol)
        if not last_train:
            return True

        days_since = (datetime.now() - last_train).days
        return days_since >= self.retrain_interval

    def _get_feature_importance(
        self, model, feature_names: list[str]
    ) -> dict[str, float]:
        """Extract top 10 feature importances."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                pairs = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1], reverse=True
                )
                return {name: round(float(imp), 4) for name, imp in pairs[:10]}
        except Exception:
            pass
        return {}

    def _save_models(self, symbol: str, models: list) -> None:
        """Save trained models to disk."""
        try:
            safe_symbol = symbol.replace(".", "_").replace("/", "_")
            path = MODEL_DIR / f"{safe_symbol}_ensemble.pkl"
            data = {
                "models": models,
                "feature_names": self._feature_names,
                "metrics": self._metrics.get(symbol),
                "trained_at": datetime.now().isoformat(),
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            log.debug(f"Saved model for {symbol} to {path}")
        except Exception as e:
            log.warning(f"Could not save model for {symbol}: {e}")

    def _load_models(self, symbol: str) -> list:
        """Load models from disk."""
        try:
            safe_symbol = symbol.replace(".", "_").replace("/", "_")
            path = MODEL_DIR / f"{safe_symbol}_ensemble.pkl"
            if not path.exists():
                return []
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._feature_names = data.get("feature_names", [])
            self._metrics[symbol] = data.get("metrics")
            trained_at = data.get("trained_at", "")
            if trained_at:
                self._last_train_date[symbol] = datetime.fromisoformat(trained_at)
            log.debug(f"Loaded model for {symbol} from disk")
            return data.get("models", [])
        except Exception as e:
            log.warning(f"Could not load model for {symbol}: {e}")
            return []

    def _neutral_prediction(self, symbol: str) -> PredictionResult:
        return PredictionResult(
            symbol=symbol,
            direction="NEUTRAL",
            probability=0.5,
            confidence=0.0,
            feature_importance={},
            model_accuracy=0.5,
            prediction_horizon_days=self.target_days,
        )

    def get_metrics(self, symbol: str) -> Optional[ModelMetrics]:
        return self._metrics.get(symbol)

    def get_all_metrics(self) -> dict[str, ModelMetrics]:
        return dict(self._metrics)
