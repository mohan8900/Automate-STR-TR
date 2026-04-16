"""
LSTM Neural Network predictor for stock price direction.
Captures temporal patterns that tree-based models (XGBoost/RF) miss.
Uses walk-forward validation to prevent overfitting.
"""
from __future__ import annotations

import joblib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.logger import get_logger
from services.swing_trading.prediction.ensemble_model import PredictionResult
from services.swing_trading.prediction.feature_engineer import FeatureEngineer

log = get_logger("lstm_model")

MODEL_DIR = Path("data/models")

# Guard torch import — system works without it
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not installed — LSTM predictions disabled. Install: pip install torch")


class _LSTMNetwork(nn.Module):
    """2-layer LSTM with binary classification head."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


class LSTMPredictor:
    """
    LSTM-based stock direction predictor.
    Same interface as EnsemblePredictor — returns PredictionResult.
    Falls back to neutral prediction if PyTorch is not installed.
    """

    def __init__(
        self,
        target_days: int = 5,
        train_window_days: int = 504,
        retrain_interval_days: int = 30,
        sequence_length: int = 30,
    ):
        self.target_days = target_days
        self.train_window = train_window_days
        self.retrain_interval = retrain_interval_days
        self.seq_len = sequence_length
        self.feature_engineer = FeatureEngineer()

        self._models: dict[str, object] = {}  # symbol -> trained _LSTMNetwork
        self._feature_names: list[str] = []
        self._last_train_date: dict[str, datetime] = {}
        self._metrics: dict[str, dict] = {}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def predict(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamental_data: Optional[dict] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """Predict direction. Trains if needed. Returns neutral if torch unavailable."""
        if not HAS_TORCH:
            return self._neutral(symbol)

        if self._needs_retrain(symbol):
            log.info(f"Training LSTM for {symbol}...")
            self._train(symbol, df, fundamental_data, benchmark_df)

        model = self._models.get(symbol)
        if model is None:
            model = self._load_model(symbol)
            if model is None:
                return self._neutral(symbol)

        # Build features for prediction (no target)
        features = self._build_features(df, fundamental_data, benchmark_df, include_target=False)
        if features is None or len(features) < self.seq_len:
            return self._neutral(symbol)

        # Prepare last sequence
        available = [f for f in self._feature_names if f in features.columns]
        if not available:
            return self._neutral(symbol)

        X = features[available].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Take last seq_len rows as input sequence
        seq = X[-self.seq_len:]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)

        model.eval()
        with torch.no_grad():
            prob = float(model(seq_tensor).item())

        # Direction
        if prob > 0.55:
            direction = "BULLISH"
        elif prob < 0.45:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        metrics = self._metrics.get(symbol, {})
        accuracy = metrics.get("accuracy", 0.5)

        return PredictionResult(
            symbol=symbol,
            direction=direction,
            probability=round(prob, 4),
            confidence=round(abs(prob - 0.5) * 2, 4),  # 0 at 0.5, 1 at 0.0/1.0
            feature_importance={},  # LSTM doesn't have simple feature importance
            model_accuracy=round(accuracy, 4),
            prediction_horizon_days=self.target_days,
        )

    def _train(
        self,
        symbol: str,
        df: pd.DataFrame,
        fundamental_data: Optional[dict] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train LSTM with walk-forward validation."""
        if not HAS_TORCH:
            return

        features = self._build_features(df, fundamental_data, benchmark_df, include_target=True)
        if features is None or len(features) < self.seq_len + 100:
            log.warning(f"Insufficient data for LSTM training on {symbol}: {len(features) if features is not None else 0}")
            return

        target_cols = ["target", "target_return"]
        feature_cols = [c for c in features.columns if c not in target_cols]
        self._feature_names = feature_cols

        X_all = features[feature_cols].values
        y_all = features["target"].values
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # Walk-forward split: train on first 80%, validate on last 20%
        split = int(len(X_all) * 0.8)
        X_train_raw, X_val_raw = X_all[:split], X_all[split:]
        y_train_raw, y_val_raw = y_all[:split], y_all[split:]

        # Build sequences
        X_train, y_train = self._make_sequences(X_train_raw, y_train_raw)
        X_val, y_val = self._make_sequences(X_val_raw, y_val_raw)

        if len(X_train) < 50 or len(X_val) < 10:
            log.warning(f"Not enough sequences for {symbol}")
            return

        input_size = X_train.shape[2]
        model = _LSTMNetwork(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # Convert to tensors
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        # Train with early stopping
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = float(criterion(val_pred, y_val_t))

            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Restore best model
        if "best_state" in dir():
            model.load_state_dict(best_state)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).numpy()
        val_preds = (val_probs > 0.5).astype(int)
        accuracy = float(np.mean(val_preds == y_val))

        self._models[symbol] = model
        self._last_train_date[symbol] = datetime.now()
        self._metrics[symbol] = {"accuracy": accuracy}
        self._save_model(symbol, model)

        log.info(f"LSTM trained for {symbol}: accuracy={accuracy:.1%}, epochs={epoch+1}")

    def _make_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from feature matrix."""
        sequences, targets = [], []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            targets.append(y[i])
        if not sequences:
            return np.array([]), np.array([])
        return np.array(sequences), np.array(targets)

    def _build_features(self, df, fundamental_data, benchmark_df, include_target):
        """Build feature matrix using shared FeatureEngineer."""
        try:
            features = self.feature_engineer.build_features(
                df, target_days=self.target_days, include_target=include_target
            )
            if fundamental_data:
                features = self.feature_engineer.add_fundamental_features(features, fundamental_data)
            if benchmark_df is not None:
                features = self.feature_engineer.add_market_context_features(features, benchmark_df)
            return features
        except Exception as e:
            log.warning(f"Feature building failed: {e}")
            return None

    def _needs_retrain(self, symbol: str) -> bool:
        if symbol not in self._models:
            model = self._load_model(symbol)
            if model:
                self._models[symbol] = model
                return False
            return True
        last = self._last_train_date.get(symbol)
        if not last:
            return True
        return (datetime.now() - last).days >= self.retrain_interval

    def _save_model(self, symbol: str, model) -> None:
        try:
            safe = symbol.replace(".", "_").replace("/", "_")
            path = MODEL_DIR / f"{safe}_lstm.joblib"
            data = {
                "state_dict": model.state_dict(),
                "input_size": model.lstm.input_size,
                "feature_names": self._feature_names,
                "metrics": self._metrics.get(symbol, {}),
                "trained_at": datetime.now().isoformat(),
            }
            joblib.dump(data, path)
        except Exception as e:
            log.warning(f"Could not save LSTM model for {symbol}: {e}")

    def _load_model(self, symbol: str):
        if not HAS_TORCH:
            return None
        try:
            safe = symbol.replace(".", "_").replace("/", "_")
            path = MODEL_DIR / f"{safe}_lstm.joblib"
            if not path.exists():
                return None
            data = joblib.load(path)
            model = _LSTMNetwork(input_size=data["input_size"])
            model.load_state_dict(data["state_dict"])
            self._feature_names = data.get("feature_names", [])
            self._metrics[symbol] = data.get("metrics", {})
            trained_at = data.get("trained_at", "")
            if trained_at:
                self._last_train_date[symbol] = datetime.fromisoformat(trained_at)
            return model
        except Exception as e:
            log.warning(f"Could not load LSTM model for {symbol}: {e}")
            return None

    def _neutral(self, symbol: str) -> PredictionResult:
        return PredictionResult(
            symbol=symbol,
            direction="NEUTRAL",
            probability=0.5,
            confidence=0.0,
            feature_importance={},
            model_accuracy=0.5,
            prediction_horizon_days=self.target_days,
        )
