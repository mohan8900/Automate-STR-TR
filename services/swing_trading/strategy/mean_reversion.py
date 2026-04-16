"""
Mean Reversion Strategy — buys oversold stocks expecting price to revert to mean.
Works best in sideways/range-bound markets on 1-5 day timeframes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from services.swing_trading.strategy.base import Strategy, TradeSignal
from core.logger import get_logger

log = get_logger("mean_reversion_strategy")


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy — buy when price deviates significantly below its mean.
    Uses Bollinger Bands, RSI, and z-score of returns.
    """

    name = "Mean Reversion"

    def __init__(
        self,
        bb_entry_threshold: float = 0.10,  # Enter when price below 10% of BB range
        rsi_threshold: float = 30,
        zscore_threshold: float = -2.0,
    ):
        self.bb_entry = bb_entry_threshold
        self.rsi_threshold = rsi_threshold
        self.zscore_threshold = zscore_threshold

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
    ) -> TradeSignal:
        if len(df) < 30:
            return self._pass(symbol, "Insufficient data")

        close = df["close"]
        price = float(close.iloc[-1])

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
        bb_pos = (price - float(bb_lower.iloc[-1])) / bb_range if bb_range > 0 else 0.5

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)
        rsi_val = float(rsi.iloc[-1])

        # Z-score of returns (how far from normal is today's price?)
        returns_20d = close.pct_change().rolling(20)
        zscore = (close.pct_change() - returns_20d.mean()) / (returns_20d.std() + 1e-10)
        zscore_val = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0

        # Distance from 20-day mean
        mean_dist = (price - float(bb_mid.iloc[-1])) / float(bb_mid.iloc[-1])

        # ── Score ─────────────────────────────────────────────────────────
        score = 0.0
        reasons = []

        # Bollinger Band oversold
        if bb_pos < self.bb_entry:
            score += 0.30
            reasons.append(f"BB position {bb_pos:.1%} (oversold)")
        elif bb_pos < 0.25:
            score += 0.15

        # RSI oversold
        if rsi_val < self.rsi_threshold:
            score += 0.25
            reasons.append(f"RSI {rsi_val:.0f} (oversold)")
        elif rsi_val < 40:
            score += 0.10

        # Z-score extreme
        if zscore_val < self.zscore_threshold:
            score += 0.20
            reasons.append(f"Z-score {zscore_val:.1f} (extreme)")

        # Mean distance (further below = stronger signal)
        if mean_dist < -0.05:
            score += 0.15
            reasons.append(f"{mean_dist:+.1%} below 20d mean")

        # Volume: low volume selloff suggests panic (mean reversion opportunity)
        volume = df["volume"]
        vol_sma = float(volume.rolling(20).mean().iloc[-1])
        rel_vol = float(volume.iloc[-1]) / vol_sma if vol_sma > 0 else 1.0
        if rel_vol > 1.5 and mean_dist < -0.03:
            score += 0.10
            reasons.append(f"High volume selloff ({rel_vol:.1f}x)")

        # ATR stops (tighter for mean reversion)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        stop_pct = min(0.06, max(0.02, (atr * 1.5) / price))
        # Target: revert to mean
        tp_pct = max(abs(mean_dist), stop_pct * 2.0)

        if score >= 0.55:
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                strength=round(min(1.0, score), 3),
                strategy_name=self.name,
                entry_price=price,
                stop_loss_pct=round(stop_pct, 4),
                take_profit_pct=round(tp_pct, 4),
                holding_period_days=5,  # Short hold for mean reversion
                reason=" | ".join(reasons),
            )

        return self._pass(symbol, f"Mean reversion score {score:.2f} insufficient")

    def is_suitable_regime(self, regime: str) -> bool:
        return regime in ("SIDEWAYS", "BULL", "BEAR")  # Oversold bounces are strong in bear markets too

    def _pass(self, symbol: str, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=symbol, action="PASS", strength=0.0,
            strategy_name=self.name, reason=reason,
        )
