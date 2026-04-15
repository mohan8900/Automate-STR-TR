"""
Momentum Strategy — buys stocks with strong recent performance.
Research shows momentum works well in Indian markets (Nifty momentum index
historically outperforms Nifty 50).

Entry: Strong 1-3 month returns + volume confirmation + trend alignment
Exit: Momentum loss or trend breakdown
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from core.logger import get_logger

log = get_logger("momentum_strategy")


class MomentumStrategy(Strategy):
    """
    Momentum strategy — ride winners, cut losers.
    Buys stocks showing strong upward momentum with volume confirmation.
    """

    name = "Momentum"

    def __init__(
        self,
        momentum_lookback: int = 20,
        min_momentum_pct: float = 0.05,
        min_volume_ratio: float = 1.3,
    ):
        self.lookback = momentum_lookback
        self.min_momentum = min_momentum_pct
        self.min_vol_ratio = min_volume_ratio

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
    ) -> TradeSignal:
        if len(df) < 60:
            return self._pass(symbol, "Insufficient data")

        close = df["close"]
        volume = df["volume"]
        price = float(close.iloc[-1])

        # Momentum at multiple timeframes
        mom_5d = float(close.iloc[-1] / close.iloc[-6] - 1)
        mom_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
        mom_60d = float(close.iloc[-1] / close.iloc[-min(61, len(close))] - 1)

        # Trend: price above key MAs
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        above_sma20 = price > sma20
        above_sma50 = price > sma50

        # Volume surge
        vol_sma = float(volume.rolling(20).mean().iloc[-1])
        rel_vol = float(volume.iloc[-1]) / vol_sma if vol_sma > 0 else 1.0

        # ADX-like trend strength
        returns = close.pct_change().tail(20)
        trend_strength = abs(float(returns.mean())) / (float(returns.std()) + 1e-10)

        # Rate of change acceleration
        roc_5 = mom_5d
        roc_10 = float(close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else 0
        roc_accelerating = roc_5 > roc_10 / 2  # Recent momentum stronger

        # Near 52-week high (breakout potential)
        high_52w = float(df["high"].max())
        near_high = (price / high_52w) > 0.95

        # ── Score the signal ──────────────────────────────────────────────
        score = 0.0
        reasons = []

        if mom_20d >= self.min_momentum:
            score += 0.25
            reasons.append(f"20d momentum {mom_20d:+.1%}")

        if mom_60d >= self.min_momentum * 2:
            score += 0.15
            reasons.append(f"60d momentum {mom_60d:+.1%}")

        if above_sma20 and above_sma50:
            score += 0.20
            reasons.append("Above SMA20 & SMA50")
        elif above_sma50:
            score += 0.10

        if rel_vol >= self.min_vol_ratio:
            score += 0.10
            reasons.append(f"Volume {rel_vol:.1f}x avg")

        if roc_accelerating and mom_5d > 0:
            score += 0.10
            reasons.append("Momentum accelerating")

        if near_high:
            score += 0.10
            reasons.append("Near 52w high (breakout)")

        if trend_strength > 0.15:
            score += 0.10
            reasons.append(f"Strong trend ({trend_strength:.2f})")

        # ATR-based stops
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / price

        stop_pct = min(0.08, max(0.04, atr_pct * 1.5))

        if score >= 0.55:
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                strength=round(min(1.0, score), 3),
                strategy_name=self.name,
                entry_price=price,
                stop_loss_pct=round(stop_pct, 4),
                take_profit_pct=round(stop_pct * 3.0, 4),  # 3:1 R:R for momentum
                holding_period_days=15,
                reason=" | ".join(reasons),
            )

        return self._pass(symbol, f"Momentum score {score:.2f} insufficient")

    def is_suitable_regime(self, regime: str) -> bool:
        return regime in ("BULL",)  # Momentum works best in bull markets

    def _pass(self, symbol: str, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=symbol, action="PASS", strength=0.0,
            strategy_name=self.name, reason=reason,
        )
