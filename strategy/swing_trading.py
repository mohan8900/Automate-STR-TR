"""
Swing Trading Strategy — optimized for small capital and 3-15 day holds.
Best strategy for retail algo traders with limited capital.

Entry: RSI oversold + MACD momentum shift + trend confirmation
Exit: ATR-based stops, take-profit targets, time-based exit
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import Strategy, TradeSignal
from core.logger import get_logger

log = get_logger("swing_strategy")


class SwingTradingStrategy(Strategy):
    """
    Swing trading strategy for 3-15 day holding periods.
    Combines mean reversion (oversold bounce) with trend following (MACD + MA).
    """

    name = "Swing Trading"

    def __init__(
        self,
        rsi_oversold: float = 35,
        rsi_overbought: float = 70,
        min_volume_ratio: float = 1.2,
        atr_stop_mult: float = 2.0,
        profit_target_mult: float = 2.5,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_volume_ratio = min_volume_ratio
        self.atr_stop_mult = atr_stop_mult
        self.profit_target_mult = profit_target_mult

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
    ) -> TradeSignal:
        if len(df) < 50:
            return self._pass_signal(symbol, "Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        price = float(close.iloc[-1])

        # ── Compute indicators ────────────────────────────────────────────
        rsi = self._rsi(close, 14)
        rsi_value = float(rsi.iloc[-1])

        sma50 = close.rolling(50).mean()
        in_uptrend = price > float(sma50.iloc[-1])

        # MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal_line
        hist_now = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        hist_2 = float(histogram.iloc[-3]) if len(histogram) >= 3 else hist_prev

        # Volume confirmation
        vol_sma = volume.rolling(20).mean()
        rel_vol = float(volume.iloc[-1]) / float(vol_sma.iloc[-1]) if float(vol_sma.iloc[-1]) > 0 else 1.0

        # ATR for stop placement
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / price

        # Bollinger Band position
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_lower = bb_mid - 2 * bb_std
        bb_pos = float((price - bb_lower.iloc[-1]) / (4 * bb_std.iloc[-1])) if float(bb_std.iloc[-1]) > 0 else 0.5

        # ── EXIT signal for existing positions ────────────────────────────
        if current_position:
            return self._check_exit(
                symbol, price, rsi_value, hist_now, current_position
            )

        # ── ENTRY signal scoring ──────────────────────────────────────────
        buy_score = 0.0
        reasons = []

        # 1. RSI oversold bounce (strongest signal)
        if rsi_value < self.rsi_oversold:
            rsi_prev = float(rsi.iloc[-2])
            if rsi_value > rsi_prev:  # RSI turning up from oversold
                buy_score += 0.30
                reasons.append(f"RSI bounce from {rsi_value:.0f}")
        elif rsi_value < 45:
            buy_score += 0.10  # Neutral-low RSI, room to run

        # 2. MACD momentum shift
        if hist_prev <= 0 and hist_now > 0:
            buy_score += 0.25  # Bullish crossover
            reasons.append("MACD bullish crossover")
        elif hist_now > hist_prev > hist_2:
            buy_score += 0.15  # Accelerating momentum
            reasons.append("MACD accelerating")

        # 3. Trend alignment
        if in_uptrend:
            buy_score += 0.15
            reasons.append("Price above SMA50")

            # EMA 9/21 alignment
            ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            if ema9 > ema21:
                buy_score += 0.05
                reasons.append("Short-term trend bullish")
        else:
            buy_score -= 0.10  # Penalty for downtrend

        # 4. Volume confirmation
        if rel_vol >= self.min_volume_ratio:
            buy_score += 0.10
            reasons.append(f"Volume {rel_vol:.1f}x avg")

        # 5. Bollinger Band near lower band (mean reversion opportunity)
        if bb_pos < 0.25:
            buy_score += 0.10
            reasons.append("Near Bollinger lower band")

        # 6. Overbought penalty
        if rsi_value > self.rsi_overbought:
            buy_score -= 0.20
            reasons.append(f"RSI overbought ({rsi_value:.0f})")

        # ── Generate signal ───────────────────────────────────────────────
        stop_pct = min(0.08, max(0.03, atr_pct * self.atr_stop_mult))
        tp_pct = stop_pct * self.profit_target_mult

        if buy_score >= 0.55:
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                strength=round(min(1.0, buy_score), 3),
                strategy_name=self.name,
                entry_price=price,
                stop_loss_pct=round(stop_pct, 4),
                take_profit_pct=round(tp_pct, 4),
                holding_period_days=10,
                reason=" | ".join(reasons),
            )

        return self._pass_signal(symbol, f"Score {buy_score:.2f} below threshold")

    def _check_exit(
        self,
        symbol: str,
        price: float,
        rsi: float,
        macd_hist: float,
        position: dict,
    ) -> TradeSignal:
        """Check if an existing position should be exited."""
        entry_price = position.get("avg_cost", price)
        gain_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
        days_held = position.get("days_held", 0)

        # Exit: RSI overbought (take profit)
        if rsi > 75 and gain_pct > 0.03:
            return TradeSignal(
                symbol=symbol, action="SELL", strength=0.8,
                strategy_name=self.name, entry_price=price,
                reason=f"RSI overbought ({rsi:.0f}) with {gain_pct:.1%} gain",
            )

        # Exit: MACD turning bearish with profit
        if macd_hist < 0 and gain_pct > 0.02:
            return TradeSignal(
                symbol=symbol, action="SELL", strength=0.7,
                strategy_name=self.name, entry_price=price,
                reason=f"MACD bearish with {gain_pct:.1%} gain",
            )

        # Time-based exit with minimal gain
        if days_held > 15 and gain_pct < 0.01:
            return TradeSignal(
                symbol=symbol, action="SELL", strength=0.6,
                strategy_name=self.name, entry_price=price,
                reason=f"Time stop: {days_held} days, {gain_pct:.1%} gain",
            )

        return TradeSignal(
            symbol=symbol, action="HOLD", strength=0.5,
            strategy_name=self.name, entry_price=price,
            reason="Hold — no exit trigger",
        )

    def is_suitable_regime(self, regime: str) -> bool:
        return regime in ("BULL", "SIDEWAYS")

    def _pass_signal(self, symbol: str, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=symbol, action="PASS", strength=0.0,
            strategy_name=self.name, reason=reason,
        )

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)
