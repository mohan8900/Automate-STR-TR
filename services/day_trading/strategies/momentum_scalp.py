"""Momentum Scalp strategy — rides short bursts of consecutive candles."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from services.day_trading.signals import IntradaySignal, IntradayPosition
from services.day_trading.indicators.intraday_indicators import IntradayTechnicals
from services.day_trading.strategies.base import IntradayStrategy
from core.logger import get_logger

logger = get_logger(__name__)


class MomentumScalpStrategy(IntradayStrategy):
    """Enter on strong consecutive-candle runs backed by volume."""

    name: str = "Momentum Scalp"

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def generate_signal(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        technicals: IntradayTechnicals,
        current_position: Optional[IntradayPosition] = None,
    ) -> IntradaySignal:
        if len(candles_1m) < 5:
            return self._pass_signal(symbol, "Not enough 1m candles for momentum check")

        last5 = candles_1m.iloc[-5:]
        current_price = float(candles_1m["close"].iloc[-1])
        atr = technicals.atr
        volume_ratio = technicals.relative_volume

        green_streak = _count_consecutive_green(last5)
        red_streak = _count_consecutive_red(last5)

        # ---- BUY scoring ----
        buy_score = 0.0
        buy_reasons: list[str] = []

        if green_streak >= 3:
            buy_score += 0.25
            buy_reasons.append(f"{green_streak} green candles")

        if volume_ratio > 2.0:
            buy_score += 0.20
            buy_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        if current_price > technicals.ema_5 and current_price > technicals.ema_9:
            buy_score += 0.20
            buy_reasons.append("above EMA5 & EMA9")

        if 50 <= technicals.rsi_7 <= 75:
            buy_score += 0.15
            buy_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        # Strong current candle body
        cur = candles_1m.iloc[-1]
        body = abs(float(cur["close"]) - float(cur["open"]))
        if atr > 0 and body > 0.5 * atr:
            buy_score += 0.10
            buy_reasons.append("strong candle body")

        if current_price > technicals.vwap:
            buy_score += 0.10
            buy_reasons.append("above VWAP")

        if buy_score >= 0.65:
            # Stop at low of first green candle in the streak
            first_green_idx = len(last5) - green_streak
            stop = float(last5.iloc[first_green_idx]["low"])
            risk = current_price - stop
            target = current_price + 2.0 * risk if risk > 0 else current_price + 2.0 * atr
            rr = (target - current_price) / risk if risk > 0 else 0.0
            return IntradaySignal(
                symbol=symbol,
                action="BUY",
                strength=round(buy_score, 2),
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss_price=round(stop, 2),
                target_price=round(target, 2),
                risk_reward_ratio=round(rr, 2),
                vwap=technicals.vwap,
                atr_1m=atr,
                volume_ratio=volume_ratio,
                expected_hold_minutes=10,
                reason="; ".join(buy_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        # ---- SELL scoring ----
        sell_score = 0.0
        sell_reasons: list[str] = []

        if red_streak >= 3:
            sell_score += 0.25
            sell_reasons.append(f"{red_streak} red candles")

        if volume_ratio > 2.0:
            sell_score += 0.20
            sell_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        if current_price < technicals.ema_5 and current_price < technicals.ema_9:
            sell_score += 0.20
            sell_reasons.append("below EMA5 & EMA9")

        if 25 <= technicals.rsi_7 <= 50:
            sell_score += 0.15
            sell_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        if atr > 0 and body > 0.5 * atr:
            sell_score += 0.10
            sell_reasons.append("strong candle body")

        if current_price < technicals.vwap:
            sell_score += 0.10
            sell_reasons.append("below VWAP")

        if sell_score >= 0.65:
            first_red_idx = len(last5) - red_streak
            stop = float(last5.iloc[first_red_idx]["high"])
            risk = stop - current_price
            target = current_price - 2.0 * risk if risk > 0 else current_price - 2.0 * atr
            rr = (current_price - target) / risk if risk > 0 else 0.0
            return IntradaySignal(
                symbol=symbol,
                action="SELL",
                strength=round(sell_score, 2),
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss_price=round(stop, 2),
                target_price=round(target, 2),
                risk_reward_ratio=round(rr, 2),
                vwap=technicals.vwap,
                atr_1m=atr,
                volume_ratio=volume_ratio,
                expected_hold_minutes=10,
                reason="; ".join(sell_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        return self._pass_signal(symbol, "No momentum setup")

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def should_exit(
        self,
        position: IntradayPosition,
        technicals: IntradayTechnicals,
        candles_1m: pd.DataFrame,
    ) -> tuple[bool, str]:
        if len(candles_1m) < 1:
            return False, ""

        last_candle = candles_1m.iloc[-1]
        is_red = float(last_candle["close"]) < float(last_candle["open"])

        # Momentum break: counter-candle with strong volume
        if position.side == "long" and is_red and technicals.relative_volume > 1.5:
            return True, "Momentum break — red candle with high volume"
        if position.side == "short" and not is_red and technicals.relative_volume > 1.5:
            return True, "Momentum break — green candle with high volume"

        # RSI exhaustion
        if position.side == "long" and technicals.rsi_7 > 80:
            return True, f"RSI exhausted ({technicals.rsi_7:.1f})"
        if position.side == "short" and technicals.rsi_7 < 20:
            return True, f"RSI oversold exhaustion ({technicals.rsi_7:.1f})"

        # Stalled momentum
        if position.minutes_held > 30:
            gain_pct = position.unrealized_pnl / (position.entry_price * position.qty) if position.qty else 0
            if gain_pct < 0.003:
                return True, "Momentum stalled (>30 min, <0.3% gain)"

        return False, ""


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------
def _count_consecutive_green(df: pd.DataFrame) -> int:
    """Count consecutive green candles from the end of *df*."""
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if float(df.iloc[i]["close"]) > float(df.iloc[i]["open"]):
            count += 1
        else:
            break
    return count


def _count_consecutive_red(df: pd.DataFrame) -> int:
    """Count consecutive red candles from the end of *df*."""
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if float(df.iloc[i]["close"]) < float(df.iloc[i]["open"]):
            count += 1
        else:
            break
    return count
