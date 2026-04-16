"""Opening Range Breakout (ORB) strategy."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from services.day_trading.signals import IntradaySignal, IntradayPosition
from services.day_trading.indicators.intraday_indicators import IntradayTechnicals
from services.day_trading.strategies.base import IntradayStrategy
from core.logger import get_logger

logger = get_logger(__name__)


class ORBStrategy(IntradayStrategy):
    """Trade breakouts of the opening range (first 15 minutes)."""

    name: str = "ORB"

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
        orb = technicals.orb

        # ORB must be calculated and valid
        if not orb.is_valid:
            return self._pass_signal(symbol, "ORB not valid yet")

        # ORB loses reliability in the afternoon (IST hour >= 12)
        if len(candles_1m) > 0:
            last_ts = candles_1m.index[-1] if isinstance(candles_1m.index, pd.DatetimeIndex) else pd.Timestamp(candles_1m["datetime"].iloc[-1] if "datetime" in candles_1m.columns else datetime.now())
            if last_ts.hour >= 12:
                return self._pass_signal(symbol, "Afternoon — ORB unreliable")

        current_price = float(candles_1m["close"].iloc[-1])
        atr = technicals.atr
        volume_ratio = technicals.relative_volume

        # 5m confirmation candle (last 5m candle close)
        candle_5m_close = float(candles_5m["close"].iloc[-1]) if len(candles_5m) > 0 else current_price

        # ---- BUY scoring ----
        buy_score = 0.0
        buy_reasons: list[str] = []

        if current_price > orb.high:
            buy_score += 0.35
            buy_reasons.append("breakout above ORB high")

        if candle_5m_close > orb.high:
            buy_score += 0.25
            buy_reasons.append("5m candle confirms above ORB")

        if volume_ratio > 2.0:
            buy_score += 0.20
            buy_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        if 50 <= technicals.rsi_7 <= 70:
            buy_score += 0.10
            buy_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        range_pct = orb.range_size / current_price if current_price > 0 else 0
        if range_pct < 0.015:
            buy_score += 0.10
            buy_reasons.append(f"tight range ({range_pct:.3f})")

        if buy_score >= 0.65:
            stop = orb.low
            target = current_price + orb.range_size  # measured move
            risk = current_price - stop
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
                expected_hold_minutes=45,
                reason="; ".join(buy_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        # ---- SELL scoring ----
        sell_score = 0.0
        sell_reasons: list[str] = []

        if current_price < orb.low:
            sell_score += 0.35
            sell_reasons.append("breakdown below ORB low")

        if candle_5m_close < orb.low:
            sell_score += 0.25
            sell_reasons.append("5m candle confirms below ORB")

        if volume_ratio > 2.0:
            sell_score += 0.20
            sell_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        if 30 <= technicals.rsi_7 <= 50:
            sell_score += 0.10
            sell_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        if range_pct < 0.015:
            sell_score += 0.10
            sell_reasons.append(f"tight range ({range_pct:.3f})")

        if sell_score >= 0.65:
            stop = orb.high
            target = current_price - orb.range_size
            risk = stop - current_price
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
                expected_hold_minutes=45,
                reason="; ".join(sell_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        return self._pass_signal(symbol, "No ORB breakout")

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def should_exit(
        self,
        position: IntradayPosition,
        technicals: IntradayTechnicals,
        candles_1m: pd.DataFrame,
    ) -> tuple[bool, str]:
        orb = technicals.orb
        price = position.current_price

        # Failed breakout — price back inside ORB range
        if position.side == "long" and orb.low <= price <= orb.high:
            return True, "Failed breakout — price retreated inside ORB"
        if position.side == "short" and orb.low <= price <= orb.high:
            return True, "Failed breakdown — price retreated inside ORB"

        # Stale trade: held > 60 min with tiny gain
        if position.minutes_held > 60:
            gain_pct = position.unrealized_pnl / (position.entry_price * position.qty) if position.qty else 0
            if gain_pct < 0.002:
                return True, "Stale ORB trade (>60 min, <0.2% gain)"

        return False, ""
