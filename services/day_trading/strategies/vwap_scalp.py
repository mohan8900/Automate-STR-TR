"""VWAP Scalp strategy — buys on VWAP cross-over, sells on cross-under."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from services.day_trading.signals import IntradaySignal, IntradayPosition
from services.day_trading.indicators.intraday_indicators import IntradayTechnicals
from services.day_trading.strategies.base import IntradayStrategy
from core.logger import get_logger

logger = get_logger(__name__)


class VWAPScalpStrategy(IntradayStrategy):
    """Scalp entries around VWAP with volume and momentum confirmation."""

    name: str = "VWAP Scalp"

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
        if len(candles_1m) < 2:
            return self._pass_signal(symbol, "Not enough 1m candles")

        current_price = float(candles_1m["close"].iloc[-1])
        prev_close = float(candles_1m["close"].iloc[-2])
        vwap = technicals.vwap
        atr = technicals.atr
        volume_ratio = technicals.relative_volume

        # ---- BUY scoring ----
        buy_score = 0.0
        buy_reasons: list[str] = []

        # Price crossed above VWAP from below
        if prev_close < vwap and current_price > vwap:
            buy_score += 0.30
            buy_reasons.append("VWAP cross-over")

        # Strong relative volume
        if volume_ratio > 1.5:
            buy_score += 0.20
            buy_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        # RSI not overbought
        if 35 <= technicals.rsi_7 <= 55:
            buy_score += 0.20
            buy_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        # Near micro support
        if technicals.micro_supports and _near_level(current_price, technicals.micro_supports, pct=0.3):
            buy_score += 0.15
            buy_reasons.append("near support")

        # Short-term EMA bullish
        if technicals.ema_5 > technicals.ema_9:
            buy_score += 0.15
            buy_reasons.append("EMA5>EMA9")

        if buy_score >= 0.65:
            stop = current_price - 1.5 * atr
            target = current_price + 2.0 * atr
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
                vwap=vwap,
                atr_1m=atr,
                volume_ratio=volume_ratio,
                expected_hold_minutes=15,
                reason="; ".join(buy_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        # ---- SELL scoring ----
        sell_score = 0.0
        sell_reasons: list[str] = []

        # Price crossed below VWAP from above
        if prev_close > vwap and current_price < vwap:
            sell_score += 0.30
            sell_reasons.append("VWAP cross-under")

        if volume_ratio > 1.5:
            sell_score += 0.20
            sell_reasons.append(f"vol_ratio={volume_ratio:.2f}")

        if technicals.rsi_7 > 65:
            sell_score += 0.20
            sell_reasons.append(f"rsi7={technicals.rsi_7:.1f}")

        if technicals.micro_resistances and _near_level(current_price, technicals.micro_resistances, pct=0.3):
            sell_score += 0.15
            sell_reasons.append("near resistance")

        if technicals.ema_5 < technicals.ema_9:
            sell_score += 0.15
            sell_reasons.append("EMA5<EMA9")

        if sell_score >= 0.65:
            stop = current_price + 1.5 * atr
            target = current_price - 2.0 * atr
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
                vwap=vwap,
                atr_1m=atr,
                volume_ratio=volume_ratio,
                expected_hold_minutes=15,
                reason="; ".join(sell_reasons),
                timestamp=datetime.now(),
                candle_timeframe="1m",
            )

        return self._pass_signal(symbol, "No VWAP scalp setup")

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def should_exit(
        self,
        position: IntradayPosition,
        technicals: IntradayTechnicals,
        candles_1m: pd.DataFrame,
    ) -> tuple[bool, str]:
        atr = technicals.atr
        vwap = technicals.vwap
        price = position.current_price

        if position.side == "long":
            # Price dropped well below VWAP
            if price < vwap - atr:
                return True, "Price below VWAP - ATR"

            # Overbought
            if technicals.rsi_7 > 75:
                return True, f"RSI overbought ({technicals.rsi_7:.1f})"

            # Target reached (unrealised PnL exceeds target distance)
            target_dist_pct = abs(position.target_price - position.entry_price) / position.entry_price
            unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.qty) if position.qty else 0
            if unrealized_pnl_pct > target_dist_pct:
                return True, "Target reached"

        if position.side == "short":
            if price > vwap + atr:
                return True, "Price above VWAP + ATR"
            if technicals.rsi_7 < 25:
                return True, f"RSI oversold ({technicals.rsi_7:.1f})"
            target_dist_pct = abs(position.entry_price - position.target_price) / position.entry_price
            unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.qty) if position.qty else 0
            if unrealized_pnl_pct > target_dist_pct:
                return True, "Target reached"

        return False, ""


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------
def _near_level(price: float, levels: list[float], pct: float = 0.3) -> bool:
    """Return True if *price* is within *pct*% of any level in *levels*."""
    threshold = pct / 100.0
    return any(abs(price - lvl) / price <= threshold for lvl in levels)
