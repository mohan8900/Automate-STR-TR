"""Strategy selector — runs enabled strategies and builds consensus."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from services.day_trading.signals import IntradaySignal, IntradayPosition
from services.day_trading.indicators.intraday_indicators import IntradayTechnicals
from services.day_trading.strategies.base import IntradayStrategy
from services.day_trading.strategies.vwap_scalp import VWAPScalpStrategy
from services.day_trading.strategies.orb import ORBStrategy
from services.day_trading.strategies.momentum_scalp import MomentumScalpStrategy
from core.logger import get_logger

logger = get_logger(__name__)

# Registry mapping config names to classes
_STRATEGY_REGISTRY: dict[str, type[IntradayStrategy]] = {
    "vwap_scalp": VWAPScalpStrategy,
    "orb": ORBStrategy,
    "momentum_scalp": MomentumScalpStrategy,
}


class IntradayStrategySelector:
    """Run enabled strategies, count votes, and return a consensus signal."""

    def __init__(self, strategies_enabled: list[str]) -> None:
        self.strategies: list[IntradayStrategy] = []
        for name in strategies_enabled:
            cls = _STRATEGY_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"Unknown strategy '{name}' — skipping")
                continue
            self.strategies.append(cls())
        logger.info(f"Selector initialised with {[s.name for s in self.strategies]}")

    # ------------------------------------------------------------------
    def generate_combined_signal(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        technicals: IntradayTechnicals,
        ml_prediction: Optional[object] = None,
        current_position: Optional[IntradayPosition] = None,
    ) -> IntradaySignal:
        """Run every strategy and return a consensus signal or PASS."""

        signals: list[IntradaySignal] = []
        for strat in self.strategies:
            try:
                sig = strat.generate_signal(
                    symbol, candles_1m, candles_5m, technicals, current_position,
                )
                signals.append(sig)
                logger.debug(f"[{strat.name}] {sig.action} strength={sig.strength:.2f}")
            except Exception:
                logger.exception(f"Strategy {strat.name} raised an exception")

        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]

        buy_count = len(buy_signals)
        sell_count = len(sell_signals)

        logger.info(
            f"{symbol} votes — BUY: {buy_count}, SELL: {sell_count}, "
            f"PASS: {len(signals) - buy_count - sell_count}"
        )

        # --- ML confirmation helper ---
        ml_dir = None
        if ml_prediction is not None:
            ml_dir = getattr(ml_prediction, "direction", None)

        # --- Consensus logic ---
        # A side wins if: 2+ strategies agree, OR 1 strategy with strength > 0.75
        # AND ml_prediction confirms the direction
        best_signal = self._pick_winner(
            buy_signals, sell_signals, buy_count, sell_count, ml_dir, symbol,
        )
        return best_signal

    # ------------------------------------------------------------------
    def _pick_winner(
        self,
        buy_signals: list[IntradaySignal],
        sell_signals: list[IntradaySignal],
        buy_count: int,
        sell_count: int,
        ml_dir: Optional[str],
        symbol: str,
    ) -> IntradaySignal:
        """Determine the winning side and return the strongest signal."""

        buy_qualifies = self._side_qualifies(buy_count, buy_signals, "BULLISH", ml_dir)
        sell_qualifies = self._side_qualifies(sell_count, sell_signals, "BEARISH", ml_dir)

        if buy_qualifies and sell_qualifies:
            # Both sides qualify — pick the one with more votes; tie-break by strength
            if buy_count > sell_count:
                return self._strongest(buy_signals)
            elif sell_count > buy_count:
                return self._strongest(sell_signals)
            else:
                best_buy = self._strongest(buy_signals)
                best_sell = self._strongest(sell_signals)
                return best_buy if best_buy.strength >= best_sell.strength else best_sell

        if buy_qualifies:
            return self._strongest(buy_signals)

        if sell_qualifies:
            return self._strongest(sell_signals)

        # No consensus
        logger.info(f"{symbol}: No strategy consensus — PASS")
        return IntradaySignal(
            symbol=symbol,
            action="PASS",
            strength=0.0,
            strategy_name="selector",
            entry_price=0.0,
            stop_loss_price=0.0,
            target_price=0.0,
            risk_reward_ratio=0.0,
            vwap=0.0,
            atr_1m=0.0,
            volume_ratio=0.0,
            expected_hold_minutes=0,
            reason="No consensus",
        )

    @staticmethod
    def _side_qualifies(
        count: int,
        signals: list[IntradaySignal],
        required_ml_dir: str,
        ml_dir: Optional[str],
    ) -> bool:
        """Check whether a side (BUY / SELL) meets the consensus bar."""
        if count == 0:
            return False

        # 2+ strategies agree — enough on its own
        if count >= 2:
            return True

        # 1 strategy with strength > 0.75 AND ML confirmation
        strongest = max(signals, key=lambda s: s.strength)
        if strongest.strength > 0.75 and ml_dir == required_ml_dir:
            return True

        return False

    @staticmethod
    def _strongest(signals: list[IntradaySignal]) -> IntradaySignal:
        return max(signals, key=lambda s: s.strength)
