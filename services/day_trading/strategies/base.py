"""Abstract base class for intraday trading strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from services.day_trading.signals import IntradaySignal, IntradayPosition
from services.day_trading.indicators.intraday_indicators import IntradayTechnicals
from core.logger import get_logger

logger = get_logger(__name__)


class IntradayStrategy(ABC):
    """Base class that every intraday strategy must extend."""

    name: str = "base_intraday"

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        technicals: IntradayTechnicals,
        current_position: Optional[IntradayPosition] = None,
    ) -> IntradaySignal:
        """Analyse current market data and return an IntradaySignal."""
        ...

    @abstractmethod
    def should_exit(
        self,
        position: IntradayPosition,
        technicals: IntradayTechnicals,
        candles_1m: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Return (should_exit, reason) for an open position."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pass_signal(self, symbol: str, reason: str) -> IntradaySignal:
        """Return a neutral PASS signal with strength 0."""
        return IntradaySignal(
            symbol=symbol,
            action="PASS",
            strength=0.0,
            strategy_name=self.name,
            entry_price=0.0,
            stop_loss_price=0.0,
            target_price=0.0,
            risk_reward_ratio=0.0,
            vwap=0.0,
            atr_1m=0.0,
            volume_ratio=0.0,
            expected_hold_minutes=0,
            reason=reason,
            timestamp=datetime.now(),
            candle_timeframe="1m",
        )
