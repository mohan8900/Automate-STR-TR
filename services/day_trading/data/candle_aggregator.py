"""Aggregates raw ticks into OHLCV candle bars."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from core.logger import get_logger

log = get_logger("candle_aggregator")


@dataclass
class CandleBar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    timeframe: str


@dataclass
class _TickAccumulator:
    """Internal buffer for accumulating ticks within a single candle period."""
    first_tick_time: Optional[datetime] = None
    open: float = 0.0
    high: float = float("-inf")
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0


# Map timeframe string to timedelta
_TIMEFRAME_DELTAS = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
}


class CandleAggregator:
    """Aggregates raw price ticks into OHLCV candle bars."""

    def __init__(self) -> None:
        # buffers[symbol][timeframe] -> _TickAccumulator (current open candle)
        self._buffers: dict[str, dict[str, _TickAccumulator]] = {}
        # completed[symbol][timeframe] -> list[CandleBar]
        self._completed: dict[str, dict[str, list[CandleBar]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> Optional[CandleBar]:
        """
        Feed a tick into the aggregator.  Returns a completed CandleBar if
        the tick caused a candle to close (for the 1m timeframe), else None.
        All configured timeframes are updated internally.
        """
        completed_1m: Optional[CandleBar] = None

        for tf in _TIMEFRAME_DELTAS:
            buf = self._ensure_buffer(symbol, tf)

            # First tick in this candle period
            if buf.tick_count == 0:
                buf.first_tick_time = self._align_timestamp(timestamp, tf)
                buf.open = price
                buf.high = price
                buf.low = price
                buf.close = price
                buf.volume = volume
                buf.tick_count = 1
            else:
                # Check if current tick belongs to a new candle period
                if self._should_close_candle(tf, buf.first_tick_time, timestamp):
                    candle = self._close_candle(symbol, tf, buf)
                    self._store_candle(symbol, tf, candle)
                    if tf == "1m":
                        completed_1m = candle

                    # Reset buffer and start new candle
                    buf.first_tick_time = self._align_timestamp(timestamp, tf)
                    buf.open = price
                    buf.high = price
                    buf.low = price
                    buf.close = price
                    buf.volume = volume
                    buf.tick_count = 1
                else:
                    # Update running candle
                    buf.high = max(buf.high, price)
                    buf.low = min(buf.low, price)
                    buf.close = price
                    buf.volume += volume
                    buf.tick_count += 1

        return completed_1m

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        count: int = 60,
    ) -> pd.DataFrame:
        """
        Return the last `count` completed candles as an OHLCV DataFrame.
        Columns: open, high, low, close, volume (with datetime index).
        """
        candles = (
            self._completed
            .get(symbol, {})
            .get(timeframe, [])
        )
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        tail = candles[-count:]
        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in tail
        ]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def reset(self) -> None:
        """Clear all buffers and completed candles."""
        self._buffers.clear()
        self._completed.clear()
        log.info("CandleAggregator reset — all buffers cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_buffer(self, symbol: str, timeframe: str) -> _TickAccumulator:
        if symbol not in self._buffers:
            self._buffers[symbol] = {}
        if timeframe not in self._buffers[symbol]:
            self._buffers[symbol][timeframe] = _TickAccumulator()
        return self._buffers[symbol][timeframe]

    def _should_close_candle(
        self,
        timeframe: str,
        first_tick_time: Optional[datetime],
        current_time: datetime,
    ) -> bool:
        """Return True if the current tick falls outside the current candle period."""
        if first_tick_time is None:
            return False
        delta = _TIMEFRAME_DELTAS.get(timeframe, timedelta(minutes=1))
        return current_time >= first_tick_time + delta

    @staticmethod
    def _align_timestamp(ts: datetime, timeframe: str) -> datetime:
        """Align a timestamp to the start of its candle period."""
        delta = _TIMEFRAME_DELTAS.get(timeframe, timedelta(minutes=1))
        minutes = int(delta.total_seconds() / 60)
        aligned_minute = (ts.minute // minutes) * minutes
        return ts.replace(minute=aligned_minute, second=0, microsecond=0)

    @staticmethod
    def _close_candle(symbol: str, timeframe: str, buf: _TickAccumulator) -> CandleBar:
        return CandleBar(
            symbol=symbol,
            open=buf.open,
            high=buf.high,
            low=buf.low,
            close=buf.close,
            volume=buf.volume,
            timestamp=buf.first_tick_time,
            timeframe=timeframe,
        )

    def _store_candle(self, symbol: str, timeframe: str, candle: CandleBar) -> None:
        if symbol not in self._completed:
            self._completed[symbol] = {}
        if timeframe not in self._completed[symbol]:
            self._completed[symbol][timeframe] = []
        self._completed[symbol][timeframe].append(candle)
