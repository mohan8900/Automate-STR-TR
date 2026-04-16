"""
Unified intraday data feed.
Prefers live WebSocket ticks (via TickFeed + CandleAggregator);
falls back to yfinance (via PriceFeed) when WebSocket is unavailable.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from core.logger import get_logger
from data.price_feed import PriceFeed
from services.day_trading.config import DayTradingConfig
from services.day_trading.data.candle_aggregator import CandleAggregator
from services.day_trading.data.tick_feed import TickData, TickFeed

log = get_logger("intraday_feed")


class IntradayFeed:
    """
    Provides a unified interface for intraday candle and price data.

    When websocket_enabled is True and credentials are available, streams
    real-time ticks via Angel One SmartAPI.  Otherwise falls back to
    yfinance intraday bars through PriceFeed.
    """

    def __init__(
        self,
        config: DayTradingConfig,
        exchange: str = "IN",
        *,
        api_key: str = "",
        client_id: str = "",
        password: str = "",
        totp_secret: str = "",
        symbols: Optional[list[str]] = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._symbols = symbols or []

        # yfinance fallback (always available)
        self._price_feed = PriceFeed(exchange=exchange)

        # Real-time components
        self._aggregator = CandleAggregator()
        self._tick_feed: Optional[TickFeed] = None

        if config.websocket_enabled and api_key:
            self._tick_feed = TickFeed(
                api_key=api_key,
                client_id=client_id,
                password=password,
                totp_secret=totp_secret,
                symbols=self._symbols,
            )
            # Wire ticks into the candle aggregator
            self._tick_feed.on_tick(self._on_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        bars: int = 60,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for the given symbol and timeframe.

        Uses live WebSocket data when connected; otherwise falls back to
        yfinance intraday data.
        """
        if self._tick_feed and self._tick_feed.is_connected():
            df = self._aggregator.get_candles(symbol, timeframe=timeframe, count=bars)
            if not df.empty:
                return df
            log.debug(
                f"No aggregated candles for {symbol} yet — "
                f"falling back to yfinance"
            )

        # Fallback: yfinance via PriceFeed
        if not self._config.websocket_fallback_to_yfinance:
            log.warning(
                f"WebSocket not connected and yfinance fallback disabled "
                f"for {symbol}"
            )
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        try:
            return self._price_feed.get_intraday(
                symbol, interval=timeframe, period="1d"
            )
        except Exception as e:
            log.error(f"yfinance fallback failed for {symbol}: {e}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

    def get_current_price(self, symbol: str) -> float:
        """
        Return the latest price for a symbol.
        Tries tick cache first, then yfinance.
        """
        if self._tick_feed and self._tick_feed.is_connected():
            tick = self._tick_feed.get_latest_tick(symbol)
            if tick is not None and tick.ltp > 0:
                return tick.ltp

        # Fallback
        try:
            return self._price_feed.get_current_price(symbol)
        except Exception as e:
            log.error(f"Cannot get current price for {symbol}: {e}")
            return 0.0

    def start_streaming(self, symbols: Optional[list[str]] = None) -> None:
        """Start the WebSocket tick stream."""
        if symbols:
            self._symbols = symbols

        if self._tick_feed is None:
            log.info(
                "WebSocket tick feed not configured — using yfinance only"
            )
            return

        log.info(
            f"Starting tick stream for {len(self._symbols)} symbols"
        )
        self._tick_feed.start()

    def stop_streaming(self) -> None:
        """Stop the WebSocket tick stream and clear aggregator buffers."""
        if self._tick_feed is not None:
            self._tick_feed.stop()
        self._aggregator.reset()
        log.info("Intraday feed streaming stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_tick(self, tick: TickData) -> None:
        """Route incoming ticks into the candle aggregator."""
        self._aggregator.add_tick(
            symbol=tick.symbol,
            price=tick.ltp,
            volume=float(tick.volume),
            timestamp=tick.timestamp,
        )
