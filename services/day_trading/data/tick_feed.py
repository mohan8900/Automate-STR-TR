"""
WebSocket tick feed using Angel One SmartAPI.
Gracefully degrades if smartapi-python is not installed.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from core.logger import get_logger

log = get_logger("tick_feed")


@dataclass
class TickData:
    symbol: str
    ltp: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime


class TickFeed:
    """
    Real-time tick stream via Angel One SmartWebSocketV2.
    Falls back gracefully (is_connected() == False) when smartapi
    is unavailable.
    """

    # Reconnection backoff settings
    _INITIAL_BACKOFF = 1.0
    _MAX_BACKOFF = 60.0
    _BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
        symbols: list[str],
    ) -> None:
        self._api_key = api_key
        self._client_id = client_id
        self._password = password
        self._totp_secret = totp_secret
        self._symbols = symbols

        self._connected = False
        self._ws = None
        self._session = None
        self._callbacks: list[Callable[[TickData], None]] = []
        self._tick_cache: dict[str, TickData] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Attempt to connect via SmartWebSocketV2.
        If smartapi-python is not installed, logs a warning and returns.
        """
        try:
            from SmartApi import SmartConnect
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError:
            log.warning(
                "smartapi-python not installed — tick feed disabled. "
                "Install with: pip install smartapi-python"
            )
            self._connected = False
            return

        try:
            self._session = SmartConnect(api_key=self._api_key)
            # Generate TOTP
            try:
                from pyotp import TOTP
                totp = TOTP(self._totp_secret).now()
            except ImportError:
                log.warning("pyotp not installed — TOTP generation skipped")
                totp = ""

            data = self._session.generateSession(
                self._client_id,
                self._password,
                totp,
            )
            if not data or data.get("status") is False:
                log.error(f"SmartAPI session generation failed: {data}")
                self._connected = False
                return

            auth_token = data["data"]["jwtToken"]
            feed_token = self._session.getfeedToken()

            self._ws = SmartWebSocketV2(
                auth_token,
                self._api_key,
                self._client_id,
                feed_token,
            )

            self._ws.on_data = self._on_ws_data
            self._ws.on_error = self._on_ws_error
            self._ws.on_close = self._on_ws_close
            self._ws.on_open = self._on_ws_open

            self._stop_event.clear()
            self._ws.connect()
            self._connected = True
            log.info(
                f"Tick feed connected for {len(self._symbols)} symbols"
            )

        except Exception as e:
            log.error(f"Failed to start tick feed: {e}")
            self._connected = False

    def stop(self) -> None:
        """Disconnect the WebSocket and stop reconnection attempts."""
        self._stop_event.set()
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close_connection()
            except Exception as e:
                log.warning(f"Error closing WebSocket: {e}")
            self._ws = None
        if self._session is not None:
            try:
                self._session.terminateSession(self._client_id)
            except Exception:
                pass
            self._session = None
        log.info("Tick feed stopped")

    def on_tick(self, callback: Callable[[TickData], None]) -> None:
        """Register a callback that fires on every incoming tick."""
        self._callbacks.append(callback)

    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Return the most recent tick for a symbol, or None."""
        with self._lock:
            return self._tick_cache.get(symbol)

    def is_connected(self) -> bool:
        """True only if the WebSocket is actively connected."""
        return self._connected

    # ------------------------------------------------------------------
    # WebSocket event handlers
    # ------------------------------------------------------------------

    def _on_ws_open(self, wsapp) -> None:
        log.info("WebSocket connection opened — subscribing to symbols")
        # Subscribe to all symbols (mode 3 = SnapQuote for LTP + depth)
        if self._ws and self._symbols:
            token_list = [
                {"exchangeType": 1, "tokens": self._symbols}
            ]
            try:
                self._ws.subscribe("abc123", 3, token_list)
            except Exception as e:
                log.error(f"Subscription failed: {e}")

    def _on_ws_data(self, wsapp, message) -> None:
        """Parse incoming tick and update cache + fire callbacks."""
        try:
            if not isinstance(message, dict):
                return

            symbol = str(message.get("token", ""))
            ltp = float(message.get("last_traded_price", 0)) / 100.0
            volume = int(message.get("volume_trade_for_the_day", 0))
            bid = float(message.get("best_bid_price", 0)) / 100.0
            ask = float(message.get("best_ask_price", 0)) / 100.0

            tick = TickData(
                symbol=symbol,
                ltp=ltp,
                volume=volume,
                bid=bid,
                ask=ask,
                timestamp=datetime.now(),
            )

            with self._lock:
                self._tick_cache[symbol] = tick

            for cb in self._callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    log.error(f"Tick callback error: {e}")

        except Exception as e:
            log.error(f"Error processing tick message: {e}")

    def _on_ws_error(self, wsapp, error) -> None:
        log.error(f"WebSocket error: {error}")
        self._connected = False

    def _on_ws_close(self, wsapp) -> None:
        log.warning("WebSocket connection closed")
        self._connected = False
        self._attempt_reconnect()

    # ------------------------------------------------------------------
    # Reconnection logic
    # ------------------------------------------------------------------

    def _attempt_reconnect(self) -> None:
        """Spawn a background thread that retries connection with backoff."""
        if self._stop_event.is_set():
            return
        if (
            self._reconnect_thread is not None
            and self._reconnect_thread.is_alive()
        ):
            return

        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        backoff = self._INITIAL_BACKOFF
        while not self._stop_event.is_set():
            log.info(
                f"Attempting WebSocket reconnection in {backoff:.1f}s ..."
            )
            time.sleep(backoff)
            if self._stop_event.is_set():
                break

            try:
                self.start()
                if self._connected:
                    log.info("WebSocket reconnected successfully")
                    return
            except Exception as e:
                log.error(f"Reconnection attempt failed: {e}")

            backoff = min(backoff * self._BACKOFF_FACTOR, self._MAX_BACKOFF)

        log.info("Reconnection loop stopped")
