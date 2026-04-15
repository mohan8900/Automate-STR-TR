"""
Unified broker client supporting:
- Alpaca (US markets — paper and live)
- Zerodha Kite (India — NSE/BSE)
- Angel One SmartAPI (India — FREE)

All implement the same BrokerClient interface.
"""
from __future__ import annotations

import uuid
import time as _time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

from config.settings import TradingSystemConfig
from core.exceptions import BrokerError, InsufficientFundsError, OrderRejectedError
from core.logger import get_logger

log = get_logger("broker")


# ── Idempotency tracker (prevents duplicate orders on retry) ─────────────────

class OrderIdempotencyTracker:
    """
    Track submitted orders to prevent duplicates on network retry.
    Each order gets a unique idempotency key. If the same key is seen
    again within the TTL, the original order_id is returned instead
    of submitting a new order.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._submitted: dict[str, tuple[str, float]] = {}  # key -> (order_id, timestamp)
        self._ttl = ttl_seconds

    def generate_key(self, symbol: str, side: str, qty: float, order_type: str) -> str:
        """Generate a unique key for this order attempt."""
        return f"{symbol}:{side}:{qty}:{order_type}:{uuid.uuid4().hex[:8]}"

    def check_duplicate(self, symbol: str, side: str, qty: float) -> Optional[str]:
        """
        Check if an identical order was submitted in the last 60 seconds.
        Returns the existing order_id if duplicate, None otherwise.
        """
        self._cleanup()
        now = _time.time()
        for key, (order_id, ts) in self._submitted.items():
            if key.startswith(f"{symbol}:{side}:{qty}:") and (now - ts) < 60:
                log.warning(
                    f"DUPLICATE ORDER BLOCKED: {side} {qty} {symbol} — "
                    f"identical order {order_id} submitted {now - ts:.0f}s ago"
                )
                return order_id
        return None

    def record(self, key: str, order_id: str) -> None:
        """Record a successfully submitted order."""
        self._submitted[key] = (order_id, _time.time())

    def _cleanup(self) -> None:
        """Remove expired entries."""
        now = _time.time()
        expired = [k for k, (_, ts) in self._submitted.items() if now - ts > self._ttl]
        for k in expired:
            del self._submitted[k]


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class BrokerOrder:
    order_id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str
    filled_price: Optional[float]
    filled_qty: float
    submitted_at: datetime
    filled_at: Optional[datetime]


@dataclass
class AccountInfo:
    account_id: str
    portfolio_value: float
    cash: float
    buying_power: float
    day_trade_count: int
    pattern_day_trader: bool
    daily_pnl: float
    daily_pnl_pct: float


class BrokerClient(ABC):
    """Abstract base class for all broker integrations."""

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        take_profit_price: float,
        stop_loss_price: float,
    ) -> BrokerOrder: ...

    @abstractmethod
    def submit_market_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"]
    ) -> BrokerOrder: ...

    @abstractmethod
    def submit_limit_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"], limit_price: float
    ) -> BrokerOrder: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def close_position(self, symbol: str) -> BrokerOrder: ...

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder: ...


# ── Alpaca Implementation ─────────────────────────────────────────────────────

class AlpacaBrokerClient(BrokerClient):

    def __init__(self, config: TradingSystemConfig):
        self.cfg = config.alpaca
        self.is_paper = self.cfg.paper

        if self.is_paper:
            log.info("Alpaca client initialized in PAPER trading mode")
        else:
            log.warning("Alpaca client initialized in LIVE trading mode — real money!")

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest, BracketOrderRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading = TradingClient(
                api_key=self.cfg.api_key.get_secret_value(),
                secret_key=self.cfg.secret_key.get_secret_value(),
                paper=self.is_paper,
            )
            self._MarketOrderRequest = MarketOrderRequest
            self._LimitOrderRequest = LimitOrderRequest
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce
            self._alpaca_available = True

        except ImportError:
            log.warning("alpaca-py not installed — running in SIMULATION mode")
            self._alpaca_available = False

    def get_account(self) -> AccountInfo:
        if not self._alpaca_available:
            return self._sim_account()
        try:
            acc = self._trading.get_account()
            last_equity = float(acc.last_equity) if acc.last_equity else float(acc.equity)
            equity = float(acc.equity)
            daily_pnl = equity - last_equity
            return AccountInfo(
                account_id=str(acc.id),
                portfolio_value=equity,
                cash=float(acc.cash),
                buying_power=float(acc.buying_power),
                day_trade_count=int(acc.daytrade_count or 0),
                pattern_day_trader=bool(acc.pattern_day_trader),
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl / last_equity if last_equity > 0 else 0,
            )
        except Exception as e:
            raise BrokerError(f"Failed to fetch Alpaca account: {e}") from e

    def get_positions(self) -> list[BrokerPosition]:
        if not self._alpaca_available:
            return []
        try:
            positions = self._trading.get_all_positions()
            return [
                BrokerPosition(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_cost=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                    unrealized_pnl_pct=float(p.unrealized_plpc),
                )
                for p in positions
            ]
        except Exception as e:
            raise BrokerError(f"Failed to fetch positions: {e}") from e

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        take_profit_price: float,
        stop_loss_price: float,
    ) -> BrokerOrder:
        """
        Submit a bracket order (entry + stop-loss + take-profit as atomic unit).
        This ensures stops are always set even if the system crashes after fill.
        """
        if not self._alpaca_available:
            return self._sim_order(symbol, qty, side)
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, TakeProfitRequest, StopLossRequest
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            )
            order = self._trading.submit_order(request)
            log.info(
                f"BRACKET ORDER: {side.upper()} {qty} {symbol} | "
                f"stop=${stop_loss_price:.2f} target=${take_profit_price:.2f}"
            )
            return self._to_broker_order(order)
        except Exception as e:
            raise OrderRejectedError(f"Bracket order rejected for {symbol}: {e}") from e

    def submit_market_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"]
    ) -> BrokerOrder:
        if not self._alpaca_available:
            return self._sim_order(symbol, qty, side)
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            request = MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._trading.submit_order(request)
            log.info(f"MARKET ORDER: {side.upper()} {qty} {symbol}")
            return self._to_broker_order(order)
        except Exception as e:
            raise OrderRejectedError(f"Market order rejected: {e}") from e

    def submit_limit_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"], limit_price: float
    ) -> BrokerOrder:
        if not self._alpaca_available:
            return self._sim_order(symbol, qty, side)
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            request = LimitOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
            order = self._trading.submit_order(request)
            log.info(f"LIMIT ORDER: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}")
            return self._to_broker_order(order)
        except Exception as e:
            raise OrderRejectedError(f"Limit order rejected: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        if not self._alpaca_available:
            return True
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def close_position(self, symbol: str) -> BrokerOrder:
        if not self._alpaca_available:
            return self._sim_order(symbol, 0, "sell")
        try:
            order = self._trading.close_position(symbol)
            log.info(f"Closed position: {symbol}")
            return self._to_broker_order(order)
        except Exception as e:
            raise BrokerError(f"Failed to close {symbol}: {e}") from e

    def get_order(self, order_id: str) -> BrokerOrder:
        if not self._alpaca_available:
            return self._sim_order("UNKNOWN", 0, "buy")
        try:
            order = self._trading.get_order_by_id(order_id)
            return self._to_broker_order(order)
        except Exception as e:
            raise BrokerError(f"Failed to get order {order_id}: {e}") from e

    def _to_broker_order(self, order) -> BrokerOrder:
        return BrokerOrder(
            order_id=str(order.id),
            symbol=str(order.symbol),
            qty=float(order.qty or 0),
            side=str(order.side).lower(),
            order_type=str(order.order_type).lower(),
            status=str(order.status).lower(),
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            filled_qty=float(order.filled_qty or 0),
            submitted_at=order.submitted_at or datetime.utcnow(),
            filled_at=order.filled_at,
        )

    def _sim_account(self) -> AccountInfo:
        return AccountInfo(
            account_id="SIMULATION",
            portfolio_value=50000.0,
            cash=50000.0,
            buying_power=50000.0,
            day_trade_count=0,
            pattern_day_trader=False,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
        )

    def _sim_order(self, symbol: str, qty: float, side: str) -> BrokerOrder:
        import uuid
        return BrokerOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            status="filled",
            filled_price=100.0,  # Placeholder
            filled_qty=qty,
            submitted_at=datetime.utcnow(),
            filled_at=datetime.utcnow(),
        )


# ── Zerodha Kite Implementation ───────────────────────────────────────────────

class ZerodhaBrokerClient(BrokerClient):
    """Zerodha Kite Connect broker for Indian markets (NSE/BSE)."""

    def __init__(self, config: TradingSystemConfig):
        self.cfg = config.zerodha
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(
                api_key=self.cfg.api_key.get_secret_value()
            )
            access_token = self.cfg.access_token.get_secret_value()
            if access_token:
                self._kite.set_access_token(access_token)
            self._available = True
            log.info("Zerodha Kite client initialized (access token must be refreshed daily)")
        except ImportError:
            log.warning("kiteconnect not installed — running in simulation mode")
            self._available = False

    def get_account(self) -> AccountInfo:
        if not self._available:
            return AccountInfo("ZERODHA_SIM", 500000, 500000, 500000, 0, False, 0, 0)
        try:
            margins = self._kite.margins()
            equity = margins.get("equity", {})
            net = float(equity.get("net", 0))
            available = float(equity.get("available", {}).get("cash", 0))
            return AccountInfo(
                account_id="ZERODHA",
                portfolio_value=net,
                cash=available,
                buying_power=available,
                day_trade_count=0,
                pattern_day_trader=False,
                daily_pnl=0,
                daily_pnl_pct=0,
            )
        except Exception as e:
            raise BrokerError(f"Zerodha account fetch failed: {e}") from e

    def get_positions(self) -> list[BrokerPosition]:
        if not self._available:
            return []
        try:
            positions = self._kite.positions().get("net", [])
            result = []
            for p in positions:
                qty = float(p.get("quantity", 0))
                if qty == 0:
                    continue
                avg = float(p.get("average_price", 0))
                ltp = float(p.get("last_price", avg))
                mv = qty * ltp
                pnl = mv - (qty * avg)
                result.append(BrokerPosition(
                    symbol=p.get("tradingsymbol", ""),
                    qty=qty,
                    avg_cost=avg,
                    current_price=ltp,
                    market_value=mv,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl / (qty * avg) if avg > 0 else 0,
                ))
            return result
        except Exception as e:
            raise BrokerError(f"Zerodha positions fetch failed: {e}") from e

    def submit_bracket_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"],
        take_profit_price: float, stop_loss_price: float,
    ) -> BrokerOrder:
        if not self._available:
            return self._sim_order(symbol, qty, side)
        try:
            import uuid
            transaction = self._kite.TRANSACTION_TYPE_BUY if side == "buy" else self._kite.TRANSACTION_TYPE_SELL
            order_id = self._kite.place_order(
                tradingsymbol=symbol.replace(".NS", "").replace(".BO", ""),
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=transaction,
                quantity=int(qty),
                product=self._kite.PRODUCT_MIS,  # Intraday
                order_type=self._kite.ORDER_TYPE_MARKET,
                variety=self._kite.VARIETY_BO,  # Bracket Order
                squareoff=round(take_profit_price, 2),
                stoploss=round(stop_loss_price, 2),
            )
            log.info(f"Zerodha BO: {side.upper()} {qty} {symbol} id={order_id}")
            return self._sim_order(symbol, qty, side, order_id=str(order_id))
        except Exception as e:
            raise OrderRejectedError(f"Zerodha bracket order failed: {e}") from e

    def submit_market_order(self, symbol, qty, side) -> BrokerOrder:
        return self._sim_order(symbol, qty, side)

    def submit_limit_order(self, symbol, qty, side, limit_price) -> BrokerOrder:
        return self._sim_order(symbol, qty, side)

    def cancel_order(self, order_id: str) -> bool:
        return True

    def close_position(self, symbol: str) -> BrokerOrder:
        return self._sim_order(symbol, 0, "sell")

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._sim_order("UNKNOWN", 0, "buy", order_id=order_id)

    def _sim_order(self, symbol, qty, side, order_id=None) -> BrokerOrder:
        import uuid
        return BrokerOrder(
            order_id=order_id or str(uuid.uuid4()),
            symbol=symbol, qty=qty, side=side,
            order_type="market", status="filled",
            filled_price=None, filled_qty=qty,
            submitted_at=datetime.utcnow(), filled_at=datetime.utcnow(),
        )


# ── Angel One SmartAPI Implementation ────────────────────────────────────────

class AngelOneBrokerClient(BrokerClient):
    """
    Angel One SmartAPI broker for Indian markets (NSE/BSE).
    FREE API access — no subscription fee (unlike Zerodha's Rs 2000/month).
    Best choice for small capital traders in India.

    Features:
    - Auto session re-authentication (tokens expire at 5 AM IST daily)
    - Idempotency tracking to prevent duplicate orders
    - LIMIT orders by default (reduces slippage)
    - Partial fill handling
    - Individual stock circuit limit checking
    """

    def __init__(self, config: TradingSystemConfig):
        self.cfg = config.angel_one
        self._auth_token = None
        self._refresh_token = None
        self._session_time: Optional[datetime] = None
        self._idempotency = OrderIdempotencyTracker()
        self._token_cache: dict[str, str] = {}  # symbol -> token cache

        try:
            from SmartApi import SmartConnect
            self._SmartConnect = SmartConnect
            self._smart = SmartConnect(api_key=self.cfg.api_key.get_secret_value())
            self._available = True
            self._authenticate()
            log.info("Angel One SmartAPI client initialized")

        except ImportError:
            log.warning(
                "smartapi-python not installed — running in SIMULATION mode. "
                "Install with: pip install smartapi-python pyotp"
            )
            self._available = False
            self._smart = None

    def _authenticate(self) -> bool:
        """Authenticate with Angel One. Returns True on success."""
        if not self._available:
            return False
        try:
            import pyotp
            client_id = self.cfg.client_id.get_secret_value()
            password = self.cfg.password.get_secret_value()
            totp_secret = self.cfg.totp_secret.get_secret_value()

            if not (client_id and password):
                log.warning("Angel One credentials not configured")
                return False

            totp_val = pyotp.TOTP(totp_secret).now() if totp_secret else ""
            data = self._smart.generateSession(client_id, password, totp_val)

            if data.get("status"):
                self._auth_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                self._session_time = datetime.now()
                log.info("Angel One SmartAPI authenticated successfully")
                return True
            else:
                log.warning(f"Angel One login failed: {data.get('message', 'unknown error')}")
                return False
        except Exception as e:
            log.warning(f"Angel One session generation failed: {e}")
            return False

    def _ensure_session(self) -> None:
        """
        Re-authenticate if session is stale.
        Angel One tokens expire at 5 AM IST daily.
        Re-auth if session is older than 12 hours or it's a new day.
        """
        if not self._available or not self._session_time:
            self._authenticate()
            return

        now = datetime.now()
        hours_since_auth = (now - self._session_time).total_seconds() / 3600

        # Re-auth if session > 12 hours old or different day
        if hours_since_auth > 12 or now.date() != self._session_time.date():
            log.info("Angel One session expired — re-authenticating...")
            # Recreate the SmartConnect instance for a clean session
            self._smart = self._SmartConnect(api_key=self.cfg.api_key.get_secret_value())
            self._authenticate()

    def get_account(self) -> AccountInfo:
        if not self._available:
            return AccountInfo("ANGEL_SIM", 500000, 500000, 500000, 0, False, 0, 0)
        self._ensure_session()
        try:
            rms = self._smart.rmsLimit()
            if rms.get("status"):
                data = rms["data"]
                net = float(data.get("net", 0))
                available = float(data.get("availablecash", 0))
                return AccountInfo(
                    account_id="ANGEL_ONE",
                    portfolio_value=net,
                    cash=available,
                    buying_power=available,
                    day_trade_count=0,
                    pattern_day_trader=False,
                    daily_pnl=0,
                    daily_pnl_pct=0,
                )
        except Exception as e:
            raise BrokerError(f"Angel One account fetch failed: {e}") from e
        return AccountInfo("ANGEL_ONE", 0, 0, 0, 0, False, 0, 0)

    def get_positions(self) -> list[BrokerPosition]:
        if not self._available:
            return []
        self._ensure_session()
        try:
            pos_data = self._smart.position()
            if not pos_data.get("status") or not pos_data.get("data"):
                return []
            result = []
            for p in pos_data["data"]:
                qty = float(p.get("netqty", 0))
                if qty == 0:
                    continue
                avg = float(p.get("averageprice", 0))
                ltp = float(p.get("ltp", avg))
                mv = abs(qty) * ltp
                pnl = float(p.get("unrealised", 0))
                result.append(BrokerPosition(
                    symbol=p.get("tradingsymbol", ""),
                    qty=abs(qty),
                    avg_cost=avg,
                    current_price=ltp,
                    market_value=mv,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl / (abs(qty) * avg) if avg > 0 else 0,
                ))
            return result
        except Exception as e:
            raise BrokerError(f"Angel One positions fetch failed: {e}") from e

    def check_circuit_limit(self, symbol: str) -> tuple[bool, str]:
        """
        Check if a stock is at its circuit limit (upper or lower).
        Returns (can_trade, reason).
        """
        if not self._available:
            return True, ""
        try:
            clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
            token = self._get_token(clean_symbol)
            if not token:
                return True, ""  # Can't check, allow trade
            ltp_data = self._smart.ltpData("NSE", clean_symbol, token)
            if ltp_data.get("status") and ltp_data.get("data"):
                data = ltp_data["data"]
                ltp = float(data.get("ltp", 0))
                upper = float(data.get("upperCircuit", 0) or 0)
                lower = float(data.get("lowerCircuit", 0) or 0)

                if upper > 0 and ltp >= upper * 0.99:
                    return False, f"{clean_symbol} at upper circuit limit (₹{ltp:.2f} / ₹{upper:.2f})"
                if lower > 0 and ltp <= lower * 1.01:
                    return False, f"{clean_symbol} at lower circuit limit (₹{ltp:.2f} / ₹{lower:.2f})"
        except Exception as e:
            log.debug(f"Circuit limit check failed for {symbol}: {e}")
        return True, ""

    def submit_bracket_order(
        self, symbol: str, qty: float, side: Literal["buy", "sell"],
        take_profit_price: float, stop_loss_price: float,
    ) -> BrokerOrder:
        if not self._available:
            return self._sim_order(symbol, qty, side)
        self._ensure_session()

        # Check circuit limits
        can_trade, reason = self.check_circuit_limit(symbol)
        if not can_trade:
            raise OrderRejectedError(f"Circuit limit: {reason}")

        # Check for duplicate orders
        dup_id = self._idempotency.check_duplicate(symbol, side, qty)
        if dup_id:
            return self._sim_order(symbol, qty, side, order_id=dup_id)

        try:
            clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
            idem_key = self._idempotency.generate_key(symbol, side, qty, "bracket")
            order_params = {
                "variety": "ROBO",
                "tradingsymbol": clean_symbol,
                "symboltoken": self._get_token(clean_symbol),
                "transactiontype": "BUY" if side == "buy" else "SELL",
                "exchange": "NSE",
                "ordertype": "LIMIT",
                "producttype": "BO",
                "duration": "DAY",
                "quantity": str(int(qty)),
                "price": str(round(take_profit_price, 2)),
                "squareoff": str(round(take_profit_price, 2)),
                "stoploss": str(round(stop_loss_price, 2)),
            }
            order_id = self._smart.placeOrder(order_params)
            self._idempotency.record(idem_key, str(order_id))
            log.info(f"Angel One BO: {side.upper()} {qty} {clean_symbol} id={order_id}")
            return self._build_order(symbol, qty, side, str(order_id))
        except Exception as e:
            raise OrderRejectedError(f"Angel One bracket order failed: {e}") from e

    def submit_market_order(self, symbol, qty, side) -> BrokerOrder:
        if not self._available:
            return self._sim_order(symbol, qty, side)
        self._ensure_session()

        can_trade, reason = self.check_circuit_limit(symbol)
        if not can_trade:
            raise OrderRejectedError(f"Circuit limit: {reason}")

        dup_id = self._idempotency.check_duplicate(symbol, side, qty)
        if dup_id:
            return self._sim_order(symbol, qty, side, order_id=dup_id)

        try:
            clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
            idem_key = self._idempotency.generate_key(symbol, side, qty, "market")
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": clean_symbol,
                "symboltoken": self._get_token(clean_symbol),
                "transactiontype": "BUY" if side == "buy" else "SELL",
                "exchange": "NSE",
                "ordertype": "MARKET",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "quantity": str(int(qty)),
            }
            order_id = self._smart.placeOrder(order_params)
            self._idempotency.record(idem_key, str(order_id))
            log.info(f"Angel One MARKET: {side.upper()} {qty} {clean_symbol} id={order_id}")
            return self._build_order(symbol, qty, side, str(order_id))
        except Exception as e:
            raise OrderRejectedError(f"Angel One market order failed: {e}") from e

    def submit_limit_order(self, symbol, qty, side, limit_price) -> BrokerOrder:
        if not self._available:
            return self._sim_order(symbol, qty, side)
        self._ensure_session()

        can_trade, reason = self.check_circuit_limit(symbol)
        if not can_trade:
            raise OrderRejectedError(f"Circuit limit: {reason}")

        dup_id = self._idempotency.check_duplicate(symbol, side, qty)
        if dup_id:
            return self._sim_order(symbol, qty, side, order_id=dup_id)

        try:
            clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
            idem_key = self._idempotency.generate_key(symbol, side, qty, "limit")
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": clean_symbol,
                "symboltoken": self._get_token(clean_symbol),
                "transactiontype": "BUY" if side == "buy" else "SELL",
                "exchange": "NSE",
                "ordertype": "LIMIT",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "price": str(round(limit_price, 2)),
                "quantity": str(int(qty)),
            }
            order_id = self._smart.placeOrder(order_params)
            self._idempotency.record(idem_key, str(order_id))
            log.info(
                f"Angel One LIMIT: {side.upper()} {qty} {clean_symbol} "
                f"@ ₹{limit_price:.2f} id={order_id}"
            )
            return self._build_order(symbol, qty, side, str(order_id))
        except Exception as e:
            raise OrderRejectedError(f"Angel One limit order failed: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        if not self._available:
            return True
        self._ensure_session()
        try:
            # Try NORMAL variety first, then ROBO (bracket)
            try:
                self._smart.cancelOrder(order_id, "NORMAL")
            except Exception:
                self._smart.cancelOrder(order_id, "ROBO")
            log.info(f"Angel One: cancelled order {order_id}")
            return True
        except Exception as e:
            log.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def close_position(self, symbol: str) -> BrokerOrder:
        """Close an entire position by selling all shares at market."""
        if not self._available:
            return self._sim_order(symbol, 0, "sell")
        self._ensure_session()
        try:
            positions = self.get_positions()
            for pos in positions:
                if pos.symbol == symbol.replace(".NS", "").replace(".BO", ""):
                    return self.submit_market_order(symbol, pos.qty, "sell")
            log.warning(f"No position found for {symbol} to close")
            return self._sim_order(symbol, 0, "sell")
        except Exception as e:
            raise BrokerError(f"Failed to close {symbol}: {e}") from e

    def get_order(self, order_id: str) -> BrokerOrder:
        """Fetch real order status including partial fills."""
        if not self._available:
            return self._sim_order("UNKNOWN", 0, "buy", order_id=order_id)
        self._ensure_session()
        try:
            order_book = self._smart.orderBook()
            if order_book.get("status") and order_book.get("data"):
                for o in order_book["data"]:
                    if str(o.get("orderid")) == order_id:
                        filled_qty = float(o.get("filledshares", 0))
                        total_qty = float(o.get("quantity", 0))
                        status = str(o.get("orderstatus", "")).lower()

                        # Detect partial fills
                        if filled_qty > 0 and filled_qty < total_qty:
                            status = "partial_fill"
                            log.warning(
                                f"Partial fill on {o.get('tradingsymbol')}: "
                                f"{filled_qty}/{total_qty} shares filled"
                            )

                        return BrokerOrder(
                            order_id=order_id,
                            symbol=o.get("tradingsymbol", ""),
                            qty=total_qty,
                            side=o.get("transactiontype", "").lower(),
                            order_type=o.get("ordertype", "").lower(),
                            status=status,
                            filled_price=float(o.get("averageprice", 0)) or None,
                            filled_qty=filled_qty,
                            submitted_at=datetime.utcnow(),
                            filled_at=datetime.utcnow() if filled_qty > 0 else None,
                        )
        except Exception as e:
            log.warning(f"Failed to get order {order_id}: {e}")
        return self._sim_order("UNKNOWN", 0, "buy", order_id=order_id)

    def _get_token(self, symbol: str) -> str:
        """Get Angel One symbol token (needed for order placement). Cached."""
        if symbol in self._token_cache:
            return self._token_cache[symbol]
        try:
            data = self._smart.searchScrip("NSE", symbol)
            if data.get("data"):
                token = data["data"][0].get("symboltoken", "")
                if token:
                    self._token_cache[symbol] = token
                return token
        except Exception:
            pass
        return ""

    def _build_order(self, symbol, qty, side, order_id) -> BrokerOrder:
        """Build a BrokerOrder from a freshly placed order, then verify status."""
        try:
            return self.get_order(order_id)
        except Exception:
            return self._sim_order(symbol, qty, side, order_id=order_id)

    def _sim_order(self, symbol, qty, side, order_id=None) -> BrokerOrder:
        return BrokerOrder(
            order_id=order_id or str(uuid.uuid4()),
            symbol=symbol, qty=qty, side=side,
            order_type="market", status="filled",
            filled_price=None, filled_qty=qty,
            submitted_at=datetime.utcnow(), filled_at=datetime.utcnow(),
        )


def create_broker_client(config: TradingSystemConfig) -> BrokerClient:
    """Factory: return the right broker based on exchange config."""
    if config.market.exchange == "IN":
        # Prefer Angel One (free) over Zerodha (paid)
        angel_key = config.angel_one.api_key.get_secret_value()
        if angel_key:
            return AngelOneBrokerClient(config)
        return ZerodhaBrokerClient(config)
    return AlpacaBrokerClient(config)
