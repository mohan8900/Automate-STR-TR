"""
Trade executor — orchestrates the final step: submitting orders to the broker.
All safety checks must pass before this code runs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from execution.broker_client import BrokerClient, BrokerOrder
from execution.order_manager import OrderManager, OrderRecord
from risk.position_sizer import SizedTrade
from database.repository import TradeRepository
from core.logger import get_logger
from core.exceptions import BrokerError

log = get_logger("executor")


class TradeExecutor:

    def __init__(
        self,
        broker: BrokerClient,
        order_manager: OrderManager,
        repository: TradeRepository,
        paper_trading: bool = True,
    ):
        self.broker = broker
        self.order_manager = order_manager
        self.repository = repository
        self.paper_trading = paper_trading

        if not paper_trading:
            log.warning(
                "⚠️  LIVE TRADING MODE — real money will be used for all orders!"
            )

    def execute(self, trade: SizedTrade) -> Optional[OrderRecord]:
        """
        Execute a SizedTrade by submitting a bracket order to the broker.
        Returns the order record, or None if execution failed.
        """
        if not trade.is_valid:
            log.warning(f"Refusing to execute invalid trade for {trade.symbol}")
            return None

        log.info(
            f"{'[PAPER] ' if self.paper_trading else '[LIVE] '}"
            f"Executing: {trade.action} {trade.shares:.0f} {trade.symbol} "
            f"@ ~${trade.entry_price:.2f} | "
            f"stop=${trade.stop_loss_price:.2f} | "
            f"value=${trade.position_value:,.0f}"
        )

        try:
            # Use bracket order for automatic stop-loss protection
            first_target = (
                trade.take_profit_prices[0]
                if trade.take_profit_prices
                else trade.entry_price * 1.05
            )

            order: BrokerOrder = self.broker.submit_bracket_order(
                symbol=trade.symbol,
                qty=trade.shares,
                side="buy" if trade.action == "BUY" else "sell",
                take_profit_price=first_target,
                stop_loss_price=trade.stop_loss_price,
            )

            # Register in order manager
            record = self.order_manager.register_order(
                order=order,
                action=trade.action,
                entry_price=trade.entry_price,
                stop_loss_price=trade.stop_loss_price,
                take_profit_prices=trade.take_profit_prices,
            )

            # Persist to database
            db_id = self.repository.save_trade(trade, order, self.paper_trading)
            record.db_id = db_id

            log.info(
                f"Order submitted: id={order.order_id} | "
                f"status={order.status} | db_id={db_id}"
            )
            return record

        except BrokerError as e:
            log.error(f"Trade execution failed for {trade.symbol}: {e}")
            self.repository.save_failed_trade(trade, str(e))
            return None
        except Exception as e:
            log.exception(f"Unexpected error executing {trade.symbol}: {e}")
            return None

    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Close an open position completely."""
        try:
            order = self.broker.close_position(symbol)
            self.repository.mark_trade_closed(
                symbol=symbol,
                exit_price=order.filled_price,
                close_reason=reason,
            )
            log.info(f"Closed {symbol}: reason={reason} price=${order.filled_price}")
            return True
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")
            return False

    def close_partial(
        self, symbol: str, shares: float, reason: str = "take_profit"
    ) -> bool:
        """Close a partial position (for take-profit scaling out)."""
        try:
            order = self.broker.submit_market_order(symbol, shares, "sell")
            log.info(
                f"Partial close {symbol}: {shares:.0f} shares | reason={reason} | "
                f"price=${order.filled_price}"
            )
            return True
        except Exception as e:
            log.error(f"Partial close failed {symbol}: {e}")
            return False
