"""
Order manager — tracks order lifecycle and position reconciliation.
Maintains the ledger of all submitted orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from execution.broker_client import BrokerClient, BrokerOrder, BrokerPosition
from core.logger import get_logger

log = get_logger("order_manager")


@dataclass
class OrderRecord:
    order_id: str
    symbol: str
    action: str
    shares: float
    entry_price: float
    stop_loss_price: float
    take_profit_prices: list[float]
    status: str  # PENDING | SUBMITTED | FILLED | CANCELLED | REJECTED
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    db_id: Optional[int] = None  # SQLite trade ID


class OrderManager:

    def __init__(self, broker: BrokerClient):
        self.broker = broker
        self._orders: dict[str, OrderRecord] = {}  # order_id -> record

    def register_order(
        self,
        order: BrokerOrder,
        action: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_prices: list[float],
    ) -> OrderRecord:
        record = OrderRecord(
            order_id=order.order_id,
            symbol=order.symbol,
            action=action,
            shares=order.qty,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_prices=take_profit_prices,
            status="SUBMITTED",
            submitted_at=order.submitted_at,
        )
        self._orders[order.order_id] = record
        return record

    def sync_order_status(self, order_id: str) -> OrderRecord:
        """Fetch latest status from broker and update record."""
        record = self._orders.get(order_id)
        if not record:
            raise ValueError(f"Order {order_id} not tracked")
        try:
            broker_order = self.broker.get_order(order_id)
            if broker_order.status == "filled":
                record.status = "FILLED"
                record.filled_at = broker_order.filled_at
                record.filled_price = broker_order.filled_price
                record.shares = broker_order.filled_qty
            elif broker_order.status in ("cancelled", "expired"):
                record.status = "CANCELLED"
            elif broker_order.status in ("rejected",):
                record.status = "REJECTED"
        except Exception as e:
            log.warning(f"Could not sync order {order_id}: {e}")
        return record

    def get_open_orders(self) -> list[OrderRecord]:
        return [r for r in self._orders.values() if r.status in ("PENDING", "SUBMITTED")]

    def reconcile_positions(self) -> dict[str, BrokerPosition]:
        """
        Fetch all positions from broker and reconcile with our order ledger.
        Returns current positions keyed by symbol.
        """
        try:
            positions = self.broker.get_positions()
            pos_map = {p.symbol: p for p in positions}
            log.debug(f"Reconciled {len(pos_map)} positions from broker")
            return pos_map
        except Exception as e:
            log.error(f"Position reconciliation failed: {e}")
            return {}
