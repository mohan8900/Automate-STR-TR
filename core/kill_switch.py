"""
Emergency kill switch — the last line of defense.
Cancels all orders, flattens all positions, halts all trading.
Can be triggered automatically (anomaly detection) or manually.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.exceptions import TradingSystemError
from core.logger import get_logger

log = get_logger("kill_switch")


class KillSwitchError(TradingSystemError):
    """Kill switch has been activated — all trading halted."""


@dataclass
class AnomalyThresholds:
    max_orders_per_minute: int = 5           # Runaway loop detection
    max_daily_loss_pct: float = 0.05         # 5% catastrophic loss
    max_same_symbol_per_5min: int = 2        # Duplicate order detection


@dataclass
class KillSwitchState:
    activated: bool = False
    activated_at: Optional[datetime] = None
    reason: Optional[str] = None
    orders_cancelled: int = 0
    positions_closed: int = 0
    errors: list[str] = field(default_factory=list)


class KillSwitch:
    """
    Emergency shutdown system. Once activated, cannot be deactivated
    without explicit manual reset.
    """

    def __init__(
        self,
        broker_client,
        circuit_breaker,
        alert_manager,
        trade_repo,
        thresholds: Optional[AnomalyThresholds] = None,
    ):
        self.broker = broker_client
        self.circuit_breaker = circuit_breaker
        self.alert_manager = alert_manager
        self.trade_repo = trade_repo
        self.thresholds = thresholds or AnomalyThresholds()
        self.state = KillSwitchState()
        self._lock = threading.Lock()

        # Anomaly tracking
        self._order_timestamps: list[float] = []
        self._symbol_order_times: dict[str, list[float]] = defaultdict(list)

    def activate(self, reason: str) -> KillSwitchState:
        """
        Execute full emergency shutdown:
        1. Cancel all open orders
        2. Close all positions
        3. Halt circuit breaker
        4. Send critical alert
        """
        with self._lock:
            if self.state.activated:
                log.warning(f"Kill switch already active (reason: {self.state.reason})")
                return self.state

            log.critical(f"KILL SWITCH ACTIVATED: {reason}")
            self.state.activated = True
            self.state.activated_at = datetime.now()
            self.state.reason = reason

        # Step 1: Cancel all open orders
        try:
            open_trades = self.trade_repo.get_open_trades()
            for trade in open_trades:
                order_id = trade.get("order_id")
                if order_id:
                    try:
                        self.broker.cancel_order(order_id)
                        self.state.orders_cancelled += 1
                        log.info(f"Kill switch: cancelled order {order_id} ({trade.get('symbol', '?')})")
                    except Exception as e:
                        err = f"Failed to cancel order {order_id}: {e}"
                        self.state.errors.append(err)
                        log.error(err)
        except Exception as e:
            err = f"Failed to fetch open trades: {e}"
            self.state.errors.append(err)
            log.error(err)

        # Step 2: Close all positions
        try:
            positions = self.broker.get_positions()
            for pos in positions:
                try:
                    self.broker.close_position(pos.symbol)
                    self.state.positions_closed += 1
                    log.info(f"Kill switch: closed position {pos.symbol} ({pos.qty} shares)")
                except Exception as e:
                    err = f"Failed to close position {pos.symbol}: {e}"
                    self.state.errors.append(err)
                    log.error(err)
        except Exception as e:
            err = f"Failed to fetch positions: {e}"
            self.state.errors.append(err)
            log.error(err)

        # Step 3: Halt circuit breaker
        try:
            self.circuit_breaker.manual_halt(f"Kill switch: {reason}")
        except Exception as e:
            self.state.errors.append(f"Circuit breaker halt failed: {e}")

        # Step 4: Send critical alert
        try:
            self.alert_manager.send_critical(
                f"KILL SWITCH ACTIVATED\n"
                f"Reason: {reason}\n"
                f"Orders cancelled: {self.state.orders_cancelled}\n"
                f"Positions closed: {self.state.positions_closed}\n"
                f"Errors: {len(self.state.errors)}"
            )
        except Exception:
            pass  # Alert failure should never block shutdown

        log.critical(
            f"Kill switch complete: {self.state.orders_cancelled} orders cancelled, "
            f"{self.state.positions_closed} positions closed, "
            f"{len(self.state.errors)} errors"
        )
        return self.state

    # ── Anomaly detection hooks ──────────────────────────────────────────

    def record_order(self, symbol: str) -> None:
        """Call this every time an order is submitted. Checks for anomalies."""
        if self.state.activated:
            raise KillSwitchError("Kill switch is active — cannot place orders")

        now = time.time()

        # Check 1: Too many orders per minute (runaway loop)
        self._order_timestamps.append(now)
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
        if len(self._order_timestamps) > self.thresholds.max_orders_per_minute:
            self.activate(
                f"Runaway loop: {len(self._order_timestamps)} orders in 60s "
                f"(limit: {self.thresholds.max_orders_per_minute})"
            )
            raise KillSwitchError(self.state.reason)

        # Check 2: Same symbol ordered too many times in 5 minutes
        self._symbol_order_times[symbol].append(now)
        self._symbol_order_times[symbol] = [
            t for t in self._symbol_order_times[symbol] if now - t < 300
        ]
        if len(self._symbol_order_times[symbol]) > self.thresholds.max_same_symbol_per_5min:
            self.activate(
                f"Duplicate orders: {symbol} ordered "
                f"{len(self._symbol_order_times[symbol])} times in 5 minutes"
            )
            raise KillSwitchError(self.state.reason)

    def check_daily_loss(self, current_loss_pct: float) -> None:
        """Call with daily loss. Triggers if loss exceeds catastrophic threshold."""
        if self.state.activated:
            return
        if abs(current_loss_pct) >= self.thresholds.max_daily_loss_pct:
            self.activate(
                f"Catastrophic daily loss: {current_loss_pct:.2%} "
                f"(limit: {self.thresholds.max_daily_loss_pct:.2%})"
            )

    def is_active(self) -> bool:
        return self.state.activated

    def reset(self, reason: str = "Manual reset") -> None:
        """Manually reset the kill switch. Requires explicit action."""
        with self._lock:
            log.warning(f"Kill switch RESET: {reason}")
            self.state = KillSwitchState()
            self._order_timestamps.clear()
            self._symbol_order_times.clear()
