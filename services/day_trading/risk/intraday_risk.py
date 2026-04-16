"""
Intraday risk manager — position sizing, daily loss limits, and trade gating.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date

from core.logger import get_logger
from services.day_trading.config import DayTradingConfig

log = get_logger("intraday_risk")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class IntradaySizedTrade:
    symbol: str
    qty: int
    entry_price: float
    stop_loss_price: float
    target_price: float
    position_value: float
    risk_amount: float
    brokerage_cost: float
    is_valid: bool
    rejection_reason: str = ""


@dataclass
class IntradayDailySummary:
    date: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_brokerage: float
    net_pnl: float
    max_drawdown_intraday: float
    win_rate: float


# ── Risk Manager ─────────────────────────────────────────────────────────────


class IntradayRiskManager:
    """
    Enforces per-day trade limits, loss limits, position sizing rules,
    and force-exit timing for intraday trading.
    """

    def __init__(self, config: DayTradingConfig) -> None:
        self.config = config
        self._trades_today: int = 0
        self._daily_pnl: float = 0.0
        self._open_positions: int = 0
        self._trade_log: list[dict] = []
        self._current_date: date = date.today()
        self._max_drawdown: float = 0.0
        self._peak_pnl: float = 0.0
        log.info(
            f"IntradayRiskManager initialized | "
            f"capital={config.capital_allocation:,.0f} | "
            f"max_trades={config.max_trades_per_day} | "
            f"max_loss={config.max_intraday_loss_pct:.1%}"
        )

    # ── Gate checks ──────────────────────────────────────────────────────

    def can_open_trade(self) -> tuple[bool, str]:
        """Check whether a new trade is allowed right now."""
        if self._trades_today >= self.config.max_trades_per_day:
            reason = (
                f"Max trades reached ({self._trades_today}/"
                f"{self.config.max_trades_per_day})"
            )
            log.warning(reason)
            return False, reason

        if self._open_positions >= self.config.max_concurrent_positions:
            reason = (
                f"Max concurrent positions reached ({self._open_positions}/"
                f"{self.config.max_concurrent_positions})"
            )
            log.warning(reason)
            return False, reason

        capital = self.config.capital_allocation
        if capital > 0 and abs(self._daily_pnl) / capital >= self.config.max_intraday_loss_pct:
            if self._daily_pnl < 0:
                reason = (
                    f"Daily loss limit hit "
                    f"({self._daily_pnl:,.2f} = "
                    f"{abs(self._daily_pnl) / capital:.2%} of capital)"
                )
                log.warning(reason)
                return False, reason

        if self.check_force_exit_time():
            reason = f"Past force exit time ({self.config.force_exit_time})"
            log.warning(reason)
            return False, reason

        return True, ""

    # ── Position sizing ──────────────────────────────────────────────────

    def size_position(
        self,
        price: float,
        stop_price: float,
        signal_strength: float,
    ) -> IntradaySizedTrade:
        """
        Calculate position size based on fixed-fraction risk (0.5% of capital).
        Caps at max_position_pct of capital and validates brokerage viability.
        """
        capital = self.config.capital_allocation
        risk_per_trade = 0.005 * capital  # 0.5% risk per trade

        distance = abs(price - stop_price)
        if distance <= 0:
            return IntradaySizedTrade(
                symbol="",
                qty=0,
                entry_price=price,
                stop_loss_price=stop_price,
                target_price=0.0,
                position_value=0.0,
                risk_amount=0.0,
                brokerage_cost=0.0,
                is_valid=False,
                rejection_reason="Stop-loss distance is zero",
            )

        qty = int(risk_per_trade / distance)
        position_value = qty * price

        # Cap at max position size
        max_position_value = self.config.max_position_pct * capital
        if position_value > max_position_value:
            qty = int(max_position_value / price)
            position_value = qty * price

        # Brokerage: min of flat fee and percentage, applied to both entry+exit
        brokerage = (
            min(self.config.brokerage_per_order, position_value * self.config.brokerage_pct)
            * 2  # both sides
        )

        # Reject if profit can't cover costs (position_value must be >= 4x brokerage)
        if position_value < brokerage * 4:
            return IntradaySizedTrade(
                symbol="",
                qty=qty,
                entry_price=price,
                stop_loss_price=stop_price,
                target_price=0.0,
                position_value=position_value,
                risk_amount=risk_per_trade,
                brokerage_cost=brokerage,
                is_valid=False,
                rejection_reason=(
                    f"Position value ({position_value:,.2f}) too small "
                    f"relative to brokerage ({brokerage:,.2f})"
                ),
            )

        if qty < 1:
            return IntradaySizedTrade(
                symbol="",
                qty=0,
                entry_price=price,
                stop_loss_price=stop_price,
                target_price=0.0,
                position_value=0.0,
                risk_amount=risk_per_trade,
                brokerage_cost=0.0,
                is_valid=False,
                rejection_reason="Calculated quantity is zero",
            )

        risk_amount = qty * distance

        return IntradaySizedTrade(
            symbol="",
            qty=qty,
            entry_price=price,
            stop_loss_price=stop_price,
            target_price=0.0,  # caller sets this from signal
            position_value=position_value,
            risk_amount=risk_amount,
            brokerage_cost=brokerage,
            is_valid=True,
        )

    # ── Tracking ─────────────────────────────────────────────────────────

    def record_trade_result(self, pnl: float, brokerage: float) -> None:
        """Record a completed trade's P&L and brokerage."""
        self._daily_pnl += pnl
        self._trades_today += 1
        self._trade_log.append({
            "pnl": pnl,
            "brokerage": brokerage,
            "net_pnl": pnl - brokerage,
            "timestamp": datetime.now().isoformat(),
        })

        # Track drawdown from peak
        if self._daily_pnl > self._peak_pnl:
            self._peak_pnl = self._daily_pnl
        drawdown = self._peak_pnl - self._daily_pnl
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        log.info(
            f"Trade result recorded | pnl={pnl:+,.2f} | "
            f"daily_pnl={self._daily_pnl:+,.2f} | "
            f"trades_today={self._trades_today}"
        )

    def record_position_open(self) -> None:
        """Track that a new position was opened."""
        self._open_positions += 1
        log.debug(f"Position opened | open_positions={self._open_positions}")

    def record_position_close(self) -> None:
        """Track that a position was closed."""
        self._open_positions = max(0, self._open_positions - 1)
        log.debug(f"Position closed | open_positions={self._open_positions}")

    # ── Time checks ──────────────────────────────────────────────────────

    def check_force_exit_time(self) -> bool:
        """Return True if current time is past the configured force_exit_time."""
        try:
            parts = self.config.force_exit_time.split(":")
            exit_hour, exit_minute = int(parts[0]), int(parts[1])
            now = datetime.now()
            return (now.hour > exit_hour) or (
                now.hour == exit_hour and now.minute >= exit_minute
            )
        except (ValueError, IndexError):
            log.error(f"Invalid force_exit_time: {self.config.force_exit_time}")
            return False

    # ── Daily reset ──────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Zero all daily counters. Called at the start of each trading day."""
        log.info(
            f"Daily reset | previous day pnl={self._daily_pnl:+,.2f} | "
            f"trades={self._trades_today}"
        )
        self._trades_today = 0
        self._daily_pnl = 0.0
        self._open_positions = 0
        self._trade_log = []
        self._max_drawdown = 0.0
        self._peak_pnl = 0.0
        self._current_date = date.today()

    # ── Summary ──────────────────────────────────────────────────────────

    def get_daily_summary(self) -> IntradayDailySummary:
        """Return a summary of today's intraday trading activity."""
        winning = sum(1 for t in self._trade_log if t["pnl"] > 0)
        losing = sum(1 for t in self._trade_log if t["pnl"] < 0)
        total_pnl = sum(t["pnl"] for t in self._trade_log)
        total_brokerage = sum(t["brokerage"] for t in self._trade_log)
        total_trades = len(self._trade_log)
        win_rate = winning / total_trades if total_trades > 0 else 0.0

        return IntradayDailySummary(
            date=self._current_date,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            total_pnl=total_pnl,
            total_brokerage=total_brokerage,
            net_pnl=total_pnl - total_brokerage,
            max_drawdown_intraday=self._max_drawdown,
            win_rate=win_rate,
        )
