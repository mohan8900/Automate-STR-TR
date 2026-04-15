"""
Circuit breaker — the safety layer that halts trading under dangerous conditions.
All checks run BEFORE any trade is submitted to the broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Optional

import pytz

from config.settings import TradingSystemConfig
from core.exceptions import CircuitBreakerError
from core.logger import get_logger

log = get_logger("circuit_breaker")

ET = pytz.timezone("America/New_York")
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class CircuitBreakerState:
    daily_pnl_pct: float = 0.0
    daily_trades_count: int = 0
    is_triggered: bool = False
    trigger_reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    last_reset_date: Optional[date] = None
    vix_level: float = 0.0
    open_positions_count: int = 0
    # Per-symbol earnings blackout
    earnings_blackout_symbols: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    passed: bool
    reason: str = ""


class CircuitBreaker:
    """
    Multi-layer safety system. All checks must pass for a trade to proceed.
    State persists in-memory and is reset daily.
    """

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self._tz = ET if config.market.exchange == "US" else IST

    # ── Public API ────────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset daily counters. Called at market open."""
        today = date.today()
        if self.state.last_reset_date != today:
            self.state.daily_pnl_pct = 0.0
            self.state.daily_trades_count = 0
            self.state.is_triggered = False
            self.state.trigger_reason = None
            self.state.triggered_at = None
            self.state.last_reset_date = today
            log.info("Circuit breaker daily counters reset")

    def update_daily_pnl(self, pnl_pct: float) -> None:
        self.state.daily_pnl_pct = pnl_pct

    def update_vix(self, vix_level: float) -> None:
        self.state.vix_level = vix_level

    def increment_trades(self) -> None:
        self.state.daily_trades_count += 1

    def set_open_positions(self, count: int) -> None:
        self.state.open_positions_count = count

    def set_earnings_blackout(self, symbols: list[str]) -> None:
        self.state.earnings_blackout_symbols = symbols

    def check_all(self, symbol: Optional[str] = None) -> tuple[bool, str]:
        """
        Run all circuit breaker checks. Returns (ok, reason).
        Raises CircuitBreakerError if already triggered.
        """
        if self.state.is_triggered:
            raise CircuitBreakerError(self.state.trigger_reason or "Previously triggered")

        self.reset_daily()

        checks = [
            self._check_market_hours(),
            self._check_daily_loss(),
            self._check_vix_extreme(),
            self._check_max_daily_trades(),
            self._check_max_positions(),
        ]

        if symbol:
            checks.append(self._check_earnings_blackout(symbol))

        failed = [r for r in checks if not r.passed]
        if failed:
            reasons = "; ".join(r.reason for r in failed)
            self._trigger(reasons)
            return False, reasons

        return True, "All checks passed"

    def check_symbol_only(self, symbol: str) -> tuple[bool, str]:
        """Symbol-specific checks (earnings blackout)."""
        result = self._check_earnings_blackout(symbol)
        return result.passed, result.reason

    def manual_halt(self, reason: str) -> None:
        """Operator-initiated trading halt."""
        self._trigger(f"Manual halt: {reason}")
        log.warning(f"Manual trading halt: {reason}")

    def resume(self) -> None:
        """Resume after manual halt (resets trigger state)."""
        self.state.is_triggered = False
        self.state.trigger_reason = None
        self.state.triggered_at = None
        log.info("Circuit breaker manually resumed")

    # ── Individual checks ─────────────────────────────────────────────────

    def _check_market_hours(self) -> CheckResult:
        """Only trade during core market hours (avoids open/close volatility)."""
        now = datetime.now(self._tz)
        wd = now.weekday()  # 0=Mon, 6=Sun

        # Weekends
        if wd >= 5:
            return CheckResult(False, f"Market closed (weekend: {now.strftime('%A')})")

        if self.config.market.exchange == "US":
            # Core hours: 9:35 AM – 3:45 PM ET (avoids first/last 15-min volatility)
            market_open = time(9, 35)
            market_close = time(15, 45)
        else:
            # NSE: 9:20 AM – 3:20 PM IST
            market_open = time(9, 20)
            market_close = time(15, 20)

        current_time = now.time()
        if not (market_open <= current_time <= market_close):
            return CheckResult(
                False,
                f"Outside market hours (now {current_time.strftime('%H:%M')} "
                f"{self._tz.zone}, open {market_open}–{market_close})"
            )
        return CheckResult(True)

    def _check_daily_loss(self) -> CheckResult:
        """Halt if daily P&L loss exceeds threshold."""
        threshold = -abs(self.config.user.max_daily_loss_pct)
        if self.state.daily_pnl_pct <= threshold:
            return CheckResult(
                False,
                f"Daily loss limit hit: {self.state.daily_pnl_pct:.2%} "
                f"(limit: {threshold:.2%})"
            )
        return CheckResult(True)

    def _check_vix_extreme(self) -> CheckResult:
        """Halt all new trades when VIX is in extreme territory."""
        if not self.config.vix.extreme_halt:
            return CheckResult(True)
        if self.state.vix_level > self.config.vix.high_threshold:
            return CheckResult(
                False,
                f"VIX extreme: {self.state.vix_level:.1f} "
                f"> threshold {self.config.vix.high_threshold}"
            )
        return CheckResult(True)

    def _check_max_daily_trades(self) -> CheckResult:
        """Prevent over-trading."""
        limit = self.config.trading.max_daily_trades
        if self.state.daily_trades_count >= limit:
            return CheckResult(
                False,
                f"Max daily trades reached: {self.state.daily_trades_count}/{limit}"
            )
        return CheckResult(True)

    def _check_max_positions(self) -> CheckResult:
        """Cap number of simultaneous open positions."""
        limit = self.config.trading.max_open_positions
        if self.state.open_positions_count >= limit:
            return CheckResult(
                False,
                f"Max open positions reached: {self.state.open_positions_count}/{limit}"
            )
        return CheckResult(True)

    def _check_earnings_blackout(self, symbol: str) -> CheckResult:
        """Block trades within N days of a company's earnings report."""
        if symbol in self.state.earnings_blackout_symbols:
            return CheckResult(
                False,
                f"{symbol} is within {self.config.trading.earnings_blackout_days}-day "
                "earnings blackout window"
            )
        return CheckResult(True)

    def _trigger(self, reason: str) -> None:
        # Only trigger persistent halt for loss limit, not temporary conditions
        if "Daily loss limit" in reason or "Manual halt" in reason:
            self.state.is_triggered = True
            self.state.trigger_reason = reason
            self.state.triggered_at = datetime.now(self._tz)
        log.warning(f"Circuit breaker: {reason}")
