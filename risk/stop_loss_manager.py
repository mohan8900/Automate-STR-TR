"""
Stop-loss and take-profit manager.
Implements ATR-based stops, trailing stops, and time-based exits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from core.logger import get_logger

log = get_logger("stop_loss")


@dataclass
class StopLossState:
    symbol: str
    entry_price: float
    entry_date: date
    initial_stop: float
    current_stop: float         # Trailing stop (moves up as price rises)
    take_profit_targets: list[float]
    take_profit_fractions: list[float]
    take_profit_hits: list[bool]   # Which targets have been hit
    trailing_active: bool = False  # Once activated, stop follows price
    holding_period_days: int = 20
    atr_at_entry: float = 0.0

    @property
    def days_held(self) -> int:
        return (date.today() - self.entry_date).days


class StopLossManager:

    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self._positions: dict[str, StopLossState] = {}

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        take_profit_targets: list[float],
        take_profit_fractions: list[float],
        holding_period_days: int = 20,
        custom_stop_pct: Optional[float] = None,
    ) -> StopLossState:
        """Register a new position with its stop parameters."""
        if custom_stop_pct:
            initial_stop = entry_price * (1 - custom_stop_pct)
        else:
            initial_stop = entry_price - (atr * self.atr_multiplier)
            # Never more than 12% below entry
            min_stop = entry_price * 0.88
            initial_stop = max(initial_stop, min_stop)

        state = StopLossState(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=date.today(),
            initial_stop=round(initial_stop, 2),
            current_stop=round(initial_stop, 2),
            take_profit_targets=take_profit_targets,
            take_profit_fractions=take_profit_fractions,
            take_profit_hits=[False] * len(take_profit_targets),
            atr_at_entry=atr,
            holding_period_days=holding_period_days,
        )
        self._positions[symbol] = state
        log.info(
            f"Registered stop for {symbol}: entry ${entry_price:.2f} | "
            f"stop ${state.current_stop:.2f} ({(entry_price - state.current_stop) / entry_price:.1%} below)"
        )
        return state

    def update_price(
        self, symbol: str, current_price: float
    ) -> tuple[bool, str, Optional[float]]:
        """
        Update price for a position. Returns (should_exit, reason, exit_fraction).
        exit_fraction: None=full exit, 0.33=partial exit at take profit
        """
        state = self._positions.get(symbol)
        if not state:
            return False, "Position not tracked", None

        # Check stop loss hit
        if current_price <= state.current_stop:
            log.warning(
                f"{symbol} STOP HIT: ${current_price:.2f} <= "
                f"stop ${state.current_stop:.2f}"
            )
            self._positions.pop(symbol, None)
            return True, f"Stop loss hit at {state.current_stop:.2f}", None

        # Check take profit targets
        for i, (target, fraction) in enumerate(
            zip(state.take_profit_targets, state.take_profit_fractions)
        ):
            if not state.take_profit_hits[i] and current_price >= target:
                state.take_profit_hits[i] = True
                log.info(
                    f"{symbol} TAKE PROFIT {i+1}: {current_price:.2f} >= "
                    f"{target:.2f} | selling {fraction:.0%}"
                )
                # Activate trailing stop after first target hit
                if i == 0:
                    state.trailing_active = True
                return True, f"Take profit {i+1} hit at {target:.2f}", fraction

        # Update trailing stop
        if state.trailing_active:
            trail_distance = state.atr_at_entry * 1.5
            new_stop = current_price - trail_distance
            if new_stop > state.current_stop:
                log.debug(
                    f"{symbol} trailing stop updated: "
                    f"{state.current_stop:.2f} → {new_stop:.2f}"
                )
                state.current_stop = round(new_stop, 2)

        # Time-based exit check
        if state.days_held >= state.holding_period_days:
            profit_pct = (current_price - state.entry_price) / state.entry_price
            if profit_pct < 0.005:  # Less than 0.5% gain after holding period
                log.info(
                    f"{symbol} TIME STOP: {state.days_held} days, "
                    f"only {profit_pct:+.1%} gain"
                )
                self._positions.pop(symbol, None)
                return True, f"Time stop: {state.days_held} days held, minimal gain", None

        return False, "Hold", None

    def get_state(self, symbol: str) -> Optional[StopLossState]:
        return self._positions.get(symbol)

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)

    def get_all_stops(self) -> dict[str, float]:
        """Return {symbol: current_stop_price} for all tracked positions."""
        return {sym: state.current_stop for sym, state in self._positions.items()}
