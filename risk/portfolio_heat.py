"""
Portfolio heat monitor — tracks total portfolio risk exposure.
Heat = sum of (position_value/portfolio_value) * stop_distance_pct for all open positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HeatContributor:
    symbol: str
    position_value: float
    stop_loss_pct: float
    heat_contribution: float


class PortfolioHeatMonitor:

    def __init__(self, max_heat: float = 0.06):
        self.max_heat = max_heat
        self._contributors: dict[str, HeatContributor] = {}

    def update_position(
        self,
        symbol: str,
        position_value: float,
        stop_loss_pct: float,
        portfolio_value: float,
    ) -> None:
        """Add or update a position's heat contribution."""
        if position_value <= 0:
            self._contributors.pop(symbol, None)
            return
        heat = (position_value / portfolio_value) * stop_loss_pct
        self._contributors[symbol] = HeatContributor(
            symbol=symbol,
            position_value=position_value,
            stop_loss_pct=stop_loss_pct,
            heat_contribution=heat,
        )

    def remove_position(self, symbol: str) -> None:
        self._contributors.pop(symbol, None)

    def total_heat(self) -> float:
        return sum(c.heat_contribution for c in self._contributors.values())

    def remaining_heat(self) -> float:
        return max(0.0, self.max_heat - self.total_heat())

    def heat_pct_of_max(self) -> float:
        return self.total_heat() / self.max_heat if self.max_heat > 0 else 0

    def can_add_position(self, additional_heat: float) -> tuple[bool, str]:
        """Check if adding a new position would breach the heat limit."""
        new_total = self.total_heat() + additional_heat
        if new_total > self.max_heat:
            return False, (
                f"Heat limit breach: current={self.total_heat():.2%} + "
                f"new={additional_heat:.2%} = {new_total:.2%} > max={self.max_heat:.2%}"
            )
        return True, "OK"

    def get_breakdown(self) -> list[dict]:
        """Return heat breakdown by position for dashboard display."""
        return [
            {
                "symbol": c.symbol,
                "position_value": c.position_value,
                "stop_pct": c.stop_loss_pct,
                "heat_contribution": c.heat_contribution,
                "pct_of_total_heat": c.heat_contribution / self.total_heat()
                if self.total_heat() > 0 else 0,
            }
            for c in sorted(
                self._contributors.values(),
                key=lambda x: x.heat_contribution,
                reverse=True,
            )
        ]

    def clear(self) -> None:
        self._contributors.clear()
