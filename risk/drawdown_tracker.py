"""
Cumulative drawdown tracker — tracks peak-to-trough portfolio drawdown.
Reduces position sizes or halts trading based on drawdown severity.

Tiers:
  0-5%   drawdown → 1.0x  (trade normally)
  5-10%  drawdown → 0.5x  (reduce size 50%)
  10-15% drawdown → 0.25x (reduce size 75%)
  >15%   drawdown → 0.0x  (halt new entries)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.logger import get_logger

log = get_logger("drawdown")

STATE_FILE = "data/drawdown_state.json"

DRAWDOWN_TIERS = [
    (0.05, 1.0, "normal"),
    (0.10, 0.5, "caution"),
    (0.15, 0.25, "severe"),
    (float("inf"), 0.0, "halt"),
]


@dataclass
class DrawdownState:
    high_water_mark: float = 0.0
    current_value: float = 0.0
    drawdown_pct: float = 0.0
    multiplier: float = 1.0
    tier_label: str = "normal"


class DrawdownTracker:
    """
    Tracks cumulative drawdown from the portfolio's peak value.
    Persists state to disk so it survives restarts.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self._state_file = Path(state_file)
        self._state = self._load_state()
        self._previous_tier = self._state.tier_label

    def update(self, portfolio_value: float) -> float:
        """
        Feed in the latest portfolio value. Returns the position size multiplier.
        """
        if portfolio_value <= 0:
            return self._state.multiplier

        # Update high-water mark
        if portfolio_value > self._state.high_water_mark:
            self._state.high_water_mark = portfolio_value

        self._state.current_value = portfolio_value

        # Calculate drawdown
        if self._state.high_water_mark > 0:
            self._state.drawdown_pct = (
                1 - portfolio_value / self._state.high_water_mark
            )
        else:
            self._state.drawdown_pct = 0.0

        # Determine tier and multiplier
        for threshold, multiplier, label in DRAWDOWN_TIERS:
            if self._state.drawdown_pct < threshold:
                self._state.multiplier = multiplier
                self._state.tier_label = label
                break

        # Log tier transitions
        if self._state.tier_label != self._previous_tier:
            self._log_tier_change()
            self._previous_tier = self._state.tier_label

        self._save_state()
        return self._state.multiplier

    @property
    def multiplier(self) -> float:
        return self._state.multiplier

    @property
    def drawdown_pct(self) -> float:
        return self._state.drawdown_pct

    @property
    def high_water_mark(self) -> float:
        return self._state.high_water_mark

    @property
    def tier_label(self) -> str:
        return self._state.tier_label

    @property
    def state(self) -> DrawdownState:
        return self._state

    def reset(self, new_hwm: Optional[float] = None) -> None:
        """Reset the high-water mark. Use after capital injection."""
        self._state = DrawdownState()
        if new_hwm and new_hwm > 0:
            self._state.high_water_mark = new_hwm
            self._state.current_value = new_hwm
        self._previous_tier = "normal"
        self._save_state()
        log.info(f"Drawdown tracker reset (new HWM: {new_hwm or 'auto'})")

    def _log_tier_change(self) -> None:
        dd = self._state.drawdown_pct
        tier = self._state.tier_label
        mult = self._state.multiplier

        if tier == "normal":
            log.info(
                f"Drawdown recovered to {dd:.1%} — resuming normal position sizing"
            )
        elif tier == "caution":
            log.warning(
                f"Drawdown at {dd:.1%} — reducing position sizes to {mult:.0%}"
            )
        elif tier == "severe":
            log.warning(
                f"SEVERE drawdown at {dd:.1%} — reducing position sizes to {mult:.0%}"
            )
        elif tier == "halt":
            log.critical(
                f"CRITICAL drawdown at {dd:.1%} — HALTING all new entries"
            )

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "high_water_mark": self._state.high_water_mark,
                "current_value": self._state.current_value,
                "drawdown_pct": self._state.drawdown_pct,
                "multiplier": self._state.multiplier,
                "tier_label": self._state.tier_label,
            }
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save drawdown state: {e}")

    def _load_state(self) -> DrawdownState:
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    data = json.load(f)
                return DrawdownState(
                    high_water_mark=data.get("high_water_mark", 0),
                    current_value=data.get("current_value", 0),
                    drawdown_pct=data.get("drawdown_pct", 0),
                    multiplier=data.get("multiplier", 1.0),
                    tier_label=data.get("tier_label", "normal"),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning(f"Corrupt drawdown state file, starting fresh: {e}")
        return DrawdownState()
