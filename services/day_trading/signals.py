"""Intraday signal and position dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class IntradaySignal:
    symbol: str
    action: Literal["BUY", "SELL", "SHORT", "COVER", "PASS"]
    strength: float  # 0-1
    strategy_name: str
    entry_price: float
    stop_loss_price: float
    target_price: float
    risk_reward_ratio: float
    vwap: float
    atr_1m: float
    volume_ratio: float
    expected_hold_minutes: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    candle_timeframe: Literal["1m", "5m"] = "1m"


@dataclass
class IntradayPosition:
    symbol: str
    entry_price: float
    entry_time: datetime
    qty: int
    side: Literal["long", "short"]
    stop_loss_price: float
    target_price: float
    current_price: float
    unrealized_pnl: float
    minutes_held: float
    strategy_name: str
