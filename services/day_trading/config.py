"""Day Trading configuration model."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DayTradingConfig(BaseModel):
    enabled: bool = False
    capital_allocation: float = Field(10000.0, gt=0)
    max_trades_per_day: int = Field(10, ge=1, le=50)
    max_intraday_loss_pct: float = Field(0.015, gt=0, le=0.10)
    max_position_pct: float = Field(0.20, gt=0, le=0.50)
    max_concurrent_positions: int = Field(3, ge=1, le=10)
    position_time_limit_minutes: int = Field(180, ge=10)
    force_exit_time: str = "15:15"
    orb_window_minutes: int = Field(15, ge=5, le=30)
    scan_interval_seconds: int = Field(60, ge=10, le=300)
    websocket_enabled: bool = True
    websocket_fallback_to_yfinance: bool = True
    llm_batch_review_interval_minutes: int = Field(15, ge=5, le=60)
    strategies_enabled: list[str] = ["vwap_scalp", "orb", "momentum_scalp"]
    min_signal_strength: float = Field(0.65, ge=0.3, le=0.95)
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0025
    paper_trading: bool = True
