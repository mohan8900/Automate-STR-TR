"""
Master configuration using Pydantic Settings v2.
Loads from .env first, then config.toml, then environment variables.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # pip install tomli for Python < 3.11

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Sub-models ──────────────────────────────────────────────────────────────

class UserConfig(BaseModel):
    investment_amount: float = Field(50000.0, gt=0)
    trading_budget: float = Field(10000.0, gt=0)
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "moderate"
    max_position_pct: float = Field(0.05, gt=0, le=0.20)
    max_daily_loss_pct: float = Field(0.02, gt=0, le=0.10)
    max_portfolio_heat: float = Field(0.06, gt=0, le=0.20)
    approval_required: bool = True
    paper_trading: bool = True

    @model_validator(mode="after")
    def trading_budget_le_investment(self) -> "UserConfig":
        if self.trading_budget > self.investment_amount:
            raise ValueError("trading_budget cannot exceed investment_amount")
        return self


class MarketConfig(BaseModel):
    exchange: Literal["US", "IN"] = "IN"


class AnthropicConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = Field(0.1, ge=0, le=1)
    max_daily_cost_usd: float = 10.0


class TradingConfig(BaseModel):
    scan_interval_minutes: int = 30
    watchlist_size: int = Field(50, ge=10, le=500)
    max_open_positions: int = Field(15, ge=1, le=50)
    max_daily_trades: int = Field(10, ge=1, le=100)
    min_composite_score: float = Field(55.0, ge=0, le=100)
    min_conviction_execute: int = Field(7, ge=1, le=10)
    min_conviction_auto: int = Field(9, ge=1, le=10)
    atr_stop_multiplier: float = Field(2.0, ge=0.5, le=5.0)
    take_profit_atr_multiples: list[float] = [2.0, 4.0, 6.0]
    min_volume_usd: float = 1_000_000.0
    min_price: float = 10.0
    max_price: float = 50_000.0
    earnings_blackout_days: int = 3


class RiskConfig(BaseModel):
    sizing_method: Literal["fixed", "atr", "kelly"] = "atr"
    risk_per_trade_pct: float = Field(0.005, gt=0, le=0.05)
    kelly_fraction: float = Field(0.5, gt=0, le=1.0)


class VixConfig(BaseModel):
    low_threshold: float = 15.0
    elevated_threshold: float = 20.0
    high_threshold: float = 30.0
    low_size_multiplier: float = 1.0
    elevated_size_multiplier: float = 0.7
    high_size_multiplier: float = 0.4
    extreme_halt: bool = True


class DatabaseConfig(BaseModel):
    path: str = "data/trading.db"
    backup_daily: bool = True


class NotificationsConfig(BaseModel):
    email_enabled: bool = False
    sms_enabled: bool = False
    slack_enabled: bool = False
    alert_on_trade: bool = True
    alert_on_circuit_break: bool = True
    daily_summary_time: str = "15:45"


class AlpacaConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    secret_key: SecretStr = SecretStr("")
    paper: bool = True

    @property
    def base_url(self) -> str:
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


class ZerodhaConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    access_token: SecretStr = SecretStr("")


class AngelOneConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    client_id: SecretStr = SecretStr("")
    password: SecretStr = SecretStr("")
    totp_secret: SecretStr = SecretStr("")  # For TOTP-based 2FA


class MLConfig(BaseModel):
    enabled: bool = True
    target_days: int = Field(5, ge=1, le=30)
    retrain_interval_days: int = Field(30, ge=7, le=90)
    min_confidence: float = Field(0.55, ge=0.5, le=0.9)
    train_window_days: int = Field(504, ge=100, le=1000)


class StrategyConfig(BaseModel):
    primary: Literal["swing", "momentum", "mean_reversion", "auto"] = "auto"
    enable_ml_boost: bool = True
    min_consensus_strength: float = Field(0.55, ge=0.3, le=0.9)
    buffett_screen_enabled: bool = True
    min_buffett_score: float = Field(50.0, ge=0, le=100)


# ── Root settings ────────────────────────────────────────────────────────────

class TradingSystemConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    user: UserConfig = UserConfig()
    market: MarketConfig = MarketConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    vix: VixConfig = VixConfig()
    database: DatabaseConfig = DatabaseConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    alpaca: AlpacaConfig = AlpacaConfig()
    zerodha: ZerodhaConfig = ZerodhaConfig()
    angel_one: AngelOneConfig = AngelOneConfig()
    ml: MLConfig = MLConfig()
    strategy: StrategyConfig = StrategyConfig()
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"


def _load_toml_config(path: str = "config.toml") -> dict:
    """Load config.toml as a dict (handles missing file gracefully)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return tomllib.load(f)


@lru_cache(maxsize=1)
def get_settings() -> TradingSystemConfig:
    """Return singleton config, merging config.toml + .env + environment."""
    toml_data = _load_toml_config()

    # Flatten toml into env-style overrides
    # Pydantic-settings will pick them up via model_validate
    cfg = TradingSystemConfig()

    # Manually overlay toml values (pydantic-settings doesn't natively read toml)
    for section, values in toml_data.items():
        if hasattr(cfg, section) and isinstance(values, dict):
            current = getattr(cfg, section)
            updated = current.model_copy(update=values)
            object.__setattr__(cfg, section, updated)

    return cfg
