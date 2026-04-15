"""
VIX monitor — determines current volatility regime.
VIX drives position sizing and strategy gating.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yfinance as yf

from config.settings import VixConfig
from config.watchlists import VIX_SYMBOL
from core.logger import get_logger

log = get_logger("vix_monitor")

_CACHE: tuple[float, float] = (0.0, 0.0)  # (timestamp, vix_value)
_CACHE_TTL = 300  # 5 minutes


class VixRegime(str, Enum):
    LOW = "LOW"               # VIX < 18  → full sizing
    ELEVATED = "ELEVATED"     # VIX 18-25 → 70% sizing
    HIGH = "HIGH"             # VIX 25-35 → 40% sizing
    EXTREME = "EXTREME"       # VIX > 35  → HALT


@dataclass
class VixReading:
    level: float
    regime: VixRegime
    size_multiplier: float
    trading_allowed: bool
    description: str


class VixMonitor:

    def __init__(self, config: VixConfig, exchange: str = "US"):
        self.config = config
        self.vix_symbol = VIX_SYMBOL.get(exchange, "^VIX")

    def get(self) -> VixReading:
        """Fetch current VIX and determine regime."""
        global _CACHE
        if (time.time() - _CACHE[0]) < _CACHE_TTL and _CACHE[1] > 0:
            level = _CACHE[1]
        else:
            level = self._fetch_vix()
            _CACHE = (time.time(), level)

        return self._classify(level)

    def _fetch_vix(self) -> float:
        try:
            ticker = yf.Ticker(self.vix_symbol)
            hist = ticker.history(period="2d", interval="1d")
            if not hist.empty:
                val = float(hist["Close"].iloc[-1])
                log.debug(f"VIX: {val:.2f}")
                return val
        except Exception as e:
            log.warning(f"VIX fetch failed: {e}")
        return 20.0  # Fallback to elevated (conservative default)

    def _classify(self, level: float) -> VixReading:
        cfg = self.config
        if level > cfg.high_threshold:
            return VixReading(
                level=level,
                regime=VixRegime.EXTREME,
                size_multiplier=0.0,
                trading_allowed=not cfg.extreme_halt,
                description=f"EXTREME volatility (VIX {level:.1f}) — trading halted",
            )
        elif level > cfg.elevated_threshold:
            return VixReading(
                level=level,
                regime=VixRegime.HIGH,
                size_multiplier=cfg.high_size_multiplier,
                trading_allowed=True,
                description=f"High volatility (VIX {level:.1f}) — position sizes reduced to {cfg.high_size_multiplier:.0%}",
            )
        elif level > cfg.low_threshold:
            return VixReading(
                level=level,
                regime=VixRegime.ELEVATED,
                size_multiplier=cfg.elevated_size_multiplier,
                trading_allowed=True,
                description=f"Elevated volatility (VIX {level:.1f}) — position sizes at {cfg.elevated_size_multiplier:.0%}",
            )
        else:
            return VixReading(
                level=level,
                regime=VixRegime.LOW,
                size_multiplier=cfg.low_size_multiplier,
                trading_allowed=True,
                description=f"Low volatility (VIX {level:.1f}) — full position sizing",
            )
