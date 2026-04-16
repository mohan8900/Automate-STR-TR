"""
Portfolio manager — handles diversification, rebalancing, and allocation.
Implements Warren Buffett value investing principles as quantitative filters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from config.settings import TradingSystemConfig
from core.logger import get_logger

log = get_logger("portfolio_manager")


@dataclass
class PortfolioAllocation:
    symbol: str
    target_weight: float        # Target % of portfolio
    current_weight: float       # Current % of portfolio
    deviation: float            # current - target
    action_needed: str          # "buy_more" | "trim" | "hold" | "new_entry" | "full_exit"
    rebalance_amount: float     # Dollar amount to buy/sell
    sector: str = ""


@dataclass
class DiversificationCheck:
    passed: bool
    max_sector_exposure: float
    max_single_stock: float
    sector_breakdown: dict[str, float]
    correlation_warning: bool
    warnings: list[str]


@dataclass
class PortfolioHealth:
    total_value: float
    cash_pct: float
    invested_pct: float
    num_positions: int
    sector_diversification_score: float  # 0-100
    concentration_score: float           # 0-100 (lower = more concentrated)
    value_quality_score: float           # 0-100 (Buffett quality score)
    overall_health: str                  # "EXCELLENT" | "GOOD" | "FAIR" | "POOR"


class PortfolioManager:
    """
    Manages portfolio-level decisions: allocation, diversification, rebalancing.
    Enforces sector limits and correlation constraints.
    """

    # Maximum exposure limits
    MAX_SINGLE_STOCK_PCT = 0.10      # 10% max in any single stock
    MAX_SECTOR_PCT = 0.30            # 30% max in any single sector
    MIN_POSITIONS = 5                # At least 5 positions for diversification
    MAX_POSITIONS = 20
    MIN_CASH_RESERVE_PCT = 0.10      # Keep 10% cash always
    REBALANCE_THRESHOLD = 0.03       # Rebalance when weight deviates > 3%

    # Sector mappings for Indian stocks
    INDIA_SECTOR_MAP = {
        "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT",
        "HCLTECH.NS": "IT", "TECHM.NS": "IT", "LTIM.NS": "IT",
        "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking",
        "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking",
        "SBIN.NS": "Banking", "BAJFINANCE.NS": "Finance",
        "BAJAJFINSV.NS": "Finance",
        "RELIANCE.NS": "Energy", "ONGC.NS": "Energy",
        "POWERGRID.NS": "Utilities", "NTPC.NS": "Utilities",
        "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG",
        "NESTLEIND.NS": "FMCG", "DABUR.NS": "FMCG",
        "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma",
        "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
        "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto",
        "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
        "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals",
        "BHARTIARTL.NS": "Telecom", "TITAN.NS": "Consumer",
        "ASIANPAINT.NS": "Consumer", "LT.NS": "Infrastructure",
        "ULTRACEMCO.NS": "Infrastructure",
    }

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.max_single = min(
            self.MAX_SINGLE_STOCK_PCT,
            config.user.max_position_pct
        )

    def check_diversification(
        self,
        positions: dict[str, dict],
        portfolio_value: float,
    ) -> DiversificationCheck:
        """
        Check if current portfolio meets diversification requirements.
        positions: {symbol: {value, sector, weight}}
        """
        warnings = []

        # Sector breakdown
        sector_values: dict[str, float] = {}
        for symbol, pos in positions.items():
            sector = pos.get("sector", self._get_sector(symbol))
            sector_values[sector] = sector_values.get(sector, 0) + pos.get("value", 0)

        sector_pcts = {
            s: v / portfolio_value for s, v in sector_values.items()
        } if portfolio_value > 0 else {}

        # Check sector concentration
        max_sector_exp = max(sector_pcts.values(), default=0)
        if max_sector_exp > self.MAX_SECTOR_PCT:
            over_sectors = [s for s, p in sector_pcts.items() if p > self.MAX_SECTOR_PCT]
            warnings.append(
                f"Sector over-concentration: {', '.join(over_sectors)} "
                f"({max_sector_exp:.1%} > {self.MAX_SECTOR_PCT:.0%} limit)"
            )

        # Check single stock concentration
        max_single = 0.0
        for symbol, pos in positions.items():
            weight = pos.get("value", 0) / portfolio_value if portfolio_value > 0 else 0
            if weight > self.max_single:
                warnings.append(
                    f"{symbol} over-weight: {weight:.1%} > {self.max_single:.0%} limit"
                )
            max_single = max(max_single, weight)

        # Under-diversification
        if len(positions) < self.MIN_POSITIONS and len(positions) > 0:
            warnings.append(
                f"Under-diversified: only {len(positions)} positions "
                f"(minimum {self.MIN_POSITIONS} recommended)"
            )

        return DiversificationCheck(
            passed=len(warnings) == 0,
            max_sector_exposure=round(max_sector_exp, 4),
            max_single_stock=round(max_single, 4),
            sector_breakdown=sector_pcts,
            correlation_warning=max_sector_exp > 0.40,
            warnings=warnings,
        )

    def can_add_position(
        self,
        symbol: str,
        position_value: float,
        current_positions: dict[str, dict],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if a new position can be added without violating constraints."""
        # Position count check
        if len(current_positions) >= self.MAX_POSITIONS:
            return False, f"Max positions ({self.MAX_POSITIONS}) reached"

        # Single stock size check
        weight = position_value / portfolio_value if portfolio_value > 0 else 1.0
        if weight > self.max_single:
            return False, (
                f"Position size {weight:.1%} exceeds limit {self.max_single:.0%}"
            )

        # Sector check
        sector = self._get_sector(symbol)
        sector_total = position_value
        for sym, pos in current_positions.items():
            if pos.get("sector", self._get_sector(sym)) == sector:
                sector_total += pos.get("value", 0)

        sector_pct = sector_total / portfolio_value if portfolio_value > 0 else 1.0
        if sector_pct > self.MAX_SECTOR_PCT:
            return False, (
                f"Sector '{sector}' would be {sector_pct:.1%} "
                f"(limit: {self.MAX_SECTOR_PCT:.0%})"
            )

        # Cash reserve check
        cash_after = (portfolio_value - sum(
            p.get("value", 0) for p in current_positions.values()
        ) - position_value) / portfolio_value
        if cash_after < self.MIN_CASH_RESERVE_PCT:
            return False, (
                f"Cash would drop to {cash_after:.1%} "
                f"(minimum: {self.MIN_CASH_RESERVE_PCT:.0%})"
            )

        return True, "OK"

    def get_rebalance_actions(
        self,
        current_positions: dict[str, dict],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[PortfolioAllocation]:
        """
        Calculate rebalancing actions to move from current to target weights.
        Only suggests rebalancing when deviation exceeds threshold.
        """
        actions = []

        all_symbols = set(current_positions.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_value = current_positions.get(symbol, {}).get("value", 0)
            current_weight = current_value / portfolio_value if portfolio_value > 0 else 0
            target_weight = target_weights.get(symbol, 0)
            deviation = current_weight - target_weight

            if abs(deviation) < self.REBALANCE_THRESHOLD:
                action = "hold"
                rebalance_amt = 0
            elif deviation > 0:
                action = "trim"
                rebalance_amt = -deviation * portfolio_value  # Negative = sell
            elif target_weight > 0 and current_weight == 0:
                action = "new_entry"
                rebalance_amt = target_weight * portfolio_value
            elif target_weight == 0:
                action = "full_exit"
                rebalance_amt = -current_value
            else:
                action = "buy_more"
                rebalance_amt = abs(deviation) * portfolio_value

            sector = current_positions.get(symbol, {}).get(
                "sector", self._get_sector(symbol)
            )

            actions.append(PortfolioAllocation(
                symbol=symbol,
                target_weight=round(target_weight, 4),
                current_weight=round(current_weight, 4),
                deviation=round(deviation, 4),
                action_needed=action,
                rebalance_amount=round(rebalance_amt, 2),
                sector=sector,
            ))

        # Sort: exits first, then trims, then buys
        priority = {"full_exit": 0, "trim": 1, "hold": 2, "buy_more": 3, "new_entry": 4}
        actions.sort(key=lambda a: priority.get(a.action_needed, 5))

        return actions

    def get_portfolio_health(
        self,
        positions: dict[str, dict],
        portfolio_value: float,
        cash: float,
    ) -> PortfolioHealth:
        """Calculate overall portfolio health score."""
        if portfolio_value <= 0:
            return PortfolioHealth(
                total_value=0, cash_pct=1.0, invested_pct=0,
                num_positions=0, sector_diversification_score=0,
                concentration_score=0, value_quality_score=0,
                overall_health="POOR",
            )

        invested = sum(p.get("value", 0) for p in positions.values())
        cash_pct = cash / portfolio_value
        invested_pct = invested / portfolio_value

        # Sector diversification score (Herfindahl index)
        div_check = self.check_diversification(positions, portfolio_value)
        if div_check.sector_breakdown:
            weights = list(div_check.sector_breakdown.values())
            hhi = sum(w ** 2 for w in weights)
            # Perfect diversification (equal weight) HHI = 1/n
            n = len(weights)
            min_hhi = 1 / n if n > 0 else 1
            div_score = max(0, min(100, (1 - hhi) / (1 - min_hhi) * 100)) if min_hhi < 1 else 50
        else:
            div_score = 0

        # Concentration score (inverse of largest position)
        if positions:
            max_weight = max(
                p.get("value", 0) / portfolio_value
                for p in positions.values()
            )
            conc_score = max(0, min(100, (1 - max_weight) * 100))
        else:
            conc_score = 0

        # Value quality score (average fundamental score)
        fund_scores = [
            p.get("fundamental_score", 50)
            for p in positions.values()
            if p.get("fundamental_score")
        ]
        value_score = np.mean(fund_scores) if fund_scores else 50

        # Overall health
        avg_score = (div_score + conc_score + value_score) / 3
        if avg_score >= 75:
            health = "EXCELLENT"
        elif avg_score >= 55:
            health = "GOOD"
        elif avg_score >= 35:
            health = "FAIR"
        else:
            health = "POOR"

        return PortfolioHealth(
            total_value=portfolio_value,
            cash_pct=round(cash_pct, 4),
            invested_pct=round(invested_pct, 4),
            num_positions=len(positions),
            sector_diversification_score=round(div_score, 1),
            concentration_score=round(conc_score, 1),
            value_quality_score=round(value_score, 1),
            overall_health=health,
        )

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.INDIA_SECTOR_MAP.get(symbol, "Unknown")
