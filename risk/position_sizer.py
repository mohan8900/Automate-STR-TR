"""
Position sizer — determines how many shares to buy/sell.
Three methods: Fixed %, ATR-based volatility sizing, Half-Kelly criterion.
VIX and regime multipliers applied at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from analysis.aggregator import AnalysisBundle
from analysis.technical.market_regime import MarketRegime
from analysis.technical.symbol_regime import SymbolRegime
from config.settings import TradingSystemConfig
from llm.response_parser import TradeRecommendation, RiskReview
from core.logger import get_logger

log = get_logger("position_sizer")


@dataclass
class SizedTrade:
    symbol: str
    action: str
    shares: float
    entry_price: float
    position_value: float
    stop_loss_price: float
    stop_loss_pct: float
    take_profit_prices: list[float]
    take_profit_fractions: list[float]
    portfolio_heat_contribution: float  # How much heat this adds
    is_valid: bool
    invalid_reason: str = ""
    # From LLM
    conviction: int = 0
    primary_thesis: str = ""
    key_risks: list[str] = None
    holding_period_days: int = 10

    def __post_init__(self):
        if self.key_risks is None:
            self.key_risks = []


class PositionSizer:

    def __init__(self, config: TradingSystemConfig):
        self.config = config

    def size_trade(
        self,
        bundle: AnalysisBundle,
        recommendation: TradeRecommendation,
        review: RiskReview,
        portfolio_value: float,
        cash_available: float,
        current_heat: float,
    ) -> SizedTrade:
        """
        Calculate final position size applying all multipliers.
        Returns SizedTrade with is_valid=False if trade should not proceed.
        """
        price = bundle.current_price
        if price <= 0:
            return self._invalid(bundle.symbol, "Price is zero or negative")

        # Symbol-regime hard gate: reject trades the stock's own behavior rules out,
        # even if the market regime and LLM thesis say otherwise.
        sr = bundle.symbol_regime
        if sr.regime == SymbolRegime.CHOPPY_VOLATILE:
            return self._invalid(
                bundle.symbol,
                f"Symbol regime CHOPPY_VOLATILE (vol pctile {sr.vol_percentile:.0f}, ADX {sr.adx:.0f}) — chop eats edge"
            )
        if recommendation.action == "BUY" and sr.regime == SymbolRegime.DOWNTREND:
            return self._invalid(
                bundle.symbol,
                f"BUY blocked: symbol in DOWNTREND (ADX {sr.adx:.0f}, -DI {sr.minus_di:.0f} > +DI {sr.plus_di:.0f})"
            )

        # Use modified stop from risk reviewer if provided
        stop_pct = review.modified_stop_loss_pct or recommendation.stop_loss_pct
        # Use modified size from risk reviewer if provided
        max_pos_pct = review.modified_position_pct or recommendation.max_position_pct

        # ── Base sizing ───────────────────────────────────────────────────
        method = self.config.risk.sizing_method
        if method == "fixed":
            base_value = portfolio_value * max_pos_pct
        elif method == "atr":
            base_value = self._atr_based_size(
                portfolio_value, stop_pct, bundle.technical.atr_pct
            )
            base_value = min(base_value, portfolio_value * max_pos_pct)
        else:  # kelly
            base_value = self._kelly_based_size(portfolio_value)
            base_value = min(base_value, portfolio_value * max_pos_pct)

        # ── Multipliers ───────────────────────────────────────────────────
        # VIX multiplier
        vix_mult = bundle.vix.size_multiplier

        # Market regime multiplier
        regime_mult = bundle.market_regime.position_size_multiplier

        # Conviction multiplier (scale by conviction 6-10 → 0.6x-1.0x)
        conviction_mult = min(1.0, recommendation.conviction / 10.0)

        final_value = base_value * vix_mult * regime_mult * conviction_mult
        # Minimum trade floor: 500 currency units, but never more than 10% of portfolio.
        # Keeps small paper-trading sandboxes (e.g. ₹1000) usable without forcing
        # oversized bets, while preserving the ₹500 minimum for normal-sized portfolios.
        min_floor = min(500, portfolio_value * 0.10)
        final_value = max(min_floor, final_value)

        # ── Constraints ───────────────────────────────────────────────────
        # Portfolio heat check
        heat_contribution = (final_value / portfolio_value) * stop_pct
        if current_heat + heat_contribution > self.config.user.max_portfolio_heat:
            # Reduce size to fit within heat budget
            remaining_heat = max(0, self.config.user.max_portfolio_heat - current_heat)
            if remaining_heat < 0.001:
                return self._invalid(
                    bundle.symbol,
                    f"Portfolio heat budget exhausted ({current_heat:.1%} / "
                    f"{self.config.user.max_portfolio_heat:.1%})"
                )
            final_value = (remaining_heat / stop_pct) * portfolio_value
            heat_contribution = remaining_heat

        # Cash check
        if final_value > cash_available * 0.95:
            final_value = cash_available * 0.90
            if final_value < 500:
                return self._invalid(
                    bundle.symbol,
                    f"Insufficient cash: ${cash_available:,.0f} available"
                )

        shares = final_value / price
        shares = max(1, round(shares))  # At least 1 share, whole shares
        actual_value = shares * price
        actual_heat = (actual_value / portfolio_value) * stop_pct

        # Stop loss price
        stop_price = price * (1 - stop_pct)

        # Take profit prices
        tp_prices = []
        tp_fractions = []
        for target in recommendation.take_profit_targets:
            if target.price > 0 and target.price > price:
                tp_prices.append(target.price)
                tp_fractions.append(target.sell_fraction)
            else:
                # Use ATR-based targets if LLM targets are invalid
                atr = bundle.technical.atr_14
                multiples = self.config.trading.take_profit_atr_multiples
                for i, mult in enumerate(multiples):
                    tp_prices.append(round(price + atr * mult, 2))
                    tp_fractions.append(1.0 / len(multiples))
                break

        log.info(
            f"Sized {bundle.symbol}: {shares:.0f} shares @ ${price:.2f} "
            f"= ${actual_value:,.0f} | stop ${stop_price:.2f} | "
            f"heat +{actual_heat:.2%} (VIX×{vix_mult:.0%} regime×{regime_mult:.0%})"
        )

        return SizedTrade(
            symbol=bundle.symbol,
            action=recommendation.action,
            shares=shares,
            entry_price=price,
            position_value=actual_value,
            stop_loss_price=round(stop_price, 2),
            stop_loss_pct=stop_pct,
            take_profit_prices=tp_prices,
            take_profit_fractions=tp_fractions,
            portfolio_heat_contribution=actual_heat,
            is_valid=True,
            conviction=recommendation.conviction,
            primary_thesis=recommendation.primary_thesis,
            key_risks=recommendation.key_risks,
            holding_period_days=recommendation.holding_period_days,
        )

    # ── Sizing methods ────────────────────────────────────────────────────

    def _atr_based_size(
        self, portfolio_value: float, stop_pct: float, atr_pct: float
    ) -> float:
        """
        Risk a fixed % of portfolio per trade.
        Size = (portfolio * risk_per_trade) / stop_distance
        """
        risk_amount = portfolio_value * self.config.risk.risk_per_trade_pct
        if stop_pct <= 0:
            stop_pct = atr_pct * 2.0
        return risk_amount / stop_pct

    def _kelly_based_size(self, portfolio_value: float) -> float:
        """Half-Kelly with conservative fallback."""
        # Without historical win rate data, use fixed fraction
        # Full Kelly would require real backtest data
        kelly_f = 0.10  # Conservative base Kelly fraction
        half_kelly = kelly_f * self.config.risk.kelly_fraction
        return portfolio_value * half_kelly

    def _invalid(self, symbol: str, reason: str) -> SizedTrade:
        log.warning(f"Trade sizing rejected for {symbol}: {reason}")
        return SizedTrade(
            symbol=symbol, action="PASS", shares=0, entry_price=0,
            position_value=0, stop_loss_price=0, stop_loss_pct=0,
            take_profit_prices=[], take_profit_fractions=[],
            portfolio_heat_contribution=0, is_valid=False,
            invalid_reason=reason,
        )
