"""
Warren Buffett Value Screener — quantifies Buffett's investment principles.
Screens for companies with durable competitive advantages, low debt,
strong cash flow, and margin of safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yfinance as yf

from core.logger import get_logger

log = get_logger("value_screener")


@dataclass
class BuffettScore:
    symbol: str
    total_score: float          # 0-100 composite Buffett quality score

    # Individual criteria scores (0-100 each)
    moat_score: float           # Consistent high ROE = competitive advantage
    earnings_quality: float     # Predictable, growing earnings
    debt_health: float          # Low debt = margin of safety
    cash_generation: float      # Strong free cash flow
    margin_quality: float       # High and expanding margins
    valuation_score: float      # Margin of safety vs intrinsic value
    management_score: float     # Capital allocation efficiency

    # Raw data
    roe_10yr_avg: Optional[float] = None
    debt_to_equity: Optional[float] = None
    fcf_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    intrinsic_value: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    current_price: Optional[float] = None

    # Flags
    passes_buffett_screen: bool = False
    disqualifiers: list[str] = None

    def __post_init__(self):
        if self.disqualifiers is None:
            self.disqualifiers = []


class ValueScreener:
    """
    Quantitative implementation of Warren Buffett's value investing criteria.
    Each criterion is scored 0-100 and combined into a composite score.
    """

    # Buffett's minimum thresholds
    MIN_ROE = 0.15              # ROE > 15% consistently
    MIN_OPERATING_MARGIN = 0.15 # Operating margin > 15%
    MAX_DEBT_TO_EQUITY = 0.50   # D/E < 0.5 (Buffett prefers < 0.3)
    MIN_FCF_YIELD = 0.04        # FCF yield > 4%
    MIN_MARGIN_OF_SAFETY = 0.20 # 20% below intrinsic value
    MAX_PE_RATIO = 25.0         # Reasonable P/E

    def screen(self, symbol: str) -> BuffettScore:
        """Run full Buffett screen on a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow

            current_price = self._safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

            # 1. MOAT: Consistent high ROE
            moat_score, roe_avg = self._score_moat(info, financials, balance_sheet)

            # 2. EARNINGS QUALITY: Predictable, growing
            earnings_score = self._score_earnings_quality(info)

            # 3. DEBT HEALTH
            debt_score, de_ratio = self._score_debt(info)

            # 4. CASH GENERATION
            cash_score, fcf_yield = self._score_cash_generation(info, cashflow)

            # 5. MARGIN QUALITY
            margin_score = self._score_margins(info)

            # 6. VALUATION: Margin of safety
            val_score, intrinsic, mos = self._score_valuation(info, current_price)

            # 7. MANAGEMENT: Capital allocation
            mgmt_score = self._score_management(info)

            # Composite score: weighted average
            weights = {
                "moat": 0.20, "earnings": 0.15, "debt": 0.15,
                "cash": 0.15, "margins": 0.10, "valuation": 0.15,
                "management": 0.10,
            }
            total = (
                moat_score * weights["moat"]
                + earnings_score * weights["earnings"]
                + debt_score * weights["debt"]
                + cash_score * weights["cash"]
                + margin_score * weights["margins"]
                + val_score * weights["valuation"]
                + mgmt_score * weights["management"]
            )

            # Disqualifiers
            disqualifiers = []
            if de_ratio is not None and de_ratio > 1.0:
                disqualifiers.append(f"High debt: D/E={de_ratio:.1f}")
            if roe_avg is not None and roe_avg < 0.08:
                disqualifiers.append(f"Low ROE: {roe_avg:.1%}")
            pe = self._safe_float(info.get("trailingPE"))
            if pe is not None and pe > 50:
                disqualifiers.append(f"Very high P/E: {pe:.1f}")
            if pe is not None and pe < 0:
                disqualifiers.append("Negative earnings")

            passes = total >= 65 and len(disqualifiers) == 0

            return BuffettScore(
                symbol=symbol,
                total_score=round(total, 1),
                moat_score=round(moat_score, 1),
                earnings_quality=round(earnings_score, 1),
                debt_health=round(debt_score, 1),
                cash_generation=round(cash_score, 1),
                margin_quality=round(margin_score, 1),
                valuation_score=round(val_score, 1),
                management_score=round(mgmt_score, 1),
                roe_10yr_avg=round(roe_avg, 4) if roe_avg else None,
                debt_to_equity=round(de_ratio, 2) if de_ratio else None,
                fcf_yield=round(fcf_yield, 4) if fcf_yield else None,
                pe_ratio=pe,
                intrinsic_value=round(intrinsic, 2) if intrinsic else None,
                margin_of_safety_pct=round(mos, 4) if mos else None,
                current_price=current_price,
                passes_buffett_screen=passes,
                disqualifiers=disqualifiers,
            )

        except Exception as e:
            log.warning(f"Buffett screen failed for {symbol}: {e}")
            return BuffettScore(
                symbol=symbol, total_score=0,
                moat_score=0, earnings_quality=0, debt_health=50,
                cash_generation=0, margin_quality=0,
                valuation_score=0, management_score=0,
            )

    def screen_multiple(self, symbols: list[str]) -> list[BuffettScore]:
        """Screen multiple symbols and return sorted by total score."""
        results = []
        for sym in symbols:
            score = self.screen(sym)
            results.append(score)
            log.debug(
                f"{sym}: Buffett score={score.total_score:.0f} "
                f"{'PASS' if score.passes_buffett_screen else 'FAIL'}"
            )

        results.sort(key=lambda s: s.total_score, reverse=True)
        return results

    # ── Individual scoring methods ────────────────────────────────────────

    def _score_moat(self, info, financials, balance_sheet) -> tuple[float, Optional[float]]:
        """Score competitive advantage based on ROE consistency."""
        roe = self._safe_float(info.get("returnOnEquity"))
        if roe is None:
            return 50.0, None

        # Score based on ROE level
        if roe > 0.25:
            score = 90
        elif roe > 0.20:
            score = 80
        elif roe > 0.15:
            score = 65
        elif roe > 0.10:
            score = 45
        elif roe > 0.05:
            score = 25
        else:
            score = 10

        return float(score), roe

    def _score_earnings_quality(self, info) -> float:
        """Score earnings predictability and growth."""
        score = 50.0

        earnings_growth = self._safe_float(info.get("earningsGrowth"))
        revenue_growth = self._safe_float(info.get("revenueGrowth"))
        quarterly_growth = self._safe_float(info.get("earningsQuarterlyGrowth"))

        if earnings_growth is not None:
            if earnings_growth > 0.20:
                score += 25
            elif earnings_growth > 0.10:
                score += 15
            elif earnings_growth > 0:
                score += 5
            else:
                score -= 20

        if revenue_growth is not None:
            if revenue_growth > 0.15:
                score += 15
            elif revenue_growth > 0.05:
                score += 8
            elif revenue_growth < 0:
                score -= 15

        if quarterly_growth is not None and quarterly_growth > 0:
            score += 10

        return max(0, min(100, score))

    def _score_debt(self, info) -> tuple[float, Optional[float]]:
        """Score financial health — lower debt is better."""
        de = self._safe_float(info.get("debtToEquity"))
        if de is None:
            return 50.0, None

        de = de / 100 if de > 10 else de  # Some sources report as %

        if de < 0.1:
            score = 95  # Almost debt-free (Buffett's favorite)
        elif de < 0.3:
            score = 85
        elif de < 0.5:
            score = 70
        elif de < 0.8:
            score = 50
        elif de < 1.0:
            score = 30
        else:
            score = 10

        # Current ratio bonus
        cr = self._safe_float(info.get("currentRatio"))
        if cr is not None:
            if cr > 2.0:
                score += 5
            elif cr < 1.0:
                score -= 10

        return max(0, min(100, score)), de

    def _score_cash_generation(self, info, cashflow) -> tuple[float, Optional[float]]:
        """Score free cash flow generation ability."""
        fcf = self._safe_float(info.get("freeCashflow"))
        market_cap = self._safe_float(info.get("marketCap"))

        if fcf is None or market_cap is None or market_cap <= 0:
            return 50.0, None

        fcf_yield = fcf / market_cap

        if fcf_yield > 0.10:
            score = 95
        elif fcf_yield > 0.07:
            score = 85
        elif fcf_yield > 0.05:
            score = 70
        elif fcf_yield > 0.03:
            score = 55
        elif fcf_yield > 0:
            score = 35
        else:
            score = 10  # Negative FCF is bad

        return float(score), fcf_yield

    def _score_margins(self, info) -> float:
        """Score profitability margins."""
        score = 50.0

        gross = self._safe_float(info.get("grossMargins"))
        operating = self._safe_float(info.get("operatingMargins"))
        net = self._safe_float(info.get("profitMargins"))

        if gross is not None:
            if gross > 0.60:
                score += 20  # Strong moat indicator
            elif gross > 0.40:
                score += 12
            elif gross > 0.20:
                score += 5
            else:
                score -= 10

        if operating is not None:
            if operating > 0.25:
                score += 15
            elif operating > 0.15:
                score += 8
            elif operating < 0:
                score -= 20

        if net is not None:
            if net > 0.20:
                score += 10
            elif net > 0.10:
                score += 5
            elif net < 0:
                score -= 15

        return max(0, min(100, score))

    def _score_valuation(self, info, current_price) -> tuple[float, Optional[float], Optional[float]]:
        """Score valuation using simplified DCF for margin of safety."""
        pe = self._safe_float(info.get("trailingPE"))
        forward_pe = self._safe_float(info.get("forwardPE"))
        target = self._safe_float(info.get("targetMeanPrice"))

        score = 50.0
        intrinsic = None
        mos = None

        # P/E based valuation
        if pe is not None:
            if 5 < pe < 15:
                score += 25  # Great value
            elif 15 <= pe < 20:
                score += 10  # Fair value
            elif 20 <= pe < 25:
                score += 0
            elif pe >= 25:
                score -= 15  # Expensive
            elif pe <= 0:
                score -= 25  # Negative earnings

        # Forward P/E improvement
        if forward_pe is not None and pe is not None:
            if forward_pe < pe * 0.8:
                score += 10  # Earnings expected to grow

        # Analyst target as intrinsic value proxy
        if target and current_price and current_price > 0:
            intrinsic = target
            mos = (target - current_price) / target
            if mos > 0.30:
                score += 20  # > 30% margin of safety
            elif mos > 0.20:
                score += 12
            elif mos > 0.10:
                score += 5
            elif mos < -0.10:
                score -= 10  # Overvalued

        # Simple DCF: intrinsic = EPS * (8.5 + 2*growth_rate) * 4.4/current_yield
        eps = self._safe_float(info.get("trailingEps"))
        growth = self._safe_float(info.get("earningsGrowth"))
        if eps and growth and eps > 0:
            # Benjamin Graham formula (simplified)
            graham_value = eps * (8.5 + 2 * growth * 100) * 4.4 / 6.0  # ~6% yield
            if graham_value > 0 and current_price:
                intrinsic = graham_value
                mos = (graham_value - current_price) / graham_value
                if mos > 0.25:
                    score += 10

        return max(0, min(100, score)), intrinsic, mos

    def _score_management(self, info) -> float:
        """Score management quality through capital allocation metrics."""
        score = 50.0

        roa = self._safe_float(info.get("returnOnAssets"))
        roe = self._safe_float(info.get("returnOnEquity"))
        payout = self._safe_float(info.get("payoutRatio"))

        # ROIC proxy (ROE with debt check)
        if roe is not None:
            if roe > 0.20:
                score += 15
            elif roe > 0.15:
                score += 8

        if roa is not None:
            if roa > 0.10:
                score += 10
            elif roa > 0.05:
                score += 5

        # Reasonable payout (not too high, not zero)
        if payout is not None:
            if 0.20 <= payout <= 0.60:
                score += 10  # Healthy dividend policy
            elif payout > 0.90:
                score -= 10  # Unsustainable

        return max(0, min(100, score))

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None or val == "N/A":
            return None
        try:
            f = float(val)
            return f if not np.isnan(f) else None
        except (TypeError, ValueError):
            return None
