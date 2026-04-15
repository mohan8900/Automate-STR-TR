"""
Fundamental data fetcher using yfinance.
Provides sector-relative scoring for each stock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import yfinance as yf
import numpy as np

from core.logger import get_logger
from core.exceptions import MarketDataError

log = get_logger("fundamentals")

_CACHE: dict[str, tuple[float, "FundamentalData"]] = {}
_CACHE_TTL = 3600  # 1 hour — fundamentals don't change often


@dataclass
class FundamentalData:
    symbol: str
    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    # Growth
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    earnings_quarterly_growth: Optional[float] = None
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    # Financial Health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_yield: Optional[float] = None
    # Dividend
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    # Market
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    # Analyst
    analyst_target_price: Optional[float] = None
    analyst_recommendation: Optional[str] = None
    # Computed score
    fundamental_score: float = 50.0


class FundamentalFetcher:

    def __init__(self):
        self._sector_medians: dict[str, dict[str, float]] = {}

    def fetch(self, symbol: str) -> FundamentalData:
        cached = _CACHE.get(symbol)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]

        try:
            info = yf.Ticker(symbol).info
            data = self._parse_info(symbol, info)
            data.fundamental_score = self._score(data)
            _CACHE[symbol] = (time.time(), data)
            return data
        except Exception as e:
            log.warning(f"Fundamental fetch failed for {symbol}: {e}")
            return FundamentalData(symbol=symbol)

    def fetch_multiple(self, symbols: list[str]) -> dict[str, FundamentalData]:
        results = {}
        for sym in symbols:
            results[sym] = self.fetch(sym)
        return results

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_info(self, symbol: str, info: dict) -> FundamentalData:
        def safe(key: str) -> Optional[float]:
            v = info.get(key)
            if v is None or v == "N/A":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        market_cap = safe("marketCap")
        fcf = safe("freeCashflow")
        fcf_yield = None
        if fcf and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap

        return FundamentalData(
            symbol=symbol,
            pe_ratio=safe("trailingPE"),
            forward_pe=safe("forwardPE"),
            peg_ratio=safe("pegRatio"),
            price_to_book=safe("priceToBook"),
            price_to_sales=safe("priceToSalesTrailing12Months"),
            ev_to_ebitda=safe("enterpriseToEbitda"),
            revenue_growth_yoy=safe("revenueGrowth"),
            earnings_growth_yoy=safe("earningsGrowth"),
            earnings_quarterly_growth=safe("earningsQuarterlyGrowth"),
            gross_margin=safe("grossMargins"),
            operating_margin=safe("operatingMargins"),
            net_margin=safe("profitMargins"),
            return_on_equity=safe("returnOnEquity"),
            return_on_assets=safe("returnOnAssets"),
            debt_to_equity=safe("debtToEquity"),
            current_ratio=safe("currentRatio"),
            quick_ratio=safe("quickRatio"),
            free_cash_flow=fcf,
            fcf_yield=fcf_yield,
            dividend_yield=safe("dividendYield"),
            payout_ratio=safe("payoutRatio"),
            market_cap=market_cap,
            beta=safe("beta"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            analyst_target_price=safe("targetMeanPrice"),
            analyst_recommendation=info.get("recommendationKey"),
        )

    # ── Scoring (0–100) ───────────────────────────────────────────────────

    def _score(self, d: FundamentalData) -> float:
        """
        Composite fundamental quality score.
        Higher = better quality / more attractive stock.
        Uses a weighted scorecard approach.
        """
        scores: list[tuple[float, float]] = []  # (score, weight)

        def add(value: Optional[float], fn, weight: float):
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                scores.append((fn(value), weight))

        # Valuation — lower P/E is better (up to a point)
        add(d.pe_ratio, lambda v: max(0, min(100, 100 - (v - 10) * 2)), 1.5)
        add(d.peg_ratio, lambda v: max(0, min(100, 100 - v * 20)), 1.5)
        add(d.price_to_book, lambda v: max(0, min(100, 100 - v * 10)), 0.5)
        add(d.fcf_yield, lambda v: min(100, v * 1000), 2.0)  # 10% FCF yield = 100

        # Growth — higher is better
        add(d.revenue_growth_yoy, lambda v: min(100, max(0, 50 + v * 200)), 2.0)
        add(d.earnings_growth_yoy, lambda v: min(100, max(0, 50 + v * 150)), 2.0)

        # Profitability — higher is better
        add(d.gross_margin, lambda v: min(100, v * 150), 1.5)
        add(d.operating_margin, lambda v: min(100, max(0, 50 + v * 300)), 1.5)
        add(d.return_on_equity, lambda v: min(100, max(0, 50 + v * 200)), 1.5)

        # Financial health
        add(d.debt_to_equity, lambda v: max(0, min(100, 100 - v * 5)), 1.5)
        add(d.current_ratio, lambda v: min(100, max(0, v * 40)), 1.0)

        # Analyst sentiment
        rec_map = {
            "strong_buy": 90, "buy": 75, "hold": 50,
            "underperform": 30, "sell": 15,
        }
        if d.analyst_recommendation:
            rec_score = rec_map.get(d.analyst_recommendation.lower(), 50)
            scores.append((rec_score, 1.5))

        # Analyst price target upside
        if d.analyst_target_price:
            # Will be compared to current price in aggregator
            pass

        if not scores:
            return 50.0

        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)
        return round(weighted_sum / total_weight, 1)
