"""
Fundamental scorer — wraps FundamentalFetcher with sector-relative Z-score comparison.
"""
from __future__ import annotations

from data.fundamental_fetcher import FundamentalData, FundamentalFetcher


class FundamentalScorer:
    """Thin wrapper — fundamental scoring is already in FundamentalFetcher._score()."""

    def __init__(self):
        self.fetcher = FundamentalFetcher()

    def fetch_and_score(self, symbol: str) -> FundamentalData:
        return self.fetcher.fetch(symbol)

    def get_upside_pct(self, data: FundamentalData, current_price: float) -> float:
        """Calculate analyst price target upside as a percentage."""
        if data.analyst_target_price and current_price > 0:
            return (data.analyst_target_price / current_price) - 1
        return 0.0

    def analyst_rating_to_score(self, recommendation: str | None) -> float:
        """Convert analyst recommendation string to 0-100 score."""
        mapping = {
            "strong_buy": 95, "strongbuy": 95,
            "buy": 75, "outperform": 70, "overweight": 70,
            "hold": 50, "neutral": 50, "equalweight": 50,
            "underperform": 30, "underweight": 30,
            "sell": 15, "strongsell": 10, "strong_sell": 10,
        }
        if not recommendation:
            return 50.0
        return mapping.get(recommendation.lower().replace(" ", ""), 50.0)
