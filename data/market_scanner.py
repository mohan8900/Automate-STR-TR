"""
Market scanner — filters the universe to top candidate stocks for analysis.
Applies pre-filters (liquidity, momentum, volume) before expensive LLM analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import TradingSystemConfig
from config.watchlists import US_WATCHLIST, IN_WATCHLIST
from data.price_feed import PriceFeed
from data.earnings_calendar import EarningsCalendar
from core.logger import get_logger

log = get_logger("market_scanner")


@dataclass
class ScanResult:
    symbol: str
    price: float
    market_cap: Optional[float]
    avg_daily_volume_usd: float
    relative_volume: float
    momentum_5d: float
    momentum_20d: float
    pct_from_52w_high: float
    pre_filter_score: float  # 0–100
    pass_filter: bool
    fail_reason: Optional[str] = None


class MarketScanner:

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.price_feed = PriceFeed(exchange=config.market.exchange)
        self.earnings_cal = EarningsCalendar(
            blackout_days=config.trading.earnings_blackout_days
        )

    def scan(self) -> list[ScanResult]:
        """
        Scan the full watchlist, apply pre-filters, return top N candidates
        sorted by pre-filter score.
        """
        universe = self._get_universe()
        log.info(f"Scanning {len(universe)} symbols...")

        results: list[ScanResult] = []
        for symbol in universe:
            result = self._scan_symbol(symbol)
            if result:
                results.append(result)

        # Sort by pre_filter_score descending, keep only passing ones
        passing = [r for r in results if r.pass_filter]
        passing.sort(key=lambda r: r.pre_filter_score, reverse=True)

        top_n = passing[: self.config.trading.watchlist_size]
        log.info(
            f"Scan complete: {len(passing)} passed filters, "
            f"returning top {len(top_n)}"
        )
        return top_n

    def get_full_scan_with_failures(self) -> list[ScanResult]:
        """Return all scan results including failed filters (for dashboard)."""
        universe = self._get_universe()
        results = []
        for symbol in universe:
            result = self._scan_symbol(symbol)
            if result:
                results.append(result)
        results.sort(key=lambda r: r.pre_filter_score, reverse=True)
        return results

    # ── Private ───────────────────────────────────────────────────────────

    def _get_universe(self) -> list[str]:
        if self.config.market.exchange == "US":
            base = list(US_WATCHLIST)
            # Optionally augment with dynamic S&P 500 from Wikipedia
            sp500 = self._fetch_sp500_symbols()
            combined = list(dict.fromkeys(base + sp500))  # deduplicate, preserve order
            return combined
        else:
            return list(IN_WATCHLIST)

    def _fetch_sp500_symbols(self) -> list[str]:
        """Fetch current S&P 500 constituents from Wikipedia."""
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            sp500_df = tables[0]
            symbols = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
            return symbols[:300]  # Top 300 by index weight
        except Exception as e:
            log.warning(f"Could not fetch S&P 500 list: {e}")
            return []

    def _scan_symbol(self, symbol: str) -> Optional[ScanResult]:
        try:
            cfg = self.config.trading
            df = self.price_feed.get_historical(symbol, period="1y", interval="1d")
            if df.empty or len(df) < 50:
                return None

            price = float(df["close"].iloc[-1])
            volume_today = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].tail(20).mean())
            avg_vol_usd = avg_vol * price

            # Hard filters
            if price < cfg.min_price or price > cfg.max_price:
                return ScanResult(
                    symbol=symbol, price=price, market_cap=None,
                    avg_daily_volume_usd=avg_vol_usd, relative_volume=1.0,
                    momentum_5d=0, momentum_20d=0, pct_from_52w_high=0,
                    pre_filter_score=0, pass_filter=False,
                    fail_reason=f"Price ${price:.2f} outside range"
                )
            if avg_vol_usd < cfg.min_volume_usd:
                return ScanResult(
                    symbol=symbol, price=price, market_cap=None,
                    avg_daily_volume_usd=avg_vol_usd, relative_volume=1.0,
                    momentum_5d=0, momentum_20d=0, pct_from_52w_high=0,
                    pre_filter_score=0, pass_filter=False,
                    fail_reason=f"Low liquidity (${avg_vol_usd/1e6:.1f}M daily)"
                )

            # Compute metrics
            relative_vol = volume_today / avg_vol if avg_vol > 0 else 1.0
            mom_5d = float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) >= 6 else 0
            mom_20d = float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1) if len(df) >= 21 else 0
            high_52w = float(df["high"].max())
            pct_from_high = (price - high_52w) / high_52w  # negative = below high

            # Pre-filter score (0–100)
            score = 50.0
            score += min(20, relative_vol * 5)      # Relative volume boost
            score += min(15, mom_5d * 100)          # Short momentum
            score += min(10, mom_20d * 50)          # Medium momentum
            # Near 52w high is bullish (breakout potential)
            if pct_from_high > -0.05:
                score += 10
            elif pct_from_high > -0.15:
                score += 5
            score = max(0, min(100, score))

            return ScanResult(
                symbol=symbol,
                price=price,
                market_cap=None,  # Fetched lazily in fundamental analysis
                avg_daily_volume_usd=avg_vol_usd,
                relative_volume=relative_vol,
                momentum_5d=mom_5d,
                momentum_20d=mom_20d,
                pct_from_52w_high=pct_from_high,
                pre_filter_score=score,
                pass_filter=True,
            )

        except Exception as e:
            log.debug(f"Scan failed for {symbol}: {e}")
            return None
