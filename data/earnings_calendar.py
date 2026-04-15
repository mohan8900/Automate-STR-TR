"""
Earnings calendar — checks upcoming earnings dates for blackout enforcement.
Uses yfinance as primary source.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd

from core.logger import get_logger

log = get_logger("earnings_calendar")

_CACHE: dict[str, tuple[float, Optional[date]]] = {}
_CACHE_TTL = 43200  # 12 hours


@dataclass
class EarningsInfo:
    symbol: str
    next_earnings_date: Optional[date]
    within_3_days: bool
    within_7_days: bool
    post_earnings_window: bool  # Within 2 days after earnings (drift opportunity)
    days_until_earnings: Optional[int]


class EarningsCalendar:

    def __init__(self, blackout_days: int = 3):
        self.blackout_days = blackout_days

    def get_earnings_info(self, symbol: str) -> EarningsInfo:
        """Return earnings timing info for a symbol."""
        next_date = self._get_next_earnings_date(symbol)
        today = date.today()

        within_3 = False
        within_7 = False
        post_window = False
        days_until = None

        if next_date:
            days_until = (next_date - today).days
            within_3 = 0 <= days_until <= 3
            within_7 = 0 <= days_until <= 7
            # Post-earnings drift: 1–2 days after earnings
            post_window = -2 <= days_until < 0

        return EarningsInfo(
            symbol=symbol,
            next_earnings_date=next_date,
            within_3_days=within_3,
            within_7_days=within_7,
            post_earnings_window=post_window,
            days_until_earnings=days_until,
        )

    def get_blackout_symbols(self, symbols: list[str]) -> list[str]:
        """Return subset of symbols that are within the earnings blackout window."""
        blackout = []
        for sym in symbols:
            info = self.get_earnings_info(sym)
            if info.within_3_days:
                blackout.append(sym)
        return blackout

    def get_upcoming_earnings(
        self, symbols: list[str], days_ahead: int = 14
    ) -> list[dict]:
        """Return sorted list of upcoming earnings events."""
        events = []
        for sym in symbols:
            info = self.get_earnings_info(sym)
            if (
                info.next_earnings_date
                and info.days_until_earnings is not None
                and 0 <= info.days_until_earnings <= days_ahead
            ):
                events.append({
                    "symbol": sym,
                    "earnings_date": info.next_earnings_date,
                    "days_until": info.days_until_earnings,
                    "blackout": info.within_3_days,
                })
        events.sort(key=lambda x: x["days_until"])
        return events

    # ── Private ───────────────────────────────────────────────────────────

    def _get_next_earnings_date(self, symbol: str) -> Optional[date]:
        cached = _CACHE.get(symbol)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]

        result = None
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is not None and not calendar.empty:
                # calendar is a DataFrame with index like 'Earnings Date'
                if "Earnings Date" in calendar.index:
                    raw = calendar.loc["Earnings Date"]
                    if isinstance(raw, pd.Series):
                        raw = raw.iloc[0]
                    if raw is not None:
                        if isinstance(raw, (datetime, pd.Timestamp)):
                            result = raw.date()
                        elif isinstance(raw, date):
                            result = raw

            # Fallback: check upcoming earnings dates
            if result is None:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    future = earnings_dates[earnings_dates.index.date >= date.today()]
                    if not future.empty:
                        result = future.index[0].date()

        except Exception as e:
            log.debug(f"Earnings date lookup failed for {symbol}: {e}")

        _CACHE[symbol] = (time.time(), result)
        return result
