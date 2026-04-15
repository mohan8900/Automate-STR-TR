"""
Market data service — OHLCV bars via yfinance (free, both US & India).
Caches results in-memory with TTL to reduce API hammering.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.exceptions import MarketDataError
from core.logger import get_logger

log = get_logger("price_feed")

_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}  # key -> (timestamp, df)
_CACHE_TTL_SECONDS = 300  # 5 minutes


class PriceFeed:
    """Fetch and cache OHLCV data for both US and Indian stocks."""

    def __init__(self, exchange: str = "US"):
        self.exchange = exchange

    # ── Public API ────────────────────────────────────────────────────────

    def get_historical(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for the given symbol.
        Columns: open, high, low, close, volume (lowercase)
        """
        cache_key = f"{symbol}:{period}:{interval}"
        cached = _CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1].copy()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise MarketDataError(f"No data returned for {symbol}")

            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            df = self._validate_data(symbol, df)

            _CACHE[cache_key] = (time.time(), df)
            log.debug(f"Fetched {len(df)} bars for {symbol} ({period}/{interval})")
            return df.copy()

        except MarketDataError:
            raise
        except Exception as e:
            raise MarketDataError(f"Failed to fetch data for {symbol}: {e}") from e

    def get_current_price(self, symbol: str) -> float:
        """Return the latest close price."""
        df = self.get_historical(symbol, period="5d", interval="1d")
        if df.empty:
            raise MarketDataError(f"Cannot get price for {symbol}")
        return float(df["close"].iloc[-1])

    def get_multiple(
        self,
        symbols: list[str],
        period: str = "6mo",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch multiple symbols. Skips failures silently."""
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym] = self.get_historical(sym, period, interval)
            except MarketDataError as e:
                log.warning(f"Skipping {sym}: {e}")
        return results

    def get_intraday(
        self,
        symbol: str,
        interval: str = "15m",
        period: str = "5d",
    ) -> pd.DataFrame:
        """Intraday bars (15m, 30m, 1h). Max 60 days lookback from yfinance."""
        return self.get_historical(symbol, period=period, interval=interval)

    def get_52week_range(self, symbol: str) -> tuple[float, float]:
        """Return (52w_low, 52w_high)."""
        df = self.get_historical(symbol, period="1y", interval="1d")
        return float(df["low"].min()), float(df["high"].max())

    def get_average_volume(self, symbol: str, days: int = 20) -> float:
        """Average daily volume over last N days."""
        df = self.get_historical(symbol, period="3mo", interval="1d")
        return float(df["volume"].tail(days).mean())

    def get_relative_volume(self, symbol: str) -> float:
        """Today's volume vs. 20-day average volume."""
        df = self.get_historical(symbol, period="3mo", interval="1d")
        if len(df) < 2:
            return 1.0
        avg_vol = float(df["volume"].iloc[:-1].tail(20).mean())
        today_vol = float(df["volume"].iloc[-1])
        return today_vol / avg_vol if avg_vol > 0 else 1.0

    def get_momentum(self, symbol: str, days: int = 5) -> float:
        """N-day price return as a decimal (e.g., 0.03 = +3%)."""
        df = self.get_historical(symbol, period="1mo", interval="1d")
        if len(df) < days + 1:
            return 0.0
        return float(df["close"].iloc[-1] / df["close"].iloc[-(days + 1)] - 1)

    # ── Data validation (Indian market protections) ────────────────────

    def _validate_data(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run sanity checks on fetched OHLCV data and return a cleaned
        DataFrame. Catches common issues with Indian stock data
        (splits, bonus issues, data gaps) without crashing.
        """
        if df.empty:
            return df

        # 1. Corporate action detection (suspicious single-day jumps)
        pct_change = df["close"].pct_change().abs()
        spike_mask = pct_change > 0.40
        for dt in df.index[spike_mask]:
            prev_close = df["close"].shift(1).loc[dt]
            curr_close = df["close"].loc[dt]
            log.warning(
                f"[{symbol}] Suspicious price jump on {dt.date()}: "
                f"{prev_close:.2f} -> {curr_close:.2f} "
                f"({pct_change.loc[dt]:.1%} change). "
                f"Possible stock split or bonus issue."
            )

        # 2. Missing data handling
        if len(df) >= 2:
            start, end = df.index.min(), df.index.max()
            expected = max(int(np.busday_count(start.date(), end.date())) + 1, 1)
            actual = len(df)
            missing_ratio = 1 - (actual / expected)

            if missing_ratio > 0.10:
                log.warning(
                    f"[{symbol}] {missing_ratio:.1%} of expected trading days "
                    f"missing ({actual}/{expected} days)."
                )

            # Forward-fill small gaps (1-2 consecutive missing days)
            full_idx = pd.bdate_range(start=start, end=end)
            df = df.reindex(full_idx)
            df = df.ffill(limit=2)
            df = df.dropna()

        # 3. Price sanity check (OHLC consistency & volume >= 0)
        valid_mask = (
            (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["open"] <= df["high"])
            & (df["close"] <= df["high"])
            & (df["volume"] >= 0)
        )
        bad_rows = (~valid_mask).sum()
        if bad_rows > 0:
            log.warning(
                f"[{symbol}] Dropping {bad_rows} rows with inconsistent OHLC values."
            )
            df = df[valid_mask]

        # 4. Stale data detection (latest point > 3 trading days old)
        if len(df) > 0:
            latest = df.index.max()
            now = pd.Timestamp.now(tz=latest.tz) if latest.tz else pd.Timestamp.now()
            days_since = int(np.busday_count(latest.date(), now.date()))
            if days_since > 3:
                log.warning(
                    f"[{symbol}] Stale data: latest bar is from "
                    f"{latest.date()} ({days_since} trading days ago)."
                )

        return df

    def clear_cache(self) -> None:
        _CACHE.clear()
