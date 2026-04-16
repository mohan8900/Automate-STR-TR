"""
Dynamic watchlists for US and Indian markets.
Fetches live Nifty 50/100 and S&P 500 from web sources.
Falls back to cached/static lists if fetch fails.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.logger import get_logger

log = get_logger("watchlists")

_CACHE_DIR = Path("data")
_CACHE_TTL = 86400  # 24 hours


# ── Dynamic fetchers ─────────────────────────────────────────────────────────

def _fetch_nifty_symbols() -> list[str]:
    """Fetch current Nifty 50 + Nifty Next 50 symbols from Wikipedia/NSE."""
    try:
        import pandas as pd
        # Nifty 50 from Wikipedia
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/NIFTY_50", match="Symbol"
        )
        if tables:
            symbols = tables[0]["Symbol"].tolist()
            nse_symbols = [f"{s.strip()}.NS" for s in symbols if isinstance(s, str)]
            log.info(f"Fetched {len(nse_symbols)} Nifty 50 symbols from Wikipedia")
            return nse_symbols
    except Exception as e:
        log.debug(f"Wikipedia Nifty fetch failed: {e}")

    try:
        import requests
        # Fallback: NSE direct API
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        resp = requests.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            symbols = [
                f"{item['symbol']}.NS"
                for item in data.get("data", [])
                if item.get("symbol") and item["symbol"] != "NIFTY 50"
            ]
            if symbols:
                log.info(f"Fetched {len(symbols)} Nifty 50 symbols from NSE")
                return symbols
    except Exception as e:
        log.debug(f"NSE API Nifty fetch failed: {e}")

    return []


def _fetch_sp500_symbols() -> list[str]:
    """Fetch current S&P 500 symbols from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        if tables:
            symbols = tables[0]["Symbol"].tolist()
            clean = [s.strip().replace(".", "-") for s in symbols if isinstance(s, str)]
            log.info(f"Fetched {len(clean)} S&P 500 symbols from Wikipedia")
            return clean
    except Exception as e:
        log.debug(f"S&P 500 fetch failed: {e}")
    return []


def _load_cached(name: str) -> list[str]:
    """Load cached watchlist from disk."""
    cache_file = _CACHE_DIR / f"watchlist_{name}.json"
    try:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            age = time.time() - data.get("timestamp", 0)
            if age < _CACHE_TTL:
                return data["symbols"]
            log.debug(f"Cache expired for {name} (age: {age/3600:.0f}h)")
    except Exception:
        pass
    return []


def _save_cache(name: str, symbols: list[str]) -> None:
    """Save watchlist to disk cache."""
    cache_file = _CACHE_DIR / f"watchlist_{name}.json"
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "symbols": symbols,
            "timestamp": time.time(),
            "count": len(symbols),
        }, indent=2))
    except Exception:
        pass


# ── Fallback static lists (used when fetch fails) ───────────────────────────

_FALLBACK_NIFTY = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "MARICO.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "MARUTI.NS", "TATMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    "BHARTIARTL.NS",
    "TITAN.NS", "ASIANPAINT.NS", "BRITANNIA.NS",
    "ULTRACEMCO.NS", "GRASIM.NS", "LT.NS", "ADANIPORTS.NS",
    "NIFTYBEES.NS", "JUNIORBEES.NS",
]

_FALLBACK_US = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "AMD",
    "ORCL", "QCOM", "CRM", "ADBE", "PANW", "CRWD",
    "JPM", "BAC", "GS", "MS", "BLK",
    "UNH", "JNJ", "LLY", "ABBV", "MRK",
    "TSLA", "HD", "MCD", "COST",
    "XOM", "CVX",
    "CAT", "HON", "GE",
    "SPY", "QQQ", "IWM",
]


# ── Public API ───────────────────────────────────────────────────────────────

def get_watchlist(exchange: str = "IN") -> list[str]:
    """
    Get the current watchlist for the given exchange.
    Tries: cache -> live fetch -> fallback static list.
    """
    name = "nifty" if exchange == "IN" else "sp500"

    # Try cache first
    cached = _load_cached(name)
    if cached:
        return cached

    # Try live fetch
    if exchange == "IN":
        symbols = _fetch_nifty_symbols()
        fallback = _FALLBACK_NIFTY
    else:
        symbols = _fetch_sp500_symbols()
        fallback = _FALLBACK_US

    if symbols:
        _save_cache(name, symbols)
        return symbols

    # Fallback
    log.warning(f"Using fallback {name} watchlist ({len(fallback)} symbols)")
    return fallback


# ── Backward-compatible exports (used throughout the codebase) ───────────────
# These are loaded once at import time. The dynamic fetcher runs on first access.

IN_WATCHLIST: list[str] = get_watchlist("IN")
US_WATCHLIST: list[str] = get_watchlist("US")

# Sector mapping for Indian stocks
IN_SECTOR_ETF_PROXIES: dict[str, str] = {
    "IT": "TCS.NS",
    "Banking": "HDFCBANK.NS",
    "Energy": "RELIANCE.NS",
    "FMCG": "HINDUNILVR.NS",
    "Pharma": "SUNPHARMA.NS",
    "Auto": "MARUTI.NS",
}

# Sector ETF mapping for US
US_SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Communication Services": "XLC",
}

# VIX symbols by market
VIX_SYMBOL: dict[str, str] = {
    "US": "^VIX",
    "IN": "^INDIAVIX",
}

# Benchmark index by market
BENCHMARK: dict[str, str] = {
    "US": "SPY",
    "IN": "^NSEI",  # Nifty 50
}
