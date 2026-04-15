"""
Curated watchlists for US and Indian markets.
S&P 500 full list is fetched dynamically from Wikipedia.
These are curated high-liquidity starting points.
"""
from __future__ import annotations

# Top 100 US large-cap tech + blue-chip stocks (high liquidity)
US_WATCHLIST: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "AMD",
    "ORCL", "QCOM", "TXN", "CRM", "INTC", "AMAT", "MU", "KLAC", "LRCX",
    "SNPS", "CDNS", "ADBE", "PANW", "CRWD", "ZS", "FTNT", "NET", "DDOG",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SPGI", "ICE", "CME",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "ISRG",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "DE", "HON", "GE", "BA", "RTX", "LMT",
    # ETFs for regime analysis
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
]

# NSE India top 100 (Nifty 100 components)
IN_WATCHLIST: list[str] = [
    # Large Cap IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS",
    # Banking & Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS",
    # Energy
    "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS",
    # FMCG
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "MARICO.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Auto
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    # Telecom
    "BHARTIARTL.NS",
    # Consumer
    "TITAN.NS", "ASIANPAINT.NS", "BRITANNIA.NS",
    # Infrastructure
    "ULTRACEMCO.NS", "GRASIM.NS", "LT.NS", "ADANIPORTS.NS",
    # Index ETFs for regime
    "NIFTYBEES.NS", "JUNIORBEES.NS",
]

# Sector ETF mapping for correlation and regime analysis
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

# Sector mapping for Indian stocks (simplified)
IN_SECTOR_ETF_PROXIES: dict[str, str] = {
    "IT": "TCS.NS",
    "Banking": "HDFCBANK.NS",
    "Energy": "RELIANCE.NS",
    "FMCG": "HINDUNILVR.NS",
    "Pharma": "SUNPHARMA.NS",
    "Auto": "MARUTI.NS",
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
