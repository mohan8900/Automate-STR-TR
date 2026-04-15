"""
News fetcher — pulls recent headlines from NewsAPI and RSS feeds.
Used for sentiment analysis and LLM context injection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests
import feedparser

from core.logger import get_logger

log = get_logger("news")

_CACHE: dict[str, tuple[float, list["NewsArticle"]]] = {}
_CACHE_TTL = 900  # 15 minutes


@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    published_at: Optional[datetime]
    url: str
    sentiment_score: float = 0.0  # Filled by sentiment analyzer


class NewsFetcher:

    # Free RSS feeds for financial news (no API key needed)
    RSS_FEEDS_US = {
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "CNBC Markets": "https://search.cnbc.com/rs/search/combinedcombined/view/rss/",
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    }

    RSS_FEEDS_IN = {
        "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Economic Times Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
        "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
        "LiveMint Markets": "https://www.livemint.com/rss/markets",
        "Business Standard Markets": "https://www.business-standard.com/rss/markets-106.rss",
        "NDTV Profit": "https://feeds.feedburner.com/ndtvprofit-latest",
    }

    def __init__(self, newsapi_key: Optional[str] = None, exchange: str = "US"):
        self.newsapi_key = newsapi_key
        self.exchange = exchange

    def fetch_for_symbol(self, symbol: str, days_back: int = 3) -> list[NewsArticle]:
        """Fetch news articles mentioning the given stock symbol."""
        cache_key = f"{symbol}:{days_back}"
        cached = _CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < _CACHE_TTL:
            return cached[1]

        articles: list[NewsArticle] = []

        # Try NewsAPI first (better quality, requires key)
        if self.newsapi_key:
            articles.extend(self._fetch_newsapi(symbol, days_back))

        # Always try yfinance news (free)
        articles.extend(self._fetch_yfinance_news(symbol))

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique = []
        for a in articles:
            key = a.title[:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(a)

        # Sort by recency
        unique.sort(key=lambda x: x.published_at or datetime.min, reverse=True)
        result = unique[:10]  # Top 10 most recent

        _CACHE[cache_key] = (time.time(), result)
        return result

    def fetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """General market news for regime/sentiment analysis."""
        articles: list[NewsArticle] = []
        feeds = self.RSS_FEEDS_IN if self.exchange == "IN" else self.RSS_FEEDS_US

        for name, url in feeds.items():
            articles.extend(self._fetch_rss(url, name))

        articles.sort(key=lambda x: x.published_at or datetime.min, reverse=True)
        return articles[:limit]

    # ── Private helpers ───────────────────────────────────────────────────

    def _clean_symbol_for_search(self, symbol: str) -> str:
        """Strip exchange suffixes (.NS, .BO) for search queries."""
        return symbol.replace(".NS", "").replace(".BO", "")

    def _fetch_newsapi(self, symbol: str, days_back: int) -> list[NewsArticle]:
        try:
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            query = self._clean_symbol_for_search(symbol)
            # For Indian stocks, add "NSE" or "stock" to improve relevance
            if self.exchange == "IN":
                query = f"{query} AND (NSE OR BSE OR stock OR share)"
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey": self.newsapi_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                pub = item.get("publishedAt")
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    published_at=datetime.fromisoformat(pub.replace("Z", "+00:00")) if pub else None,
                    url=item.get("url", ""),
                ))
            return articles
        except Exception as e:
            log.debug(f"NewsAPI fetch failed for {symbol}: {e}")
            return []

    def _fetch_yfinance_news(self, symbol: str) -> list[NewsArticle]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            articles = []
            for item in news[:10]:
                pub_ts = item.get("providerPublishTime")
                pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else None
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("summary", ""),
                    source=item.get("publisher", "Yahoo Finance"),
                    published_at=pub_dt,
                    url=item.get("link", ""),
                ))
            return articles
        except Exception as e:
            log.debug(f"yfinance news failed for {symbol}: {e}")
            return []

    def _fetch_rss(self, url: str, source_name: str) -> list[NewsArticle]:
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:10]:
                pub_dt = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_dt = datetime(*entry.published_parsed[:6])
                articles.append(NewsArticle(
                    title=entry.get("title", ""),
                    description=entry.get("summary", ""),
                    source=source_name,
                    published_at=pub_dt,
                    url=entry.get("link", ""),
                ))
            return articles
        except Exception as e:
            log.debug(f"RSS fetch failed ({source_name}): {e}")
            return []
