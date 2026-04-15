"""
Sentiment analyzer using VADER (rule-based, no model download needed).
Falls back to TextBlob for additional signal.
Outputs a normalized sentiment score: -1.0 (very negative) to +1.0 (very positive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.news_fetcher import NewsArticle
from core.logger import get_logger

log = get_logger("sentiment")


@dataclass
class SentimentResult:
    symbol: str
    overall_score: float        # -1.0 to +1.0
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    top_headlines: list[str]    # Top 5 headlines for LLM context


class SentimentAnalyzer:

    def __init__(self):
        self._vader = None
        self._vader_loaded = False

    def _load_vader(self):
        if self._vader_loaded:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._vader_loaded = True
        except ImportError:
            log.warning("VADER not available — using basic keyword matching")
            self._vader_loaded = True

    def analyze(self, symbol: str, articles: list[NewsArticle]) -> SentimentResult:
        """Compute sentiment score from a list of news articles."""
        self._load_vader()

        if not articles:
            return SentimentResult(
                symbol=symbol, overall_score=0.0, article_count=0,
                positive_count=0, negative_count=0, neutral_count=0,
                top_headlines=[],
            )

        scores = []
        for article in articles:
            text = f"{article.title}. {article.description or ''}"
            score = self._score_text(text)
            article.sentiment_score = score
            scores.append(score)

        overall = sum(scores) / len(scores) if scores else 0.0

        pos = sum(1 for s in scores if s > 0.1)
        neg = sum(1 for s in scores if s < -0.1)
        neu = len(scores) - pos - neg

        headlines = [a.title for a in articles[:5] if a.title]

        return SentimentResult(
            symbol=symbol,
            overall_score=round(overall, 3),
            article_count=len(articles),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            top_headlines=headlines,
        )

    def _score_text(self, text: str) -> float:
        """Return sentiment score -1 to +1."""
        if self._vader and hasattr(self._vader, "polarity_scores"):
            scores = self._vader.polarity_scores(text)
            return scores["compound"]  # Already -1 to +1

        # Basic keyword fallback
        text_lower = text.lower()
        positive_words = [
            "strong", "growth", "beat", "record", "surge", "rally", "upgrade",
            "buy", "bullish", "profit", "revenue", "earnings beat", "raised",
        ]
        negative_words = [
            "weak", "miss", "decline", "fall", "drop", "downgrade", "sell",
            "bearish", "loss", "cut", "layoff", "lawsuit", "fraud", "warning",
        ]
        pos_score = sum(1 for w in positive_words if w in text_lower)
        neg_score = sum(1 for w in negative_words if w in text_lower)
        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return (pos_score - neg_score) / total
