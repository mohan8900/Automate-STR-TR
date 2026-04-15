"""
Analysis Aggregator — builds the AnalysisBundle data contract.
This is the central data object passed to the LLM for decision making.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from analysis.technical.indicators import TechnicalAnalyzer, TechnicalSignals
from analysis.technical.support_resistance import SupportResistanceAnalyzer
from analysis.technical.market_regime import MarketRegimeClassifier, MarketRegime, RegimeReading
from analysis.fundamental.scorer import FundamentalScorer
from analysis.sentiment.news_sentiment import SentimentAnalyzer, SentimentResult
from data.price_feed import PriceFeed
from data.fundamental_fetcher import FundamentalData
from data.news_fetcher import NewsFetcher
from data.earnings_calendar import EarningsCalendar, EarningsInfo
from data.vix_monitor import VixMonitor, VixReading
from config.settings import TradingSystemConfig
from core.logger import get_logger

log = get_logger("aggregator")


@dataclass
class CurrentPosition:
    """State of an existing open position in this symbol."""
    shares: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    days_held: int
    stop_loss_price: float
    take_profit_price: float


@dataclass
class AnalysisBundle:
    """
    Complete analysis package for a single stock symbol.
    This is the unified data contract fed to the LLM analyst.
    """
    symbol: str
    timestamp: datetime
    current_price: float

    # Market context
    market_regime: RegimeReading
    vix: VixReading

    # Analysis layers
    technical: TechnicalSignals
    fundamental: FundamentalData
    sentiment: SentimentResult
    earnings: EarningsInfo

    # Existing position (None if not held)
    current_position: Optional[CurrentPosition]

    # Pre-computed composite score (used for LLM pre-filtering)
    technical_score: float
    fundamental_score: float
    sentiment_score_normalized: float   # 0–100 (was -1 to +1)
    composite_score: float

    # Analyst data
    analyst_upside_pct: float

    def to_llm_prompt_text(self) -> str:
        """
        Serialize this bundle into human-readable text for the LLM prompt.
        Pre-formats numbers with context labels so the LLM doesn't have to interpret raw values.
        """
        t = self.technical
        f = self.fundamental
        s = self.sentiment
        e = self.earnings

        position_text = "None — no open position"
        if self.current_position:
            p = self.current_position
            position_text = (
                f"{p.shares:.0f} shares @ ${p.avg_cost:.2f} avg cost | "
                f"P&L: {p.unrealized_pnl_pct:+.1%} | {p.days_held} days held | "
                f"Stop: ${p.stop_loss_price:.2f} | Target: ${p.take_profit_price:.2f}"
            )

        earnings_text = "No upcoming earnings within 30 days"
        if e.days_until_earnings is not None:
            flag = " ⚠️ BLACKOUT" if e.within_3_days else (" ⚠️" if e.within_7_days else "")
            earnings_text = f"Earnings in {e.days_until_earnings} days ({e.next_earnings_date}){flag}"

        return f"""
=== STOCK ANALYSIS: {self.symbol} ===
Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M')}
Current Price: ${self.current_price:.2f}
Current Position: {position_text}

--- MARKET CONTEXT ---
Market Regime: {self.market_regime.regime.value} — {self.market_regime.description}
VIX Level: {self.vix.level:.1f} ({self.vix.regime.value}) — {self.vix.description}
Earnings: {earnings_text}

--- TECHNICAL ANALYSIS (Score: {self.technical_score:.0f}/100) ---
Trend: Price {t.price_vs_sma50_pct:+.1%} vs SMA50, {t.price_vs_sma200_pct:+.1%} vs SMA200
MA Cross: {'Golden Cross (BULLISH)' if t.golden_cross else 'Death Cross (BEARISH)'}
RSI(14): {t.rsi_14:.1f} — {t.rsi_signal.upper()} {('| Divergence: ' + t.rsi_divergence) if t.rsi_divergence != 'none' else ''}
MACD: Histogram {t.macd_histogram:+.4f}, Signal {t.macd_signal_line:.4f} {('| ' + t.macd_crossover.upper() + ' CROSSOVER') if t.macd_crossover != 'none' else ''}
Bollinger Bands: Position {t.bb_position:.1%} of band range (0%=lower, 100%=upper)
ATR(14): ${t.atr_14:.2f} ({t.atr_pct:.1%} of price) — volatility measure
Volume: {t.relative_volume:.1f}x average | OBV trend: {t.obv_trend.upper()}
Stochastic: K={t.stoch_k:.0f}, D={t.stoch_d:.0f}
{'Nearest Support: $' + str(t.nearest_support) if t.nearest_support else ''}
{'Nearest Resistance: $' + str(t.nearest_resistance) if t.nearest_resistance else ''}
{'Risk/Reward to Resistance: 1:' + str(t.risk_reward_to_resistance) if t.risk_reward_to_resistance else ''}

--- FUNDAMENTAL ANALYSIS (Score: {self.fundamental_score:.0f}/100) ---
Sector: {f.sector or 'Unknown'} | Industry: {f.industry or 'Unknown'}
Valuation: P/E={f.pe_ratio or 'N/A'}, Forward P/E={f.forward_pe or 'N/A'}, PEG={f.peg_ratio or 'N/A'}, P/B={f.price_to_book or 'N/A'}
Growth: Revenue {(f.revenue_growth_yoy or 0):+.1%} YoY, Earnings {(f.earnings_growth_yoy or 0):+.1%} YoY
Margins: Gross {(f.gross_margin or 0):.1%}, Operating {(f.operating_margin or 0):.1%}, Net {(f.net_margin or 0):.1%}
FCF Yield: {(f.fcf_yield or 0):.1%} | ROE: {(f.return_on_equity or 0):.1%}
Balance Sheet: D/E={f.debt_to_equity or 'N/A'}, Current Ratio={f.current_ratio or 'N/A'}
Dividend Yield: {(f.dividend_yield or 0):.1%} | Beta: {f.beta or 'N/A'}
Analyst Target: ${f.analyst_target_price or 'N/A'} ({self.analyst_upside_pct:+.1%} upside) | Rating: {f.analyst_recommendation or 'N/A'}

--- SENTIMENT ANALYSIS (Score: {self.sentiment_score_normalized:.0f}/100) ---
Overall Sentiment: {self.sentiment.overall_score:+.2f} (scale: -1 very negative to +1 very positive)
Articles: {s.article_count} analyzed | Positive: {s.positive_count} | Negative: {s.negative_count} | Neutral: {s.neutral_count}
Recent Headlines:
{chr(10).join('  • ' + h for h in s.top_headlines[:5])}

--- COMPOSITE SCORES ---
Technical: {self.technical_score:.0f}/100 | Fundamental: {self.fundamental_score:.0f}/100 | Sentiment: {self.sentiment_score_normalized:.0f}/100
Composite (40/35/25 weighted): {self.composite_score:.0f}/100
""".strip()


class AnalysisAggregator:
    """Orchestrates all analysis layers and builds the AnalysisBundle."""

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        exchange = config.market.exchange
        self.price_feed = PriceFeed(exchange=exchange)
        self.tech_analyzer = TechnicalAnalyzer()
        self.sr_analyzer = SupportResistanceAnalyzer()
        self.regime_classifier = MarketRegimeClassifier(exchange=exchange)
        self.fundamental_scorer = FundamentalScorer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_fetcher = NewsFetcher(exchange=exchange)
        self.earnings_cal = EarningsCalendar(
            blackout_days=config.trading.earnings_blackout_days
        )
        self.vix_monitor = VixMonitor(config.vix, exchange=exchange)

        # Cache regime/VIX — only recalculate once per cycle
        self._cached_regime: Optional[RegimeReading] = None
        self._cached_vix: Optional[VixReading] = None

    def refresh_market_context(self) -> None:
        """Call once per scan cycle to refresh regime and VIX."""
        self._cached_regime = self.regime_classifier.classify()
        self._cached_vix = self.vix_monitor.get()
        log.info(
            f"Market context: {self._cached_regime.regime.value} | "
            f"VIX {self._cached_vix.level:.1f} ({self._cached_vix.regime.value})"
        )

    def build_bundle(
        self,
        symbol: str,
        current_position: Optional[CurrentPosition] = None,
    ) -> AnalysisBundle:
        """Build a complete AnalysisBundle for a single symbol."""
        log.debug(f"Building analysis bundle for {symbol}")

        regime = self._cached_regime or self.regime_classifier.classify()
        vix = self._cached_vix or self.vix_monitor.get()

        # Fetch price data
        df = self.price_feed.get_historical(symbol, period="1y", interval="1d")
        current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0

        # Technical analysis
        tech = self.tech_analyzer.analyze(symbol, df)
        sr = self.sr_analyzer.analyze(symbol, df)
        tech.nearest_support = sr.nearest_support
        tech.nearest_resistance = sr.nearest_resistance
        tech.risk_reward_to_resistance = sr.risk_reward_to_resistance

        # Fundamental analysis
        fundamental = self.fundamental_scorer.fetch_and_score(symbol)
        analyst_upside = self.fundamental_scorer.get_upside_pct(fundamental, current_price)

        # Sentiment analysis
        articles = self.news_fetcher.fetch_for_symbol(symbol, days_back=3)
        sentiment = self.sentiment_analyzer.analyze(symbol, articles)

        # Earnings
        earnings = self.earnings_cal.get_earnings_info(symbol)

        # Normalize sentiment to 0–100
        sentiment_normalized = (sentiment.overall_score + 1) / 2 * 100

        # Composite score: 40% technical, 35% fundamental, 25% sentiment
        composite = (
            0.40 * tech.technical_score
            + 0.35 * fundamental.fundamental_score
            + 0.25 * sentiment_normalized
        )

        return AnalysisBundle(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            market_regime=regime,
            vix=vix,
            technical=tech,
            fundamental=fundamental,
            sentiment=sentiment,
            earnings=earnings,
            current_position=current_position,
            technical_score=tech.technical_score,
            fundamental_score=fundamental.fundamental_score,
            sentiment_score_normalized=round(sentiment_normalized, 1),
            composite_score=round(composite, 1),
            analyst_upside_pct=analyst_upside,
        )
