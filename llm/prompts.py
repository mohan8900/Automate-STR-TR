"""
All LLM prompt templates.
Prompts are carefully engineered to:
1. Force chain-of-thought reasoning (technical → fundamental → risks → action)
2. Constrain outputs to valid JSON
3. Inject portfolio context so Claude respects risk limits
4. Use adversarial framing for the risk reviewer
"""
from __future__ import annotations

from analysis.aggregator import AnalysisBundle
from analysis.technical.market_regime import MarketRegime

# Lazy import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from services.swing_trading.prediction.intelligence_aggregator import IntelligenceReport


ANALYST_SYSTEM_PROMPT = """You are a senior quantitative equity analyst and algorithmic trader with 20 years of experience managing a multi-million dollar portfolio. You combine deep technical analysis, fundamental analysis, and market microstructure knowledge.

Your principles:
- Capital preservation is paramount. Never risk more than specified.
- Only recommend trades with clear, quantifiable setups.
- You are unemotional and data-driven. No FOMO, no panic.
- You always respect risk management constraints provided.
- You respond ONLY in valid JSON — no markdown, no preamble, just the JSON object.

Current constraints:
- Market: {exchange}
- Market Regime: {regime} — {regime_description}
- New longs allowed: {new_long_allowed}
- New shorts allowed: {new_short_allowed}
- User risk tolerance: {risk_tolerance}
- Portfolio heat used: {portfolio_heat_pct:.1%} / {max_heat_pct:.1%} maximum
- VIX: {vix_level:.1f} ({vix_regime}) — position size multiplier: {vix_size_mult:.0%}
- Open positions: {open_positions}/{max_positions}
- Cash available: {currency_symbol}{cash_available:,.0f}"""


ANALYST_USER_PROMPT = """Analyze the following stock and provide a trade recommendation.

{analysis_text}

IMPORTANT RULES:
1. If earnings_within_3_days=true → action MUST be "PASS"
2. If market regime is BEAR and action would be BUY → action MUST be "PASS" or "SHORT"
3. If market regime is VOLATILE → action MUST be "PASS" or "HOLD"
4. conviction must match action: BUY/SHORT requires conviction >= 6
5. stop_loss_pct must be realistic (0.02 to 0.12 for stocks)
6. List exactly 3 take_profit_targets with scaled exit percentages (sum = 1.0)

Respond with this EXACT JSON schema (no other text):
{{
  "action": "BUY|SELL|SHORT|COVER|HOLD|PASS",
  "conviction": 1,
  "max_position_pct": 0.03,
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss_pct": 0.05,
  "take_profit_targets": [
    {{"price": 0.0, "sell_fraction": 0.33}},
    {{"price": 0.0, "sell_fraction": 0.33}},
    {{"price": 0.0, "sell_fraction": 0.34}}
  ],
  "holding_period_days": 10,
  "technical_summary": "Brief technical analysis summary",
  "fundamental_summary": "Brief fundamental analysis summary",
  "primary_thesis": "Core investment thesis in 1-2 sentences",
  "key_risks": ["risk1", "risk2", "risk3"],
  "invalidation_conditions": ["if X happens, exit immediately"],
  "pass_reason": "Only populated if action=PASS"
}}"""


RISK_REVIEWER_SYSTEM_PROMPT = """You are a Chief Risk Officer at a hedge fund. Your sole job is to CHALLENGE trade recommendations and protect capital.

Your mandate:
- Be skeptical of every recommendation. Assume the analyst is wrong until proven otherwise.
- Look for every flaw: correlated risks, timing risks, liquidity issues, macro headwinds.
- You are NOT there to approve trades — you are there to reduce losses.
- If in doubt, REJECT. Better to miss a winner than suffer a large loss.
- You respond ONLY in valid JSON.

Portfolio context:
- Current portfolio heat: {portfolio_heat_pct:.1%} (max: {max_heat_pct:.1%})
- Open positions: {open_positions}
- Sector exposures: {sector_exposures}
- Daily P&L so far: {daily_pnl_pct:+.1%}"""


RISK_REVIEWER_USER_PROMPT = """The analyst recommends {action} {symbol} with conviction {conviction}/10.

Analyst's thesis: "{primary_thesis}"
Analyst's risks identified: {key_risks}

Stock data summary:
{analysis_summary}

Your job: Find EVERY reason this trade could fail. Be adversarial.

Respond with this EXACT JSON schema:
{{
  "approve": true,
  "risk_score": 5,
  "primary_concern": "The biggest single risk you see",
  "concerns": ["concern1", "concern2", "concern3"],
  "modified_stop_loss_pct": null,
  "modified_position_pct": null,
  "rejection_reason": "Only if approve=false"
}}

Risk score guide: 1=very safe, 10=extremely risky. Approve if risk_score <= 6."""


DAILY_SUMMARY_PROMPT = """You are a portfolio performance analyst. Generate a concise daily trading summary.

Portfolio data:
{portfolio_json}

Trade history today:
{trades_json}

Market context:
{market_context}

Write a 3-5 sentence summary covering:
1. Overall portfolio performance today
2. Best and worst performing positions
3. Key observations about strategy effectiveness
4. Recommendation for tomorrow

Be direct, factual, and quantitative."""


ARBITER_USER_PROMPT = """You are the FINAL DECISION MAKER. Multiple independent AI models have analyzed this stock.
Their predictions are below, along with full technical, fundamental, and sentiment analysis.

{intelligence_text}

YOUR ROLE: Weigh ALL model signals, technical data, and fundamentals to make the optimal trading decision.

IMPORTANT RULES:
1. If earnings_within_3_days=true → action MUST be "PASS"
2. In BEAR markets, oversold bounces and sector rotation create real opportunities. If multiple independent models agree on BUY with high confidence, the signal is likely valid despite the broader regime. Size conservatively.
3. In VOLATILE markets, only act when model consensus is strong (agreement > 0.7). Require tighter stops.
4. conviction must match action: BUY/SHORT requires conviction >= 6
5. stop_loss_pct must be realistic (0.02 to 0.12 for stocks)
6. List exactly 3 take_profit_targets with scaled exit percentages (sum = 1.0)
7. If models DISAGREE significantly (agreement < 0.4), prefer PASS unless one signal is overwhelming.

Respond with this EXACT JSON schema (no other text):
{{
  "action": "BUY|SELL|SHORT|COVER|HOLD|PASS",
  "conviction": 1,
  "max_position_pct": 0.03,
  "entry_price_low": 0.0,
  "entry_price_high": 0.0,
  "stop_loss_pct": 0.05,
  "take_profit_targets": [
    {{"price": 0.0, "sell_fraction": 0.33}},
    {{"price": 0.0, "sell_fraction": 0.33}},
    {{"price": 0.0, "sell_fraction": 0.34}}
  ],
  "holding_period_days": 10,
  "technical_summary": "Brief technical analysis summary",
  "fundamental_summary": "Brief fundamental analysis summary",
  "primary_thesis": "Core investment thesis in 1-2 sentences",
  "key_risks": ["risk1", "risk2", "risk3"],
  "invalidation_conditions": ["if X happens, exit immediately"],
  "pass_reason": "Only populated if action=PASS"
}}"""


def build_arbiter_prompt(
    bundle: AnalysisBundle,
    intelligence_text: str,
    portfolio_heat_pct: float,
    max_heat_pct: float,
    cash_available: float,
    open_positions: int,
    max_positions: int,
    risk_tolerance: str,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM arbiter with full intelligence report."""
    currency_symbol = "₹" if "IN" in str(bundle.market_regime.regime.value) or bundle.symbol.endswith((".NS", ".BO")) else "$"
    system = ANALYST_SYSTEM_PROMPT.format(
        exchange=bundle.market_regime.regime.value,
        regime=bundle.market_regime.regime.value,
        regime_description=bundle.market_regime.description,
        new_long_allowed=bundle.market_regime.new_long_allowed,
        new_short_allowed=bundle.market_regime.new_short_allowed,
        risk_tolerance=risk_tolerance,
        portfolio_heat_pct=portfolio_heat_pct,
        max_heat_pct=max_heat_pct,
        vix_level=bundle.vix.level,
        vix_regime=bundle.vix.regime.value,
        vix_size_mult=bundle.vix.size_multiplier,
        open_positions=open_positions,
        max_positions=max_positions,
        cash_available=cash_available,
        currency_symbol=currency_symbol,
    )
    user = ARBITER_USER_PROMPT.format(intelligence_text=intelligence_text)
    return system, user


def build_analyst_prompt(
    bundle: AnalysisBundle,
    portfolio_heat_pct: float,
    max_heat_pct: float,
    cash_available: float,
    open_positions: int,
    max_positions: int,
    risk_tolerance: str,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the analyst role."""
    currency_symbol = "₹" if "IN" in str(bundle.market_regime.regime.value) or bundle.symbol.endswith((".NS", ".BO")) else "$"
    system = ANALYST_SYSTEM_PROMPT.format(
        exchange=bundle.market_regime.regime.value,
        regime=bundle.market_regime.regime.value,
        regime_description=bundle.market_regime.description,
        new_long_allowed=bundle.market_regime.new_long_allowed,
        new_short_allowed=bundle.market_regime.new_short_allowed,
        risk_tolerance=risk_tolerance,
        portfolio_heat_pct=portfolio_heat_pct,
        max_heat_pct=max_heat_pct,
        vix_level=bundle.vix.level,
        vix_regime=bundle.vix.regime.value,
        vix_size_mult=bundle.vix.size_multiplier,
        open_positions=open_positions,
        max_positions=max_positions,
        cash_available=cash_available,
        currency_symbol=currency_symbol,
    )
    user = ANALYST_USER_PROMPT.format(analysis_text=bundle.to_llm_prompt_text())
    return system, user


def build_risk_reviewer_prompt(
    bundle: AnalysisBundle,
    recommendation,
    portfolio_heat_pct: float,
    max_heat_pct: float,
    open_positions: int,
    sector_exposures: dict,
    daily_pnl_pct: float,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the risk reviewer role."""
    system = RISK_REVIEWER_SYSTEM_PROMPT.format(
        portfolio_heat_pct=portfolio_heat_pct,
        max_heat_pct=max_heat_pct,
        open_positions=open_positions,
        sector_exposures=str(sector_exposures),
        daily_pnl_pct=daily_pnl_pct,
    )
    # Compact summary for reviewer (not full bundle — saves tokens)
    currency = "₹" if bundle.symbol.endswith((".NS", ".BO")) else "$"
    summary = (
        f"Price: {currency}{bundle.current_price:.2f} | "
        f"Technical Score: {bundle.technical_score:.0f}/100 | "
        f"Fundamental Score: {bundle.fundamental_score:.0f}/100 | "
        f"Composite: {bundle.composite_score:.0f}/100 | "
        f"Earnings in {bundle.earnings.days_until_earnings or 'N/A'} days | "
        f"VIX: {bundle.vix.level:.1f} | "
        f"Regime: {bundle.market_regime.regime.value}"
    )
    user = RISK_REVIEWER_USER_PROMPT.format(
        action=recommendation.action,
        symbol=bundle.symbol,
        conviction=recommendation.conviction,
        primary_thesis=recommendation.primary_thesis,
        key_risks=recommendation.key_risks,
        analysis_summary=summary,
    )
    return system, user
