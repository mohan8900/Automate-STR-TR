"""
LLM Analyst — first-pass trade idea generation using Claude.
"""
from __future__ import annotations

from analysis.aggregator import AnalysisBundle
from llm.client import ClaudeClient
from llm.prompts import build_analyst_prompt
from llm.response_parser import ResponseParser, TradeRecommendation
from core.logger import get_logger

log = get_logger("llm_analyst")


class LLMAnalyst:

    def __init__(self, client: ClaudeClient):
        self.client = client
        self.parser = ResponseParser()

    def analyze(
        self,
        bundle: AnalysisBundle,
        portfolio_heat_pct: float = 0.0,
        max_heat_pct: float = 0.06,
        cash_available: float = 0.0,
        open_positions: int = 0,
        max_positions: int = 15,
        risk_tolerance: str = "moderate",
    ) -> TradeRecommendation:
        """
        Analyze a stock bundle and return a trade recommendation.
        This is the primary LLM call — uses full context window.
        """
        log.info(f"Analyzing {bundle.symbol} (composite score: {bundle.composite_score:.0f})")

        system, user = build_analyst_prompt(
            bundle=bundle,
            portfolio_heat_pct=portfolio_heat_pct,
            max_heat_pct=max_heat_pct,
            cash_available=cash_available,
            open_positions=open_positions,
            max_positions=max_positions,
            risk_tolerance=risk_tolerance,
        )

        raw = self.client.complete(system_prompt=system, user_prompt=user)
        rec = self.parser.parse_recommendation(raw, bundle.symbol)

        log.info(
            f"{bundle.symbol}: {rec.action} | conviction={rec.conviction}/10 | "
            f"{'ACTIONABLE' if rec.is_actionable else 'PASS'} | "
            f"thesis={rec.primary_thesis[:60]}"
        )
        return rec
