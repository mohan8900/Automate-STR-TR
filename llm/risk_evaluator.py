"""
LLM Risk Evaluator — second-pass adversarial review of trade recommendations.
Challenges the analyst's recommendation from a risk management perspective.
"""
from __future__ import annotations

from analysis.aggregator import AnalysisBundle
from llm.client import ClaudeClient
from llm.prompts import build_risk_reviewer_prompt
from llm.response_parser import ResponseParser, TradeRecommendation, RiskReview
from core.logger import get_logger

log = get_logger("llm_risk")


class LLMRiskEvaluator:

    def __init__(self, client: ClaudeClient):
        self.client = client
        self.parser = ResponseParser()

    def review(
        self,
        bundle: AnalysisBundle,
        recommendation: TradeRecommendation,
        portfolio_heat_pct: float = 0.0,
        max_heat_pct: float = 0.06,
        open_positions: int = 0,
        sector_exposures: dict | None = None,
        daily_pnl_pct: float = 0.0,
    ) -> RiskReview:
        """
        Adversarial risk review. Only called for actionable recommendations.
        Uses a compact prompt (saves tokens vs. analyst call).
        """
        log.info(f"Risk review: {bundle.symbol} {recommendation.action} conviction={recommendation.conviction}")

        system, user = build_risk_reviewer_prompt(
            bundle=bundle,
            recommendation=recommendation,
            portfolio_heat_pct=portfolio_heat_pct,
            max_heat_pct=max_heat_pct,
            open_positions=open_positions,
            sector_exposures=sector_exposures or {},
            daily_pnl_pct=daily_pnl_pct,
        )

        raw = self.client.complete(
            system_prompt=system,
            user_prompt=user,
            max_tokens=1024,  # Shorter response needed
        )
        review = self.parser.parse_risk_review(raw)

        status = "APPROVED" if review.approve else "REJECTED"
        log.info(
            f"{bundle.symbol} risk review: {status} | "
            f"risk_score={review.risk_score}/10 | "
            f"concern={review.primary_concern[:60]}"
        )
        return review
