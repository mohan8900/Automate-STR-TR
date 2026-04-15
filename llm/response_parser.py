"""
Structured output parser for LLM responses.
Never trust raw LLM output — always validate with Pydantic.
Falls back to PASS action on any parse failure.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from core.logger import get_logger

log = get_logger("llm_parser")


class TakeProfitTarget(BaseModel):
    price: float = Field(gt=0)
    sell_fraction: float = Field(gt=0, le=1.0)


class TradeRecommendation(BaseModel):
    action: Literal["BUY", "SELL", "SHORT", "COVER", "HOLD", "PASS"]
    conviction: int = Field(ge=1, le=10)
    max_position_pct: float = Field(ge=0.005, le=0.20, default=0.03)
    entry_price_low: float = Field(ge=0, default=0)
    entry_price_high: float = Field(ge=0, default=0)
    stop_loss_pct: float = Field(ge=0.01, le=0.15, default=0.05)
    take_profit_targets: list[TakeProfitTarget] = Field(default_factory=list)
    holding_period_days: int = Field(ge=1, le=365, default=10)
    technical_summary: str = ""
    fundamental_summary: str = ""
    primary_thesis: str = Field(default="", max_length=500)
    key_risks: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    pass_reason: str = ""

    @model_validator(mode="after")
    def validate_conviction_action_alignment(self) -> "TradeRecommendation":
        """BUY/SHORT with low conviction should be degraded to PASS."""
        if self.action in ("BUY", "SHORT") and self.conviction < 6:
            log.warning(
                f"Degrading action {self.action} to PASS: "
                f"conviction {self.conviction} < 6"
            )
            self.action = "PASS"
            self.pass_reason = f"Conviction too low ({self.conviction}/10) for execution"
        return self

    @field_validator("take_profit_targets")
    @classmethod
    def validate_targets(cls, v: list) -> list:
        if len(v) > 5:
            return v[:5]
        return v

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL", "SHORT", "COVER")

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        if not self.take_profit_targets or self.stop_loss_pct <= 0:
            return None
        first_target = self.take_profit_targets[0]
        if first_target.price <= 0:
            return None
        return None  # Computed externally with actual prices


class RiskReview(BaseModel):
    approve: bool
    risk_score: int = Field(ge=1, le=10)
    primary_concern: str = ""
    concerns: list[str] = Field(default_factory=list)
    modified_stop_loss_pct: Optional[float] = Field(None, ge=0.01, le=0.20)
    modified_position_pct: Optional[float] = Field(None, ge=0.005, le=0.15)
    rejection_reason: str = ""

    @model_validator(mode="after")
    def high_risk_auto_reject(self) -> "RiskReview":
        """Auto-reject if risk score is very high."""
        if self.risk_score > 7 and self.approve:
            log.warning(f"Auto-rejecting: risk_score={self.risk_score} > 7")
            self.approve = False
            self.rejection_reason = f"Risk score {self.risk_score}/10 exceeds threshold"
        return self


class ResponseParser:

    @staticmethod
    def parse_recommendation(raw_response: str, symbol: str) -> TradeRecommendation:
        """Parse LLM response into a TradeRecommendation. Returns PASS on failure."""
        try:
            data = ResponseParser._extract_json(raw_response)
            return TradeRecommendation(**data)
        except Exception as e:
            log.error(
                f"Failed to parse trade recommendation for {symbol}: {e}\n"
                f"Raw: {raw_response[:500]}"
            )
            return TradeRecommendation(
                action="PASS",
                conviction=1,
                pass_reason=f"LLM response parse error: {str(e)[:100]}",
                primary_thesis="Parse failed",
            )

    @staticmethod
    def parse_risk_review(raw_response: str) -> RiskReview:
        """Parse LLM response into a RiskReview. Returns rejection on failure."""
        try:
            data = ResponseParser._extract_json(raw_response)
            return RiskReview(**data)
        except Exception as e:
            log.error(f"Failed to parse risk review: {e}\nRaw: {raw_response[:500]}")
            return RiskReview(
                approve=False,
                risk_score=10,
                rejection_reason=f"Risk review parse error: {str(e)[:100]}",
            )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Robustly extract a JSON object from LLM output.
        Handles markdown code blocks, extra text, trailing commas.
        """
        # Remove markdown code blocks
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        # Find outermost JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in response")

        json_str = match.group(0)

        # Fix common LLM JSON errors
        # Trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # Single quotes to double quotes (sometimes models use them)
        # Be careful not to break legitimate string content
        # Only do basic cleanup
        json_str = json_str.replace("\n", " ").replace("\t", " ")

        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        return data
