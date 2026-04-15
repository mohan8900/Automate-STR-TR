"""
Claude API client with exponential backoff retry logic.
Tracks token usage for daily cost management.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from config.settings import AnthropicConfig
from core.exceptions import LLMError
from core.logger import get_logger

log = get_logger("llm_client")
_tenacity_log = logging.getLogger("tenacity")


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0


# Claude Sonnet pricing (approximate, per 1M tokens)
_COST_PER_1M_INPUT = 3.0
_COST_PER_1M_OUTPUT = 15.0


class ClaudeClient:

    def __init__(self, config: AnthropicConfig):
        self.config = config
        self._client = anthropic.Anthropic(
            api_key=config.api_key.get_secret_value()
        )
        self.usage = TokenUsage()
        self._daily_cost_usd = 0.0
        self._last_reset_date: str = ""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Send a message to Claude and return the text response.
        Includes retry logic for transient failures.
        """
        self._check_daily_cost_limit()

        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )),
            before_sleep=before_sleep_log(_tenacity_log, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Track usage
            usage = response.usage
            self.usage.input_tokens += usage.input_tokens
            self.usage.output_tokens += usage.output_tokens
            self.usage.total_calls += 1

            cost = (
                usage.input_tokens * _COST_PER_1M_INPUT / 1_000_000
                + usage.output_tokens * _COST_PER_1M_OUTPUT / 1_000_000
            )
            self.usage.estimated_cost_usd += cost
            self._daily_cost_usd += cost

            log.debug(
                f"LLM call #{self.usage.total_calls}: "
                f"in={usage.input_tokens} out={usage.output_tokens} "
                f"cost=${cost:.4f} daily=${self._daily_cost_usd:.3f}"
            )

            return response.content[0].text

        try:
            return _call()
        except anthropic.AuthenticationError as e:
            raise LLMError(f"Claude authentication failed — check ANTHROPIC_API_KEY: {e}") from e
        except Exception as e:
            raise LLMError(f"Claude API call failed: {e}") from e

    def _check_daily_cost_limit(self) -> None:
        """Reset daily counter at midnight, enforce cost limit."""
        today = time.strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_cost_usd = 0.0
            self._last_reset_date = today

        if self._daily_cost_usd >= self.config.max_daily_cost_usd:
            raise LLMError(
                f"Daily LLM cost limit reached: ${self._daily_cost_usd:.2f} "
                f">= ${self.config.max_daily_cost_usd}"
            )

    def get_usage_summary(self) -> dict:
        return {
            "total_calls": self.usage.total_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "estimated_cost_usd": round(self.usage.estimated_cost_usd, 4),
            "daily_cost_usd": round(self._daily_cost_usd, 4),
        }
