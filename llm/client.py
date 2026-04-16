"""
LLM client — supports multiple providers:
1. Anthropic Claude (paid)
2. LiteLLM proxy / OpenAI-compatible APIs (free)
3. Google Gemini (free, 1500 req/day)

Auto-selects based on which API keys are configured in .env
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

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


class ClaudeClient:
    """
    Multi-provider LLM client. Tries providers in order:
    1. LiteLLM proxy (if LITELLM_API_BASE is set) — FREE
    2. Google Gemini (if GEMINI_API_KEY is set) — FREE
    3. Anthropic Claude (if ANTHROPIC__API_KEY is set) — PAID
    """

    def __init__(self, config: AnthropicConfig):
        # Ensure .env is loaded into os.environ for non-pydantic vars
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self.config = config
        self.usage = TokenUsage()
        self._daily_cost_usd = 0.0
        self._last_reset_date: str = ""

        # Determine which provider to use
        self._provider = self._detect_provider()
        self._init_provider()

    def _detect_provider(self) -> str:
        """Auto-detect which LLM provider to use based on env vars."""
        litellm_base = os.getenv("LITELLM_API_BASE", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        anthropic_key = self.config.api_key.get_secret_value()

        if litellm_base:
            log.info(f"LLM Provider: LiteLLM proxy at {litellm_base}")
            return "litellm"
        elif gemini_key:
            log.info("LLM Provider: Google Gemini (free)")
            return "gemini"
        elif anthropic_key and not anthropic_key.startswith("sk-ant-PASTE"):
            log.info("LLM Provider: Anthropic Claude")
            return "anthropic"
        else:
            # Check if LiteLLM env is set without the env var name
            log.warning(
                "No LLM API key configured. Set one of:\n"
                "  LITELLM_API_BASE=http://... (free proxy)\n"
                "  GEMINI_API_KEY=... (free, 1500 req/day)\n"
                "  ANTHROPIC__API_KEY=sk-ant-... (paid)\n"
                "Falling back to mock mode."
            )
            return "mock"

    def _init_provider(self) -> None:
        """Initialize the selected provider."""
        if self._provider == "anthropic":
            import anthropic
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key.get_secret_value()
            )

        elif self._provider == "litellm":
            import httpx
            self._httpx = httpx
            self._litellm_base = os.getenv("LITELLM_API_BASE", "").rstrip("/")
            self._litellm_model = os.getenv("BEDROCK_MODEL", "qwen3-vl-235b")
            self._litellm_key = os.getenv("LITELLM_API_KEY", "no-key-needed")

        elif self._provider == "gemini":
            from google import genai
            self._genai = genai
            self._gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a message to the LLM and return the text response."""
        self._check_daily_cost_limit()

        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        if self._provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, max_tok, temp)
        elif self._provider == "litellm":
            return self._call_litellm(system_prompt, user_prompt, max_tok, temp)
        elif self._provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt, max_tok, temp)
        else:
            return self._call_mock(system_prompt, user_prompt)

    # ── Anthropic Claude ─────────────────────────────────────────────────

    def _call_anthropic(self, system: str, user: str, max_tok: int, temp: float) -> str:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((
                self._anthropic.RateLimitError,
                self._anthropic.APIConnectionError,
                self._anthropic.InternalServerError,
            )),
            before_sleep=before_sleep_log(_tenacity_log, logging.WARNING),
            reraise=True,
        )
        def _call():
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            usage = response.usage
            self._track_usage(usage.input_tokens, usage.output_tokens, cost_per_1m_in=3.0, cost_per_1m_out=15.0)
            return response.content[0].text

        try:
            return _call()
        except self._anthropic.AuthenticationError as e:
            raise LLMError(f"Claude auth failed — check ANTHROPIC__API_KEY: {e}") from e
        except Exception as e:
            raise LLMError(f"Claude API failed: {e}") from e

    # ── LiteLLM / OpenAI-compatible ──────────────────────────────────────

    def _call_litellm(self, system: str, user: str, max_tok: int, temp: float) -> str:
        try:
            response = self._httpx.post(
                f"{self._litellm_base}/chat/completions",
                json={
                    "model": self._litellm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tok,
                    "temperature": temp,
                },
                headers={
                    "Authorization": f"Bearer {self._litellm_key}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # Track usage (free, but count tokens)
            usage = data.get("usage", {})
            self._track_usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                cost_per_1m_in=0, cost_per_1m_out=0,  # Free
            )

            text = data["choices"][0]["message"]["content"]
            log.debug(f"LiteLLM response ({self._litellm_model}): {len(text)} chars")
            return text

        except Exception as e:
            raise LLMError(f"LiteLLM API failed ({self._litellm_base}): {e}") from e

    # ── Google Gemini ────────────────────────────────────────────────────

    def _call_gemini(self, system: str, user: str, max_tok: int, temp: float) -> str:
        try:
            full_prompt = f"{system}\n\n{user}"
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config={
                    "max_output_tokens": max_tok,
                    "temperature": temp,
                },
            )

            text = response.text
            self._track_usage(0, 0, cost_per_1m_in=0, cost_per_1m_out=0)  # Free
            log.debug(f"Gemini response: {len(text)} chars")
            return text

        except Exception as e:
            raise LLMError(f"Gemini API failed: {e}") from e

    # ── Mock (no API key) ────────────────────────────────────────────────

    def _call_mock(self, system: str, user: str) -> str:
        log.warning("Using MOCK LLM — no real analysis. Set an API key in .env")
        return '''{
            "action": "PASS",
            "conviction": 1,
            "max_position_pct": 0,
            "entry_price_low": 0, "entry_price_high": 0,
            "stop_loss_pct": 0.05,
            "take_profit_targets": [],
            "holding_period_days": 0,
            "technical_summary": "Mock mode - no LLM configured",
            "fundamental_summary": "Mock mode",
            "primary_thesis": "No LLM API key configured",
            "key_risks": ["No analysis available"],
            "invalidation_conditions": [],
            "pass_reason": "No LLM provider configured. Set LITELLM_API_BASE, GEMINI_API_KEY, or ANTHROPIC__API_KEY in .env"
        }'''

    # ── Usage tracking ───────────────────────────────────────────────────

    def _track_usage(self, in_tokens: int, out_tokens: int, cost_per_1m_in: float, cost_per_1m_out: float) -> None:
        self.usage.input_tokens += in_tokens
        self.usage.output_tokens += out_tokens
        self.usage.total_calls += 1
        cost = (in_tokens * cost_per_1m_in / 1_000_000) + (out_tokens * cost_per_1m_out / 1_000_000)
        self.usage.estimated_cost_usd += cost
        self._daily_cost_usd += cost
        log.debug(
            f"LLM call #{self.usage.total_calls}: "
            f"in={in_tokens} out={out_tokens} cost=${cost:.4f}"
        )

    def _check_daily_cost_limit(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_cost_usd = 0.0
            self._last_reset_date = today
        if self._provider == "anthropic" and self._daily_cost_usd >= self.config.max_daily_cost_usd:
            raise LLMError(f"Daily cost limit: ${self._daily_cost_usd:.2f} >= ${self.config.max_daily_cost_usd}")

    def get_usage_summary(self) -> dict:
        return {
            "provider": self._provider,
            "total_calls": self.usage.total_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "estimated_cost_usd": round(self.usage.estimated_cost_usd, 4),
            "daily_cost_usd": round(self._daily_cost_usd, 4),
        }
