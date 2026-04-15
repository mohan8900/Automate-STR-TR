"""Custom exception hierarchy for the trading system."""
from __future__ import annotations


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""


class ConfigurationError(TradingSystemError):
    """Invalid or missing configuration."""


class MarketDataError(TradingSystemError):
    """Failed to fetch or parse market data."""


class AnalysisError(TradingSystemError):
    """Error during technical/fundamental/sentiment analysis."""


class LLMError(TradingSystemError):
    """Claude API call failed or returned unparseable response."""


class LLMParseError(LLMError):
    """Could not parse structured JSON from LLM response."""


class BrokerError(TradingSystemError):
    """Broker API error (Alpaca / Zerodha)."""


class OrderRejectedError(BrokerError):
    """Broker rejected the order."""


class InsufficientFundsError(BrokerError):
    """Not enough buying power for the order."""


class CircuitBreakerError(TradingSystemError):
    """Trading halted by circuit breaker."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Circuit breaker triggered: {reason}")


class RiskManagementError(TradingSystemError):
    """Trade blocked by risk management rules."""


class PositionNotFoundError(TradingSystemError):
    """Attempted operation on a non-existent position."""
