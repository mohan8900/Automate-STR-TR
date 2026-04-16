"""
Strategy Selector — chooses which strategy to apply based on market regime.
Combines signals from multiple strategies using a voting mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from services.swing_trading.strategy.base import Strategy, TradeSignal
from services.swing_trading.strategy.swing_trading import SwingTradingStrategy
from services.swing_trading.strategy.momentum import MomentumStrategy
from services.swing_trading.strategy.mean_reversion import MeanReversionStrategy
from services.swing_trading.prediction.ensemble_model import PredictionResult
from core.logger import get_logger

log = get_logger("strategy_selector")


@dataclass
class StrategyVote:
    """Combined signal from multiple strategies."""
    symbol: str
    consensus_action: str           # BUY | SELL | PASS
    consensus_strength: float       # 0.0 to 1.0
    ml_prediction: Optional[PredictionResult]
    strategy_signals: dict[str, TradeSignal]
    active_strategies: list[str]
    combined_stop_loss_pct: float
    combined_take_profit_pct: float
    combined_holding_days: int
    reason: str


class StrategySelector:
    """
    Selects and combines strategies based on market regime.
    Uses a voting mechanism when multiple strategies agree.
    """

    def __init__(self):
        self.strategies: dict[str, Strategy] = {
            "swing": SwingTradingStrategy(),
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

    def select_strategies(self, market_regime: str) -> list[str]:
        """Select which strategies are active for the current regime."""
        active = []
        for name, strategy in self.strategies.items():
            if strategy.is_suitable_regime(market_regime):
                active.append(name)

        if not active:
            # Fallback: always use swing trading as minimum
            active = ["swing"]

        log.info(
            f"Market regime: {market_regime} -> active strategies: {active}"
        )
        return active

    def generate_combined_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_regime: str,
        ml_prediction: Optional[PredictionResult] = None,
        current_position: Optional[dict] = None,
    ) -> StrategyVote:
        """
        Generate a combined signal from all active strategies + ML prediction.
        Uses consensus voting: trade only when multiple signals agree.
        """
        active = self.select_strategies(market_regime)
        signals: dict[str, TradeSignal] = {}

        # Get signals from each active strategy
        for name in active:
            strategy = self.strategies[name]
            try:
                signal = strategy.generate_signal(symbol, df, current_position)
                signals[name] = signal
                log.debug(
                    f"{symbol} | {name}: {signal.action} "
                    f"(strength={signal.strength:.2f}) — {signal.reason}"
                )
            except Exception as e:
                log.warning(f"Strategy {name} failed for {symbol}: {e}")

        # Count votes
        buy_votes = 0
        sell_votes = 0
        total_strength = 0.0
        buy_strength = 0.0

        for name, signal in signals.items():
            if signal.action == "BUY":
                buy_votes += 1
                buy_strength += signal.strength
                total_strength += signal.strength
            elif signal.action in ("SELL", "SHORT"):
                sell_votes += 1
                total_strength += signal.strength

        # ML prediction boost
        ml_boost = 0.0
        if ml_prediction and ml_prediction.direction == "BULLISH":
            ml_boost = min(0.20, ml_prediction.confidence * 0.3)
            if ml_prediction.probability > 0.60:
                buy_votes += 1
                buy_strength += ml_prediction.probability * 0.5
        elif ml_prediction and ml_prediction.direction == "BEARISH":
            if ml_prediction.probability < 0.40:
                sell_votes += 1

        # Consensus decision
        total_votes = len(signals) + (1 if ml_prediction else 0)
        if total_votes == 0:
            return self._neutral_vote(symbol, active, ml_prediction)

        # In BEAR regime, accept a single strong signal (0.60+) since opportunities are rarer
        single_vote_threshold = 0.60 if market_regime == "BEAR" else 0.70
        if buy_votes >= 2 or (buy_votes >= 1 and buy_strength > single_vote_threshold):
            consensus_action = "BUY"
            consensus_strength = min(1.0, (buy_strength / max(1, buy_votes)) + ml_boost)
        elif sell_votes >= 2:
            consensus_action = "SELL"
            consensus_strength = 0.7
        else:
            consensus_action = "PASS"
            consensus_strength = 0.0

        # Aggregate stop/tp from strongest buy signal
        stop_pct = 0.05
        tp_pct = 0.10
        hold_days = 10
        buy_signals = [s for s in signals.values() if s.action == "BUY"]
        if buy_signals:
            strongest = max(buy_signals, key=lambda s: s.strength)
            stop_pct = strongest.stop_loss_pct
            tp_pct = strongest.take_profit_pct
            hold_days = strongest.holding_period_days

        # Build reason string
        reasons = []
        for name, sig in signals.items():
            if sig.action != "PASS":
                reasons.append(f"{name}: {sig.action} ({sig.strength:.2f}) - {sig.reason}")
        if ml_prediction and ml_prediction.direction != "NEUTRAL":
            reasons.append(
                f"ML: {ml_prediction.direction} "
                f"(prob={ml_prediction.probability:.2f}, "
                f"conf={ml_prediction.confidence:.2f})"
            )

        return StrategyVote(
            symbol=symbol,
            consensus_action=consensus_action,
            consensus_strength=round(consensus_strength, 3),
            ml_prediction=ml_prediction,
            strategy_signals=signals,
            active_strategies=active,
            combined_stop_loss_pct=stop_pct,
            combined_take_profit_pct=tp_pct,
            combined_holding_days=hold_days,
            reason=" || ".join(reasons) if reasons else "No actionable signals",
        )

    def _neutral_vote(
        self,
        symbol: str,
        active: list[str],
        ml_prediction: Optional[PredictionResult],
    ) -> StrategyVote:
        return StrategyVote(
            symbol=symbol,
            consensus_action="PASS",
            consensus_strength=0.0,
            ml_prediction=ml_prediction,
            strategy_signals={},
            active_strategies=active,
            combined_stop_loss_pct=0.05,
            combined_take_profit_pct=0.10,
            combined_holding_days=10,
            reason="No signals generated",
        )
