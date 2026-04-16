"""
Intelligence Aggregator — runs all AI/ML models in parallel and builds
a comprehensive report for the LLM arbiter.

Replaces the rigid strategy consensus filter with a data-rich intelligence
layer that lets the LLM make the final trading decision.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from analysis.aggregator import AnalysisBundle
from config.settings import TradingSystemConfig
from core.logger import get_logger
from services.swing_trading.prediction.ensemble_model import EnsemblePredictor, PredictionResult
from services.swing_trading.prediction.fuzzy_logic import FuzzyTradingSystem, FuzzySignal
from services.swing_trading.prediction.lstm_model import LSTMPredictor
from services.swing_trading.strategy.base import TradeSignal
from services.swing_trading.strategy.selector import StrategySelector

log = get_logger("intelligence")


@dataclass
class IntelligenceReport:
    """All AI/ML model outputs for a single stock, ready for LLM review."""
    symbol: str
    timestamp: datetime

    # Model predictions
    ensemble_prediction: Optional[PredictionResult] = None
    lstm_prediction: Optional[PredictionResult] = None
    fuzzy_signal: Optional[FuzzySignal] = None

    # Strategy signals (informational, not gating)
    strategy_signals: dict[str, TradeSignal] = field(default_factory=dict)

    # Meta
    models_succeeded: list[str] = field(default_factory=list)
    models_failed: list[str] = field(default_factory=list)

    # Consensus metrics
    model_agreement_score: float = 0.0
    weighted_bullish_prob: float = 0.5


class IntelligenceAggregator:
    """
    Orchestrates all AI/ML models in parallel and produces an IntelligenceReport.
    Each model runs in its own thread with a timeout. Failures are graceful.
    """

    def __init__(
        self,
        config: TradingSystemConfig,
        ensemble_predictor: Optional[EnsemblePredictor] = None,
    ):
        self.config = config
        self.timeout = config.ml.parallel_timeout_seconds

        # Initialize models
        self.ensemble = ensemble_predictor or EnsemblePredictor(
            target_days=config.ml.target_days,
            train_window_days=config.ml.train_window_days,
            retrain_interval_days=config.ml.retrain_interval_days,
        )

        self.lstm = LSTMPredictor(
            target_days=config.ml.target_days,
            train_window_days=config.ml.train_window_days,
            retrain_interval_days=config.ml.retrain_interval_days,
        ) if config.ml.lstm_enabled else None

        self.fuzzy = FuzzyTradingSystem() if config.ml.fuzzy_enabled else None

        self.strategy_selector = StrategySelector()

    def gather(
        self,
        symbol: str,
        df: pd.DataFrame,
        bundle: AnalysisBundle,
        market_regime: str,
    ) -> IntelligenceReport:
        """
        Run all models in parallel and collect results into an IntelligenceReport.
        Each model gets `timeout` seconds. Failures are logged, not fatal.
        """
        report = IntelligenceReport(symbol=symbol, timestamp=datetime.now())

        futures = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Thread 1: ML Ensemble (XGBoost + RF)
            if self.config.ml.enabled:
                futures["ML Ensemble"] = executor.submit(
                    self._run_ensemble, symbol, df, bundle
                )

            # Thread 2: LSTM Neural Network
            if self.lstm is not None:
                futures["LSTM Neural Net"] = executor.submit(
                    self._run_lstm, symbol, df, bundle
                )

            # Thread 3: Fuzzy Logic
            if self.fuzzy is not None:
                futures["Fuzzy Logic"] = executor.submit(
                    self._run_fuzzy, bundle
                )

            # Thread 4: Strategy Signals
            futures["Strategy Signals"] = executor.submit(
                self._run_strategies, symbol, df, market_regime
            )

            # Collect results with timeout
            for name, future in futures.items():
                try:
                    result = future.result(timeout=self.timeout)
                    if name == "ML Ensemble" and result is not None:
                        report.ensemble_prediction = result
                        report.models_succeeded.append(name)
                    elif name == "LSTM Neural Net" and result is not None:
                        report.lstm_prediction = result
                        report.models_succeeded.append(name)
                    elif name == "Fuzzy Logic" and result is not None:
                        report.fuzzy_signal = result
                        report.models_succeeded.append(name)
                    elif name == "Strategy Signals" and result is not None:
                        report.strategy_signals = result
                        report.models_succeeded.append(name)
                    else:
                        report.models_failed.append(name)
                except Exception as e:
                    log.warning(f"{name} failed for {symbol}: {e}")
                    report.models_failed.append(name)

        # Compute consensus metrics
        report.model_agreement_score = self._compute_agreement(report)
        report.weighted_bullish_prob = self._compute_weighted_prob(report)

        return report

    # ── Model runners ─────────────────────────────────────────────────────

    def _run_ensemble(self, symbol: str, df: pd.DataFrame, bundle: AnalysisBundle) -> Optional[PredictionResult]:
        try:
            fund_dict = self._bundle_to_fund_dict(bundle)
            return self.ensemble.predict(symbol, df, fundamental_data=fund_dict)
        except Exception as e:
            log.warning(f"Ensemble failed for {symbol}: {e}")
            return None

    def _run_lstm(self, symbol: str, df: pd.DataFrame, bundle: AnalysisBundle) -> Optional[PredictionResult]:
        try:
            fund_dict = self._bundle_to_fund_dict(bundle)
            return self.lstm.predict(symbol, df, fundamental_data=fund_dict)
        except Exception as e:
            log.warning(f"LSTM failed for {symbol}: {e}")
            return None

    def _run_fuzzy(self, bundle: AnalysisBundle) -> Optional[FuzzySignal]:
        try:
            t = bundle.technical
            return self.fuzzy.evaluate(
                rsi=t.rsi_14,
                macd_histogram=t.macd_histogram,
                price_vs_sma50_pct=t.price_vs_sma50_pct / 100.0 if abs(t.price_vs_sma50_pct) > 1 else t.price_vs_sma50_pct,
                relative_volume=t.relative_volume,
                fundamental_score=bundle.fundamental_score,
                market_regime=bundle.market_regime.regime.value,
            )
        except Exception as e:
            log.warning(f"Fuzzy logic failed for {bundle.symbol}: {e}")
            return None

    def _run_strategies(self, symbol: str, df: pd.DataFrame, market_regime: str) -> Optional[dict[str, TradeSignal]]:
        try:
            vote = self.strategy_selector.generate_combined_signal(
                symbol=symbol, df=df, market_regime=market_regime,
            )
            return vote.strategy_signals
        except Exception as e:
            log.warning(f"Strategy signals failed for {symbol}: {e}")
            return None

    # ── Consensus computation ─────────────────────────────────────────────

    def _compute_agreement(self, report: IntelligenceReport) -> float:
        """
        Compute how much models agree on direction.
        +1 = bullish, 0 = neutral, -1 = bearish
        Agreement = 1 - normalized_std_dev (1.0 = perfect agreement)
        """
        votes = []

        if report.ensemble_prediction:
            p = report.ensemble_prediction
            if p.direction == "BULLISH":
                votes.append(1.0)
            elif p.direction == "BEARISH":
                votes.append(-1.0)
            else:
                votes.append(0.0)

        if report.lstm_prediction:
            p = report.lstm_prediction
            if p.direction == "BULLISH":
                votes.append(1.0)
            elif p.direction == "BEARISH":
                votes.append(-1.0)
            else:
                votes.append(0.0)

        if report.fuzzy_signal:
            f = report.fuzzy_signal
            if f.action == "BUY":
                votes.append(1.0)
            elif f.action == "SELL":
                votes.append(-1.0)
            else:
                votes.append(0.0)

        # Count strategy BUY votes
        buy_count = sum(1 for s in report.strategy_signals.values() if s.action == "BUY")
        sell_count = sum(1 for s in report.strategy_signals.values() if s.action in ("SELL", "SHORT"))
        if buy_count > sell_count:
            votes.append(1.0)
        elif sell_count > buy_count:
            votes.append(-1.0)
        elif report.strategy_signals:
            votes.append(0.0)

        if len(votes) < 2:
            return 0.0

        std = float(np.std(votes))
        # Max possible std for votes in [-1, 1] is 1.0
        return round(max(0.0, 1.0 - std), 2)

    def _compute_weighted_prob(self, report: IntelligenceReport) -> float:
        """Weighted average of bullish probabilities across models."""
        probs = []
        weights = []

        if report.ensemble_prediction:
            probs.append(report.ensemble_prediction.probability)
            weights.append(1.0)  # Base weight

        if report.lstm_prediction:
            probs.append(report.lstm_prediction.probability)
            weights.append(0.8)  # Slightly lower — newer, less proven

        if report.fuzzy_signal:
            # Convert fuzzy score (0-100) to probability (0-1)
            probs.append(report.fuzzy_signal.score / 100.0)
            weights.append(0.6)  # Rule-based, less adaptive

        if not probs:
            return 0.5

        weighted = sum(p * w for p, w in zip(probs, weights))
        total_w = sum(weights)
        return round(weighted / total_w, 4)

    # ── Report formatting for LLM ─────────────────────────────────────────

    @staticmethod
    def to_llm_text(report: IntelligenceReport, bundle: AnalysisBundle) -> str:
        """Format the full intelligence report as text for the LLM prompt."""
        total = len(report.models_succeeded) + len(report.models_failed)
        lines = [
            f"=== INTELLIGENCE REPORT: {report.symbol} ===",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"--- AI MODEL PREDICTIONS ({len(report.models_succeeded)}/{total} succeeded) ---",
            "",
        ]

        # ML Ensemble
        if report.ensemble_prediction:
            p = report.ensemble_prediction
            top_features = ", ".join(list(p.feature_importance.keys())[:5]) if p.feature_importance else "N/A"
            lines.extend([
                "## ML Ensemble (XGBoost + Random Forest)",
                f"Direction: {p.direction} | Probability: {p.probability:.2f} | Confidence: {p.confidence:.2f}",
                f"Model Accuracy: {p.model_accuracy:.0%} | Top features: {top_features}",
                "",
            ])
        elif "ML Ensemble" in report.models_failed:
            lines.extend(["## ML Ensemble (XGBoost + Random Forest)", "Status: FAILED", ""])

        # LSTM
        if report.lstm_prediction:
            p = report.lstm_prediction
            lines.extend([
                "## LSTM Neural Network",
                f"Direction: {p.direction} | Probability: {p.probability:.2f} | Confidence: {p.confidence:.2f}",
                f"Model Accuracy: {p.model_accuracy:.0%}",
                "",
            ])
        elif "LSTM Neural Net" in report.models_failed:
            lines.extend(["## LSTM Neural Network", "Status: FAILED", ""])

        # Fuzzy Logic
        if report.fuzzy_signal:
            f = report.fuzzy_signal
            active_rules = ", ".join(
                f"{name}({strength})" for name, strength in sorted(
                    f.rule_activations.items(), key=lambda x: x[1], reverse=True
                )[:5]
            )
            lines.extend([
                "## Fuzzy Logic System",
                f"Signal: {f.action} | Strength: {f.strength:.2f} | Score: {f.score:.0f}/100",
                f"Active rules: {active_rules or 'none'}",
                "",
            ])
        elif "Fuzzy Logic" in report.models_failed:
            lines.extend(["## Fuzzy Logic System", "Status: FAILED", ""])

        # Strategy Signals
        if report.strategy_signals:
            strat_parts = []
            for name, sig in report.strategy_signals.items():
                if sig.action != "PASS":
                    strat_parts.append(f"{name}: {sig.action} ({sig.strength:.2f}) — {sig.reason[:60]}")
                else:
                    strat_parts.append(f"{name}: PASS")
            lines.extend([
                "## Strategy Signals (informational, not gating)",
                *strat_parts,
                "",
            ])

        # Consensus
        bullish_count = 0
        total_models = 0
        for pred in [report.ensemble_prediction, report.lstm_prediction]:
            if pred:
                total_models += 1
                if pred.direction == "BULLISH":
                    bullish_count += 1
        if report.fuzzy_signal:
            total_models += 1
            if report.fuzzy_signal.action == "BUY":
                bullish_count += 1

        lines.extend([
            "## Model Consensus",
            f"Agreement: {report.model_agreement_score:.2f} ({bullish_count}/{total_models} bullish) | "
            f"Weighted Bullish Probability: {report.weighted_bullish_prob:.2f}",
            "",
        ])

        # Append existing analysis
        lines.extend([
            "--- STOCK ANALYSIS ---",
            bundle.to_llm_prompt_text(),
        ])

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _bundle_to_fund_dict(bundle: AnalysisBundle) -> dict:
        """Extract fundamental data dict from bundle for ML models."""
        f = bundle.fundamental
        return {
            "pe_ratio": f.pe_ratio,
            "peg_ratio": f.peg_ratio,
            "debt_to_equity": f.debt_to_equity,
            "return_on_equity": f.return_on_equity,
            "operating_margin": f.operating_margin,
            "revenue_growth_yoy": f.revenue_growth_yoy,
            "fcf_yield": f.fcf_yield,
            "beta": f.beta,
        }
