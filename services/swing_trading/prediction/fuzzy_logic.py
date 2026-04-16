"""
Fuzzy Logic Trading System — captures expert trading rules using fuzzy inference.
Encodes domain knowledge that ML models can't learn from data alone:
  "if RSI is somewhat oversold AND volume is high, that's a moderately strong buy"

Uses pure numpy — zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.logger import get_logger

log = get_logger("fuzzy_logic")


@dataclass
class FuzzySignal:
    """Output of the fuzzy inference system."""
    action: str             # BUY | HOLD | SELL
    strength: float         # 0.0 to 1.0
    score: float            # 0 to 100 (defuzzified output)
    rule_activations: dict  # {rule_name: activation_strength}


# ── Membership functions (trapezoidal) ──────────────────────────────────────

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function. Returns 0-1."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    # c < x < d
    return (d - x) / (d - c)


def _grade_up(x: float, a: float, b: float) -> float:
    """Rising grade function (open right). a=0%, b=100%."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def _grade_down(x: float, a: float, b: float) -> float:
    """Falling grade function (open left). a=100%, b=0%."""
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


# ── Fuzzy variable definitions ──────────────────────────────────────────────

def _rsi_oversold(rsi: float) -> float:
    return _grade_down(rsi, 20.0, 35.0)

def _rsi_neutral(rsi: float) -> float:
    return _trapezoid(rsi, 30.0, 40.0, 60.0, 70.0)

def _rsi_overbought(rsi: float) -> float:
    return _grade_up(rsi, 65.0, 80.0)


def _macd_negative(hist: float) -> float:
    return _grade_down(hist, -0.02, 0.0)

def _macd_near_zero(hist: float) -> float:
    return _trapezoid(hist, -0.01, -0.002, 0.002, 0.01)

def _macd_positive(hist: float) -> float:
    return _grade_up(hist, 0.0, 0.02)


def _price_far_below(dist: float) -> float:
    """dist = (price - sma50) / sma50, typically -0.2 to +0.2"""
    return _grade_down(dist, -0.10, -0.03)

def _price_below(dist: float) -> float:
    return _trapezoid(dist, -0.08, -0.04, -0.01, 0.02)

def _price_above(dist: float) -> float:
    return _trapezoid(dist, -0.02, 0.01, 0.04, 0.08)

def _price_far_above(dist: float) -> float:
    return _grade_up(dist, 0.03, 0.10)


def _volume_low(rel_vol: float) -> float:
    return _grade_down(rel_vol, 0.5, 0.8)

def _volume_normal(rel_vol: float) -> float:
    return _trapezoid(rel_vol, 0.7, 0.9, 1.2, 1.5)

def _volume_high(rel_vol: float) -> float:
    return _grade_up(rel_vol, 1.2, 2.0)


def _fund_poor(score: float) -> float:
    return _grade_down(score, 25.0, 40.0)

def _fund_average(score: float) -> float:
    return _trapezoid(score, 35.0, 45.0, 60.0, 70.0)

def _fund_good(score: float) -> float:
    return _grade_up(score, 60.0, 75.0)


# ── Fuzzy rules ─────────────────────────────────────────────────────────────

@dataclass
class _FuzzyRule:
    name: str
    antecedent: float       # Activation strength (min of inputs)
    consequent: str         # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"

# Consequent centroids on 0-100 scale
_CONSEQUENT_CENTERS = {
    "strong_buy": 90.0,
    "buy": 72.0,
    "hold": 50.0,
    "sell": 28.0,
    "strong_sell": 10.0,
}


class FuzzyTradingSystem:
    """
    Fuzzy inference engine for trading signals.
    Evaluates ~15 expert rules and defuzzifies to a trading decision.
    """

    def evaluate(
        self,
        rsi: float,
        macd_histogram: float,
        price_vs_sma50_pct: float,
        relative_volume: float,
        fundamental_score: float,
        market_regime: str,
    ) -> FuzzySignal:
        """
        Evaluate all fuzzy rules and produce a trading signal.

        Args:
            rsi: RSI(14) value, 0-100
            macd_histogram: MACD histogram value (normalized by price)
            price_vs_sma50_pct: (price - sma50) / sma50, e.g. -0.05 = 5% below
            relative_volume: current_vol / 20d_avg, e.g. 1.5 = 50% above average
            fundamental_score: 0-100 fundamental score
            market_regime: "BULL" | "BEAR" | "SIDEWAYS" | "VOLATILE"
        """
        # Compute membership degrees
        rsi_os = _rsi_oversold(rsi)
        rsi_n = _rsi_neutral(rsi)
        rsi_ob = _rsi_overbought(rsi)

        macd_neg = _macd_negative(macd_histogram)
        macd_nz = _macd_near_zero(macd_histogram)
        macd_pos = _macd_positive(macd_histogram)

        p_far_below = _price_far_below(price_vs_sma50_pct)
        p_below = _price_below(price_vs_sma50_pct)
        p_above = _price_above(price_vs_sma50_pct)
        p_far_above = _price_far_above(price_vs_sma50_pct)

        vol_low = _volume_low(relative_volume)
        vol_norm = _volume_normal(relative_volume)
        vol_high = _volume_high(relative_volume)

        fund_poor = _fund_poor(fundamental_score)
        fund_avg = _fund_average(fundamental_score)
        fund_good = _fund_good(fundamental_score)

        is_bear = 1.0 if market_regime == "BEAR" else 0.0
        is_bull = 1.0 if market_regime == "BULL" else 0.0
        is_sideways = 1.0 if market_regime == "SIDEWAYS" else 0.0

        # ── Evaluate rules (AND = min, OR = max) ────────────────────────

        rules: list[_FuzzyRule] = []

        # BUY rules
        rules.append(_FuzzyRule(
            "oversold_bounce",
            min(rsi_os, macd_pos, vol_high),
            "strong_buy",
        ))
        rules.append(_FuzzyRule(
            "oversold_macd_turn",
            min(rsi_os, macd_nz),
            "buy",
        ))
        rules.append(_FuzzyRule(
            "value_opportunity",
            min(fund_good, p_far_below),
            "buy",
        ))
        rules.append(_FuzzyRule(
            "trend_continuation",
            min(p_above, macd_pos, vol_norm, is_bull),
            "buy",
        ))
        rules.append(_FuzzyRule(
            "mean_reversion_buy",
            min(p_far_below, rsi_os, max(is_sideways, is_bear)),
            "strong_buy",
        ))
        rules.append(_FuzzyRule(
            "volume_breakout",
            min(vol_high, macd_pos, p_above),
            "buy",
        ))
        rules.append(_FuzzyRule(
            "fundamental_momentum",
            min(fund_good, macd_pos, rsi_n),
            "buy",
        ))

        # HOLD rules
        rules.append(_FuzzyRule(
            "neutral_conditions",
            min(rsi_n, macd_nz, vol_norm),
            "hold",
        ))
        rules.append(_FuzzyRule(
            "bear_no_signal",
            min(is_bear, vol_low, rsi_n),
            "hold",
        ))
        rules.append(_FuzzyRule(
            "weak_fundamentals_hold",
            min(fund_poor, rsi_n),
            "hold",
        ))

        # SELL rules
        rules.append(_FuzzyRule(
            "overbought_reversal",
            min(rsi_ob, macd_neg),
            "strong_sell",
        ))
        rules.append(_FuzzyRule(
            "overbought_bear",
            min(rsi_ob, vol_high, is_bear),
            "strong_sell",
        ))
        rules.append(_FuzzyRule(
            "trend_breakdown",
            min(p_far_below, macd_neg, vol_high),
            "sell",
        ))
        rules.append(_FuzzyRule(
            "bear_overbought",
            min(rsi_ob, is_bear),
            "sell",
        ))
        rules.append(_FuzzyRule(
            "poor_fundamentals_decline",
            min(fund_poor, macd_neg, p_below),
            "sell",
        ))

        # ── Defuzzify using weighted average ────────────────────────────

        numerator = 0.0
        denominator = 0.0
        activations = {}

        for rule in rules:
            if rule.antecedent > 0.01:  # Only count meaningfully activated rules
                center = _CONSEQUENT_CENTERS[rule.consequent]
                numerator += rule.antecedent * center
                denominator += rule.antecedent
                activations[rule.name] = round(rule.antecedent, 3)

        if denominator < 0.01:
            # No rules activated — default hold
            return FuzzySignal(action="HOLD", strength=0.0, score=50.0, rule_activations={})

        score = numerator / denominator

        # Map score to action
        if score >= 65:
            action = "BUY"
            strength = min(1.0, (score - 50) / 40)
        elif score <= 35:
            action = "SELL"
            strength = min(1.0, (50 - score) / 40)
        else:
            action = "HOLD"
            strength = 1.0 - abs(score - 50) / 15  # Stronger hold near 50

        return FuzzySignal(
            action=action,
            strength=round(strength, 3),
            score=round(score, 1),
            rule_activations=activations,
        )
