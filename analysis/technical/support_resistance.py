"""
Support & Resistance detection using pivot point analysis.
Identifies key price levels the stock has respected historically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

log = get_logger("support_resistance")


@dataclass
class SRLevel:
    price: float
    strength: float     # 0–100, based on number of touches + recency + volume
    touches: int
    level_type: str     # "support" | "resistance"


@dataclass
class SRAnalysis:
    symbol: str
    current_price: float
    supports: list[SRLevel]        # Sorted descending (nearest first)
    resistances: list[SRLevel]     # Sorted ascending (nearest first)
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    risk_reward_to_resistance: Optional[float]  # (R - price) / (price - S)


class SupportResistanceAnalyzer:

    def __init__(self, lookback: int = 252, cluster_pct: float = 0.005):
        self.lookback = lookback     # Trading days (~1 year)
        self.cluster_pct = cluster_pct  # Cluster levels within 0.5%

    def analyze(self, symbol: str, df: pd.DataFrame) -> SRAnalysis:
        if len(df) < 30:
            price = float(df["close"].iloc[-1]) if not df.empty else 0
            return SRAnalysis(symbol=symbol, current_price=price,
                              supports=[], resistances=[],
                              nearest_support=None, nearest_resistance=None,
                              risk_reward_to_resistance=None)

        df = df.tail(self.lookback).copy()
        price = float(df["close"].iloc[-1])

        # Find pivot highs and lows (5-bar lookback each side)
        pivots = self._find_pivots(df, left=5, right=5)

        # Cluster nearby pivots
        levels = self._cluster_levels(pivots, df)

        # Classify as support or resistance relative to current price
        supports = sorted(
            [l for l in levels if l.price < price],
            key=lambda x: x.price, reverse=True  # nearest first
        )[:5]
        resistances = sorted(
            [l for l in levels if l.price > price],
            key=lambda x: x.price  # nearest first
        )[:5]

        nearest_sup = supports[0].price if supports else price * 0.95
        nearest_res = resistances[0].price if resistances else price * 1.05

        rr = None
        if nearest_res > price and price > nearest_sup:
            potential_gain = nearest_res - price
            potential_loss = price - nearest_sup
            rr = potential_gain / potential_loss if potential_loss > 0 else None

        for s in supports:
            s.level_type = "support"
        for r in resistances:
            r.level_type = "resistance"

        return SRAnalysis(
            symbol=symbol,
            current_price=price,
            supports=supports,
            resistances=resistances,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            risk_reward_to_resistance=round(rr, 2) if rr else None,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int) -> list[tuple[float, float, int]]:
        """Find swing high/low pivots. Returns list of (price, volume, bar_index)."""
        pivots = []
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values
        n = len(highs)

        for i in range(left, n - right):
            # Pivot high
            if all(highs[i] >= highs[i - j] for j in range(1, left + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, right + 1)):
                pivots.append((highs[i], volumes[i], i))
            # Pivot low
            if all(lows[i] <= lows[i - j] for j in range(1, left + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, right + 1)):
                pivots.append((lows[i], volumes[i], i))

        return pivots

    def _cluster_levels(
        self, pivots: list[tuple[float, float, int]], df: pd.DataFrame
    ) -> list[SRLevel]:
        """Cluster nearby pivots into single S/R levels."""
        if not pivots:
            return []

        pivots_sorted = sorted(pivots, key=lambda x: x[0])
        clusters: list[list[tuple[float, float, int]]] = []
        current_cluster = [pivots_sorted[0]]

        for pivot in pivots_sorted[1:]:
            ref_price = current_cluster[0][0]
            if abs(pivot[0] - ref_price) / ref_price <= self.cluster_pct:
                current_cluster.append(pivot)
            else:
                clusters.append(current_cluster)
                current_cluster = [pivot]
        clusters.append(current_cluster)

        total_bars = len(df)
        levels = []
        for cluster in clusters:
            avg_price = np.mean([c[0] for c in cluster])
            touches = len(cluster)
            # Recency score — more recent = higher weight
            recency = np.mean([c[2] / total_bars for c in cluster])
            volume_weight = np.mean([c[1] for c in cluster])
            strength = min(100, touches * 15 + recency * 30)
            levels.append(SRLevel(
                price=round(avg_price, 2),
                strength=round(strength, 1),
                touches=touches,
                level_type="unknown",
            ))

        return levels
