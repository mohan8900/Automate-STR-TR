"""
Performance tracker — computes Sharpe, Sortino, drawdown, win rate, and other metrics.
Compares against benchmark (SPY / NIFTY).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from database.manager import DatabaseManager
from core.logger import get_logger

log = get_logger("performance")


@dataclass
class PerformanceMetrics:
    # Returns
    total_return_pct: float
    daily_return_pct: float
    monthly_return_pct: float
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Drawdown
    max_drawdown_pct: float
    current_drawdown_pct: float
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct: float
    avg_holding_days: float
    largest_win_pct: float
    largest_loss_pct: float
    # vs Benchmark
    benchmark_return_pct: float
    alpha: float
    beta: float


class PerformanceTracker:

    RISK_FREE_RATE = 0.05  # 5% annual

    def __init__(self, db: DatabaseManager):
        self.db = db

    def calculate(self, days_back: int = 30) -> PerformanceMetrics:
        """Calculate performance metrics over the last N days."""
        history = self._get_portfolio_history(days_back)
        trades = self._get_closed_trades(days_back)

        returns = self._calc_returns(history)
        trade_stats = self._calc_trade_stats(trades)
        benchmark_return = self._get_benchmark_return(days_back)

        if len(returns) < 2:
            return self._empty_metrics(trade_stats, benchmark_return)

        sharpe = self._sharpe(returns)
        sortino = self._sortino(returns)
        max_dd, current_dd = self._drawdown(history)
        calmar = (returns.sum() * (252 / len(returns))) / abs(max_dd) if max_dd != 0 else 0

        total_return = (
            (history[-1]["total_value"] - history[0]["total_value"]) / history[0]["total_value"]
            if history else 0
        )
        daily_return = float(returns.mean()) if len(returns) > 0 else 0
        monthly_return = daily_return * 21  # ~21 trading days per month

        # Alpha & Beta (simplified — linear regression vs benchmark)
        alpha, beta = self._alpha_beta(returns, benchmark_return, days_back)

        return PerformanceMetrics(
            total_return_pct=round(total_return, 4),
            daily_return_pct=round(daily_return, 4),
            monthly_return_pct=round(monthly_return, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd, 4),
            current_drawdown_pct=round(current_dd, 4),
            total_trades=trade_stats["total"],
            winning_trades=trade_stats["wins"],
            losing_trades=trade_stats["losses"],
            win_rate=round(trade_stats["win_rate"], 3),
            avg_win_pct=round(trade_stats["avg_win"], 4),
            avg_loss_pct=round(trade_stats["avg_loss"], 4),
            profit_factor=round(trade_stats["profit_factor"], 2),
            expectancy_pct=round(trade_stats["expectancy"], 4),
            avg_holding_days=round(trade_stats["avg_holding_days"], 1),
            largest_win_pct=round(trade_stats["largest_win"], 4),
            largest_loss_pct=round(trade_stats["largest_loss"], 4),
            benchmark_return_pct=round(benchmark_return, 4),
            alpha=round(alpha, 4),
            beta=round(beta, 2),
        )

    # ── Calculation helpers ───────────────────────────────────────────────

    def _get_portfolio_history(self, days: int) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM portfolio_snapshots
                WHERE snapshot_date >= date('now', ?)
                ORDER BY snapshot_date
            """, (f"-{days} days",)).fetchall()
            return [dict(r) for r in rows]

    def _get_closed_trades(self, days: int) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM trades WHERE status='CLOSED'
                AND closed_at >= date('now', ?)
            """, (f"-{days} days",)).fetchall()
            return [dict(r) for r in rows]

    def _calc_returns(self, history: list[dict]) -> pd.Series:
        if len(history) < 2:
            return pd.Series(dtype=float)
        values = [h["total_value"] for h in history]
        return pd.Series(values).pct_change().dropna()

    def _calc_trade_stats(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                "expectancy": 0, "avg_holding_days": 0,
                "largest_win": 0, "largest_loss": 0,
            }

        pnl_pcts = [t["realized_pnl_pct"] for t in trades if t.get("realized_pnl_pct") is not None]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p <= 0]

        win_rate = len(wins) / len(pnl_pcts) if pnl_pcts else 0
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        holding_days = [t["holding_days"] for t in trades if t.get("holding_days")]
        avg_holding = float(np.mean(holding_days)) if holding_days else 0

        return {
            "total": len(pnl_pcts),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_holding_days": avg_holding,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
        }

    def _sharpe(self, returns: pd.Series) -> float:
        if returns.std() == 0:
            return 0
        daily_rf = self.RISK_FREE_RATE / 252
        excess = returns - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _sortino(self, returns: pd.Series) -> float:
        daily_rf = self.RISK_FREE_RATE / 252
        excess = returns - daily_rf
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if excess.mean() > 0 else 0
        return float(excess.mean() / downside.std() * np.sqrt(252))

    def _drawdown(self, history: list[dict]) -> tuple[float, float]:
        if not history:
            return 0.0, 0.0
        values = pd.Series([h["total_value"] for h in history])
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])
        return max_dd, current_dd

    def _get_benchmark_return(self, days: int) -> float:
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period=f"{days + 5}d")
            if len(hist) < 2:
                return 0.0
            return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
        except Exception:
            return 0.0

    def _alpha_beta(
        self, returns: pd.Series, benchmark_return: float, days: int
    ) -> tuple[float, float]:
        """Simplified alpha/beta calculation."""
        if len(returns) < 5 or benchmark_return == 0:
            return 0.0, 1.0
        portfolio_annualized = float(returns.mean()) * 252
        benchmark_annualized = benchmark_return * (252 / days)
        beta = float(returns.std() / (benchmark_annualized / 252) if benchmark_annualized else 1.0)
        beta = max(0.1, min(3.0, beta))
        alpha = portfolio_annualized - (self.RISK_FREE_RATE + beta * (benchmark_annualized - self.RISK_FREE_RATE))
        return alpha, beta

    def _empty_metrics(self, trade_stats: dict, benchmark: float) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return_pct=0, daily_return_pct=0, monthly_return_pct=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown_pct=0, current_drawdown_pct=0,
            total_trades=trade_stats["total"],
            winning_trades=trade_stats["wins"],
            losing_trades=trade_stats["losses"],
            win_rate=trade_stats["win_rate"],
            avg_win_pct=trade_stats["avg_win"],
            avg_loss_pct=trade_stats["avg_loss"],
            profit_factor=trade_stats["profit_factor"],
            expectancy_pct=trade_stats["expectancy"],
            avg_holding_days=trade_stats["avg_holding_days"],
            largest_win_pct=trade_stats["largest_win"],
            largest_loss_pct=trade_stats["largest_loss"],
            benchmark_return_pct=benchmark,
            alpha=0, beta=1,
        )
