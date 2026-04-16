"""
Repository — CRUD operations for all database tables.
Single source of truth for database access.
"""
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Optional

from database.manager import DatabaseManager
from core.logger import get_logger

log = get_logger("repository")


class TradeRepository:

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ── Trades ────────────────────────────────────────────────────────────

    def save_trade(self, trade, order, paper_trading: bool) -> int:
        sql = """
        INSERT INTO trades (
            symbol, action, shares, entry_price, stop_loss_price,
            take_profit_prices, order_id, status, paper_trading,
            conviction, primary_thesis, key_risks,
            technical_score, fundamental_score, composite_score,
            market_regime, vix_at_entry, opened_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, (
                trade.symbol,
                trade.action,
                trade.shares,
                trade.entry_price,
                trade.stop_loss_price,
                json.dumps(trade.take_profit_prices),
                order.order_id,
                1 if paper_trading else 0,
                trade.conviction,
                trade.primary_thesis,
                json.dumps(trade.key_risks or []),
                None,  # technical_score filled via analysis log
                None,
                None,
                None,  # market_regime
                None,  # vix
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
            return cur.lastrowid

    def save_failed_trade(self, trade, error_message: str) -> int:
        sql = """
        INSERT INTO trades (symbol, action, shares, entry_price, stop_loss_price,
            status, paper_trading, error_message, opened_at)
        VALUES (?, ?, ?, ?, ?, 'FAILED', 1, ?, ?)
        """
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, (
                trade.symbol, trade.action, trade.shares,
                trade.entry_price, trade.stop_loss_price,
                error_message, datetime.utcnow().isoformat(),
            ))
            conn.commit()
            return cur.lastrowid

    def mark_trade_closed(
        self,
        symbol: str,
        exit_price: Optional[float],
        close_reason: str = "manual",
    ) -> None:
        with self.db.get_connection() as conn:
            # Find the open trade
            row = conn.execute(
                "SELECT * FROM trades WHERE symbol=? AND status='OPEN' ORDER BY opened_at DESC LIMIT 1",
                (symbol,)
            ).fetchone()
            if not row:
                log.warning(f"No open trade found for {symbol}")
                return

            if exit_price and row["entry_price"]:
                pnl = (exit_price - row["entry_price"]) * row["shares"]
                pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"]
                opened = datetime.fromisoformat(row["opened_at"])
                days = (datetime.utcnow() - opened).days
            else:
                pnl = None
                pnl_pct = None
                days = None

            conn.execute("""
                UPDATE trades SET
                    status='CLOSED', exit_price=?, close_reason=?,
                    closed_at=?, realized_pnl=?, realized_pnl_pct=?, holding_days=?
                WHERE id=?
            """, (exit_price, close_reason, datetime.utcnow().isoformat(),
                  pnl, pnl_pct, days, row["id"]))
            conn.commit()

    def get_open_trades(self) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='OPEN' ORDER BY opened_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trade_history(self, limit: int = 100, symbol: Optional[str] = None) -> list[dict]:
        with self.db.get_connection() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE symbol=? ORDER BY opened_at DESC LIMIT ?",
                    (symbol, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_closed_trades(self, days_back: int = 30) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM trades WHERE status='CLOSED'
                AND closed_at >= date('now', ?)
                ORDER BY closed_at DESC
            """, (f"-{days_back} days",)).fetchall()
            return [dict(r) for r in rows]

    # ── Analysis Logs ─────────────────────────────────────────────────────

    def log_analysis(
        self,
        symbol: str,
        action: str,
        conviction: int,
        composite_score: float,
        technical_score: float,
        fundamental_score: float,
        sentiment_score: float,
        market_regime: str,
        vix_level: float,
        thesis: str,
        risks: list[str],
        approved: bool = False,
        trade_id: Optional[int] = None,
        llm_full_response: Optional[str] = None,
        model_signals: Optional[str] = None,
    ) -> int:
        sql = """
        INSERT INTO analysis_logs (
            symbol, action_recommended, conviction, composite_score,
            technical_score, fundamental_score, sentiment_score,
            market_regime, vix_level, llm_thesis, llm_risks,
            llm_full_response, model_signals,
            execution_approved, trade_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, (
                symbol, action, conviction, composite_score,
                technical_score, fundamental_score, sentiment_score,
                market_regime, vix_level, thesis, json.dumps(risks),
                llm_full_response, model_signals,
                1 if approved else 0, trade_id,
            ))
            conn.commit()
            return cur.lastrowid

    def get_analysis_logs(self, limit: int = 50) -> list[dict]:
        """Get recent LLM analysis logs."""
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM analysis_logs
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── Approval Queue ────────────────────────────────────────────────────

    def add_to_approval_queue(self, trade, risk_score: int, analysis_scores: dict) -> int:
        sql = """
        INSERT INTO approval_queue (
            symbol, action, shares, entry_price, stop_loss_price,
            take_profit_prices, position_value, portfolio_heat_add,
            conviction, primary_thesis, key_risks, llm_risk_score,
            technical_score, fundamental_score, composite_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, (
                trade.symbol, trade.action, trade.shares, trade.entry_price,
                trade.stop_loss_price,
                json.dumps(trade.take_profit_prices),
                trade.position_value, trade.portfolio_heat_contribution,
                trade.conviction, trade.primary_thesis,
                json.dumps(trade.key_risks or []),
                risk_score,
                analysis_scores.get("technical"),
                analysis_scores.get("fundamental"),
                analysis_scores.get("composite"),
            ))
            conn.commit()
            return cur.lastrowid

    def get_pending_approvals(self) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM approval_queue WHERE status='PENDING' ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def approve_trade(self, approval_id: int, note: str = "") -> None:
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE approval_queue SET status='APPROVED',
                decided_at=?, decision_note=? WHERE id=?
            """, (datetime.utcnow().isoformat(), note, approval_id))
            conn.commit()

    def reject_trade(self, approval_id: int, reason: str = "") -> None:
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE approval_queue SET status='REJECTED',
                decided_at=?, decision_note=? WHERE id=?
            """, (datetime.utcnow().isoformat(), reason, approval_id))
            conn.commit()

    # ── Portfolio Snapshots ───────────────────────────────────────────────

    def save_portfolio_snapshot(self, snapshot: dict) -> None:
        sql = """
        INSERT OR REPLACE INTO portfolio_snapshots (
            snapshot_date, total_value, cash, invested_value,
            unrealized_pnl, unrealized_pnl_pct, daily_pnl, daily_pnl_pct,
            cumulative_return, open_positions, portfolio_heat, vix_level, market_regime
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(sql, (
                snapshot.get("date", date.today().isoformat()),
                snapshot.get("total_value"), snapshot.get("cash"),
                snapshot.get("invested_value"), snapshot.get("unrealized_pnl"),
                snapshot.get("unrealized_pnl_pct"), snapshot.get("daily_pnl"),
                snapshot.get("daily_pnl_pct"), snapshot.get("cumulative_return"),
                snapshot.get("open_positions"), snapshot.get("portfolio_heat"),
                snapshot.get("vix_level"), snapshot.get("market_regime"),
            ))
            conn.commit()

    def get_portfolio_history(self, days: int = 30) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM portfolio_snapshots
                WHERE snapshot_date >= date('now', ?)
                ORDER BY snapshot_date
            """, (f"-{days} days",)).fetchall()
            return [dict(r) for r in rows]
