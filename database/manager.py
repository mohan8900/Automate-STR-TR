"""
Database manager — initializes SQLite schema and provides a connection factory.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from core.logger import get_logger

log = get_logger("database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    shares              REAL NOT NULL,
    entry_price         REAL NOT NULL,
    exit_price          REAL,
    stop_loss_price     REAL NOT NULL,
    take_profit_prices  TEXT,           -- JSON array
    order_id            TEXT,
    status              TEXT DEFAULT 'OPEN',   -- OPEN | CLOSED | CANCELLED
    close_reason        TEXT,           -- stop_loss | take_profit | manual | time_stop
    opened_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at           TIMESTAMP,
    realized_pnl        REAL,
    realized_pnl_pct    REAL,
    holding_days        INTEGER,
    paper_trading       INTEGER DEFAULT 1,
    -- LLM metadata
    conviction          INTEGER,
    primary_thesis      TEXT,
    key_risks           TEXT,           -- JSON array
    technical_score     REAL,
    fundamental_score   REAL,
    composite_score     REAL,
    market_regime       TEXT,
    vix_at_entry        REAL,
    -- Error tracking
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS analysis_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    timestamp           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action_recommended  TEXT,
    conviction          INTEGER,
    composite_score     REAL,
    technical_score     REAL,
    fundamental_score   REAL,
    sentiment_score     REAL,
    market_regime       TEXT,
    vix_level           REAL,
    llm_thesis          TEXT,
    llm_risks           TEXT,           -- JSON array
    execution_approved  INTEGER DEFAULT 0,
    trade_id            INTEGER REFERENCES trades(id)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date       DATE NOT NULL,
    snapshot_time       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_value         REAL NOT NULL,
    cash                REAL NOT NULL,
    invested_value      REAL NOT NULL,
    unrealized_pnl      REAL,
    unrealized_pnl_pct  REAL,
    daily_pnl           REAL,
    daily_pnl_pct       REAL,
    cumulative_return   REAL,
    open_positions      INTEGER,
    portfolio_heat      REAL,
    vix_level           REAL,
    market_regime       TEXT
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_date         DATE PRIMARY KEY,
    daily_return_pct    REAL,
    cumulative_return   REAL,
    sharpe_30d          REAL,
    sortino_30d         REAL,
    max_drawdown_30d    REAL,
    win_rate_30         REAL,
    profit_factor_30    REAL,
    avg_holding_days    REAL,
    total_trades        INTEGER,
    winning_trades      INTEGER,
    benchmark_return    REAL,
    alpha               REAL,
    beta                REAL
);

CREATE TABLE IF NOT EXISTS approval_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    shares              REAL,
    entry_price         REAL,
    stop_loss_price     REAL,
    take_profit_prices  TEXT,
    position_value      REAL,
    portfolio_heat_add  REAL,
    conviction          INTEGER,
    primary_thesis      TEXT,
    key_risks           TEXT,
    llm_risk_score      INTEGER,
    technical_score     REAL,
    fundamental_score   REAL,
    composite_score     REAL,
    status              TEXT DEFAULT 'PENDING',  -- PENDING | APPROVED | REJECTED | EXPIRED
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at          TIMESTAMP,
    decision_note       TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_queue(status);
"""


class DatabaseManager:

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        log.info(f"Database initialized at {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
