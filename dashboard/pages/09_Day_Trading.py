"""Day Trading Dashboard — Real-time intraday scalping monitor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Day Trading", page_icon="", layout="wide")
from dashboard.theme import (
    apply_theme, apply_plotly_theme, COLORS,
    section_header, styled_metric, status_badge,
)
apply_theme()

# ── Auto-refresh every 10 seconds ───────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, key="daytrading_refresh")
except ImportError:
    pass


# ── Dependencies ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_deps():
    from database.manager import DatabaseManager
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    return cfg, db


cfg, db = get_deps()
currency = "\u20b9" if cfg.market.exchange == "IN" else "$"


# ── Ensure intraday tables exist ─────────────────────────────────────────────
def _ensure_intraday_tables():
    """Create intraday_trades and intraday_signals tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS intraday_trades (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        trading_date      DATE NOT NULL DEFAULT (date('now')),
        symbol            TEXT NOT NULL,
        action            TEXT NOT NULL,
        qty               REAL NOT NULL,
        entry_price       REAL NOT NULL,
        exit_price        REAL,
        stop_loss_price   REAL,
        target_price      REAL,
        strategy_name     TEXT,
        status            TEXT DEFAULT 'OPEN',
        signal_strength   REAL,
        opened_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at         TIMESTAMP,
        realized_pnl      REAL,
        unrealized_pnl    REAL,
        net_pnl           REAL,
        minutes_held      REAL,
        brokerage_cost    REAL DEFAULT 0,
        paper_trading     INTEGER DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS intraday_signals (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        trading_date      DATE NOT NULL DEFAULT (date('now')),
        timestamp         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        symbol            TEXT NOT NULL,
        strategy_name     TEXT,
        action            TEXT,
        strength          REAL,
        entry_price       REAL,
        vwap              REAL,
        xgboost_direction TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_intraday_trades_date ON intraday_trades(trading_date);
    CREATE INDEX IF NOT EXISTS idx_intraday_signals_date ON intraday_signals(trading_date);
    """
    with db.get_connection() as conn:
        conn.executescript(ddl)
        conn.commit()


_ensure_intraday_tables()


# ── IST helpers ──────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(IST)


def _market_is_open() -> bool:
    now = _now_ist()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _minutes_until_force_exit() -> int:
    """Minutes until 15:15 IST force-exit deadline."""
    now = _now_ist()
    deadline = now.replace(hour=15, minute=15, second=0, microsecond=0)
    diff = (deadline - now).total_seconds() / 60
    return max(0, int(diff))


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEADER + STATUS
# ══════════════════════════════════════════════════════════════════════════════
section_header("Intraday Day Trading", "Real-time scalping \u2014 VWAP, ORB, Momentum strategies")

col_mode, col_market, col_exit = st.columns(3)
with col_mode:
    if cfg.user.paper_trading:
        status_badge("PAPER TRADING", "info")
    else:
        status_badge("LIVE TRADING", "danger")
with col_market:
    if _market_is_open():
        status_badge("MARKET OPEN", "success")
    else:
        status_badge("MARKET CLOSED", "default")
with col_exit:
    mins = _minutes_until_force_exit()
    if mins > 60:
        status_badge(f"Force-exit in {mins} min", "default")
    elif mins > 0:
        status_badge(f"Force-exit in {mins} min", "warning")
    else:
        status_badge("Force-exit deadline passed", "danger")

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TODAY'S P&L SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section_header("Today's P&L Summary")


@st.cache_data(ttl=10)
def get_pnl_summary():
    with db.get_connection() as conn:
        # Realized P&L from closed trades
        row = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN status='CLOSED' THEN realized_pnl ELSE 0 END), 0) AS realized,
                COALESCE(SUM(CASE WHEN status='OPEN' THEN unrealized_pnl ELSE 0 END), 0) AS unrealized,
                COUNT(*) AS trade_count,
                SUM(CASE WHEN status='CLOSED' AND realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END) AS closed_count,
                COALESCE(AVG(CASE WHEN status='CLOSED' THEN minutes_held END), 0) AS avg_hold,
                COALESCE(SUM(brokerage_cost), 0) AS total_brokerage
            FROM intraday_trades
            WHERE trading_date = date('now')
              AND status IN ('OPEN', 'CLOSED')
        """).fetchone()
        return dict(row) if row else {}


pnl = get_pnl_summary()
total_pnl = pnl.get("realized", 0) + pnl.get("unrealized", 0)
trade_count = pnl.get("trade_count", 0)
closed_count = pnl.get("closed_count", 0)
wins = pnl.get("wins", 0)
win_rate = (wins / closed_count * 100) if closed_count > 0 else 0
avg_hold = pnl.get("avg_hold", 0)
total_brokerage = pnl.get("total_brokerage", 0)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    pnl_delta = f"{'+' if total_pnl >= 0 else ''}{currency}{total_pnl:,.2f}"
    styled_metric(
        "Total P&L", f"{currency}{total_pnl:,.2f}",
        delta=pnl_delta,
        delta_color="green" if total_pnl >= 0 else "red",
    )
with c2:
    styled_metric("Trades Today", str(trade_count))
with c3:
    styled_metric(
        "Win Rate", f"{win_rate:.1f}%",
        delta=f"{wins}W / {closed_count - wins}L" if closed_count > 0 else "No closed trades",
        delta_color="green" if win_rate >= 50 else "red" if closed_count > 0 else "muted",
    )
with c4:
    styled_metric("Avg Hold Time", f"{avg_hold:.1f} min")
with c5:
    styled_metric("Total Brokerage", f"{currency}{total_brokerage:,.2f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 3. OPEN POSITIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════
section_header("Open Positions")


@st.cache_data(ttl=10)
def get_open_positions():
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT symbol, action, qty, entry_price, stop_loss_price,
                   target_price, strategy_name, opened_at, signal_strength
            FROM intraday_trades
            WHERE status = 'OPEN' AND trading_date = date('now')
            ORDER BY opened_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


open_positions = get_open_positions()

if open_positions:
    df_open = pd.DataFrame(open_positions)
    col_config = {
        "symbol": st.column_config.TextColumn("Symbol"),
        "action": st.column_config.TextColumn("Action"),
        "qty": st.column_config.NumberColumn("Qty", format="%.0f"),
        "entry_price": st.column_config.NumberColumn("Entry", format="%.2f"),
        "stop_loss_price": st.column_config.NumberColumn("Stop Loss", format="%.2f"),
        "target_price": st.column_config.NumberColumn("Target", format="%.2f"),
        "strategy_name": st.column_config.TextColumn("Strategy"),
        "opened_at": st.column_config.TextColumn("Opened At"),
        "signal_strength": st.column_config.NumberColumn("Strength", format="%.2f"),
    }
    st.dataframe(df_open, use_container_width=True, hide_index=True, column_config=col_config)
else:
    st.info("No open intraday positions right now.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LIVE 1-MINUTE CHART (OPTIONAL)
# ══════════════════════════════════════════════════════════════════════════════
section_header("Live 1-Minute Chart", "Select a symbol to view intraday candlestick with VWAP overlay")


@st.cache_data(ttl=10)
def get_todays_symbols():
    """Get unique symbols from today's trades and signals."""
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT DISTINCT symbol FROM (
                SELECT symbol FROM intraday_trades WHERE trading_date = date('now')
                UNION
                SELECT symbol FROM intraday_signals WHERE trading_date = date('now')
            ) ORDER BY symbol
        """).fetchall()
        return [r["symbol"] for r in rows]


today_symbols = get_todays_symbols()

# Allow manual entry too
symbol_options = today_symbols if today_symbols else []
chart_symbol = st.selectbox(
    "Select symbol",
    options=symbol_options,
    index=0 if symbol_options else None,
    placeholder="Choose a symbol from today's trades/signals...",
)

if chart_symbol:
    try:
        from data.price_feed import PriceFeed
        feed = PriceFeed(exchange=cfg.market.exchange)

        # yfinance: 1m data requires period <= 7d
        suffix = ".NS" if cfg.market.exchange == "IN" and not chart_symbol.endswith(".NS") else ""
        yf_symbol = chart_symbol + suffix

        df_1m = feed.get_historical(yf_symbol, period="1d", interval="1m")

        if not df_1m.empty:
            # Compute VWAP
            typical_price = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3
            cum_tp_vol = (typical_price * df_1m["volume"]).cumsum()
            cum_vol = df_1m["volume"].cumsum()
            df_1m["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_1m.index,
                open=df_1m["open"],
                high=df_1m["high"],
                low=df_1m["low"],
                close=df_1m["close"],
                name="Price",
                increasing_line_color=COLORS["green"],
                decreasing_line_color=COLORS["red"],
            ))

            # VWAP overlay
            fig.add_trace(go.Scatter(
                x=df_1m.index,
                y=df_1m["vwap"],
                name="VWAP",
                line=dict(color=COLORS["accent2"], width=2, dash="dot"),
            ))

            fig.update_layout(
                title=f"{chart_symbol} \u2014 1-Minute Candles + VWAP",
                height=400,
                xaxis_rangeslider_visible=False,
                xaxis_title="Time",
                yaxis_title="Price",
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No 1-minute data available for {chart_symbol}. Market may be closed.")
    except ImportError:
        st.error("PriceFeed module not available. Ensure yfinance is installed.")
    except Exception as e:
        st.error(f"Chart error: {e}")
else:
    st.info("Select a symbol above to view the intraday chart, or trade/signal data is needed first.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RECENT SIGNALS FEED
# ══════════════════════════════════════════════════════════════════════════════
section_header("Recent Signals Feed", "Latest intraday signals from all strategies")


@st.cache_data(ttl=10)
def get_recent_signals():
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT symbol, timestamp, strategy_name, action, strength,
                   entry_price, vwap, xgboost_direction
            FROM intraday_signals
            WHERE trading_date = date('now')
            ORDER BY timestamp DESC
            LIMIT 30
        """).fetchall()
        return [dict(r) for r in rows]


signals = get_recent_signals()

if signals:
    df_signals = pd.DataFrame(signals)

    # Color the action column using emoji prefixes
    def _action_display(action):
        if action is None:
            return action
        a = str(action).upper()
        if a == "BUY":
            return f"\U0001f7e2 {a}"
        elif a == "SELL":
            return f"\U0001f534 {a}"
        else:
            return f"\u26aa {a}"

    df_signals["action"] = df_signals["action"].apply(_action_display)

    col_config_sig = {
        "symbol": st.column_config.TextColumn("Symbol"),
        "timestamp": st.column_config.TextColumn("Time"),
        "strategy_name": st.column_config.TextColumn("Strategy"),
        "action": st.column_config.TextColumn("Action"),
        "strength": st.column_config.NumberColumn("Strength", format="%.2f"),
        "entry_price": st.column_config.NumberColumn("Entry", format="%.2f"),
        "vwap": st.column_config.NumberColumn("VWAP", format="%.2f"),
        "xgboost_direction": st.column_config.TextColumn("XGBoost"),
    }
    st.dataframe(df_signals, use_container_width=True, hide_index=True, column_config=col_config_sig)
else:
    st.info("No intraday signals generated today yet.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRATEGY PERFORMANCE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
section_header("Strategy Performance Breakdown", "Closed intraday trades grouped by strategy")


@st.cache_data(ttl=10)
def get_strategy_performance():
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT
                strategy_name,
                COUNT(*) AS trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS win_rate,
                ROUND(COALESCE(SUM(realized_pnl), 0), 2) AS total_pnl,
                ROUND(COALESCE(AVG(realized_pnl), 0), 2) AS avg_pnl
            FROM intraday_trades
            WHERE status = 'CLOSED'
            GROUP BY strategy_name
            ORDER BY total_pnl DESC
        """).fetchall()
        return [dict(r) for r in rows]


strategy_perf = get_strategy_performance()

if strategy_perf:
    df_strat = pd.DataFrame(strategy_perf)
    col_config_strat = {
        "strategy_name": st.column_config.TextColumn("Strategy"),
        "trades": st.column_config.NumberColumn("Trades", format="%d"),
        "wins": st.column_config.NumberColumn("Wins", format="%d"),
        "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.1f%%"),
        "total_pnl": st.column_config.NumberColumn("Total P&L", format="%.2f"),
        "avg_pnl": st.column_config.NumberColumn("Avg P&L", format="%.2f"),
    }
    st.dataframe(df_strat, use_container_width=True, hide_index=True, column_config=col_config_strat)
else:
    st.info("No closed intraday trades to analyze yet.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 7. 30-DAY CALENDAR HEATMAP (Bar chart)
# ══════════════════════════════════════════════════════════════════════════════
section_header("30-Day P&L Calendar", "Daily net P&L for the last 30 trading days")


@st.cache_data(ttl=10)
def get_daily_pnl_30d():
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT trading_date,
                   ROUND(COALESCE(SUM(net_pnl), 0), 2) AS daily_pnl
            FROM intraday_trades
            WHERE status = 'CLOSED'
              AND trading_date >= date('now', '-30 days')
            GROUP BY trading_date
            ORDER BY trading_date
        """).fetchall()
        return [dict(r) for r in rows]


daily_pnl_data = get_daily_pnl_30d()

if daily_pnl_data:
    df_daily = pd.DataFrame(daily_pnl_data)
    df_daily["trading_date"] = pd.to_datetime(df_daily["trading_date"])

    bar_colors = [
        COLORS["green"] if v >= 0 else COLORS["red"]
        for v in df_daily["daily_pnl"]
    ]

    fig_cal = go.Figure(go.Bar(
        x=df_daily["trading_date"],
        y=df_daily["daily_pnl"],
        marker_color=bar_colors,
        name="Daily P&L",
    ))
    fig_cal.update_layout(
        title="Daily Net P&L (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title=f"P&L ({currency})",
        height=320,
    )
    fig_cal.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,0.4)")
    apply_plotly_theme(fig_cal)
    st.plotly_chart(fig_cal, use_container_width=True)
else:
    st.info("No closed intraday trades in the last 30 days to display.")
