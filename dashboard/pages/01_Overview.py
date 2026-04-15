"""Portfolio Overview — P&L, open positions, equity curve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("Portfolio Overview")

@st.cache_resource
def get_deps():
    from database.manager import DatabaseManager
    from database.repository import TradeRepository
    from database.performance_tracker import PerformanceTracker
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    repo = TradeRepository(db)
    perf = PerformanceTracker(db)
    return cfg, repo, perf

cfg, repo, perf = get_deps()

# Auto-refresh every 30 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="overview_refresh")
except ImportError:
    pass

# ── Performance metrics ───────────────────────────────────────────────────────
metrics = perf.calculate(days_back=30)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Return", f"{metrics.total_return_pct:+.1%}")
col2.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
col3.metric("Win Rate", f"{metrics.win_rate:.0%}")
col4.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1%}")
col5.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
col6.metric("Total Trades", metrics.total_trades)

st.markdown("---")

# ── Equity curve ──────────────────────────────────────────────────────────────
history = repo.get_portfolio_history(days=90)
if history:
    df_hist = pd.DataFrame(history)
    df_hist["snapshot_date"] = pd.to_datetime(df_hist["snapshot_date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["snapshot_date"],
        y=df_hist["total_value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.1)",
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title=f"Value ({'₹' if cfg.market.exchange == 'IN' else '$'})",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No portfolio history yet. Start the trading loop.")

# ── Daily P&L bar chart ───────────────────────────────────────────────────────
if history:
    df_hist["daily_pnl"] = df_hist["daily_pnl"].fillna(0)
    colors = ["green" if v >= 0 else "red" for v in df_hist["daily_pnl"]]
    fig2 = go.Figure(go.Bar(
        x=df_hist["snapshot_date"].tail(30),
        y=df_hist["daily_pnl"].tail(30),
        marker_color=colors,
        name="Daily P&L",
    ))
    fig2.update_layout(
        title="Daily P&L (Last 30 Days)",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Open positions ────────────────────────────────────────────────────────────
st.subheader("Open Positions")
open_trades = repo.get_open_trades()
if open_trades:
    df_pos = pd.DataFrame(open_trades)
    display_cols = ["symbol", "action", "shares", "entry_price", "stop_loss_price",
                    "conviction", "primary_thesis", "opened_at"]
    available = [c for c in display_cols if c in df_pos.columns]
    st.dataframe(
        df_pos[available].rename(columns={
            "entry_price": f"Entry {'₹' if cfg.market.exchange == 'IN' else '$'}",
            "stop_loss_price": f"Stop {'₹' if cfg.market.exchange == 'IN' else '$'}",
            "conviction": "Conviction",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No open positions currently.")

# ── Portfolio heat breakdown ──────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Trade Statistics")
    stats_df = pd.DataFrame([
        ("Winning Trades", metrics.winning_trades),
        ("Losing Trades", metrics.losing_trades),
        ("Avg Win", f"{metrics.avg_win_pct:+.1%}"),
        ("Avg Loss", f"{metrics.avg_loss_pct:+.1%}"),
        ("Largest Win", f"{metrics.largest_win_pct:+.1%}"),
        ("Largest Loss", f"{metrics.largest_loss_pct:+.1%}"),
        ("Avg Holding Days", f"{metrics.avg_holding_days:.1f}"),
        ("Expectancy", f"{metrics.expectancy_pct:+.2%}"),
        ("Sortino Ratio", f"{metrics.sortino_ratio:.2f}"),
        ("vs Benchmark (SPY)", f"{metrics.benchmark_return_pct:+.1%}"),
        ("Alpha", f"{metrics.alpha:+.2%}"),
        ("Beta", f"{metrics.beta:.2f}"),
    ], columns=["Metric", "Value"])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Recent Activity")
    recent_trades = repo.get_trade_history(limit=10)
    if recent_trades:
        df_recent = pd.DataFrame(recent_trades)
        show_cols = ["symbol", "action", "status", "realized_pnl_pct", "opened_at"]
        available = [c for c in show_cols if c in df_recent.columns]
        st.dataframe(df_recent[available], use_container_width=True, hide_index=True)
    else:
        st.info("No trade history yet.")
