"""Trade history and performance analytics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="History", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("Trade History & Analytics")

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
    return repo, perf

repo, perf = get_deps()

# ── Performance overview ──────────────────────────────────────────────────────
time_range = st.selectbox("Time Range", ["7 days", "30 days", "90 days", "All time"], index=1)
days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All time": 3650}
days_back = days_map[time_range]

metrics = perf.calculate(days_back=days_back)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Returns")
    st.metric("Total Return", f"{metrics.total_return_pct:+.1%}")
    st.metric("vs Benchmark (SPY)", f"{metrics.benchmark_return_pct:+.1%}")
    st.metric("Alpha", f"{metrics.alpha:+.2%}")
    st.metric("Beta", f"{metrics.beta:.2f}")

with col2:
    st.subheader("Risk Metrics")
    st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1%}")
    st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")

with col3:
    st.subheader("Trade Stats")
    st.metric("Win Rate", f"{metrics.win_rate:.0%}", f"{metrics.winning_trades}/{metrics.total_trades}")
    st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
    st.metric("Expectancy", f"{metrics.expectancy_pct:+.2%} per trade")
    st.metric("Avg Holding Days", f"{metrics.avg_holding_days:.1f}")

st.markdown("---")

# ── Cumulative returns chart ──────────────────────────────────────────────────
history = repo.get_portfolio_history(days=days_back)
if len(history) > 1:
    df_hist = pd.DataFrame(history)
    df_hist["snapshot_date"] = pd.to_datetime(df_hist["snapshot_date"])
    start_val = df_hist["total_value"].iloc[0]
    df_hist["cumulative_return"] = (df_hist["total_value"] / start_val - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["snapshot_date"], y=df_hist["cumulative_return"],
        name="Portfolio", line=dict(color="#00b4d8", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Cumulative Return ({time_range})",
        yaxis_title="Return (%)", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Trade history table ───────────────────────────────────────────────────────
st.subheader("Closed Trades")
closed_trades = repo.get_closed_trades(days_back=days_back)

if closed_trades:
    df_trades = pd.DataFrame(closed_trades)

    # Filter
    col1, col2 = st.columns(2)
    with col1:
        symbols = ["All"] + sorted(df_trades["symbol"].unique().tolist())
        sym_filter = st.selectbox("Filter by Symbol", symbols)
    with col2:
        outcome_filter = st.selectbox("Filter by Outcome", ["All", "Winners", "Losers"])

    filtered = df_trades.copy()
    if sym_filter != "All":
        filtered = filtered[filtered["symbol"] == sym_filter]
    if outcome_filter == "Winners":
        filtered = filtered[filtered["realized_pnl_pct"] > 0]
    elif outcome_filter == "Losers":
        filtered = filtered[filtered["realized_pnl_pct"] <= 0]

    # Format
    if "realized_pnl_pct" in filtered.columns:
        filtered["P&L %"] = filtered["realized_pnl_pct"].apply(
            lambda x: f"{x:+.1%}" if x is not None else "N/A"
        )

    display_cols = [
        "symbol", "action", "shares", "entry_price", "exit_price",
        "P&L %", "realized_pnl", "holding_days", "close_reason",
        "conviction", "opened_at",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available], use_container_width=True, hide_index=True)

    # P&L distribution chart
    if "realized_pnl_pct" in filtered.columns:
        pnl_data = filtered["realized_pnl_pct"].dropna()
        if not pnl_data.empty:
            fig = px.histogram(
                pnl_data * 100, nbins=20,
                title="P&L Distribution (%)",
                color_discrete_sequence=["#00b4d8"],
                labels={"value": "P&L %"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No closed trades yet in the selected time range.")

# ── LLM accuracy analysis ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("AI Conviction vs Actual Performance")

try:
    from database.manager import DatabaseManager
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT t.conviction, t.realized_pnl_pct, t.symbol, t.market_regime
            FROM trades t
            WHERE t.status='CLOSED' AND t.realized_pnl_pct IS NOT NULL
            AND t.conviction IS NOT NULL
        """).fetchall()
        conviction_data = [dict(r) for r in rows]

    if conviction_data:
        df_conv = pd.DataFrame(conviction_data)
        fig = px.scatter(
            df_conv, x="conviction", y=df_conv["realized_pnl_pct"] * 100,
            color="market_regime", title="AI Conviction vs Trade P&L",
            labels={"conviction": "AI Conviction (1-10)", "y": "Actual P&L (%)"},
            trendline="ols",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher correlation between conviction and P&L = AI model is well-calibrated"
        )
    else:
        st.info("Need more closed trades to show conviction vs performance analysis.")
except Exception as e:
    st.caption(f"Analysis unavailable: {e}")
