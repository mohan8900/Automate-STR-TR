"""Portfolio Overview — Investment tracking, P&L, positions, and LLM results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Overview", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header, styled_metric
apply_theme()
section_header("Portfolio Overview", "Investment tracking, profit/loss, and AI analysis results")

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
currency = "\u20b9" if cfg.market.exchange == "IN" else "$"

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="overview_refresh")
except ImportError:
    pass

# ── Investment vs Profit (the key info user wants) ───────────────────────────
section_header("Investment & Profit")

history = repo.get_portfolio_history(days=90)
initial_investment = cfg.user.investment_amount

if history:
    latest = history[-1]
    current_value = latest.get("total_value", initial_investment)
    cash = latest.get("cash", 0)
    invested = latest.get("invested_value", 0)
    total_profit = current_value - initial_investment
    total_return_pct = (total_profit / initial_investment) * 100 if initial_investment > 0 else 0
    daily_pnl = latest.get("daily_pnl", 0) or 0
    daily_pnl_pct = latest.get("daily_pnl_pct", 0) or 0
else:
    current_value = initial_investment
    cash = initial_investment
    invested = 0
    total_profit = 0
    total_return_pct = 0
    daily_pnl = 0
    daily_pnl_pct = 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    styled_metric("Initial Investment", f"{currency}{initial_investment:,.0f}")
with c2:
    styled_metric("Current Value", f"{currency}{current_value:,.0f}",
                  delta=f"{total_return_pct:+.2f}%",
                  delta_color="green" if total_profit >= 0 else "red")
with c3:
    profit_sign = "+" if total_profit >= 0 else ""
    styled_metric("Total Profit / Loss", f"{profit_sign}{currency}{total_profit:,.0f}",
                  delta=f"{total_return_pct:+.2f}%",
                  delta_color="green" if total_profit >= 0 else "red")
with c4:
    styled_metric("Cash Available", f"{currency}{cash:,.0f}")
with c5:
    styled_metric("Today's P&L", f"{currency}{daily_pnl:,.0f}",
                  delta=f"{daily_pnl_pct:+.2f}%",
                  delta_color="green" if daily_pnl >= 0 else "red")

st.markdown("")

# Invested vs Cash breakdown
c1, c2 = st.columns(2)
with c1:
    if invested > 0 or cash > 0:
        fig_pie = go.Figure(go.Pie(
            labels=["Invested in Stocks", "Cash"],
            values=[max(0, invested), max(0, cash)],
            hole=0.6,
            marker=dict(colors=[COLORS["accent"], COLORS["card_alt"]],
                        line=dict(color="#1a1f2e", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#1e293b"),
        ))
        fig_pie.update_layout(
            title="Capital Allocation",
            height=280,
            showlegend=False,
            annotations=[dict(text=f"{currency}{current_value:,.0f}", x=0.5, y=0.5,
                             font_size=18, font_color="#e2e8f0", showarrow=False)],
        )
        apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    # Profit gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_return_pct,
        number=dict(suffix="%", font=dict(size=36, color="#e2e8f0")),
        delta=dict(reference=0, increasing_color=COLORS["green"],
                   decreasing_color=COLORS["red"]),
        gauge=dict(
            axis=dict(range=[-20, 20], tickfont=dict(color="#94a3b8")),
            bar=dict(color=COLORS["green"] if total_return_pct >= 0 else COLORS["red"]),
            bgcolor="#232a3b",
            borderwidth=0,
            steps=[
                dict(range=[-20, -5], color="rgba(239,68,68,0.15)"),
                dict(range=[-5, 0], color="rgba(239,68,68,0.08)"),
                dict(range=[0, 5], color="rgba(16,185,129,0.08)"),
                dict(range=[5, 20], color="rgba(16,185,129,0.15)"),
            ],
        ),
        title=dict(text="Total Return", font=dict(color="#94a3b8", size=14)),
    ))
    fig_gauge.update_layout(height=280)
    apply_plotly_theme(fig_gauge)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ── Performance metrics ──────────────────────────────────────────────────────
section_header("Performance Metrics")
metrics = perf.calculate(days_back=30)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Return", f"{metrics.total_return_pct:+.1%}")
c2.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
c3.metric("Win Rate", f"{metrics.win_rate:.0%}")
c4.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.1%}")
c5.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
c6.metric("Total Trades", metrics.total_trades)

st.markdown("---")

# ── Equity curve ─────────────────────────────────────────────────────────────
if history:
    df_hist = pd.DataFrame(history)
    df_hist["snapshot_date"] = pd.to_datetime(df_hist["snapshot_date"])

    fig = go.Figure()
    # Add initial investment reference line
    fig.add_hline(y=initial_investment, line_dash="dot",
                  line_color="rgba(148,163,184,0.4)",
                  annotation_text=f"Initial: {currency}{initial_investment:,.0f}",
                  annotation_font_color="#94a3b8")
    fig.add_trace(go.Scatter(
        x=df_hist["snapshot_date"], y=df_hist["total_value"],
        mode="lines", name="Portfolio Value",
        line=dict(color=COLORS["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title=f"Value ({currency})",
        height=320,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Daily P&L
    df_hist["daily_pnl"] = df_hist["daily_pnl"].fillna(0)
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df_hist["daily_pnl"]]
    fig2 = go.Figure(go.Bar(
        x=df_hist["snapshot_date"].tail(30), y=df_hist["daily_pnl"].tail(30),
        marker_color=colors, name="Daily P&L",
    ))
    fig2.update_layout(title="Daily P&L (Last 30 Days)", height=220)
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No portfolio history yet. Start the trading loop to begin.")

st.markdown("---")

# ── Open positions ───────────────────────────────────────────────────────────
section_header("Open Positions")
open_trades = repo.get_open_trades()
if open_trades:
    df_pos = pd.DataFrame(open_trades)
    display_cols = ["symbol", "action", "shares", "entry_price", "stop_loss_price",
                    "conviction", "primary_thesis", "opened_at"]
    available = [c for c in display_cols if c in df_pos.columns]
    st.dataframe(df_pos[available], use_container_width=True, hide_index=True)
else:
    st.info("No open positions currently.")

st.markdown("---")

# ── LLM Analysis Results ────────────────────────────────────────────────────
section_header("AI / LLM Analysis Results", "Recent recommendations from the LLM arbiter with model signals")

try:
    analysis_logs = repo.get_analysis_logs(limit=20)
except Exception:
    analysis_logs = []

if analysis_logs:
    for log_entry in analysis_logs[:10]:
        symbol = log_entry.get("symbol", "?")
        action = log_entry.get("action_recommended", "PASS")
        conviction = log_entry.get("conviction", 0)
        thesis = log_entry.get("llm_thesis", "")
        timestamp = log_entry.get("timestamp", "")
        composite = log_entry.get("composite_score", 0) or 0
        tech_score = log_entry.get("technical_score", 0) or 0
        fund_score = log_entry.get("fundamental_score", 0) or 0
        regime = log_entry.get("market_regime", "")

        # Color by action
        action_colors = {
            "BUY": COLORS["green"], "SELL": COLORS["red"],
            "SHORT": COLORS["orange"], "PASS": "#64748b", "HOLD": "#94a3b8"
        }
        action_color = action_colors.get(action, "#94a3b8")

        with st.expander(f"**{symbol}** — {action} (conviction {conviction}/10) — {timestamp}", expanded=False):
            # Scores row
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Action", action)
            sc2.metric("Conviction", f"{conviction}/10")
            sc3.metric("Technical", f"{tech_score:.0f}")
            sc4.metric("Fundamental", f"{fund_score:.0f}")
            sc5.metric("Composite", f"{composite:.0f}")

            # Thesis
            if thesis:
                st.markdown(f"**AI Thesis:** {thesis}")

            # Risks
            risks_raw = log_entry.get("llm_risks", "[]")
            try:
                risks = json.loads(risks_raw) if isinstance(risks_raw, str) else risks_raw
                if risks:
                    st.markdown("**Key Risks:** " + " | ".join(str(r) for r in risks))
            except Exception:
                pass

            # Model signals (from intelligence aggregator)
            signals_raw = log_entry.get("model_signals")
            if signals_raw:
                try:
                    signals = json.loads(signals_raw) if isinstance(signals_raw, str) else signals_raw
                    st.markdown("---")
                    st.markdown("**AI Model Signals:**")
                    ms1, ms2, ms3, ms4 = st.columns(4)

                    ensemble = signals.get("ensemble", {})
                    lstm = signals.get("lstm", {})
                    fuzzy = signals.get("fuzzy", {})

                    with ms1:
                        direction = ensemble.get("direction", "N/A")
                        prob = ensemble.get("probability")
                        st.metric("ML Ensemble", direction,
                                  delta=f"{prob:.0%}" if prob else "")
                    with ms2:
                        direction = lstm.get("direction", "N/A")
                        prob = lstm.get("probability")
                        st.metric("LSTM Neural Net", direction,
                                  delta=f"{prob:.0%}" if prob else "")
                    with ms3:
                        faction = fuzzy.get("action", "N/A")
                        fscore = fuzzy.get("score")
                        st.metric("Fuzzy Logic", faction,
                                  delta=f"{fscore:.0f}/100" if fscore else "")
                    with ms4:
                        agreement = signals.get("agreement", 0)
                        bullish = signals.get("bullish_prob", 0.5)
                        st.metric("Model Agreement", f"{agreement:.0%}",
                                  delta=f"Bullish: {bullish:.0%}")

                except Exception:
                    pass

            # Full LLM response
            full_resp = log_entry.get("llm_full_response")
            if full_resp:
                try:
                    resp = json.loads(full_resp) if isinstance(full_resp, str) else full_resp
                    with st.expander("Full LLM Response"):
                        st.json(resp)
                except Exception:
                    pass
else:
    st.info("No LLM analysis results yet. Run a backtest or trading cycle first.")

st.markdown("---")

# ── Trade Statistics ─────────────────────────────────────────────────────────
section_header("Trade Statistics")
col_left, col_right = st.columns(2)

with col_left:
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
        ("Alpha", f"{metrics.alpha:+.2%}"),
        ("Beta", f"{metrics.beta:.2f}"),
    ], columns=["Metric", "Value"])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("**Recent Activity**")
    recent_trades = repo.get_trade_history(limit=10)
    if recent_trades:
        df_recent = pd.DataFrame(recent_trades)
        show_cols = ["symbol", "action", "status", "realized_pnl_pct", "opened_at"]
        available = [c for c in show_cols if c in df_recent.columns]
        st.dataframe(df_recent[available], use_container_width=True, hide_index=True)
    else:
        st.info("No trade history yet.")
