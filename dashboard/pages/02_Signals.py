"""Signals page — live analysis results and stock charts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="Signals", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("Live Market Signals")

@st.cache_resource
def get_deps():
    from database.manager import DatabaseManager
    from database.repository import TradeRepository
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    return cfg, TradeRepository(db)

cfg, repo = get_deps()

# ── Recent analysis logs ──────────────────────────────────────────────────────
st.subheader("Latest AI Analysis Results")

with cfg.database.path.__class__(cfg.database.path):
    pass  # Just ensure path is used

try:
    from database.manager import DatabaseManager
    db = DatabaseManager(cfg.database.path)
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM analysis_logs
            ORDER BY timestamp DESC LIMIT 50
        """).fetchall()
        logs = [dict(r) for r in rows]
except Exception:
    logs = []

if logs:
    df_logs = pd.DataFrame(logs)

    # Color coding
    def action_color(action):
        colors = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🟠", "HOLD": "🟡", "PASS": "⚫"}
        return colors.get(action, "⚪")

    df_logs["Action"] = df_logs["action_recommended"].apply(
        lambda a: f"{action_color(a)} {a}"
    )

    display = df_logs[[
        "symbol", "Action", "conviction", "composite_score",
        "technical_score", "fundamental_score", "market_regime", "timestamp"
    ]].copy()
    display.columns = [
        "Symbol", "Action", "Conviction", "Composite", "Technical",
        "Fundamental", "Regime", "Time"
    ]

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        action_filter = st.multiselect(
            "Filter by Action",
            ["BUY", "SELL", "SHORT", "HOLD", "PASS"],
            default=["BUY", "SHORT"],
        )
    with col2:
        min_score = st.slider("Min Composite Score", 0, 100, 55)
    with col3:
        min_conviction = st.slider("Min Conviction", 1, 10, 6)

    mask = display["Action"].apply(lambda x: any(a in x for a in action_filter))
    filtered = display[mask & (df_logs["composite_score"] >= min_score)]
    if "Conviction" in filtered.columns:
        filtered = filtered[filtered["Conviction"] >= min_conviction]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
else:
    st.info("No analysis logs yet. Start the trading loop to generate signals.")

# ── Stock chart for selected symbol ──────────────────────────────────────────
st.markdown("---")
st.subheader("Technical Chart")

symbol_input = st.text_input("Enter symbol to chart", value="AAPL").upper()
period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

if st.button("Load Chart") or symbol_input:
    try:
        from data.price_feed import PriceFeed
        from analysis.technical.indicators import TechnicalAnalyzer

        feed = PriceFeed()
        df = feed.get_historical(symbol_input, period=period, interval="1d")

        if not df.empty:
            analyzer = TechnicalAnalyzer()
            signals = analyzer.analyze(symbol_input, df)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=[f"{symbol_input} Price", "RSI", "MACD"],
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="Price"
            ), row=1, col=1)

            # Moving averages
            import pandas as pd
            close = df["close"]
            fig.add_trace(go.Scatter(
                x=df.index, y=close.rolling(20).mean(),
                name="SMA20", line=dict(color="blue", width=1)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=close.rolling(50).mean(),
                name="SMA50", line=dict(color="orange", width=1)
            ), row=1, col=1)

            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            fig.add_trace(go.Scatter(
                x=df.index, y=sma20 + 2 * std20,
                name="BB Upper", line=dict(color="gray", width=1, dash="dash")
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=sma20 - 2 * std20,
                name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)"
            ), row=1, col=1)

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
            fig.add_trace(go.Scatter(
                x=df.index, y=rsi, name="RSI", line=dict(color="purple")
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            hist = macd - macd_signal
            fig.add_trace(go.Bar(
                x=df.index, y=hist, name="MACD Hist",
                marker_color=["green" if v >= 0 else "red" for v in hist]
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=macd, name="MACD", line=dict(color="blue")
            ), row=3, col=1)

            fig.update_layout(
                height=700, showlegend=True,
                xaxis_rangeslider_visible=False,
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Technical signals summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI(14)", f"{signals.rsi_14:.1f}", signals.rsi_signal)
            col2.metric("Technical Score", f"{signals.technical_score:.0f}/100")
            col3.metric("MACD Crossover", signals.macd_crossover)
            col4.metric("Relative Volume", f"{signals.relative_volume:.1f}x")

        else:
            st.error(f"No data found for {symbol_input}")
    except Exception as e:
        st.error(f"Chart error: {e}")
