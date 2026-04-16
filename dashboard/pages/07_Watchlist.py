"""
Live Watchlist — Real-time view of all tracked stocks with scores and signals.
Shows what the scanner sees before it goes to the LLM.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Watchlist", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("Live Watchlist Monitor")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="watchlist_refresh")
except ImportError:
    pass


@st.cache_resource
def get_deps():
    from config.settings import get_settings
    from data.price_feed import PriceFeed
    from analysis.technical.indicators import TechnicalAnalyzer
    cfg = get_settings()
    pf = PriceFeed(exchange=cfg.market.exchange)
    ta = TechnicalAnalyzer()
    return cfg, pf, ta


cfg, price_feed, tech_analyzer = get_deps()
currency = "₹" if cfg.market.exchange == "IN" else "$"

# ── Watchlist Selection ──────────────────────────────────────────────────────
from config.watchlists import IN_WATCHLIST, US_WATCHLIST

watchlist = IN_WATCHLIST if cfg.market.exchange == "IN" else US_WATCHLIST

col1, col2 = st.columns([3, 1])
with col1:
    selected_view = st.radio(
        "View",
        ["Full Watchlist", "Top Movers", "Oversold (RSI < 30)", "Overbought (RSI > 70)"],
        horizontal=True,
    )
with col2:
    max_stocks = st.slider("Max stocks", 10, 50, 20)

# ── Scan Watchlist ───────────────────────────────────────────────────────────
st.markdown("---")

@st.cache_data(ttl=300)
def scan_watchlist(symbols, max_count):
    """Fetch price data and compute technical signals for all symbols."""
    results = []
    progress = st.progress(0, text="Scanning watchlist...")

    for i, symbol in enumerate(symbols[:max_count]):
        try:
            df = price_feed.get_historical(symbol, period="6mo", interval="1d")
            if df.empty or len(df) < 50:
                continue

            signals = tech_analyzer.analyze(symbol, df)
            price = signals.current_price
            prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else price
            change_pct = (price - prev_close) / prev_close if prev_close > 0 else 0

            results.append({
                "Symbol": symbol,
                "Price": price,
                "Change %": change_pct,
                "RSI": signals.rsi_14,
                "RSI Signal": signals.rsi_signal,
                "MACD Cross": signals.macd_crossover,
                "Tech Score": signals.technical_score,
                "Rel Volume": signals.relative_volume,
                "BB Position": signals.bb_position,
                "SMA50": signals.sma_50,
                "SMA200": signals.sma_200,
                "ATR %": signals.atr_pct * 100,
                "Golden Cross": signals.golden_cross,
                "OBV Trend": signals.obv_trend,
            })
        except Exception as e:
            pass  # Skip failed symbols silently
        progress.progress((i + 1) / min(len(symbols), max_count))

    progress.empty()
    return results


with st.spinner("Loading watchlist data..."):
    data = scan_watchlist(watchlist, max_stocks)

if not data:
    st.warning("No data available. Check your internet connection and try again.")
    st.stop()

df = pd.DataFrame(data)

# ── Apply Filters ────────────────────────────────────────────────────────────
if selected_view == "Top Movers":
    df = df.sort_values("Change %", key=abs, ascending=False).head(15)
elif selected_view == "Oversold (RSI < 30)":
    df = df[df["RSI"] < 30].sort_values("RSI")
elif selected_view == "Overbought (RSI > 70)":
    df = df[df["RSI"] > 70].sort_values("RSI", ascending=False)
else:
    df = df.sort_values("Tech Score", ascending=False)

# ── Summary Metrics ──────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

gainers = len(df[df["Change %"] > 0])
losers = len(df[df["Change %"] < 0])
avg_rsi = df["RSI"].mean()
bullish_macd = len(df[df["MACD Cross"] == "bullish"])
golden_crosses = len(df[df["Golden Cross"]])

col1.metric("Stocks Scanned", len(df))
col2.metric("Gainers / Losers", f"{gainers} / {losers}")
col3.metric("Avg RSI", f"{avg_rsi:.0f}")
col4.metric("Bullish MACD", bullish_macd)
col5.metric("Golden Crosses", golden_crosses)

st.markdown("---")

# ── Main Table ───────────────────────────────────────────────────────────────
st.subheader(f"{selected_view} ({len(df)} stocks)")


def color_change(val):
    if val > 0:
        return "color: #00c853"
    elif val < 0:
        return "color: #ff1744"
    return ""


def color_rsi(val):
    if val < 30:
        return "color: #00c853; font-weight: bold"
    elif val > 70:
        return "color: #ff1744; font-weight: bold"
    return ""


def color_score(val):
    if val >= 65:
        return "background-color: rgba(0, 200, 83, 0.2)"
    elif val <= 35:
        return "background-color: rgba(255, 23, 68, 0.2)"
    return ""


display_df = df[["Symbol", "Price", "Change %", "RSI", "RSI Signal", "MACD Cross",
                  "Tech Score", "Rel Volume", "BB Position", "ATR %", "Golden Cross", "OBV Trend"]].copy()

display_df["Price"] = display_df["Price"].apply(lambda x: f"{currency}{x:,.2f}")
display_df["Change %"] = display_df["Change %"].apply(lambda x: f"{x:+.2%}")
display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.0f}")
display_df["Tech Score"] = display_df["Tech Score"].apply(lambda x: f"{x:.0f}")
display_df["Rel Volume"] = display_df["Rel Volume"].apply(lambda x: f"{x:.1f}x")
display_df["BB Position"] = display_df["BB Position"].apply(lambda x: f"{x:.2f}")
display_df["ATR %"] = display_df["ATR %"].apply(lambda x: f"{x:.1f}%")
display_df["Golden Cross"] = display_df["Golden Cross"].apply(lambda x: "Yes" if x else "—")

st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

st.markdown("---")

# ── Quick Chart for Selected Stock ───────────────────────────────────────────
st.subheader("Quick Chart")

symbols_in_view = df["Symbol"].tolist()
if symbols_in_view:
    selected_symbol = st.selectbox("Select stock", symbols_in_view)

    if selected_symbol:
        try:
            chart_df = price_feed.get_historical(selected_symbol, period="3mo", interval="1d")

            if not chart_df.empty:
                sma20 = chart_df["close"].rolling(20).mean()
                sma50 = chart_df["close"].rolling(50).mean()

                fig = go.Figure()

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["open"],
                    high=chart_df["high"],
                    low=chart_df["low"],
                    close=chart_df["close"],
                    name="OHLC",
                ))

                # SMAs
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=sma20,
                    line=dict(color="orange", width=1),
                    name="SMA20",
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=sma50,
                    line=dict(color="blue", width=1),
                    name="SMA50",
                ))

                # Volume bars at bottom
                colors = ["green" if c >= o else "red"
                          for c, o in zip(chart_df["close"], chart_df["open"])]

                fig.update_layout(
                    title=f"{selected_symbol} — 3 Month Chart",
                    yaxis_title=f"Price ({currency})",
                    height=450,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Stock info row
                stock_data = df[df["Symbol"] == selected_symbol].iloc[0]
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("RSI", stock_data["RSI"])
                c2.metric("MACD", stock_data["MACD Cross"])
                c3.metric("Tech Score", f"{stock_data['Tech Score']:.0f}/100")
                c4.metric("Rel Volume", f"{stock_data['Rel Volume']:.1f}x")
                c5.metric("BB Position", f"{stock_data['BB Position']:.2f}")
                c6.metric("OBV Trend", stock_data["OBV Trend"])

        except Exception as e:
            st.error(f"Chart failed: {e}")

# ── Signal Legend ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Signal Guide"):
    st.markdown("""
    | Indicator | Bullish | Bearish |
    |-----------|---------|---------|
    | **RSI** | < 30 (oversold bounce) | > 70 (overbought) |
    | **MACD Cross** | bullish crossover | bearish crossover |
    | **Tech Score** | > 65 (strong buy zone) | < 35 (weak) |
    | **Rel Volume** | > 1.5x (unusual interest) | < 0.5x (low interest) |
    | **BB Position** | < 0.2 (near lower band) | > 0.8 (near upper band) |
    | **Golden Cross** | SMA50 > SMA200 | SMA50 < SMA200 (Death Cross) |
    | **OBV Trend** | up (accumulation) | down (distribution) |
    """)
