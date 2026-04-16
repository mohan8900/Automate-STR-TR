"""
ML Predictions — View machine learning model predictions, confidence, and performance.
Shows XGBoost + Random Forest ensemble results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ML Predictions", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("ML Prediction Dashboard")


@st.cache_resource
def get_deps():
    from config.settings import get_settings
    from data.price_feed import PriceFeed
    from services.swing_trading.prediction.market_predictor import MarketPredictor
    cfg = get_settings()
    pf = PriceFeed(exchange=cfg.market.exchange)
    mp = MarketPredictor(exchange=cfg.market.exchange)
    return cfg, pf, mp


cfg, price_feed, market_predictor = get_deps()
currency = "₹" if cfg.market.exchange == "IN" else "$"

if not cfg.ml.enabled:
    st.warning("ML is disabled in config.toml. Enable it with `ml.enabled = true`")
    st.stop()

# ── Market-Level Prediction ──────────────────────────────────────────────────
st.subheader("Market Direction Prediction (NIFTY 50)")

try:
    with st.spinner("Running market prediction..."):
        market_pred = market_predictor.predict_market()

    col1, col2, col3, col4 = st.columns(4)

    direction_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}
    direction_color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}

    with col1:
        emoji = direction_emoji.get(market_pred.direction, "❓")
        st.metric("Direction", f"{emoji} {market_pred.direction}")

    with col2:
        st.metric("Probability", f"{market_pred.probability:.1%}")

    with col3:
        st.metric("Confidence", f"{market_pred.confidence:.1%}")

    with col4:
        st.metric("Recommended Exposure", f"{market_pred.recommended_exposure:.0%}")

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=market_pred.probability * 100,
        title={"text": "Bullish Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green" if market_pred.probability > 0.55 else "red"},
            "steps": [
                {"range": [0, 35], "color": "rgba(255,0,0,0.1)"},
                {"range": [35, 45], "color": "rgba(255,165,0,0.1)"},
                {"range": [45, 55], "color": "rgba(128,128,128,0.1)"},
                {"range": [55, 65], "color": "rgba(144,238,144,0.1)"},
                {"range": [65, 100], "color": "rgba(0,128,0,0.1)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Market prediction failed: {e}")
    st.info("This is normal on first run — the model needs to train. Run `python main.py --backtest-ml` first.")

st.markdown("---")

# ── Individual Stock Predictions ─────────────────────────────────────────────
st.subheader("Stock-Level Predictions")

from config.watchlists import IN_WATCHLIST, US_WATCHLIST
watchlist = IN_WATCHLIST if cfg.market.exchange == "IN" else US_WATCHLIST

col1, col2 = st.columns([3, 1])
with col1:
    selected_stocks = st.multiselect(
        "Select stocks to predict",
        watchlist,
        default=watchlist[:5],
        max_selections=10,
    )
with col2:
    run_predictions = st.button("Run Predictions", type="primary", use_container_width=True)

if run_predictions and selected_stocks:
    predictions = []
    progress = st.progress(0, text="Running ML predictions...")

    for i, symbol in enumerate(selected_stocks):
        try:
            df = price_feed.get_historical(symbol, period="2y", interval="1d")
            if len(df) < 100:
                continue
            pred = market_predictor.predict_stock(symbol, df)
            predictions.append({
                "Symbol": symbol,
                "Direction": pred.direction,
                "Probability": pred.probability,
                "Confidence": pred.confidence,
                "Model Accuracy": pred.model_accuracy,
            })
        except Exception as e:
            predictions.append({
                "Symbol": symbol,
                "Direction": "ERROR",
                "Probability": 0,
                "Confidence": 0,
                "Model Accuracy": 0,
            })
        progress.progress((i + 1) / len(selected_stocks))

    progress.empty()

    if predictions:
        pred_df = pd.DataFrame(predictions)

        # Color-coded table
        st.dataframe(
            pred_df.style.apply(
                lambda row: [
                    "",
                    "background-color: rgba(0,200,83,0.2)" if row["Direction"] == "BULLISH"
                    else "background-color: rgba(255,23,68,0.2)" if row["Direction"] == "BEARISH"
                    else "",
                    "",
                    "",
                    "",
                ] if len(row) == 5 else [""] * len(row),
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart of probabilities
        fig = go.Figure()
        colors = [
            "#00c853" if p > 0.55 else "#ff1744" if p < 0.45 else "#ff9800"
            for p in pred_df["Probability"]
        ]
        fig.add_trace(go.Bar(
            x=pred_df["Symbol"],
            y=pred_df["Probability"],
            marker_color=colors,
            text=pred_df["Direction"],
            textposition="outside",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral (50%)")
        fig.update_layout(
            title="5-Day Bullish Probability by Stock",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        bullish = len(pred_df[pred_df["Direction"] == "BULLISH"])
        bearish = len(pred_df[pred_df["Direction"] == "BEARISH"])
        neutral = len(pred_df[pred_df["Direction"] == "NEUTRAL"])
        st.info(
            f"Summary: {bullish} Bullish, {bearish} Bearish, {neutral} Neutral "
            f"out of {len(pred_df)} stocks"
        )

st.markdown("---")

# ── Feature Importance (Last Trained Model) ──────────────────────────────────
st.subheader("Feature Importance")

st.caption(
    "Shows which technical indicators the ML model relies on most. "
    "Run a prediction above to see feature importances."
)

# Try to get feature importance from the last prediction
try:
    if run_predictions and selected_stocks and predictions:
        # Get a detailed prediction with feature importance
        sample_symbol = selected_stocks[0]
        sample_df = price_feed.get_historical(sample_symbol, period="2y", interval="1d")
        if len(sample_df) >= 100:
            detailed_pred = market_predictor.predict_stock(sample_symbol, sample_df)
            if hasattr(detailed_pred, "feature_importance") and detailed_pred.feature_importance:
                fi = detailed_pred.feature_importance
                fi_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
                ])

                fig = go.Figure(go.Bar(
                    x=fi_df["Importance"],
                    y=fi_df["Feature"],
                    orientation="h",
                    marker_color="#1f77b4",
                ))
                fig.update_layout(
                    title=f"Top 15 Features ({sample_symbol})",
                    xaxis_title="Importance",
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=150, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available — run a prediction first.")
except Exception as e:
    st.caption(f"Feature importance unavailable: {e}")

st.markdown("---")

# ── ML Model Configuration ──────────────────────────────────────────────────
st.subheader("Model Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Ensemble Models**")
    st.text("1. XGBoost (200 trees, depth 6)")
    st.text("2. Random Forest (200 trees, depth 10)")
    st.text("Prediction: average of both")

with col2:
    st.markdown("**Parameters**")
    st.text(f"Prediction horizon: {cfg.ml.target_days} days")
    st.text(f"Training window: {cfg.ml.train_window_days} days (~2 years)")
    st.text(f"Retrain interval: {cfg.ml.retrain_interval_days} days")
    st.text(f"Min confidence: {cfg.ml.min_confidence}")

with col3:
    st.markdown("**Features Used (50+)**")
    st.text("Price returns (1/2/3/5/10/20d)")
    st.text("Moving averages (SMA/EMA)")
    st.text("RSI, MACD, Stochastic")
    st.text("Bollinger Bands, ATR")
    st.text("Volume patterns, OBV")
    st.text("Fundamental ratios")
