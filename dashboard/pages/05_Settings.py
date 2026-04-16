"""
Settings page — configure trading parameters.
Changes are saved to config.toml.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Settings", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("Trading System Settings")

@st.cache_resource
def get_config():
    from config.settings import get_settings
    return get_settings()

cfg = get_config()

st.warning(
    "Changes are applied on next trading cycle restart. "
    "Settings are saved to config.toml."
)

# ── User configuration ────────────────────────────────────────────────────────
st.subheader("Capital & Risk")
col1, col2 = st.columns(2)

with col1:
    currency = "₹" if cfg.market.exchange == "IN" else "$"
    investment = st.number_input(
        f"Total Investment Amount ({currency})",
        min_value=1000.0, max_value=10_000_000.0,
        value=float(cfg.user.investment_amount), step=1000.0,
    )
    trading_budget = st.number_input(
        f"Per-Cycle Trading Budget ({currency})",
        min_value=500.0, max_value=float(investment),
        value=float(cfg.user.trading_budget), step=500.0,
    )
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=["conservative", "moderate", "aggressive"].index(cfg.user.risk_tolerance),
    )

with col2:
    max_position_pct = st.slider(
        "Max Position Size (% of portfolio)",
        min_value=1, max_value=15,
        value=int(cfg.user.max_position_pct * 100),
    ) / 100

    max_daily_loss = st.slider(
        "Circuit Breaker: Max Daily Loss (%)",
        min_value=1, max_value=10,
        value=int(cfg.user.max_daily_loss_pct * 100),
    ) / 100

    max_heat = st.slider(
        "Max Portfolio Heat (%)",
        min_value=2, max_value=20,
        value=int(cfg.user.max_portfolio_heat * 100),
    ) / 100

# ── Trading mode ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Trading Mode")
col1, col2 = st.columns(2)

with col1:
    paper_trading = st.toggle(
        "Paper Trading Mode (Recommended to start)",
        value=cfg.user.paper_trading,
    )
    if not paper_trading:
        st.error(
            "⚠️ LIVE MODE: Real money will be used! "
            "Only disable paper trading when you are confident in the system."
        )
        confirm_live = st.checkbox("I understand this uses real money")
        if not confirm_live:
            paper_trading = True
            st.info("Reverting to paper trading mode.")

with col2:
    approval_required = st.toggle(
        "Require Human Approval Before Trades",
        value=cfg.user.approval_required,
    )
    if not approval_required:
        st.warning(
            "⚠️ Auto-execution enabled. Trades will execute without your review. "
            "Only disable if you trust the system."
        )

# ── Execution parameters ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Execution Parameters")
col1, col2 = st.columns(2)

with col1:
    scan_interval = st.number_input(
        "Scan Interval (minutes)",
        min_value=5, max_value=120,
        value=int(cfg.trading.scan_interval_minutes),
    )
    watchlist_size = st.number_input(
        "Watchlist Size (stocks per scan)",
        min_value=10, max_value=200,
        value=int(cfg.trading.watchlist_size),
    )
    max_positions = st.number_input(
        "Max Open Positions",
        min_value=1, max_value=50,
        value=int(cfg.trading.max_open_positions),
    )

with col2:
    min_conviction = st.slider(
        "Min Conviction to Show in Queue",
        min_value=1, max_value=10,
        value=int(cfg.trading.min_conviction_execute),
    )
    auto_conviction = st.slider(
        "Min Conviction for Auto-Execution",
        min_value=min_conviction, max_value=10,
        value=max(min_conviction, int(cfg.trading.min_conviction_auto)),
    )
    min_composite = st.slider(
        "Min Composite Score for LLM Analysis",
        min_value=30, max_value=80,
        value=int(cfg.trading.min_composite_score),
    )

# ── Market selection ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Market")
exchange = st.radio(
    "Exchange",
    ["US (NYSE/NASDAQ via Alpaca)", "IN (NSE/BSE via Angel One / Zerodha)"],
    index=0 if cfg.market.exchange == "US" else 1,
)
exchange_code = "US" if "US" in exchange else "IN"

# ── LLM configuration ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("AI Model Configuration")
col1, col2 = st.columns(2)
with col1:
    llm_model = st.selectbox(
        "Claude Model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        index=0,
    )
    max_daily_cost = st.number_input(
        "Max Daily LLM Cost ($)",
        min_value=1.0, max_value=100.0,
        value=float(cfg.anthropic.max_daily_cost_usd),
    )

# ── Save button ───────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("💾 Save Settings", type="primary"):
    try:
        config_content = f"""# AI Stock Trader — Master Configuration
# Auto-generated by Settings page

[user]
investment_amount     = {investment}
trading_budget        = {trading_budget}
risk_tolerance        = "{risk_tolerance}"
max_position_pct      = {max_position_pct:.3f}
max_daily_loss_pct    = {max_daily_loss:.3f}
max_portfolio_heat    = {max_heat:.3f}
approval_required     = {str(approval_required).lower()}
paper_trading         = {str(paper_trading).lower()}

[market]
exchange              = "{exchange_code}"

[anthropic]
model                 = "{llm_model}"
max_tokens            = 4096
temperature           = 0.1
max_daily_cost_usd    = {max_daily_cost}

[trading]
scan_interval_minutes     = {scan_interval}
watchlist_size            = {watchlist_size}
max_open_positions        = {max_positions}
max_daily_trades          = 10
min_composite_score       = {min_composite}
min_conviction_execute    = {min_conviction}
min_conviction_auto       = {auto_conviction}
atr_stop_multiplier       = 2.0
take_profit_atr_multiples = [2.0, 4.0, 6.0]
min_volume_usd            = 1000000
min_price                 = 10.0
max_price                 = 50000.0
earnings_blackout_days    = 3

[risk]
sizing_method          = "atr"
risk_per_trade_pct     = 0.01
kelly_fraction         = 0.25

[vix]
low_threshold          = 15.0
elevated_threshold     = 20.0
high_threshold         = 30.0
low_size_multiplier    = 1.0
elevated_size_multiplier = 0.7
high_size_multiplier   = 0.4
extreme_halt           = true

[database]
path                   = "data/trading.db"
backup_daily           = true

[notifications]
email_enabled          = false
sms_enabled            = false
slack_enabled          = false
alert_on_trade         = true
alert_on_circuit_break = true
daily_summary_time     = "15:45"
"""
        with open("config.toml", "w") as f:
            f.write(config_content)

        # Clear settings cache so new values are picked up
        get_config.clear()
        st.success("✅ Settings saved to config.toml. Restart the trading loop to apply.")
    except Exception as e:
        st.error(f"Save failed: {e}")

# ── API Key status ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("API Key Status")
import os

keys = {
    "ANTHROPIC_API_KEY": "Claude AI",
    "ANGEL_ONE__API_KEY": "Angel One SmartAPI (India — FREE)",
    "KITE_API_KEY": "Zerodha Kite (India — paid)",
    "ALPACA_API_KEY": "Alpaca Broker (US)",
    "NEWS_API_KEY": "NewsAPI",
}

for env_key, name in keys.items():
    val = os.getenv(env_key, "")
    if val and len(val) > 4:
        st.success(f"✅ {name}: configured ({val[:4]}...)")
    else:
        st.error(f"❌ {name}: NOT configured (set {env_key} in .env)")
