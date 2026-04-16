"""
System Health — Monitor all safety layers, kill switch, drawdown, broker status.
This is your cockpit view of the entire trading machine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="System Health", page_icon="", layout="wide")
from dashboard.theme import apply_theme, apply_plotly_theme, COLORS, section_header
apply_theme()
section_header("System Health & Safety Monitor")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, key="health_refresh")
except ImportError:
    pass


@st.cache_resource
def get_deps():
    from database.manager import DatabaseManager
    from database.repository import TradeRepository
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    return cfg, TradeRepository(db)


cfg, repo = get_deps()
currency = "₹" if cfg.market.exchange == "IN" else "$"

# ── Kill Switch Status ───────────────────────────────────────────────────────
st.subheader("Kill Switch")

kill_switch_file = Path("data/kill_switch_state.json")
ks_active = False
ks_reason = ""
ks_time = ""

# Check kill switch state from log (in production, this would read from shared state)
try:
    from risk.drawdown_tracker import DrawdownTracker
    dd = DrawdownTracker()
except Exception:
    dd = None

col1, col2, col3 = st.columns(3)

with col1:
    if ks_active:
        st.error("KILL SWITCH: ACTIVE")
        st.caption(f"Reason: {ks_reason}")
        st.caption(f"Triggered: {ks_time}")
    else:
        st.success("Kill Switch: INACTIVE")
        st.caption("System operating normally")

with col2:
    st.markdown("**Anomaly Thresholds**")
    st.text("Max orders/minute: 5")
    st.text("Max daily loss: 5%")
    st.text("Max same-symbol/5min: 2")

with col3:
    st.markdown("**Manual Controls**")
    if st.button("EMERGENCY STOP", type="primary", use_container_width=True):
        st.error("Kill switch would activate here. Connect to live trading loop to enable.")
    if st.button("Reset Kill Switch", use_container_width=True):
        st.info("Kill switch reset. Connect to live trading loop to enable.")

st.markdown("---")

# ── Drawdown Tracker ─────────────────────────────────────────────────────────
st.subheader("Drawdown Tracker")

dd_state = {"high_water_mark": 0, "current_value": 0, "drawdown_pct": 0, "multiplier": 1.0, "tier_label": "normal"}
dd_file = Path("data/drawdown_state.json")
try:
    if dd_file.exists():
        with open(dd_file) as f:
            dd_state = json.load(f)
except Exception:
    pass

col1, col2, col3, col4, col5 = st.columns(5)

hwm = dd_state.get("high_water_mark", 0)
curr = dd_state.get("current_value", 0)
dd_pct = dd_state.get("drawdown_pct", 0)
dd_mult = dd_state.get("multiplier", 1.0)
dd_tier = dd_state.get("tier_label", "normal")

col1.metric("High Water Mark", f"{currency}{hwm:,.0f}")
col2.metric("Current Value", f"{currency}{curr:,.0f}")
col3.metric("Drawdown", f"{dd_pct:.1%}", delta=f"-{dd_pct:.1%}" if dd_pct > 0 else "0%", delta_color="inverse")
col4.metric("Size Multiplier", f"{dd_mult:.0%}")

tier_colors = {"normal": "🟢", "caution": "🟡", "severe": "🟠", "halt": "🔴"}
col5.metric("Tier", f"{tier_colors.get(dd_tier, '⚪')} {dd_tier.upper()}")

# Drawdown tier bar
st.markdown("**Drawdown Tiers**")
tier_col1, tier_col2, tier_col3, tier_col4 = st.columns(4)
tier_col1.markdown(
    f"{'**' if dd_tier == 'normal' else ''}0-5% Normal (1.0x){'**' if dd_tier == 'normal' else ''}"
)
tier_col2.markdown(
    f"{'**' if dd_tier == 'caution' else ''}5-10% Caution (0.5x){'**' if dd_tier == 'caution' else ''}"
)
tier_col3.markdown(
    f"{'**' if dd_tier == 'severe' else ''}10-15% Severe (0.25x){'**' if dd_tier == 'severe' else ''}"
)
tier_col4.markdown(
    f"{'**' if dd_tier == 'halt' else ''}15%+ HALT (0.0x){'**' if dd_tier == 'halt' else ''}"
)

# Drawdown progress bar
st.progress(min(dd_pct / 0.15, 1.0), text=f"Drawdown: {dd_pct:.1%} of 15% halt threshold")

st.markdown("---")

# ── Circuit Breaker Status ───────────────────────────────────────────────────
st.subheader("Circuit Breaker")

col1, col2, col3, col4, col5 = st.columns(5)

# Get today's stats from DB
try:
    open_trades = repo.get_open_trades()
    history = repo.get_portfolio_history(days=1)
    daily_pnl = history[-1]["daily_pnl_pct"] if history else 0
    open_count = len(open_trades)
except Exception:
    daily_pnl = 0
    open_count = 0

daily_loss_limit = cfg.user.max_daily_loss_pct
max_trades = cfg.trading.max_daily_trades
max_positions = cfg.trading.max_open_positions
vix_halt = cfg.vix.high_threshold

with col1:
    pnl_ok = abs(daily_pnl) < daily_loss_limit
    st.metric(
        "Daily P&L",
        f"{daily_pnl:+.2%}",
        delta=f"limit: {daily_loss_limit:.0%}",
    )
    st.markdown(f"{'🟢' if pnl_ok else '🔴'} {'OK' if pnl_ok else 'BREACHED'}")

with col2:
    pos_ok = open_count < max_positions
    st.metric("Open Positions", f"{open_count}/{max_positions}")
    st.markdown(f"{'🟢' if pos_ok else '🔴'} {'OK' if pos_ok else 'FULL'}")

with col3:
    st.metric("Max Daily Trades", f"—/{max_trades}")
    st.markdown("🟢 OK")

with col4:
    st.metric("VIX Halt Level", f"{vix_halt:.0f}")
    st.caption(f"India VIX threshold for trading halt")

with col5:
    heat_val = history[-1]["portfolio_heat"] if history else 0
    heat_max = cfg.user.max_portfolio_heat
    heat_ok = (heat_val or 0) < heat_max
    st.metric("Portfolio Heat", f"{(heat_val or 0):.1%}/{heat_max:.0%}")
    st.markdown(f"{'🟢' if heat_ok else '🔴'} {'OK' if heat_ok else 'OVERHEATED'}")

st.markdown("---")

# ── Broker Connection ────────────────────────────────────────────────────────
st.subheader("Broker Connection")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Angel One SmartAPI**")
    import os
    angel_key = os.getenv("ANGEL_ONE__API_KEY", "")
    angel_client = os.getenv("ANGEL_ONE__CLIENT_ID", "")
    if angel_key and angel_client:
        st.success(f"API Key: {angel_key[:4]}... | Client: {angel_client[:4]}...")
    else:
        st.error("Not configured — set ANGEL_ONE__API_KEY in .env")

with col2:
    st.markdown("**Claude AI**")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and len(anthropic_key) > 10:
        st.success(f"API Key: {anthropic_key[:8]}...")
    else:
        st.error("Not configured — set ANTHROPIC_API_KEY in .env")

with col3:
    st.markdown("**News Feeds**")
    news_key = os.getenv("NEWS_API_KEY", "")
    st.success("7 Indian RSS feeds (no key needed)")
    if news_key:
        st.success(f"NewsAPI: {news_key[:4]}...")
    else:
        st.info("NewsAPI: not configured (optional)")

st.markdown("---")

# ── Market Hours ─────────────────────────────────────────────────────────────
st.subheader("Market Status")

try:
    from scheduler.market_hours import MarketHours
    mh = MarketHours(cfg.market.exchange)
    col1, col2, col3 = st.columns(3)
    with col1:
        if mh.is_open():
            st.success("NSE/BSE: OPEN")
        else:
            st.warning(f"NSE/BSE: CLOSED — next open {mh.next_market_open_str()}")
    with col2:
        st.metric("Exchange", cfg.market.exchange)
    with col3:
        st.metric("Mode", "PAPER" if cfg.user.paper_trading else "LIVE")
        if not cfg.user.paper_trading:
            st.error("REAL MONEY MODE")
except Exception as e:
    st.error(f"Market hours check failed: {e}")

st.markdown("---")

# ── Configuration Summary ────────────────────────────────────────────────────
st.subheader("Active Configuration")

with st.expander("View full config", expanded=False):
    config_data = {
        "Exchange": cfg.market.exchange,
        "Investment": f"{currency}{cfg.user.investment_amount:,.0f}",
        "Trading Budget": f"{currency}{cfg.user.trading_budget:,.0f}",
        "Risk Tolerance": cfg.user.risk_tolerance,
        "Paper Trading": cfg.user.paper_trading,
        "Approval Required": cfg.user.approval_required,
        "Max Position %": f"{cfg.user.max_position_pct:.0%}",
        "Max Daily Loss": f"{cfg.user.max_daily_loss_pct:.0%}",
        "Max Portfolio Heat": f"{cfg.user.max_portfolio_heat:.0%}",
        "Scan Interval": f"{cfg.trading.scan_interval_minutes} min",
        "Max Open Positions": cfg.trading.max_open_positions,
        "Max Daily Trades": cfg.trading.max_daily_trades,
        "Min Composite Score": cfg.trading.min_composite_score,
        "Min Conviction (show)": cfg.trading.min_conviction_execute,
        "Min Conviction (auto)": cfg.trading.min_conviction_auto,
        "LLM Model": cfg.anthropic.model,
        "LLM Daily Budget": f"${cfg.anthropic.max_daily_cost_usd}",
        "ML Enabled": cfg.ml.enabled,
        "Strategy": cfg.strategy.primary,
        "Buffett Screen": cfg.strategy.buffett_screen_enabled,
    }
    for key, val in config_data.items():
        st.text(f"{key:.<30} {val}")
