"""
AI Stock Trader — Streamlit Dashboard
Main entry point. Run with: streamlit run dashboard/app.py

Pages:
  1. Overview      — Portfolio value, P&L, positions
  2. Signals       — Live analysis and scores
  3. Trade Approval — Human-in-the-loop approval queue
  4. History       — Trade history and analytics
  5. Settings      — User configuration
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AI Stock Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared state initialization ───────────────────────────────────────────────

@st.cache_resource
def get_db():
    from database.manager import DatabaseManager
    from config.settings import get_settings
    cfg = get_settings()
    return DatabaseManager(cfg.database.path)

@st.cache_resource
def get_repository():
    from database.repository import TradeRepository
    return TradeRepository(get_db())

@st.cache_resource
def get_config():
    from config.settings import get_settings
    return get_settings()

# ── Main landing page ─────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    repo = get_repository()

    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("AI Stock Trader")
        st.caption("Powered by Claude AI • Autonomous Trading System")
    with col2:
        mode_color = "red" if not cfg.user.paper_trading else "green"
        mode_text = "LIVE" if not cfg.user.paper_trading else "PAPER"
        st.markdown(
            f"<div style='background:{mode_color};color:white;padding:8px;border-radius:5px;"
            f"text-align:center;font-weight:bold;font-size:18px'>{mode_text} MODE</div>",
            unsafe_allow_html=True,
        )
    with col3:
        pending = repo.get_pending_approvals()
        if pending:
            st.warning(f"🔔 {len(pending)} trades awaiting approval")
        else:
            st.success("✅ No pending approvals")

    st.markdown("---")

    # Quick stats
    try:
        open_trades = repo.get_open_trades()
        history = repo.get_portfolio_history(days=1)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            pv = history[-1]["total_value"] if history else cfg.user.investment_amount
            currency = "₹" if cfg.market.exchange == "IN" else "$"
            st.metric("Portfolio Value", f"{currency}{pv:,.0f}")
        with col2:
            dpnl = history[-1]["daily_pnl_pct"] if history else 0
            st.metric("Today's P&L", f"{dpnl:+.2%}")
        with col3:
            st.metric("Open Positions", len(open_trades))
        with col4:
            heat = history[-1]["portfolio_heat"] if history else 0
            st.metric("Portfolio Heat", f"{(heat or 0):.1%}")
        with col5:
            regime = history[-1]["market_regime"] if history else "UNKNOWN"
            regime_emoji = {"BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "↔️", "VOLATILE": "⚡"}.get(str(regime), "❓")
            st.metric("Market Regime", f"{regime_emoji} {regime}")

    except Exception as e:
        st.info("No trading data yet. Start the trading loop to begin.")

    st.markdown("---")
    st.markdown("""
    ### Navigation
    Use the **sidebar** to navigate between pages:
    - **Overview** — Portfolio snapshot and open positions
    - **Signals** — Live market analysis and AI signals
    - **Trade Approval** — Review and approve/reject pending trades
    - **History** — Complete trade history and performance analytics
    - **Settings** — Configure trading parameters
    """)

    # System status
    with st.expander("System Status"):
        st.json({
            "exchange": cfg.market.exchange,
            "paper_trading": cfg.user.paper_trading,
            "investment_amount": cfg.user.investment_amount,
            "trading_budget": cfg.user.trading_budget,
            "risk_tolerance": cfg.user.risk_tolerance,
            "approval_required": cfg.user.approval_required,
            "scan_interval_minutes": cfg.trading.scan_interval_minutes,
            "max_positions": cfg.trading.max_open_positions,
            "llm_model": cfg.anthropic.model,
        })


if __name__ == "__main__":
    main()
