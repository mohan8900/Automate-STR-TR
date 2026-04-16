"""
AI Stock Trader — Streamlit Dashboard
Main entry point. Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AI Stock Trader",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply theme before anything else
from dashboard.theme import apply_theme, styled_metric, status_badge, section_header, COLORS
apply_theme()


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


def _update_config_field(field: str, value) -> bool:
    """Update a single field in config.toml without rewriting the entire file."""
    import re
    config_path = Path("config.toml")
    if not config_path.exists():
        return False
    content = config_path.read_text(encoding="utf-8")
    # Match: field_name = value (with optional comment)
    pattern = rf"^(\s*{field}\s*=\s*).*$"
    if isinstance(value, bool):
        replacement = rf"\g<1>{str(value).lower()}"
    elif isinstance(value, str):
        replacement = rf'\g<1>"{value}"'
    else:
        replacement = rf"\g<1>{value}"
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        return False
    config_path.write_text(new_content, encoding="utf-8")
    # Clear cached config so next read picks up the change
    get_config.clear()
    return True


def main():
    cfg = get_config()
    repo = get_repository()
    currency = "\u20b9" if cfg.market.exchange == "IN" else "$"

    # ── Hero header ───────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>AI Stock Trader</h1>
        <p>Autonomous Trading System &bull; ML Ensemble + LSTM + Fuzzy Logic + LLM Arbiter</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Trading Mode Control Panel ────────────────────────────────────
    st.markdown("""
    <div class="card-3d" style="padding:24px 28px;">
        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:14px;">
            Trading Controls
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_paper, col_approval, col_status = st.columns(3)

    # ── Paper / Live toggle ───────────────────────────────────────
    with col_paper:
        st.markdown("**Trading Mode**")
        paper_on = st.toggle(
            "Paper Trading (simulated, no real money)",
            value=cfg.user.paper_trading,
            key="main_paper_toggle",
        )

        if paper_on:
            st.success("PAPER MODE — Safe, no real money used")
        else:
            st.error("LIVE MODE — Real money on Angel One!")

        # Detect change
        if paper_on != cfg.user.paper_trading:
            if not paper_on:
                # Switching to LIVE — need confirmation
                st.session_state["confirm_live_pending"] = True
            else:
                # Switching to PAPER — always safe
                _update_config_field("paper_trading", True)
                st.rerun()

    # ── Live mode confirmation ────────────────────────────────────
    if st.session_state.get("confirm_live_pending"):
        st.markdown("---")
        st.markdown(
            '<div class="card-3d" style="border-left:4px solid #ef4444;padding:20px;">'
            '<div style="color:#ef4444;font-weight:700;font-size:1.1rem;">'
            'Are you sure you want to enable LIVE TRADING?</div>'
            '<div style="color:#64748b;margin-top:6px;">'
            'Real money will be used. Orders will be sent to Angel One. '
            'Make sure you have sufficient funds and have tested with paper trading first.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        btn1, btn2, _ = st.columns([1, 1, 3])
        with btn1:
            if st.button("Yes, enable LIVE trading", type="primary", key="confirm_live_yes"):
                _update_config_field("paper_trading", False)
                st.session_state["confirm_live_pending"] = False
                st.rerun()
        with btn2:
            if st.button("Cancel — stay on Paper", key="confirm_live_no"):
                _update_config_field("paper_trading", True)
                st.session_state["confirm_live_pending"] = False
                st.rerun()

    # ── Approval toggle ───────────────────────────────────────────
    with col_approval:
        st.markdown("**Trade Approval**")
        approval_on = st.toggle(
            "Require manual approval for every trade",
            value=cfg.user.approval_required,
            key="main_approval_toggle",
        )

        if approval_on:
            st.info("You will review and approve each trade before execution")
        else:
            st.warning("AI will auto-execute trades without your approval!")

        if approval_on != cfg.user.approval_required:
            _update_config_field("approval_required", approval_on)
            st.rerun()

    # ── Pending alerts ────────────────────────────────────────────
    with col_status:
        st.markdown("**Status**")
        pending = repo.get_pending_approvals()
        if pending:
            st.warning(f"{len(pending)} trade(s) awaiting your approval")
        else:
            st.markdown("No pending approvals")

        regime_str = "UNKNOWN"
        try:
            history_check = repo.get_portfolio_history(days=1)
            if history_check:
                regime_str = str(history_check[-1].get("market_regime", "UNKNOWN"))
        except Exception:
            pass
        regime_emoji = {"BULL": "BULL", "BEAR": "BEAR", "SIDEWAYS": "SIDEWAYS", "VOLATILE": "VOLATILE"}
        st.markdown(f"Market Regime: **{regime_emoji.get(regime_str, regime_str)}**")

    st.markdown("")

    # ── Quick stats ───────────────────────────────────────────────────
    try:
        open_trades = repo.get_open_trades()
        history = repo.get_portfolio_history(days=1)

        pv = history[-1]["total_value"] if history else cfg.user.investment_amount
        dpnl = history[-1]["daily_pnl_pct"] if history else 0
        heat = history[-1]["portfolio_heat"] if history else 0
        regime = history[-1]["market_regime"] if history else "UNKNOWN"
        regime_map = {"BULL": "BULL", "BEAR": "BEAR", "SIDEWAYS": "SIDEWAYS", "VOLATILE": "VOLATILE"}

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            styled_metric("Portfolio Value", f"{currency}{pv:,.0f}")
        with c2:
            styled_metric("Today's P&L", f"{dpnl:+.2%}",
                          delta=f"{dpnl:+.2%}", delta_color="green" if dpnl >= 0 else "red")
        with c3:
            styled_metric("Open Positions", str(len(open_trades)))
        with c4:
            styled_metric("Portfolio Heat", f"{(heat or 0):.1%}")
        with c5:
            regime_colors = {"BULL": "green", "BEAR": "red", "SIDEWAYS": "muted", "VOLATILE": "red"}
            styled_metric("Market Regime", str(regime),
                          delta=regime_map.get(str(regime), ""),
                          delta_color=regime_colors.get(str(regime), "muted"))

    except Exception:
        st.info("No trading data yet. Start the trading loop to begin.")

    st.markdown("")

    # ── Navigation cards ──────────────────────────────────────────────
    section_header("Dashboard", "Navigate using the sidebar or the cards below")

    nav_items = [
        ("Overview", "Equity curve, positions, and performance metrics", "#6366f1"),
        ("Signals", "Live AI signals with technical charts and scores", "#06b6d4"),
        ("Trade Approval", "Review and approve/reject AI recommendations", "#10b981"),
        ("History", "Complete trade analytics and P&L distribution", "#f59e0b"),
        ("System Health", "Circuit breakers, kill switch, safety monitors", "#ef4444"),
        ("ML Predictions", "LSTM, XGBoost, Random Forest, Fuzzy Logic", "#8b5cf6"),
    ]
    c1, c2, c3 = st.columns(3)
    for i, (title, desc, color) in enumerate(nav_items):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div class="card-3d" style="border-left: 4px solid {color}; cursor:pointer;">
                <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:4px">{title}</div>
                <div style="color:#64748b;font-size:0.82rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── System status ─────────────────────────────────────────────────
    section_header("System Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        styled_metric("Exchange", cfg.market.exchange)
    with c2:
        styled_metric("Investment", f"{currency}{cfg.user.investment_amount:,.0f}")
    with c3:
        styled_metric("Scan Interval", f"{cfg.trading.scan_interval_minutes}m")
    with c4:
        styled_metric("LLM Provider", "Gemini Free" if not cfg.anthropic.api_key.get_secret_value() else "Claude")

    with st.expander("Full Configuration"):
        st.json({
            "exchange": cfg.market.exchange,
            "paper_trading": cfg.user.paper_trading,
            "investment_amount": cfg.user.investment_amount,
            "trading_budget": cfg.user.trading_budget,
            "risk_tolerance": cfg.user.risk_tolerance,
            "approval_required": cfg.user.approval_required,
            "scan_interval_minutes": cfg.trading.scan_interval_minutes,
            "max_positions": cfg.trading.max_open_positions,
            "ml_lstm_enabled": cfg.ml.lstm_enabled,
            "ml_fuzzy_enabled": cfg.ml.fuzzy_enabled,
        })


if __name__ == "__main__":
    main()
