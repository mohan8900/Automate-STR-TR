"""
Trade Approval — Human-in-the-loop review queue.
This is the most critical page: approve, modify, or reject pending trades.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Trade Approval", page_icon="✅", layout="wide")
st.title("Trade Approval Queue")

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

# Auto-refresh every 15 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15_000, key="approval_refresh")
except ImportError:
    pass

pending = repo.get_pending_approvals()

if not pending:
    st.success("✅ No trades pending approval.")
    st.info("Pending trades will appear here when the trading loop generates recommendations requiring human review.")
else:
    st.warning(f"🔔 {len(pending)} trade(s) awaiting your decision")

    for item in pending:
        symbol = item["symbol"]
        action = item["action"]
        conviction = item.get("conviction", 0)
        thesis = item.get("primary_thesis", "")
        risks_json = item.get("key_risks", "[]")
        try:
            risks = json.loads(risks_json) if risks_json else []
        except Exception:
            risks = []

        tp_prices_json = item.get("take_profit_prices", "[]")
        try:
            tp_prices = json.loads(tp_prices_json) if tp_prices_json else []
        except Exception:
            tp_prices = []

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🟠", "COVER": "🔵"}.get(action, "⚪")
        created_str = item.get("created_at", "")

        with st.expander(
            f"{action_emoji} {action} {symbol} | Conviction: {conviction}/10 | "
            f"{currency}{item.get('position_value', 0):,.0f} | Created: {created_str}",
            expanded=True,
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Entry Price", f"{currency}{item.get('entry_price', 0):.2f}")
                st.metric("Shares", f"{item.get('shares', 0):.0f}")
                st.metric("Position Value", f"{currency}{item.get('position_value', 0):,.0f}")

            with col2:
                stop = item.get("stop_loss_price", 0)
                entry = item.get("entry_price", 1)
                stop_pct = (entry - stop) / entry if entry > 0 else 0
                st.metric("Stop Loss", f"{currency}{stop:.2f}", f"-{stop_pct:.1%}")
                if tp_prices:
                    st.metric("Take Profit 1", f"{currency}{tp_prices[0]:.2f}" if tp_prices else "N/A")
                    if len(tp_prices) > 1:
                        st.metric("Take Profit 2", f"{currency}{tp_prices[1]:.2f}")

            with col3:
                rr = (tp_prices[0] - entry) / (entry - stop) if (tp_prices and stop < entry) else 0
                st.metric("Risk/Reward", f"1:{rr:.1f}" if rr > 0 else "N/A")
                st.metric("Heat Added", f"{item.get('portfolio_heat_add', 0):.2%}")
                st.metric("Composite Score", f"{item.get('composite_score', 0):.0f}/100")

            with col4:
                st.metric("Technical Score", f"{item.get('technical_score', 0) or 0:.0f}/100")
                st.metric("Fundamental Score", f"{item.get('fundamental_score', 0) or 0:.0f}/100")
                st.metric("AI Risk Score", f"{item.get('llm_risk_score', 5)}/10")

            # Thesis and risks
            st.markdown(f"**AI Thesis:** {thesis}")
            if risks:
                st.markdown(f"**Key Risks:** {' • '.join(risks)}")

            st.markdown("---")

            # Decision buttons
            btn_col1, btn_col2, btn_col3, spacer = st.columns([1, 1, 1, 3])

            with btn_col1:
                if st.button(
                    f"✅ Approve",
                    key=f"approve_{item['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    repo.approve_trade(item["id"], "Approved via dashboard")
                    st.success(f"✅ {symbol} trade approved! Will execute next cycle.")
                    st.rerun()

            with btn_col2:
                if st.button(
                    f"❌ Reject",
                    key=f"reject_{item['id']}",
                    use_container_width=True,
                ):
                    reason = st.session_state.get(f"reject_reason_{item['id']}", "Rejected by user")
                    repo.reject_trade(item["id"], reason)
                    st.error(f"❌ {symbol} trade rejected.")
                    st.rerun()

            with btn_col3:
                if st.button(
                    f"🔍 Chart",
                    key=f"chart_{item['id']}",
                    use_container_width=True,
                ):
                    st.session_state[f"show_chart_{symbol}"] = True

            # Optional rejection reason
            st.text_input(
                "Rejection reason (optional)",
                key=f"reject_reason_{item['id']}",
                placeholder="Why are you rejecting this trade?",
            )

            # Show quick chart if requested
            if st.session_state.get(f"show_chart_{symbol}"):
                try:
                    from data.price_feed import PriceFeed
                    import plotly.graph_objects as go
                    feed = PriceFeed()
                    df = feed.get_historical(symbol, period="3mo", interval="1d")
                    if not df.empty:
                        close = df["close"]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, y=close, name=symbol,
                            line=dict(color="#00b4d8")
                        ))
                        fig.add_trace(go.Scatter(
                            x=df.index, y=close.rolling(20).mean(),
                            name="SMA20", line=dict(color="orange", dash="dash")
                        ))
                        # Mark entry and stop
                        fig.add_hline(
                            y=item.get("entry_price"), line_color="green",
                            annotation_text="Entry"
                        )
                        fig.add_hline(
                            y=item.get("stop_loss_price"), line_color="red",
                            annotation_text="Stop"
                        )
                        if tp_prices:
                            fig.add_hline(
                                y=tp_prices[0], line_color="blue",
                                annotation_text="Target"
                            )
                        fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Chart unavailable: {e}")

# ── Recently decided trades ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Recently Decided")
try:
    from database.manager import DatabaseManager
    from config.settings import get_settings
    cfg = get_settings()
    db = DatabaseManager(cfg.database.path)
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM approval_queue
            WHERE status != 'PENDING'
            ORDER BY decided_at DESC LIMIT 20
        """).fetchall()
        decided = [dict(r) for r in rows]

    if decided:
        import pandas as pd
        df_dec = pd.DataFrame(decided)
        display_cols = ["symbol", "action", "status", "conviction", "position_value", "decided_at", "decision_note"]
        available = [c for c in display_cols if c in df_dec.columns]
        st.dataframe(df_dec[available], use_container_width=True, hide_index=True)
    else:
        st.info("No recent decisions.")
except Exception as e:
    st.caption(f"Could not load history: {e}")
