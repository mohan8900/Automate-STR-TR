"""
AI Stock Trader — Web Dashboard
FastAPI backend serving a modern dark-theme trading UI.

Run: python ui/server.py
Open: http://localhost:8000
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path and set working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.settings import get_settings
from config.watchlists import IN_WATCHLIST, US_WATCHLIST, VIX_SYMBOL, BENCHMARK

app = FastAPI(title="AI Stock Trader")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

cfg = get_settings()
CURRENCY = "₹" if cfg.market.exchange == "IN" else "$"


# ── Lazy-loaded singletons ───────────────────────────────────────────────────

_db = None
_repo = None
_perf = None
_price_feed = None
_tech_analyzer = None


def get_db():
    global _db
    if _db is None:
        from database.manager import DatabaseManager
        _db = DatabaseManager(cfg.database.path)
    return _db


def get_repo():
    global _repo
    if _repo is None:
        from database.repository import TradeRepository
        _repo = TradeRepository(get_db())
    return _repo


def get_perf():
    global _perf
    if _perf is None:
        from database.performance_tracker import PerformanceTracker
        _perf = PerformanceTracker(get_db())
    return _perf


def get_price_feed():
    global _price_feed
    if _price_feed is None:
        from data.price_feed import PriceFeed
        _price_feed = PriceFeed(exchange=cfg.market.exchange)
    return _price_feed


def get_tech_analyzer():
    global _tech_analyzer
    if _tech_analyzer is None:
        from analysis.technical.indicators import TechnicalAnalyzer
        _tech_analyzer = TechnicalAnalyzer()
    return _tech_analyzer


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", context={
        "currency": CURRENCY,
        "exchange": cfg.market.exchange,
        "paper_trading": cfg.user.paper_trading,
    })


@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(request: Request):
    return templates.TemplateResponse(request, "watchlist.html", context={
        "currency": CURRENCY,
        "exchange": cfg.market.exchange,
    })


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    return templates.TemplateResponse(request, "trades.html", context={
        "currency": CURRENCY,
    })


@app.get("/health", response_class=HTMLResponse)
async def health_page(request: Request):
    return templates.TemplateResponse(request, "health.html", context={
        "currency": CURRENCY,
        "exchange": cfg.market.exchange,
        "paper_trading": cfg.user.paper_trading,
    })


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
async def api_portfolio():
    """Portfolio summary — value, P&L, positions."""
    repo = get_repo()
    try:
        history = repo.get_portfolio_history(days=1)
        open_trades = repo.get_open_trades()
        snapshot = history[-1] if history else {}
        return {
            "total_value": snapshot.get("total_value", cfg.user.investment_amount),
            "cash": snapshot.get("cash", cfg.user.investment_amount),
            "daily_pnl": snapshot.get("daily_pnl", 0),
            "daily_pnl_pct": snapshot.get("daily_pnl_pct", 0),
            "open_positions": len(open_trades),
            "portfolio_heat": snapshot.get("portfolio_heat", 0),
            "market_regime": snapshot.get("market_regime", "—"),
            "vix_level": snapshot.get("vix_level", 0),
            "currency": CURRENCY,
        }
    except Exception as e:
        return {
            "total_value": cfg.user.investment_amount,
            "cash": cfg.user.investment_amount,
            "daily_pnl": 0, "daily_pnl_pct": 0,
            "open_positions": 0, "portfolio_heat": 0,
            "market_regime": "—", "vix_level": 0,
            "currency": CURRENCY,
        }


@app.get("/api/positions")
async def api_positions():
    """Open positions list."""
    try:
        trades = get_repo().get_open_trades()
        return [
            {
                "symbol": t.get("symbol", ""),
                "action": t.get("action", ""),
                "shares": t.get("shares", 0),
                "entry_price": t.get("entry_price", 0),
                "stop_loss": t.get("stop_loss_price", 0),
                "conviction": t.get("conviction", 0),
                "thesis": t.get("primary_thesis", ""),
                "opened_at": str(t.get("opened_at", "")),
            }
            for t in trades
        ]
    except Exception:
        return []


@app.get("/api/performance")
async def api_performance():
    """Performance metrics."""
    try:
        m = get_perf().calculate(days_back=30)
        return {
            "total_return": m.total_return_pct,
            "sharpe": m.sharpe_ratio,
            "win_rate": m.win_rate,
            "max_drawdown": m.max_drawdown_pct,
            "profit_factor": m.profit_factor,
            "total_trades": m.total_trades,
            "sortino": m.sortino_ratio,
            "avg_holding_days": m.avg_holding_days,
        }
    except Exception:
        return {
            "total_return": 0, "sharpe": 0, "win_rate": 0,
            "max_drawdown": 0, "profit_factor": 0, "total_trades": 0,
            "sortino": 0, "avg_holding_days": 0,
        }


@app.get("/api/history")
async def api_history():
    """Portfolio history for charts."""
    try:
        history = get_repo().get_portfolio_history(days=90)
        return [
            {
                "date": str(h.get("snapshot_date", "")),
                "value": h.get("total_value", 0),
                "pnl": h.get("daily_pnl", 0),
            }
            for h in history
        ]
    except Exception:
        return []


@app.get("/api/pending")
async def api_pending():
    """Pending trade approvals."""
    try:
        pending = get_repo().get_pending_approvals()
        return [
            {
                "id": p.get("id"),
                "symbol": p.get("symbol", ""),
                "action": p.get("action", ""),
                "conviction": p.get("conviction", 0),
                "entry_price": p.get("entry_price", 0),
                "stop_loss": p.get("stop_loss_price", 0),
                "position_value": p.get("position_value", 0),
                "thesis": p.get("primary_thesis", ""),
                "risks": p.get("key_risks", "[]"),
                "composite_score": p.get("composite_score", 0),
                "created_at": str(p.get("created_at", "")),
            }
            for p in pending
        ]
    except Exception:
        return []


@app.post("/api/approve/{trade_id}")
async def api_approve(trade_id: int):
    """Approve a pending trade."""
    try:
        get_repo().approve_trade(trade_id, "Approved via web UI")
        return {"status": "ok", "message": f"Trade {trade_id} approved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/reject/{trade_id}")
async def api_reject(trade_id: int):
    """Reject a pending trade."""
    try:
        get_repo().reject_trade(trade_id, "Rejected via web UI")
        return {"status": "ok", "message": f"Trade {trade_id} rejected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/watchlist")
async def api_watchlist():
    """Scan watchlist with technical signals."""
    pf = get_price_feed()
    ta = get_tech_analyzer()
    watchlist = IN_WATCHLIST if cfg.market.exchange == "IN" else US_WATCHLIST
    results = []

    for symbol in watchlist[:30]:
        try:
            df = pf.get_historical(symbol, period="6mo", interval="1d")
            if df.empty or len(df) < 50:
                continue
            sig = ta.analyze(symbol, df)
            prev = float(df["close"].iloc[-2]) if len(df) > 1 else sig.current_price
            change = (sig.current_price - prev) / prev if prev > 0 else 0
            results.append({
                "symbol": symbol,
                "price": round(sig.current_price, 2),
                "change_pct": round(change * 100, 2),
                "rsi": sig.rsi_14,
                "rsi_signal": sig.rsi_signal,
                "macd_cross": sig.macd_crossover,
                "tech_score": sig.technical_score,
                "rel_volume": sig.relative_volume,
                "bb_position": round(sig.bb_position, 2),
                "golden_cross": sig.golden_cross,
                "obv_trend": sig.obv_trend,
            })
        except Exception:
            pass

    return results


@app.get("/api/drawdown")
async def api_drawdown():
    """Drawdown tracker state."""
    dd_file = PROJECT_ROOT / "data" / "drawdown_state.json"
    try:
        if dd_file.exists():
            with open(dd_file) as f:
                return json.load(f)
    except Exception:
        pass
    return {"high_water_mark": 0, "current_value": 0, "drawdown_pct": 0, "multiplier": 1.0, "tier_label": "normal"}


@app.get("/api/market-status")
async def api_market_status():
    """Market open/closed status."""
    try:
        from scheduler.market_hours import MarketHours
        mh = MarketHours(cfg.market.exchange)
        return {
            "is_open": mh.is_open(),
            "next_open": mh.next_market_open_str(),
            "exchange": cfg.market.exchange,
            "mode": "PAPER" if cfg.user.paper_trading else "LIVE",
        }
    except Exception:
        return {"is_open": False, "next_open": "unknown", "exchange": cfg.market.exchange, "mode": "PAPER"}


@app.get("/api/config")
async def api_config():
    """Current configuration summary."""
    return {
        "exchange": cfg.market.exchange,
        "investment": cfg.user.investment_amount,
        "budget": cfg.user.trading_budget,
        "risk_tolerance": cfg.user.risk_tolerance,
        "paper_trading": cfg.user.paper_trading,
        "approval_required": cfg.user.approval_required,
        "max_position_pct": cfg.user.max_position_pct,
        "max_daily_loss": cfg.user.max_daily_loss_pct,
        "max_heat": cfg.user.max_portfolio_heat,
        "scan_interval": cfg.trading.scan_interval_minutes,
        "max_positions": cfg.trading.max_open_positions,
        "ml_enabled": cfg.ml.enabled,
        "llm_model": cfg.anthropic.model,
    }


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  AI Stock Trader — Dashboard")
    print(f"  Open: http://localhost:8000")
    print(f"  Exchange: {cfg.market.exchange} | Mode: {'PAPER' if cfg.user.paper_trading else 'LIVE'}")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
