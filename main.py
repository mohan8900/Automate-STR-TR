"""
AI Stock Trader — Main entry point.

Usage:
  python main.py              # Start the full trading loop
  python main.py --dashboard  # Launch the Streamlit dashboard only
  python main.py --backtest   # Run a single analysis cycle (no trading)
  python main.py --day-trade  # Start intraday day trading loop
  python main.py --check      # Verify all APIs and configuration
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fix Windows console encoding for ₹ and other Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description="AI Stock Trader")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--backtest", action="store_true", help="Single analysis cycle, no execution")
    parser.add_argument("--backtest-ml", action="store_true", help="Run full ML backtesting on historical data")
    parser.add_argument("--backtest-technical", action="store_true", help="Run technical analysis backtesting")
    parser.add_argument("--buffett-screen", action="store_true", help="Run Warren Buffett value screen on watchlist")
    parser.add_argument("--check", action="store_true", help="Check API keys and configuration")
    parser.add_argument("--once", action="store_true", help="Run one trading cycle and exit")
    parser.add_argument("--day-trade", action="store_true", help="Start intraday day trading loop")
    return parser.parse_args()


def check_config():
    """Verify all configuration and API keys."""
    from config.settings import get_settings
    from core.logger import get_logger

    log = get_logger("setup")
    cfg = get_settings()

    print("\n" + "="*60)
    print("AI STOCK TRADER — CONFIGURATION CHECK")
    print("="*60)

    # Check API keys
    # At least one LLM key is required
    has_llm = bool(
        os.getenv("GEMINI_API_KEY") or
        os.getenv("LITELLM_API_BASE") or
        os.getenv("ANTHROPIC__API_KEY", "").startswith("sk-ant-api")
    )
    checks = {
        "GEMINI_API_KEY": ("Google Gemini (FREE LLM)", not has_llm),
        "ANGEL_ONE__API_KEY": ("Angel One SmartAPI (India)", cfg.market.exchange == "IN"),
        "NEWS_API_KEY": ("NewsAPI (optional)", False),
    }

    all_required_ok = True
    for env_key, (name, required) in checks.items():
        val = os.getenv(env_key, "")
        status = "[OK]" if val else ("[MISSING]" if required else "[SKIP]")
        suffix = " (REQUIRED)" if required and not val else ""
        print(f"  {status} {name}: {'set' if val else 'NOT SET'}{suffix}")
        if required and not val:
            all_required_ok = False

    print("\nConfiguration:")
    currency = "₹" if cfg.market.exchange == "IN" else "$"
    print(f"  Exchange:      {cfg.market.exchange}")
    print(f"  Paper Trading: {cfg.user.paper_trading}")
    print(f"  Investment:    {currency}{cfg.user.investment_amount:,.0f}")
    print(f"  Budget:        {currency}{cfg.user.trading_budget:,.0f}")
    print(f"  Risk Tolerance:{cfg.user.risk_tolerance}")
    print(f"  LLM Model:     {cfg.anthropic.model}")
    print(f"  Scan Interval: {cfg.trading.scan_interval_minutes}m")

    if not all_required_ok:
        print("\n[WARNING]  Missing required API keys. Copy .env.example to .env and fill in values.")
        return False

    # Try connecting to data feed
    try:
        import yfinance as yf
        test_symbol = "TCS.NS" if cfg.market.exchange == "IN" else "AAPL"
        ticker = yf.Ticker(test_symbol)
        hist = ticker.history(period="2d")
        print(f"\n[OK] Yahoo Finance: connected ({test_symbol} @ {currency}{hist['Close'].iloc[-1]:.2f})")
    except Exception as e:
        print(f"\n[FAIL] Yahoo Finance: {e}")

    # Try LLM
    try:
        from llm.client import ClaudeClient
        client = ClaudeClient(cfg.anthropic)
        test = client.complete("You are a test.", "Say 'OK' and nothing else.", max_tokens=10)
        print(f"[OK] Claude API: connected (response: {test.strip()[:20]})")
    except Exception as e:
        print(f"[FAIL] Claude API: {e}")

    print("\n" + "="*60 + "\n")
    return all_required_ok


def run_single_cycle(execute: bool = False):
    """Run one complete analysis cycle."""
    from config.settings import get_settings
    from services.swing_trading.trading_loop import TradingLoop
    from core.logger import get_logger

    log = get_logger("main")
    cfg = get_settings()

    if not execute:
        # Force approval required so no trades are auto-executed
        cfg.user.approval_required = True

    log.info(f"Starting {'analysis' if not execute else 'trading'} cycle...")
    loop = TradingLoop(cfg)

    try:
        loop.run_cycle()
    except KeyboardInterrupt:
        log.info("Interrupted by user")


def run_trading_loop():
    """Start the continuous trading loop."""
    from config.settings import get_settings
    from services.swing_trading.trading_loop import TradingLoop
    from core.logger import get_logger

    log = get_logger("main")
    cfg = get_settings()

    # Safety confirmation for live mode
    if not cfg.user.paper_trading:
        print("\n" + "!"*60)
        print("  WARNING: LIVE TRADING MODE ENABLED")
        print("  Real money will be used for all trades!")
        print("!"*60)
        confirm = input("\nType 'LIVE' to confirm live trading, anything else to abort: ")
        if confirm.strip() != "LIVE":
            print("Aborted. Switch to paper_trading=true in config.toml to use paper mode.")
            sys.exit(1)

    log.info("Starting AI Stock Trader...")
    log.info(
        f"Exchange: {cfg.market.exchange} | "
        f"{'PAPER' if cfg.user.paper_trading else 'LIVE'} | "
        f"{'₹' if cfg.market.exchange == 'IN' else '$'}{cfg.user.investment_amount:,.0f} allocated | "
        f"Approval required: {cfg.user.approval_required}"
    )

    loop = TradingLoop(cfg)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        log.info("Trading loop stopped by user (Ctrl+C)")
        loop.stop()
    except Exception as e:
        log.exception(f"Fatal trading loop error: {e}")
        sys.exit(1)


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    print(f"Launching dashboard at http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "false",
        "--server.port", "8501",
    ])


def run_ml_backtest():
    """Run ML ensemble backtesting on historical data."""
    from config.settings import get_settings
    from config.watchlists import IN_WATCHLIST, US_WATCHLIST
    from services.swing_trading.backtesting.runner import BacktestRunner

    cfg = get_settings()
    runner = BacktestRunner(cfg)

    watchlist = IN_WATCHLIST[:15] if cfg.market.exchange == "IN" else US_WATCHLIST[:15]
    print(f"\nRunning ML backtest on {len(watchlist)} stocks...")
    result = runner.run_ml_backtest(watchlist, period="2y")
    return result


def run_technical_backtest():
    """Run technical analysis backtesting."""
    from config.settings import get_settings
    from config.watchlists import IN_WATCHLIST, US_WATCHLIST
    from services.swing_trading.backtesting.runner import BacktestRunner

    cfg = get_settings()
    runner = BacktestRunner(cfg)

    watchlist = IN_WATCHLIST[:15] if cfg.market.exchange == "IN" else US_WATCHLIST[:15]
    print(f"\nRunning technical analysis backtest on {len(watchlist)} stocks...")
    result = runner.run_technical_backtest(watchlist, period="2y")
    return result


def run_day_trading_loop():
    """Start the intraday day-trading loop."""
    from config.settings import get_settings
    from services.day_trading.loop import IntradayTradingLoop
    from core.logger import get_logger

    log = get_logger("main")
    cfg = get_settings()

    if not cfg.day_trading.enabled:
        print(
            "\n[ERROR] Day trading is disabled in configuration.\n"
            "Set [day_trading] enabled = true in config.toml to enable it."
        )
        sys.exit(1)

    # Safety confirmation for live mode
    if not cfg.day_trading.paper_trading:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE INTRADAY TRADING MODE ENABLED")
        print("  Real money will be used for all day trades!")
        print("!" * 60)
        confirm = input("\nType 'LIVE' to confirm live day trading, anything else to abort: ")
        if confirm.strip() != "LIVE":
            print("Aborted. Set paper_trading = true under [day_trading] in config.toml.")
            sys.exit(1)

    log.info("Starting Intraday Day Trading System...")
    log.info(
        f"Exchange: {cfg.market.exchange} | "
        f"{'PAPER' if cfg.day_trading.paper_trading else 'LIVE'} | "
        f"Capital: {'₹' if cfg.market.exchange == 'IN' else '$'}"
        f"{cfg.day_trading.capital_allocation:,.0f} | "
        f"Max trades/day: {cfg.day_trading.max_trades_per_day}"
    )

    loop = IntradayTradingLoop(cfg)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        log.info("Day trading loop stopped by user (Ctrl+C)")
        loop.stop()
    except Exception as e:
        log.exception(f"Fatal day trading loop error: {e}")
        sys.exit(1)


def run_buffett_screen():
    """Run Warren Buffett value screen on the watchlist."""
    from config.settings import get_settings
    from config.watchlists import IN_WATCHLIST, US_WATCHLIST
    from services.swing_trading.portfolio.value_screener import ValueScreener

    cfg = get_settings()
    screener = ValueScreener()

    watchlist = IN_WATCHLIST[:20] if cfg.market.exchange == "IN" else US_WATCHLIST[:20]
    print(f"\nRunning Buffett value screen on {len(watchlist)} stocks...")
    print("=" * 80)

    results = screener.screen_multiple(watchlist)

    print(f"\n{'Symbol':<20} {'Score':>6} {'Moat':>6} {'Debt':>6} {'FCF':>6} {'Value':>6} {'Status'}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r.passes_buffett_screen else "FAIL"
        print(
            f"{r.symbol:<20} {r.total_score:>5.0f} "
            f"{r.moat_score:>5.0f} {r.debt_health:>5.0f} "
            f"{r.cash_generation:>5.0f} {r.valuation_score:>5.0f} "
            f"  {status}"
        )
        if r.disqualifiers:
            for d in r.disqualifiers:
                print(f"  {'':>20} -> {d}")

    passes = [r for r in results if r.passes_buffett_screen]
    print(f"\n{len(passes)}/{len(results)} stocks passed the Buffett screen")
    print("=" * 80)


def main():
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = parse_args()

    if args.dashboard:
        launch_dashboard()
    elif args.check:
        ok = check_config()
        sys.exit(0 if ok else 1)
    elif args.backtest_ml:
        run_ml_backtest()
    elif args.backtest_technical:
        run_technical_backtest()
    elif args.buffett_screen:
        run_buffett_screen()
    elif args.backtest:
        run_single_cycle(execute=False)
    elif args.day_trade:
        run_day_trading_loop()
    elif args.once:
        run_single_cycle(execute=True)
    else:
        # Default: run the full trading loop
        run_trading_loop()


if __name__ == "__main__":
    main()
