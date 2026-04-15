# AI Stock Trader — End-to-End Autonomous Trading System

An OpenHands-style AI agent for the stock market. Analyzes markets using Claude AI,
makes investment decisions, manages risk, and executes trades automatically.

---

## Architecture

```
Exchange Market Data (Yahoo Finance / Alpaca)
        ↓
Market Scanner (pre-filter: price, volume, momentum)
        ↓
Analysis Pipeline (parallel)
  ├── Technical Analysis (RSI, MACD, BB, ATR, SMA, Volume, S/R)
  ├── Fundamental Analysis (P/E, PEG, FCF, Margins, Growth, Debt)
  └── Sentiment Analysis (News, VADER, Analyst ratings)
        ↓
AnalysisBundle (composite score 0-100)
        ↓  (only if composite_score ≥ 55)
LLM Analyst — Claude (trade recommendation + thesis)
        ↓  (only if actionable, conviction ≥ 7)
LLM Risk Reviewer — Claude (adversarial challenge)
        ↓  (only if approved)
Risk Management
  ├── Position Sizer (ATR-based / Fixed % / Half-Kelly)
  ├── Portfolio Heat Check (max 6% total at risk)
  ├── VIX Adjustment (reduce size in high-vol regimes)
  └── Circuit Breaker (daily loss limit, market hours, earnings)
        ↓
Human Approval Queue (Streamlit dashboard) OR Auto-Execute
        ↓
Broker API (Alpaca US / Zerodha India)
  └── Bracket Order (entry + stop-loss + take-profit, atomic)
        ↓
Position Monitor (trailing stops, take-profit scaling, time exits)
        ↓
Portfolio & Performance Tracking (Sharpe, Sortino, drawdown, win rate)
```

---

## Quick Start

### 1. Install dependencies
```bash
cd ai-stock-trader
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your actual keys
```

Minimum required:
- `ANTHROPIC_API_KEY` — Claude API key (get from console.anthropic.com)
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` — Alpaca paper trading (free at alpaca.markets)

### 3. Check configuration
```bash
python main.py --check
```

### 4. Start with paper trading (ALWAYS start here)
```bash
# Terminal 1: Start the trading loop
python main.py

# Terminal 2: Launch the dashboard
python main.py --dashboard
# Open http://localhost:8501
```

### 5. Run a single analysis cycle (no trades)
```bash
python main.py --backtest
```

---

## Configuration

Edit `config.toml` or use the Settings page in the dashboard.

Key settings:
```toml
[user]
investment_amount = 50000.0   # Your total capital
trading_budget    = 10000.0   # Capital to deploy per cycle
risk_tolerance    = "moderate"
paper_trading     = true      # ALWAYS start true
approval_required = true      # Require your approval before trades

[trading]
scan_interval_minutes = 30    # How often to scan the market
min_conviction_execute = 7    # AI conviction threshold (1-10)
```

---

## How It Works — Full Cycle

Every N minutes (default 30), the system:

1. **Checks market hours** — only trades 9:35 AM–3:45 PM ET
2. **Refreshes market context** — determines market regime (BULL/BEAR/SIDEWAYS/VOLATILE) and VIX level
3. **Monitors existing positions** — checks stop-losses and take-profit targets
4. **Scans the universe** — filters S&P 500 + watchlist by price, volume, momentum
5. **Builds analysis bundles** — technical + fundamental + sentiment for top candidates
6. **LLM analysis** — Claude analyzes each candidate and recommends BUY/SELL/SHORT/PASS
7. **Risk review** — second Claude call challenges the recommendation adversarially
8. **Position sizing** — determines exact share count using ATR-based volatility sizing
9. **Queues for approval** — shows trade in dashboard for human review (if approval_required=true)
10. **Executes trades** — submits bracket orders (entry + stop-loss + take-profit) to broker
11. **Saves portfolio snapshot** — tracks performance over time

---

## Risk Management

The system has multiple layers of protection:

| Layer | Mechanism |
|-------|-----------|
| VIX Regime | Reduce position size by 30-60% in high-vol |
| Market Regime | No new longs in BEAR, no new positions in VOLATILE |
| Portfolio Heat | Max 6% of portfolio at risk at any time |
| Position Limit | Max 5% per single position (configurable) |
| Stop-Loss | ATR-based dynamic stop (2× ATR below entry) |
| Take-Profit | 3 scaled targets (2×, 4×, 6× ATR) |
| Trailing Stop | Activates after first take-profit hit |
| Earnings Blackout | No new trades within 3 days of earnings |
| Daily Loss Circuit | Halt at -2% daily P&L (configurable) |
| Market Hours | No trading outside 9:35 AM–3:45 PM ET |
| Human Approval | All trades reviewed before execution (optional) |

---

## Indian Market Support (NSE/BSE)

Set in config.toml:
```toml
[market]
exchange = "IN"
```

And add Zerodha credentials to .env:
```
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
KITE_ACCESS_TOKEN=your_daily_token
```

Note: Zerodha access tokens expire daily and must be refreshed via OAuth.

---

## Disclaimer

This system is for educational and research purposes. Past performance does not
guarantee future results. Stock trading involves significant risk of loss.
Always start with paper trading. Never invest money you cannot afford to lose.
