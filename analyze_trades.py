"""
analyze_trades.py — Post-paper-trading leak hunt.

Reads the SQLite trading database and prints a report answering the specific
questions that identify where your paper-trading system leaks money.

Usage:
    python analyze_trades.py                   # all-time, both modes
    python analyze_trades.py --days 30         # last 30 days only
    python analyze_trades.py --mode swing      # swing trades only
    python analyze_trades.py --mode intraday   # day trades only
    python analyze_trades.py --db path/to.db   # custom DB path

Run this AFTER paper-trading for at least 2–4 weeks. With less data the numbers
are noise.
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DEFAULT_DB = "data/trading.db"


def fmt_money(x: Optional[float], currency: str = "₹") -> str:
    if x is None:
        return "—"
    return f"{currency}{x:+,.2f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x*100:+.2f}%"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def load_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def swing_report(conn: sqlite3.Connection, since: Optional[str]) -> dict:
    section("SWING TRADES")
    if not table_exists(conn, "trades"):
        print("  `trades` table not found — run the swing loop at least once to create schema.")
        return {}

    where = "status = 'CLOSED'"
    params: tuple = ()
    if since:
        where += " AND closed_at >= ?"
        params = (since,)

    rows = load_rows(
        conn,
        f"SELECT * FROM trades WHERE {where} ORDER BY closed_at ASC",
        params,
    )
    if not rows:
        print("  No closed swing trades in the selected window.")
        return {}

    pnls = [r["realized_pnl"] or 0 for r in rows]
    pnl_pcts = [r["realized_pnl_pct"] or 0 for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = statistics.mean(wins) if wins else 0
    avg_loss = statistics.mean(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) else float("inf")

    # Drawdown from cumulative pnl curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Worst symbols
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_symbol[r["symbol"]].append(r["realized_pnl"] or 0)
    worst = sorted(
        ((sym, sum(ps), len(ps)) for sym, ps in by_symbol.items()),
        key=lambda x: x[1],
    )[:5]

    # Close reasons
    reasons = Counter(r["close_reason"] or "unknown" for r in rows)

    # Holding period
    holds = [r["holding_days"] for r in rows if r["holding_days"] is not None]

    print(f"  Total closed trades : {len(pnls)}")
    print(f"  Total P&L           : {fmt_money(total_pnl)}")
    print(f"  Win rate            : {win_rate:.1f}%  ({len(wins)} W / {len(losses)} L)")
    print(f"  Avg winner          : {fmt_money(avg_win)}  |  Avg loser: {fmt_money(avg_loss)}")
    print(f"  Profit factor       : {profit_factor:.2f}  (gross wins / gross losses)")
    print(f"  Avg return / trade  : {fmt_pct(statistics.mean(pnl_pcts))}")
    print(f"  Max drawdown        : {fmt_money(-max_dd)}")
    if holds:
        print(f"  Avg holding days    : {statistics.mean(holds):.1f}")
    print(f"  Close reasons       : {dict(reasons)}")
    print(f"  Worst 5 symbols     :")
    for sym, total, n in worst:
        print(f"    {sym:20s} {fmt_money(total):>14s}  ({n} trades)")
    return {
        "count": len(pnls), "win_rate": win_rate, "total_pnl": total_pnl,
        "profit_factor": profit_factor, "max_dd": max_dd,
    }


def intraday_report(conn: sqlite3.Connection, since: Optional[str]) -> dict:
    section("INTRADAY TRADES")
    if not table_exists(conn, "intraday_trades"):
        print("  `intraday_trades` table not found — run `python main.py --day-trade` once to create schema.")
        return {}

    where = "status = 'CLOSED'"
    params: tuple = ()
    if since:
        where += " AND closed_at >= ?"
        params = (since,)

    rows = load_rows(
        conn,
        f"SELECT * FROM intraday_trades WHERE {where} ORDER BY closed_at ASC",
        params,
    )
    if not rows:
        print("  No closed intraday trades in the selected window.")
        return {}

    gross_pnls = [r["realized_pnl"] or 0 for r in rows]
    net_pnls = [r["net_pnl"] or 0 for r in rows]
    fees = [r["brokerage_cost"] or 0 for r in rows]

    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]

    gross_total = sum(gross_pnls)
    net_total = sum(net_pnls)
    total_fees = sum(fees)
    fee_drag_pct = (total_fees / sum(w for w in gross_pnls if w > 0) * 100) if any(p > 0 for p in gross_pnls) else 0
    win_rate = len(wins) / len(net_pnls) * 100

    # Drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in net_pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Time of day: bucket by opening hour
    hour_pnl: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        ts = r["opened_at"]
        if not ts:
            continue
        try:
            h = datetime.fromisoformat(ts).hour
        except Exception:
            continue
        hour_pnl[h].append(r["net_pnl"] or 0)

    # By strategy
    by_strat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_strat[r["strategy_name"] or "unknown"].append(r["net_pnl"] or 0)

    # Close reasons
    reasons = Counter(r["close_reason"] or "unknown" for r in rows)

    # Worst symbols
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_symbol[r["symbol"]].append(r["net_pnl"] or 0)
    worst = sorted(
        ((sym, sum(ps), len(ps)) for sym, ps in by_symbol.items()),
        key=lambda x: x[1],
    )[:5]

    print(f"  Total closed trades : {len(net_pnls)}")
    print(f"  Gross P&L           : {fmt_money(gross_total)}")
    print(f"  Brokerage paid      : {fmt_money(-total_fees)}  ({fee_drag_pct:.1f}% of gross wins)")
    print(f"  NET P&L             : {fmt_money(net_total)}")
    print(f"  Win rate            : {win_rate:.1f}%  ({len(wins)} W / {len(losses)} L)")
    print(f"  Max drawdown        : {fmt_money(-max_dd)}")
    print(f"  Close reasons       : {dict(reasons)}")

    print(f"  By strategy         :")
    for strat, pnls in sorted(by_strat.items(), key=lambda x: -sum(x[1])):
        pf_wins = sum(p for p in pnls if p > 0)
        pf_losses = abs(sum(p for p in pnls if p < 0))
        pf = pf_wins / pf_losses if pf_losses else float("inf")
        print(f"    {strat:18s} trades={len(pnls):3d}  pnl={fmt_money(sum(pnls)):>12s}  pf={pf:.2f}")

    print(f"  By opening hour (IST)  (watch for negative hours — chop windows):")
    for h in sorted(hour_pnl):
        pnls = hour_pnl[h]
        w = sum(1 for p in pnls if p > 0)
        print(f"    {h:02d}:00  trades={len(pnls):3d}  pnl={fmt_money(sum(pnls)):>12s}  wr={w/len(pnls)*100:.0f}%")

    print(f"  Worst 5 symbols     :")
    for sym, total, n in worst:
        print(f"    {sym:20s} {fmt_money(total):>14s}  ({n} trades)")

    return {
        "count": len(net_pnls), "win_rate": win_rate,
        "net_pnl": net_total, "fee_drag": fee_drag_pct, "max_dd": max_dd,
    }


def rejection_report(conn: sqlite3.Connection, since: Optional[str]) -> None:
    if not table_exists(conn, "analysis_logs"):
        return
    where = "execution_approved = 0"
    params: tuple = ()
    if since:
        where += " AND timestamp >= ?"
        params = (since,)

    rows = load_rows(
        conn,
        f"SELECT symbol, action_recommended, conviction, composite_score, market_regime "
        f"FROM analysis_logs WHERE {where}",
        params,
    )
    if not rows:
        return

    by_regime = Counter(r["market_regime"] or "unknown" for r in rows)
    by_action = Counter(r["action_recommended"] or "unknown" for r in rows)
    convictions = [r["conviction"] for r in rows if r["conviction"] is not None]

    section("LLM SIGNALS NOT EXECUTED")
    print(f"  Total analyses not executed : {len(rows)}")
    print(f"  By market regime            : {dict(by_regime)}")
    print(f"  By action recommended       : {dict(by_action)}")
    if convictions:
        print(f"  Avg conviction when rejected: {statistics.mean(convictions):.1f}/10")


def leak_hunt(swing: dict, intra: dict) -> None:
    """Run the 7-question leak diagnostic against collected metrics."""
    section("LEAK HUNT — what to fix FIRST")

    issues: list[str] = []

    def check(d: dict, label: str):
        if not d:
            return
        if d.get("win_rate", 100) < 40:
            issues.append(f"[{label}] Win rate {d['win_rate']:.0f}% < 40%  "
                          f"→ signal generation is weak, not execution. Re-tune strategy before going live.")
        if d.get("profit_factor") is not None and d["profit_factor"] < 1.0:
            issues.append(f"[{label}] Profit factor {d['profit_factor']:.2f} < 1.0  "
                          f"→ avg loser is eating avg winner. Stops too wide OR exits too early.")
        if d.get("fee_drag", 0) > 20:
            issues.append(f"[{label}] Fees are {d['fee_drag']:.0f}% of gross wins  "
                          f"→ overtrading. Raise min_signal_strength or reduce max_daily_trades.")
        total_pnl = d.get("total_pnl", d.get("net_pnl", 0))
        if d.get("max_dd", 0) > abs(total_pnl) and total_pnl > 0:
            issues.append(f"[{label}] Max drawdown {d['max_dd']:.0f} > total profit {total_pnl:.0f}  "
                          f"→ equity curve is a roller-coaster. Cut position size.")
        if d.get("count", 0) < 20:
            issues.append(f"[{label}] Only {d['count']} trades — need 20+ before any conclusion is meaningful.")

    check(swing, "SWING")
    check(intra, "INTRADAY")

    if not issues:
        print("  No loud leaks detected in the current window. If equity curve still feels bad,")
        print("  the leak is psychological (overriding signals, panic-closing) — not mechanical.")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print()
    print("  Rule: fix ONE leak at a time. Re-run this report after each change.")


def main():
    parser = argparse.ArgumentParser(description="Paper-trading leak hunt")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument("--days", type=int, default=None, help="Look back N days (default: all)")
    parser.add_argument("--mode", choices=["both", "swing", "intraday"], default="both")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database at {db_path}. Run paper-trading first.")
        return

    since = None
    if args.days:
        since = (datetime.now() - timedelta(days=args.days)).isoformat()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print(f"Database : {db_path}")
    print(f"Window   : last {args.days} days" if args.days else "Window   : all-time")
    print(f"Mode     : {args.mode}")

    swing_stats: dict = {}
    intra_stats: dict = {}

    if args.mode in ("both", "swing"):
        swing_stats = swing_report(conn, since)
        rejection_report(conn, since)
    if args.mode in ("both", "intraday"):
        intra_stats = intraday_report(conn, since)

    leak_hunt(swing_stats, intra_stats)
    conn.close()


if __name__ == "__main__":
    main()
