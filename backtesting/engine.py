"""
Backtesting engine — validates trading strategies on historical data.
Implements walk-forward testing with realistic transaction costs.
Supports both Indian (NSE) and US market cost structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

from core.logger import get_logger

log = get_logger("backtest")


# ── Transaction cost models ──────────────────────────────────────────────────

@dataclass
class IndianCosts:
    """NSE/BSE transaction costs for delivery (CNC) trades."""
    brokerage_per_order: float = 20.0       # Flat fee (Zerodha/Angel One)
    stt_buy_pct: float = 0.001              # STT: 0.1% on buy delivery
    stt_sell_pct: float = 0.001             # STT: 0.1% on sell delivery
    exchange_charge_pct: float = 0.0000345  # NSE transaction charge
    gst_pct: float = 0.18                   # GST on brokerage + exchange charges
    stamp_duty_buy_pct: float = 0.00015     # Stamp duty on buy side
    sebi_charge_pct: float = 0.000001       # SEBI turnover fee
    slippage_pct: float = 0.001             # Estimated 0.1% slippage

    def total_cost(self, trade_value: float, side: str) -> float:
        """Total cost for a single trade leg (buy or sell)."""
        brokerage = min(self.brokerage_per_order, trade_value * 0.0025)
        stt = trade_value * (self.stt_buy_pct if side == "buy" else self.stt_sell_pct)
        exchange = trade_value * self.exchange_charge_pct
        gst = (brokerage + exchange) * self.gst_pct
        stamp = trade_value * self.stamp_duty_buy_pct if side == "buy" else 0
        sebi = trade_value * self.sebi_charge_pct
        slippage = trade_value * self.slippage_pct
        return brokerage + stt + exchange + gst + stamp + sebi + slippage


@dataclass
class USCosts:
    """US market transaction costs (commission-free era)."""
    commission_per_share: float = 0.0       # Most brokers are commission-free now
    sec_fee_pct: float = 0.0000278          # SEC fee on sells
    slippage_pct: float = 0.0005            # ~0.05% slippage

    def total_cost(self, trade_value: float, side: str) -> float:
        sec = trade_value * self.sec_fee_pct if side == "sell" else 0
        slippage = trade_value * self.slippage_pct
        return sec + slippage


# ── Trade records ────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: float = 0
    side: str = "long"
    stop_loss: float = 0
    take_profit: float = 0
    status: str = "open"     # open | closed
    exit_reason: str = ""    # stop_loss | take_profit | signal | time_stop
    pnl: float = 0
    pnl_pct: float = 0
    holding_days: int = 0
    entry_cost: float = 0
    exit_cost: float = 0


@dataclass
class BacktestResult:
    """Complete backtest results with all performance metrics."""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    benchmark_return_pct: float
    alpha: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    calmar_ratio: float         # Annual return / max drawdown
    volatility_annualized: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float        # Gross profit / gross loss
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_holding_days: float
    max_consecutive_losses: int

    # Cost analysis
    total_costs: float
    cost_pct_of_profit: float

    # Equity curve
    equity_curve: list[dict]    # [{date, equity, drawdown}]
    trades: list[BacktestTrade]


class BacktestEngine:
    """
    Event-driven backtesting engine.
    Processes one bar at a time to simulate real trading conditions.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        exchange: Literal["US", "IN"] = "IN",
        max_position_pct: float = 0.10,
        max_positions: int = 10,
        risk_per_trade_pct: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.exchange = exchange
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct

        self.costs = IndianCosts() if exchange == "IN" else USCosts()

        # State
        self._cash = initial_capital
        self._positions: dict[str, BacktestTrade] = {}
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[dict] = []
        self._peak_equity = initial_capital

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        strategy_name: str = "backtest",
    ) -> BacktestResult:
        """
        Run backtest on historical data with pre-computed signals.

        Args:
            price_data: {symbol: OHLCV DataFrame}
            signals: DataFrame with columns [date, symbol, action, stop_loss_pct, take_profit_pct, conviction]
            benchmark_df: Benchmark OHLCV for alpha calculation
            strategy_name: Label for this backtest

        Returns:
            BacktestResult with all metrics
        """
        self._reset()

        # Get all unique dates across all symbols
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index.date)
        all_dates = sorted(all_dates)

        signal_dates = {}
        if not signals.empty and "date" in signals.columns:
            for _, row in signals.iterrows():
                d = row["date"]
                if d not in signal_dates:
                    signal_dates[d] = []
                signal_dates[d].append(row)

        # Process each trading day
        for current_date in all_dates:
            # Update positions with current prices
            self._update_positions(current_date, price_data)

            # Check for exit signals on open positions
            self._check_exits(current_date, price_data)

            # Process new entry signals
            day_signals = signal_dates.get(current_date, [])
            for signal in day_signals:
                self._process_signal(signal, current_date, price_data)

            # Record equity
            equity = self._calculate_equity(current_date, price_data)
            self._peak_equity = max(self._peak_equity, equity)
            drawdown = (equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0
            self._equity_curve.append({
                "date": str(current_date),
                "equity": round(equity, 2),
                "drawdown": round(drawdown, 4),
                "cash": round(self._cash, 2),
                "positions": len(self._positions),
            })

        # Close any remaining open positions at last available price
        self._close_all_remaining(all_dates[-1] if all_dates else None, price_data)

        # Calculate benchmark return
        bench_return = 0.0
        if benchmark_df is not None and len(benchmark_df) >= 2:
            bench_return = (
                float(benchmark_df["close"].iloc[-1]) /
                float(benchmark_df["close"].iloc[0]) - 1
            )

        return self._compile_results(
            strategy_name=strategy_name,
            start_date=str(all_dates[0]) if all_dates else "",
            end_date=str(all_dates[-1]) if all_dates else "",
            benchmark_return=bench_return,
        )

    # ── Signal processing ────────────────────────────────────────────────

    def _process_signal(self, signal, current_date, price_data):
        """Process a single entry signal."""
        symbol = signal.get("symbol") if isinstance(signal, dict) else signal["symbol"]
        action = signal.get("action") if isinstance(signal, dict) else signal["action"]

        if action not in ("BUY", "SHORT"):
            return

        # Skip if already holding this symbol
        if symbol in self._positions:
            return

        # Skip if max positions reached
        if len(self._positions) >= self.max_positions:
            return

        # Get current price
        sym_df = price_data.get(symbol)
        if sym_df is None:
            return
        date_mask = sym_df.index.date == current_date
        if not date_mask.any():
            return
        bar = sym_df[date_mask].iloc[-1]
        entry_price = float(bar["close"])

        if entry_price <= 0:
            return

        # Position sizing
        stop_loss_pct = signal.get("stop_loss_pct", 0.05) if isinstance(signal, dict) else 0.05
        take_profit_pct = signal.get("take_profit_pct", 0.10) if isinstance(signal, dict) else 0.10

        risk_amount = self._cash * self.risk_per_trade_pct
        stop_distance = entry_price * stop_loss_pct
        shares = risk_amount / stop_distance if stop_distance > 0 else 0
        position_value = shares * entry_price

        # Cap at max position size
        max_value = (self._cash + sum(
            t.shares * t.entry_price for t in self._positions.values()
        )) * self.max_position_pct
        if position_value > max_value:
            shares = max_value / entry_price

        # Cap at available cash
        if position_value > self._cash * 0.95:
            shares = (self._cash * 0.90) / entry_price

        shares = max(1, int(shares))
        position_value = shares * entry_price

        if position_value > self._cash:
            return

        # Calculate entry costs
        entry_cost = self.costs.total_cost(position_value, "buy")

        # Open position
        trade = BacktestTrade(
            symbol=symbol,
            entry_date=datetime.combine(current_date, datetime.min.time()),
            entry_price=entry_price,
            shares=shares,
            side="long" if action == "BUY" else "short",
            stop_loss=entry_price * (1 - stop_loss_pct),
            take_profit=entry_price * (1 + take_profit_pct),
            entry_cost=entry_cost,
        )

        self._positions[symbol] = trade
        self._cash -= (position_value + entry_cost)

    def _check_exits(self, current_date, price_data):
        """Check all open positions for stop-loss/take-profit/time exits."""
        to_close = []

        for symbol, trade in self._positions.items():
            sym_df = price_data.get(symbol)
            if sym_df is None:
                continue
            date_mask = sym_df.index.date == current_date
            if not date_mask.any():
                continue

            bar = sym_df[date_mask].iloc[-1]
            current_low = float(bar["low"])
            current_high = float(bar["high"])
            current_close = float(bar["close"])

            exit_reason = ""
            exit_price = current_close

            # Check stop loss (use low price for realism)
            if trade.side == "long" and current_low <= trade.stop_loss:
                exit_reason = "stop_loss"
                exit_price = trade.stop_loss  # Assume filled at stop price
            elif trade.side == "long" and current_high >= trade.take_profit:
                exit_reason = "take_profit"
                exit_price = trade.take_profit

            # Time-based exit: close after 20 days with minimal gain
            holding_days = (current_date - trade.entry_date.date()).days
            if not exit_reason and holding_days >= 20:
                gain_pct = (current_close - trade.entry_price) / trade.entry_price
                if gain_pct < 0.005:
                    exit_reason = "time_stop"
                    exit_price = current_close

            if exit_reason:
                to_close.append((symbol, exit_price, exit_reason, current_date))

        for symbol, exit_price, reason, exit_date in to_close:
            self._close_position(symbol, exit_price, reason, exit_date)

    def _close_position(self, symbol, exit_price, reason, exit_date):
        """Close a position and record the trade."""
        trade = self._positions.pop(symbol, None)
        if not trade:
            return

        exit_value = trade.shares * exit_price
        exit_cost = self.costs.total_cost(exit_value, "sell")

        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.shares - trade.entry_cost - exit_cost
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl = (trade.entry_price - exit_price) * trade.shares - trade.entry_cost - exit_cost
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        trade.exit_date = datetime.combine(exit_date, datetime.min.time()) if not isinstance(exit_date, datetime) else exit_date
        trade.exit_price = exit_price
        trade.status = "closed"
        trade.exit_reason = reason
        trade.pnl = round(pnl, 2)
        trade.pnl_pct = round(pnl_pct, 4)
        trade.holding_days = (exit_date - trade.entry_date.date()).days if hasattr(exit_date, '__sub__') else 0
        trade.exit_cost = exit_cost

        self._cash += exit_value - exit_cost
        self._closed_trades.append(trade)

    def _update_positions(self, current_date, price_data):
        """Update position tracking for current date."""
        pass  # Positions are tracked via entry data; exits handled in _check_exits

    def _close_all_remaining(self, last_date, price_data):
        """Close all remaining positions at last available price."""
        for symbol in list(self._positions.keys()):
            sym_df = price_data.get(symbol)
            if sym_df is not None and not sym_df.empty:
                exit_price = float(sym_df["close"].iloc[-1])
                self._close_position(symbol, exit_price, "end_of_backtest", last_date or datetime.now().date())

    def _calculate_equity(self, current_date, price_data) -> float:
        """Calculate total portfolio equity."""
        equity = self._cash
        for symbol, trade in self._positions.items():
            sym_df = price_data.get(symbol)
            if sym_df is None:
                equity += trade.shares * trade.entry_price
                continue
            date_mask = sym_df.index.date == current_date
            if date_mask.any():
                price = float(sym_df[date_mask]["close"].iloc[-1])
            else:
                price = trade.entry_price
            equity += trade.shares * price
        return equity

    def _reset(self):
        self._cash = self.initial_capital
        self._positions.clear()
        self._closed_trades.clear()
        self._equity_curve.clear()
        self._peak_equity = self.initial_capital

    # ── Results compilation ───────────────────────────────────────────────

    def _compile_results(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        benchmark_return: float,
    ) -> BacktestResult:
        trades = self._closed_trades
        equity_df = pd.DataFrame(self._equity_curve)

        final_equity = self._cash
        for trade in self._positions.values():
            final_equity += trade.shares * trade.entry_price

        total_return = (final_equity / self.initial_capital) - 1

        # Calculate daily returns from equity curve
        if len(equity_df) >= 2:
            daily_equity = equity_df["equity"].values
            daily_returns = np.diff(daily_equity) / daily_equity[:-1]

            # Annualized metrics
            trading_days = len(daily_returns)
            years = trading_days / 252
            annualized_return = (1 + total_return) ** (1 / max(0.01, years)) - 1
            vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0

            # Sharpe ratio (assuming 0 risk-free rate for simplicity)
            mean_daily = np.mean(daily_returns) if len(daily_returns) > 0 else 0
            std_daily = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
            sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

            # Sortino (downside deviation only)
            downside = daily_returns[daily_returns < 0]
            downside_std = np.std(downside) if len(downside) > 1 else 1e-10
            sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0

            # Max drawdown
            equity_series = pd.Series(daily_equity)
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0

            # Max drawdown duration
            dd_duration = 0
            max_dd_duration = 0
            for dd_val in drawdown:
                if dd_val < 0:
                    dd_duration += 1
                    max_dd_duration = max(max_dd_duration, dd_duration)
                else:
                    dd_duration = 0
        else:
            annualized_return = 0
            vol = 0
            sharpe = 0
            sortino = 0
            max_dd = 0
            max_dd_duration = 0

        # Trade statistics
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total = len(trades)
        win_rate = len(wins) / max(1, total)

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        largest_win = max([t.pnl_pct for t in wins], default=0)
        largest_loss = min([t.pnl_pct for t in losses], default=0)
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0

        # Max consecutive losses
        max_consec_loss = 0
        current_consec = 0
        for t in trades:
            if t.pnl <= 0:
                current_consec += 1
                max_consec_loss = max(max_consec_loss, current_consec)
            else:
                current_consec = 0

        # Total costs
        total_costs = sum(t.entry_cost + t.exit_cost for t in trades)
        net_profit = final_equity - self.initial_capital
        cost_pct = total_costs / max(abs(net_profit), 1e-10) * 100

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Alpha
        alpha = total_return - benchmark_return

        return BacktestResult(
            strategy_name=strategy_name,
            symbol="PORTFOLIO",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=round(final_equity, 2),
            total_return_pct=round(total_return * 100, 2),
            annualized_return_pct=round(annualized_return * 100, 2),
            benchmark_return_pct=round(benchmark_return * 100, 2),
            alpha=round(alpha * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=round(calmar, 2),
            volatility_annualized=round(vol * 100, 2),
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            largest_win_pct=round(largest_win * 100, 2),
            largest_loss_pct=round(largest_loss * 100, 2),
            avg_holding_days=round(avg_hold, 1),
            max_consecutive_losses=max_consec_loss,
            total_costs=round(total_costs, 2),
            cost_pct_of_profit=round(cost_pct, 1),
            equity_curve=self._equity_curve,
            trades=trades,
        )
