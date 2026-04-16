"""
Intraday Trading Loop — the main orchestrator for day trading.

Continuously scans the watchlist, monitors open positions, enforces risk
limits, and executes (or paper-trades) intraday signals.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Optional

from config.settings import TradingSystemConfig
from config.watchlists import IN_WATCHLIST, US_WATCHLIST
from core.circuit_breaker import CircuitBreaker
from core.kill_switch import KillSwitch
from core.logger import get_logger
from database.manager import DatabaseManager
from database.repository import TradeRepository
from execution.broker_client import create_broker_client
from execution.order_manager import OrderManager
from execution.trade_executor import TradeExecutor
from notifications.alert_manager import AlertManager
from scheduler.market_hours import MarketHours
from services.day_trading.data.intraday_feed import IntradayFeed
from services.day_trading.indicators.intraday_indicators import IntradayIndicatorEngine
from services.day_trading.ml.xgboost_model import IntradayXGBoostModel
from services.day_trading.risk.intraday_risk import IntradayRiskManager
from services.day_trading.signals import IntradayPosition, IntradaySignal
from services.day_trading.strategies.selector import IntradayStrategySelector

log = get_logger("intraday_loop")


class IntradayTradingLoop:
    """
    Main orchestrator for the intraday day-trading service.

    Each call to ``run_cycle()`` performs:
    1. Market-hours check
    2. Force-exit time check
    3. Kill-switch check
    4. Daily risk reset (if new day)
    5. Position monitoring (stop-loss, target, time limit)
    6. Watchlist scan for new signals
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: TradingSystemConfig) -> None:
        self.config = config
        dt_cfg = config.day_trading

        # ── Shared infrastructure ────────────────────────────────────
        self.db = DatabaseManager(config.database.path)
        self.repository = TradeRepository(self.db)
        self.broker = create_broker_client(config)
        self.order_manager = OrderManager(self.broker)
        self.executor = TradeExecutor(
            self.broker, self.order_manager, self.repository,
            paper_trading=dt_cfg.paper_trading,
        )
        self.circuit_breaker = CircuitBreaker(config)
        self.alert_manager = AlertManager(config)
        self.kill_switch = KillSwitch(
            broker_client=self.broker,
            circuit_breaker=self.circuit_breaker,
            alert_manager=self.alert_manager,
            trade_repo=self.repository,
        )
        self.market_hours = MarketHours(exchange=config.market.exchange)

        # ── Day-trading components ───────────────────────────────────
        self.feed = IntradayFeed(dt_cfg, exchange=config.market.exchange)
        self.indicators = IntradayIndicatorEngine  # class, not instance
        self.strategy_selector = IntradayStrategySelector(dt_cfg.strategies_enabled)
        self.ml_model = IntradayXGBoostModel()
        self.risk_manager = IntradayRiskManager(dt_cfg)

        # ── Runtime state ────────────────────────────────────────────
        self._running: bool = False
        self._positions: dict[str, IntradayPosition] = {}
        self._last_llm_review: Optional[datetime] = None
        self._current_date: date = date.today()

        # ── Watchlist ────────────────────────────────────────────────
        if config.market.exchange == "IN":
            self._watchlist = IN_WATCHLIST[:20]
        else:
            self._watchlist = US_WATCHLIST[:20]

        log.info(
            f"Intraday Trading System initialized | "
            f"watchlist={len(self._watchlist)} symbols | "
            f"{'PAPER' if dt_cfg.paper_trading else 'LIVE'} mode | "
            f"scan_interval={dt_cfg.scan_interval_seconds}s"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Block and run the intraday loop until stopped."""
        self._running = True
        dt_cfg = self.config.day_trading
        log.info(
            f"Starting intraday loop | "
            f"scan_interval={dt_cfg.scan_interval_seconds}s | "
            f"{'PAPER' if dt_cfg.paper_trading else 'LIVE'} mode"
        )

        while self._running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                log.exception(f"Intraday loop error: {e}")
            time.sleep(dt_cfg.scan_interval_seconds)

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        """Execute one complete intraday scan-and-act cycle."""
        cycle_start = datetime.now()
        log.info(f"=== Intraday Cycle: {cycle_start.strftime('%H:%M:%S')} ===")
        dt_cfg = self.config.day_trading

        # 1. Market hours check
        if not self.market_hours.is_open():
            log.info("Market closed")
            return

        # 2. Force exit check (past configured time, e.g. 15:15)
        if self.risk_manager.check_force_exit_time():
            self._force_exit_all("Force exit time reached")
            return

        # 3. Kill switch
        if self.kill_switch.is_active():
            log.warning("Kill switch active — skipping cycle")
            return

        # 4. Daily risk reset if new day
        today = date.today()
        if today != self._current_date:
            self.risk_manager.reset_daily()
            self._current_date = today
            log.info(f"New trading day: {today}")

        # 5. Monitor existing positions
        for symbol, pos in list(self._positions.items()):
            try:
                current_price = self.feed.get_current_price(symbol)
                if current_price <= 0:
                    continue

                pos.current_price = current_price
                pos.unrealized_pnl = (
                    (current_price - pos.entry_price)
                    * pos.qty
                    * (1 if pos.side == "long" else -1)
                )
                pos.minutes_held = int(
                    (datetime.now() - pos.entry_time).total_seconds() / 60
                )

                # Check stop loss
                if pos.side == "long" and current_price <= pos.stop_loss_price:
                    self._close_position(symbol, "Stop loss hit")
                elif pos.side == "short" and current_price >= pos.stop_loss_price:
                    self._close_position(symbol, "Stop loss hit")
                # Check target
                elif pos.side == "long" and current_price >= pos.target_price:
                    self._close_position(symbol, "Target reached")
                elif pos.side == "short" and current_price <= pos.target_price:
                    self._close_position(symbol, "Target reached")
                # Check time limit
                elif pos.minutes_held >= dt_cfg.position_time_limit_minutes:
                    self._close_position(symbol, "Time limit reached")

            except Exception as e:
                log.warning(f"Position monitoring error {symbol}: {e}")

        # 6. Scan for new trades
        can_trade, reason = self.risk_manager.can_open_trade()
        if can_trade:
            for symbol in self._get_watchlist():
                if symbol in self._positions:
                    continue  # already have a position

                try:
                    candles_1m = self.feed.get_candles(symbol, "1m", 60)
                    candles_5m = self.feed.get_candles(symbol, "5m", 30)

                    if candles_1m is None or len(candles_1m) < 15:
                        continue
                    if candles_5m is None or len(candles_5m) < 10:
                        continue

                    technicals = self.indicators.compute_all(
                        candles_1m, candles_5m, dt_cfg.orb_window_minutes,
                    )

                    # ML prediction (non-blocking)
                    ml_pred = None
                    try:
                        ml_pred = self.ml_model.predict(candles_1m, candles_5m)
                    except Exception:
                        pass

                    # Strategy consensus
                    signal = self.strategy_selector.generate_combined_signal(
                        symbol, candles_1m, candles_5m, technicals,
                        ml_prediction=ml_pred,
                    )

                    if (
                        signal.action in ("BUY", "SELL")
                        and signal.strength >= dt_cfg.min_signal_strength
                    ):
                        # Log the signal
                        self._log_intraday_signal(signal)

                        sized = self.risk_manager.size_position(
                            signal.entry_price,
                            signal.stop_loss_price,
                            signal.strength,
                        )
                        sized.symbol = symbol
                        sized.target_price = signal.target_price

                        if sized.is_valid:
                            xgb_prob = (
                                ml_pred.probability if ml_pred else None
                            )
                            self._execute_or_queue(signal, sized, xgb_prob)

                            # Recheck if we can still open more
                            can_trade, _ = self.risk_manager.can_open_trade()
                            if not can_trade:
                                break
                except Exception as e:
                    log.warning(f"Scan failed for {symbol}: {e}")
        else:
            log.debug(f"Cannot open new trades: {reason}")

        # 7. LLM batch review (placeholder for v2)
        # Skip for now — add when positions are being opened regularly

        elapsed = (datetime.now() - cycle_start).total_seconds()
        log.info(
            f"=== Intraday Cycle done in {elapsed:.1f}s | "
            f"positions={len(self._positions)} | "
            f"trades_today={self.risk_manager._trades_today} ==="
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _close_position(self, symbol: str, reason: str) -> None:
        """Close an open intraday position and record the result."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        pnl = (
            (pos.current_price - pos.entry_price)
            * pos.qty
            * (1 if pos.side == "long" else -1)
        )
        pnl_pct = pnl / (pos.entry_price * pos.qty) * 100 if pos.entry_price > 0 else 0.0

        # Estimate brokerage
        position_value = pos.entry_price * pos.qty
        brokerage = (
            min(self.config.day_trading.brokerage_per_order,
                position_value * self.config.day_trading.brokerage_pct)
            * 2
        )

        self.risk_manager.record_trade_result(pnl, brokerage)
        self.risk_manager.record_position_close()

        log.info(
            f"CLOSED {symbol} | reason={reason} | side={pos.side} | "
            f"entry={pos.entry_price:.2f} -> exit={pos.current_price:.2f} | "
            f"pnl={pnl:+,.2f} ({pnl_pct:+.2f}%) | "
            f"brokerage={brokerage:.2f} | "
            f"held={pos.minutes_held}min"
        )

        # Update DB
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE intraday_trades
                    SET status = 'CLOSED',
                        exit_price = ?,
                        close_reason = ?,
                        closed_at = ?,
                        minutes_held = ?,
                        realized_pnl = ?,
                        realized_pnl_pct = ?,
                        brokerage_cost = ?,
                        net_pnl = ?
                    WHERE symbol = ? AND status = 'OPEN'
                    ORDER BY opened_at DESC LIMIT 1
                    """,
                    (
                        pos.current_price,
                        reason,
                        datetime.now().isoformat(),
                        pos.minutes_held,
                        pnl,
                        pnl_pct,
                        brokerage,
                        pnl - brokerage,
                        symbol,
                    ),
                )
                conn.commit()
        except Exception as e:
            log.error(f"Failed to update intraday_trades for {symbol}: {e}")

        del self._positions[symbol]

    def _force_exit_all(self, reason: str) -> None:
        """Close all open intraday positions."""
        if not self._positions:
            return

        log.warning(
            f"Force exiting all positions ({len(self._positions)}) | "
            f"reason={reason}"
        )
        for symbol in list(self._positions.keys()):
            try:
                # Refresh price before closing
                current_price = self.feed.get_current_price(symbol)
                if current_price > 0:
                    self._positions[symbol].current_price = current_price
                self._close_position(symbol, reason)
            except Exception as e:
                log.error(f"Force exit failed for {symbol}: {e}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_or_queue(
        self,
        signal: IntradaySignal,
        sized,
        xgboost_prob: Optional[float] = None,
    ) -> None:
        """
        Execute the trade (paper or live) and track the position.
        If approval_required, log it but don't execute.
        """
        dt_cfg = self.config.day_trading
        symbol = signal.symbol

        if self.config.user.approval_required and not dt_cfg.paper_trading:
            log.info(
                f"APPROVAL REQUIRED — {signal.action} {symbol} "
                f"qty={sized.qty} @ {signal.entry_price:.2f} "
                f"(signal_strength={signal.strength:.2f})"
            )
            self._log_intraday_trade(
                symbol=symbol,
                action=signal.action,
                qty=sized.qty,
                price=signal.entry_price,
                stop=signal.stop_loss_price,
                target=signal.target_price,
                strategy=signal.strategy_name,
                signal_strength=signal.strength,
                xgboost_prob=xgboost_prob,
                status="PENDING_APPROVAL",
            )
            return

        # Paper trading: simulate fill at signal price
        side = "long" if signal.action == "BUY" else "short"

        log.info(
            f"{'[PAPER] ' if dt_cfg.paper_trading else ''}"
            f"OPENING {signal.action} {symbol} | "
            f"qty={sized.qty} @ {signal.entry_price:.2f} | "
            f"SL={signal.stop_loss_price:.2f} | "
            f"target={signal.target_price:.2f} | "
            f"strategy={signal.strategy_name} | "
            f"strength={signal.strength:.2f}"
        )

        # Create position tracker
        position = IntradayPosition(
            symbol=symbol,
            entry_price=signal.entry_price,
            entry_time=datetime.now(),
            qty=sized.qty,
            side=side,
            stop_loss_price=signal.stop_loss_price,
            target_price=signal.target_price,
            current_price=signal.entry_price,
            unrealized_pnl=0.0,
            minutes_held=0,
            strategy_name=signal.strategy_name,
        )
        self._positions[symbol] = position
        self.risk_manager.record_position_open()

        # Log to DB
        self._log_intraday_trade(
            symbol=symbol,
            action=signal.action,
            qty=sized.qty,
            price=signal.entry_price,
            stop=signal.stop_loss_price,
            target=signal.target_price,
            strategy=signal.strategy_name,
            signal_strength=signal.strength,
            xgboost_prob=xgboost_prob,
        )

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def _get_watchlist(self) -> list[str]:
        """Return the active watchlist for scanning."""
        return self._watchlist

    # ------------------------------------------------------------------
    # Database logging
    # ------------------------------------------------------------------

    def _log_intraday_trade(
        self,
        symbol: str,
        action: str,
        qty: int,
        price: float,
        stop: float,
        target: float,
        strategy: str,
        signal_strength: float,
        xgboost_prob: Optional[float] = None,
        status: str = "OPEN",
    ) -> None:
        """INSERT a row into the intraday_trades table."""
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO intraday_trades (
                        symbol, action, qty, entry_price, stop_loss_price,
                        target_price, strategy_name, status,
                        paper_trading, xgboost_prob, signal_strength,
                        trading_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        action,
                        qty,
                        price,
                        stop,
                        target,
                        strategy,
                        status,
                        1 if self.config.day_trading.paper_trading else 0,
                        xgboost_prob,
                        signal_strength,
                        date.today().isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.error(f"Failed to log intraday trade for {symbol}: {e}")

    def _log_intraday_signal(self, signal: IntradaySignal) -> None:
        """INSERT a row into the intraday_signals table."""
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO intraday_signals (
                        symbol, strategy_name, action, strength,
                        entry_price, stop_loss_price, target_price,
                        vwap, rsi, volume_ratio,
                        trading_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.symbol,
                        signal.strategy_name,
                        signal.action,
                        signal.strength,
                        signal.entry_price,
                        signal.stop_loss_price,
                        signal.target_price,
                        signal.vwap,
                        0.0,  # RSI from technicals not stored on signal
                        signal.volume_ratio,
                        date.today().isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.error(f"Failed to log intraday signal for {signal.symbol}: {e}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Gracefully stop the trading loop."""
        log.info("Stopping intraday trading loop...")
        self._running = False
        self._force_exit_all("System shutdown")

        summary = self.risk_manager.get_daily_summary()
        log.info(
            f"Daily summary | trades={summary.total_trades} | "
            f"win_rate={summary.win_rate:.1%} | "
            f"pnl={summary.total_pnl:+,.2f} | "
            f"net_pnl={summary.net_pnl:+,.2f} | "
            f"max_drawdown={summary.max_drawdown_intraday:,.2f}"
        )
        log.info("Intraday trading loop stopped")
