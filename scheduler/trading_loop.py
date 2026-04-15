"""
Trading loop — the main orchestrator that runs the full pipeline every N minutes.
This is the heartbeat of the entire system.

Full cycle:
  1. Refresh market context (regime, VIX)
  2. Check circuit breakers
  3. Monitor existing positions (stops, targets)
  4. ML market prediction
  5. Scan universe for candidates
  6. Run Buffett value screen (optional)
  7. Build analysis bundles
  8. Strategy consensus signals
  9. LLM analyst call for top candidates
  10. LLM risk reviewer for actionable recommendations
  11. Portfolio diversification check
  12. Size trades
  13. Queue for approval or auto-execute
  14. Save portfolio snapshot
"""
from __future__ import annotations

import time as _time
from datetime import datetime
from typing import Optional

from analysis.aggregator import AnalysisAggregator, AnalysisBundle, CurrentPosition
from config.settings import TradingSystemConfig
from core.circuit_breaker import CircuitBreaker
from core.exceptions import CircuitBreakerError
from core.logger import get_logger
from data.earnings_calendar import EarningsCalendar
from data.market_scanner import MarketScanner
from data.vix_monitor import VixMonitor
from database.manager import DatabaseManager
from database.repository import TradeRepository
from database.performance_tracker import PerformanceTracker
from execution.broker_client import BrokerClient, BrokerPosition
from execution.order_manager import OrderManager
from execution.trade_executor import TradeExecutor
from llm.analyst import LLMAnalyst
from llm.client import ClaudeClient
from llm.response_parser import TradeRecommendation
from llm.risk_evaluator import LLMRiskEvaluator
from notifications.alert_manager import AlertManager
from portfolio.manager import PortfolioManager
from prediction.market_predictor import MarketPredictor
from core.kill_switch import KillSwitch, KillSwitchError
from risk.drawdown_tracker import DrawdownTracker
from risk.portfolio_heat import PortfolioHeatMonitor
from risk.position_sizer import PositionSizer, SizedTrade
from risk.stop_loss_manager import StopLossManager
from scheduler.market_hours import MarketHours
from strategy.selector import StrategySelector

log = get_logger("trading_loop")

MAX_LLM_CALLS_PER_CYCLE = 10  # Cost control


class TradingLoop:

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self._running = False

        # Initialize all components
        log.info("Initializing AI Trading System...")

        self.db = DatabaseManager(config.database.path)
        self.repository = TradeRepository(self.db)
        self.performance = PerformanceTracker(self.db)

        self.broker: BrokerClient = self._init_broker()
        self.order_manager = OrderManager(self.broker)
        self.executor = TradeExecutor(
            self.broker, self.order_manager, self.repository,
            paper_trading=config.user.paper_trading,
        )

        self.llm_client = ClaudeClient(config.anthropic)
        self.analyst = LLMAnalyst(self.llm_client)
        self.risk_evaluator = LLMRiskEvaluator(self.llm_client)

        self.aggregator = AnalysisAggregator(config)
        self.scanner = MarketScanner(config)
        self.earnings_cal = EarningsCalendar(config.trading.earnings_blackout_days)
        self.vix_monitor = VixMonitor(config.vix, exchange=config.market.exchange)

        self.circuit_breaker = CircuitBreaker(config)
        self.heat_monitor = PortfolioHeatMonitor(config.user.max_portfolio_heat)
        self.stop_loss_mgr = StopLossManager(config.trading.atr_stop_multiplier)
        self.position_sizer = PositionSizer(config)

        self.market_hours = MarketHours(config.market.exchange)
        self.alert_manager = AlertManager(config)

        # New modules
        self.portfolio_manager = PortfolioManager(config)
        self.strategy_selector = StrategySelector()
        self.market_predictor = MarketPredictor(exchange=config.market.exchange)

        # Kill switch and drawdown tracker
        self.kill_switch = KillSwitch(
            broker_client=self.broker,
            circuit_breaker=self.circuit_breaker,
            alert_manager=self.alert_manager,
            trade_repo=self.repository,
        )
        self.drawdown_tracker = DrawdownTracker()

        # Buffett screener (lazy init)
        self._value_screener = None

        log.info("Trading system initialized successfully (with ML + Strategy Engine + Kill Switch)")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Run the trading loop continuously. Block indefinitely."""
        self._running = True
        log.info(
            f"Starting trading loop | exchange={self.config.market.exchange} | "
            f"{'PAPER' if self.config.user.paper_trading else 'LIVE'} mode | "
            f"interval={self.config.trading.scan_interval_minutes}m"
        )

        while self._running:
            try:
                self.run_cycle()
            except KillSwitchError as e:
                log.critical(f"Kill switch activated — stopping trading loop: {e}")
                self.alert_manager.send_critical(f"Kill switch: {e}")
                self._running = False
                break
            except Exception as e:
                log.exception(f"Trading loop error: {e}")
                self.alert_manager.send_critical(f"Trading loop error: {e}")

            interval = self.config.trading.scan_interval_minutes * 60
            log.info(f"Cycle complete. Next cycle in {self.config.trading.scan_interval_minutes}m")
            _time.sleep(interval)

    def run_cycle(self) -> None:
        """Execute a single full analysis and trading cycle."""
        cycle_start = datetime.now()
        log.info(f"=== Trading Cycle Start: {cycle_start.strftime('%H:%M:%S')} ===")

        # ── Step 1: Market hours check ────────────────────────────────────
        if not self.market_hours.is_open():
            log.info(
                f"Market closed. Next open: {self.market_hours.next_market_open_str()}"
            )
            return

        # ── Step 1b: Kill switch check ────────────────────────────────────
        if self.kill_switch.is_active():
            log.warning("Kill switch is active — skipping cycle")
            return

        # ── Step 2: Refresh portfolio state ──────────────────────────────
        account = self._safe_get_account()
        broker_positions = self.order_manager.reconcile_positions()
        self.circuit_breaker.set_open_positions(len(broker_positions))
        self.circuit_breaker.update_daily_pnl(account.daily_pnl_pct if account else 0)

        # Kill switch: check for catastrophic daily loss
        if account:
            self.kill_switch.check_daily_loss(account.daily_pnl_pct)

        # Drawdown tracker: update and check position size multiplier
        drawdown_mult = 1.0
        if account:
            drawdown_mult = self.drawdown_tracker.update(account.portfolio_value)
            if drawdown_mult == 0.0:
                log.warning(
                    f"Drawdown halt: {self.drawdown_tracker.drawdown_pct:.1%} "
                    f"from peak — no new entries this cycle"
                )
                return

        # ── Step 3: Refresh market context ────────────────────────────────
        self.aggregator.refresh_market_context()
        vix = self.vix_monitor.get()
        self.circuit_breaker.update_vix(vix.level)
        market_regime = (
            self.aggregator._cached_regime.regime.value
            if self.aggregator._cached_regime else "SIDEWAYS"
        )

        # Update portfolio heat from current positions
        self._refresh_portfolio_heat(broker_positions, account)

        # ── Step 4: Monitor existing positions ────────────────────────────
        self._monitor_positions(broker_positions)

        # ── Step 5: Circuit breaker check ────────────────────────────────
        try:
            ok, reason = self.circuit_breaker.check_all()
            if not ok:
                log.warning(f"Circuit breaker: {reason} — skipping new trades this cycle")
                return
        except CircuitBreakerError as e:
            log.warning(f"Circuit breaker triggered: {e.reason}")
            return

        # ── Step 6: ML market prediction ──────────────────────────────────
        ml_market = None
        if self.config.ml.enabled:
            try:
                ml_market = self.market_predictor.predict_market()
                log.info(
                    f"ML Market Prediction: {ml_market.direction} "
                    f"(prob={ml_market.probability:.2f}, "
                    f"exposure={ml_market.recommended_exposure:.0%})"
                )
            except Exception as e:
                log.warning(f"ML market prediction failed: {e}")

        # ── Step 7: Update earnings blackout symbols ───────────────────────
        all_symbols = [r.symbol for r in self.scanner.scan()[:20]]
        blackout_symbols = self.earnings_cal.get_blackout_symbols(all_symbols)
        self.circuit_breaker.set_earnings_blackout(blackout_symbols)

        # ── Step 8: Full market scan ──────────────────────────────────────
        scan_results = self.scanner.scan()
        log.info(f"Scan found {len(scan_results)} candidates")

        if not scan_results:
            log.info("No candidates passed pre-filter this cycle")
            return

        # ── Step 9: Build analysis bundles ────────────────────────────────
        bundles: list[AnalysisBundle] = []
        for scan_result in scan_results[:20]:
            try:
                position_state = self._get_current_position(
                    scan_result.symbol, broker_positions
                )
                bundle = self.aggregator.build_bundle(
                    scan_result.symbol, position_state
                )
                if bundle.composite_score >= self.config.trading.min_composite_score:
                    bundles.append(bundle)
                else:
                    log.debug(
                        f"Skipping {scan_result.symbol}: "
                        f"composite {bundle.composite_score:.0f} < "
                        f"{self.config.trading.min_composite_score}"
                    )
            except Exception as e:
                log.warning(f"Bundle build failed for {scan_result.symbol}: {e}")

        log.info(f"{len(bundles)} bundles passed composite score filter")

        # ── Step 10: Strategy consensus + ML prediction per stock ─────────
        strategy_filtered: list[tuple[AnalysisBundle, dict]] = []
        for bundle in bundles:
            try:
                # Get ML prediction for this stock
                ml_pred = None
                if self.config.ml.enabled and self.config.strategy.enable_ml_boost:
                    try:
                        df = self.aggregator.price_feed.get_historical(
                            bundle.symbol, period="2y", interval="1d"
                        )
                        ml_pred = self.market_predictor.predict_stock(bundle.symbol, df)
                    except Exception:
                        pass

                # Get strategy consensus
                vote = self.strategy_selector.generate_combined_signal(
                    symbol=bundle.symbol,
                    df=self.aggregator.price_feed.get_historical(
                        bundle.symbol, period="1y", interval="1d"
                    ),
                    market_regime=market_regime,
                    ml_prediction=ml_pred,
                )

                if vote.consensus_action == "BUY" and vote.consensus_strength >= self.config.strategy.min_consensus_strength:
                    strategy_filtered.append((bundle, {
                        "vote": vote,
                        "ml_pred": ml_pred,
                    }))
                    log.info(
                        f"{bundle.symbol}: Strategy BUY "
                        f"(strength={vote.consensus_strength:.2f}, "
                        f"strategies={vote.active_strategies})"
                    )

            except Exception as e:
                log.warning(f"Strategy evaluation failed for {bundle.symbol}: {e}")

        log.info(f"{len(strategy_filtered)} stocks passed strategy consensus filter")

        # ── Step 11: LLM analysis (rate-limited) ─────────────────────────
        llm_calls = 0
        actionable_trades: list[tuple[AnalysisBundle, TradeRecommendation, dict]] = []

        for bundle, extra in strategy_filtered:
            if llm_calls >= MAX_LLM_CALLS_PER_CYCLE:
                log.info(f"LLM call limit ({MAX_LLM_CALLS_PER_CYCLE}) reached for this cycle")
                break

            cb_ok, cb_reason = self.circuit_breaker.check_symbol_only(bundle.symbol)
            if not cb_ok:
                log.info(f"Skipping {bundle.symbol}: {cb_reason}")
                continue

            try:
                recommendation = self.analyst.analyze(
                    bundle=bundle,
                    portfolio_heat_pct=self.heat_monitor.total_heat(),
                    max_heat_pct=self.config.user.max_portfolio_heat,
                    cash_available=account.cash if account else 0,
                    open_positions=len(broker_positions),
                    max_positions=self.config.trading.max_open_positions,
                    risk_tolerance=self.config.user.risk_tolerance,
                )
                llm_calls += 1

                self.repository.log_analysis(
                    symbol=bundle.symbol,
                    action=recommendation.action,
                    conviction=recommendation.conviction,
                    composite_score=bundle.composite_score,
                    technical_score=bundle.technical_score,
                    fundamental_score=bundle.fundamental_score,
                    sentiment_score=bundle.sentiment_score_normalized,
                    market_regime=market_regime,
                    vix_level=bundle.vix.level,
                    thesis=recommendation.primary_thesis,
                    risks=recommendation.key_risks,
                )

                if (
                    recommendation.is_actionable
                    and recommendation.conviction >= self.config.trading.min_conviction_execute
                ):
                    actionable_trades.append((bundle, recommendation, extra))

            except Exception as e:
                log.warning(f"LLM analysis failed for {bundle.symbol}: {e}")

        log.info(f"LLM produced {len(actionable_trades)} actionable recommendations")

        # ── Step 12: Risk review + portfolio diversification check ─────────
        approved_trades: list[SizedTrade] = []

        for bundle, recommendation, extra in actionable_trades:
            try:
                risk_review = self.risk_evaluator.review(
                    bundle=bundle,
                    recommendation=recommendation,
                    portfolio_heat_pct=self.heat_monitor.total_heat(),
                    max_heat_pct=self.config.user.max_portfolio_heat,
                    open_positions=len(broker_positions),
                    daily_pnl_pct=account.daily_pnl_pct if account else 0,
                )
                llm_calls += 1

                if not risk_review.approve:
                    log.info(
                        f"{bundle.symbol} rejected by risk reviewer: "
                        f"{risk_review.rejection_reason}"
                    )
                    continue

                # Portfolio diversification check
                current_pos = {
                    sym: {"value": p.market_value, "sector": ""}
                    for sym, p in broker_positions.items()
                }
                portfolio_value = account.portfolio_value if account else self.config.user.investment_amount
                position_value_est = portfolio_value * (recommendation.max_position_pct or 0.05)

                can_add, div_reason = self.portfolio_manager.can_add_position(
                    bundle.symbol, position_value_est, current_pos, portfolio_value
                )
                if not can_add:
                    log.info(f"{bundle.symbol} blocked by portfolio manager: {div_reason}")
                    continue

                # Size the trade
                sized = self.position_sizer.size_trade(
                    bundle=bundle,
                    recommendation=recommendation,
                    review=risk_review,
                    portfolio_value=portfolio_value,
                    cash_available=account.cash if account else self.config.user.trading_budget,
                    current_heat=self.heat_monitor.total_heat(),
                )

                if sized.is_valid:
                    approved_trades.append(sized)
                    currency = "₹" if self.config.market.exchange == "IN" else "$"
                    self.alert_manager.send_signal(
                        f"New signal: {sized.action} {bundle.symbol} | "
                        f"conviction={recommendation.conviction}/10 | "
                        f"{currency}{sized.position_value:,.0f}"
                    )

            except Exception as e:
                log.warning(f"Risk review failed for {bundle.symbol}: {e}")

        log.info(f"{len(approved_trades)} trades passed risk review and sizing")

        # ── Step 13: Execute or queue for approval ─────────────────────────
        for trade in approved_trades:
            try:
                if self.config.user.approval_required:
                    matching_bundle = next(
                        (b for b, _, _ in actionable_trades if b.symbol == trade.symbol),
                        None
                    )
                    scores = {}
                    if matching_bundle:
                        scores = {
                            "technical": matching_bundle.technical_score,
                            "fundamental": matching_bundle.fundamental_score,
                            "composite": matching_bundle.composite_score,
                        }
                    approval_id = self.repository.add_to_approval_queue(trade, 5, scores)
                    log.info(
                        f"Queued for approval: {trade.symbol} | "
                        f"approval_id={approval_id}"
                    )
                    currency = "₹" if self.config.market.exchange == "IN" else "$"
                    self.alert_manager.send_high(
                        f"APPROVAL NEEDED: {trade.action} {trade.symbol} | "
                        f"conviction={trade.conviction}/10 | "
                        f"{currency}{trade.position_value:,.0f} | "
                        f"stop={currency}{trade.stop_loss_price:.2f}"
                    )
                else:
                    if trade.conviction >= self.config.trading.min_conviction_auto:
                        record = self.executor.execute(trade)
                        if record:
                            self.circuit_breaker.increment_trades()
                            self.alert_manager.send_trade_executed(trade)
                    else:
                        self.repository.add_to_approval_queue(trade, 5, {})

            except Exception as e:
                log.error(f"Trade queue/execution failed for {trade.symbol}: {e}")

        # ── Step 14: Check for approved trades in queue ───────────────────
        self._process_approved_queue(account)

        # ── Step 15: Save portfolio snapshot ──────────────────────────────
        if account:
            invested = account.portfolio_value - account.cash
            self.repository.save_portfolio_snapshot({
                "total_value": account.portfolio_value,
                "cash": account.cash,
                "invested_value": invested,
                "daily_pnl": account.daily_pnl,
                "daily_pnl_pct": account.daily_pnl_pct,
                "open_positions": len(broker_positions),
                "portfolio_heat": self.heat_monitor.total_heat(),
                "vix_level": vix.level,
                "market_regime": market_regime,
            })

        elapsed = (datetime.now() - cycle_start).total_seconds()
        log.info(
            f"=== Cycle Complete in {elapsed:.1f}s | "
            f"LLM calls={llm_calls} | "
            f"Cost=${self.llm_client.get_usage_summary()['daily_cost_usd']:.3f} today ==="
        )

    # ── Helper methods ─────────────────────────────────────────────────────

    def _init_broker(self) -> BrokerClient:
        from execution.broker_client import create_broker_client
        return create_broker_client(self.config)

    def _safe_get_account(self):
        try:
            return self.broker.get_account()
        except Exception as e:
            log.warning(f"Could not get account info: {e}")
            return None

    def _refresh_portfolio_heat(self, positions: dict, account) -> None:
        if not account:
            return
        self.heat_monitor.clear()
        for symbol, pos in positions.items():
            stop_state = self.stop_loss_mgr.get_state(symbol)
            stop_pct = 0.05
            if stop_state:
                stop_pct = (pos.avg_cost - stop_state.current_stop) / pos.avg_cost
            self.heat_monitor.update_position(
                symbol=symbol,
                position_value=pos.market_value,
                stop_loss_pct=stop_pct,
                portfolio_value=account.portfolio_value,
            )

    def _monitor_positions(self, broker_positions: dict) -> None:
        for symbol, pos in broker_positions.items():
            try:
                should_exit, reason, fraction = self.stop_loss_mgr.update_price(
                    symbol, pos.current_price
                )
                if should_exit:
                    if fraction is None:
                        success = self.executor.close_position(symbol, reason)
                        if success:
                            self.heat_monitor.remove_position(symbol)
                            self.alert_manager.send_high(
                                f"EXIT: {symbol} | {reason} | "
                                f"P&L: {pos.unrealized_pnl_pct:+.1%}"
                            )
                    else:
                        shares_to_sell = pos.qty * fraction
                        self.executor.close_partial(symbol, shares_to_sell, reason)
            except Exception as e:
                log.warning(f"Position monitoring error for {symbol}: {e}")

    def _get_current_position(
        self, symbol: str, broker_positions: dict
    ) -> Optional[CurrentPosition]:
        pos: Optional[BrokerPosition] = broker_positions.get(symbol)
        if not pos:
            return None
        stop_state = self.stop_loss_mgr.get_state(symbol)
        return CurrentPosition(
            shares=pos.qty,
            avg_cost=pos.avg_cost,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            days_held=stop_state.days_held if stop_state else 0,
            stop_loss_price=stop_state.current_stop if stop_state else pos.avg_cost * 0.95,
            take_profit_price=(
                stop_state.take_profit_targets[0] if stop_state and stop_state.take_profit_targets
                else pos.avg_cost * 1.05
            ),
        )

    def _process_approved_queue(self, account) -> None:
        pending = self.repository.get_pending_approvals()
        for item in pending:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT status FROM approval_queue WHERE id=?", (item["id"],)
                ).fetchone()
                if row and row["status"] == "APPROVED":
                    from risk.position_sizer import SizedTrade
                    import json
                    tp_prices = json.loads(item.get("take_profit_prices") or "[]")
                    trade = SizedTrade(
                        symbol=item["symbol"],
                        action=item["action"],
                        shares=item["shares"],
                        entry_price=item["entry_price"],
                        position_value=item["position_value"],
                        stop_loss_price=item["stop_loss_price"],
                        stop_loss_pct=item.get("stop_loss_pct", 0.05),
                        take_profit_prices=tp_prices,
                        take_profit_fractions=[1.0 / len(tp_prices)] * len(tp_prices) if tp_prices else [],
                        portfolio_heat_contribution=item.get("portfolio_heat_add", 0),
                        is_valid=True,
                        conviction=item.get("conviction", 7),
                        primary_thesis=item.get("primary_thesis", ""),
                        key_risks=json.loads(item.get("key_risks") or "[]"),
                    )
                    record = self.executor.execute(trade)
                    if record:
                        self.circuit_breaker.increment_trades()
                        conn.execute(
                            "UPDATE approval_queue SET status='EXECUTED' WHERE id=?",
                            (item["id"],)
                        )
                        conn.commit()
                        log.info(f"Executed approved trade: {item['symbol']}")

    def stop(self) -> None:
        self._running = False
        log.info("Trading loop stopped")
