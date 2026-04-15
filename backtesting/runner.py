"""
Backtest runner — orchestrates strategy backtesting with data fetching.
Provides a simple interface to run backtests from the CLI or dashboard.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from config.settings import TradingSystemConfig
from config.watchlists import BENCHMARK
from data.price_feed import PriceFeed
from prediction.ensemble_model import EnsemblePredictor
from prediction.feature_engineer import FeatureEngineer
from core.logger import get_logger

log = get_logger("backtest_runner")


class BacktestRunner:
    """
    High-level backtest runner that combines data fetching,
    signal generation, and backtesting.
    """

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.price_feed = PriceFeed(exchange=config.market.exchange)
        self.feature_engineer = FeatureEngineer()
        self.predictor = EnsemblePredictor(target_days=5)

    def run_ml_backtest(
        self,
        symbols: list[str],
        period: str = "2y",
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run a backtest using the ML ensemble predictor.
        Generates BUY signals when ML model predicts upward movement
        with sufficient confidence.
        """
        capital = initial_capital or self.config.user.investment_amount
        engine = BacktestEngine(
            initial_capital=capital,
            exchange=self.config.market.exchange,
            max_position_pct=self.config.user.max_position_pct,
            risk_per_trade_pct=self.config.risk.risk_per_trade_pct,
        )

        # Fetch price data for all symbols
        log.info(f"Fetching data for {len(symbols)} symbols...")
        price_data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.price_feed.get_historical(symbol, period=period, interval="1d")
                if len(df) >= 100:
                    price_data[symbol] = df
            except Exception as e:
                log.warning(f"Skipping {symbol}: {e}")

        if not price_data:
            raise ValueError("No valid price data fetched")

        log.info(f"Loaded data for {len(price_data)} symbols")

        # Fetch benchmark
        benchmark_symbol = BENCHMARK.get(self.config.market.exchange, "SPY")
        benchmark_df = None
        try:
            benchmark_df = self.price_feed.get_historical(
                benchmark_symbol, period=period, interval="1d"
            )
        except Exception:
            pass

        # Generate signals using ML model
        log.info("Generating ML signals...")
        all_signals = []

        for symbol, df in price_data.items():
            try:
                features = self.feature_engineer.build_features(
                    df, target_days=5, include_target=True
                )
                if len(features) < 100:
                    continue

                # Walk-forward: train on first 60%, generate signals on remaining 40%
                train_end = int(len(features) * 0.6)
                train_features = features.iloc[:train_end]
                test_features = features.iloc[train_end:]

                target_cols = ["target", "target_return"]
                feature_cols = [c for c in features.columns if c not in target_cols]

                X_train = train_features[feature_cols].replace(
                    [float('inf'), float('-inf')], 0
                ).fillna(0)
                y_train = train_features["target"]

                # Train a simple model for this backtest
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=100, max_depth=8,
                        min_samples_leaf=20, random_state=42, n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                except ImportError:
                    log.warning("scikit-learn not available, skipping ML signals")
                    continue

                # Generate signals on test period
                X_test = test_features[feature_cols].replace(
                    [float('inf'), float('-inf')], 0
                ).fillna(0)
                probabilities = model.predict_proba(X_test)[:, 1]

                for i, (idx, prob) in enumerate(zip(test_features.index, probabilities)):
                    if prob > 0.60:  # High confidence buy signal
                        all_signals.append({
                            "date": idx.date(),
                            "symbol": symbol,
                            "action": "BUY",
                            "stop_loss_pct": 0.05,
                            "take_profit_pct": 0.10,
                            "conviction": int(prob * 10),
                        })

            except Exception as e:
                log.warning(f"Signal generation failed for {symbol}: {e}")

        if not all_signals:
            log.warning("No signals generated — backtest will have no trades")

        signals_df = pd.DataFrame(all_signals)
        log.info(f"Generated {len(all_signals)} BUY signals across {len(price_data)} symbols")

        # Run backtest
        result = engine.run(
            price_data=price_data,
            signals=signals_df,
            benchmark_df=benchmark_df,
            strategy_name="ML Ensemble (Walk-Forward)",
        )

        self._print_summary(result)
        return result

    def run_technical_backtest(
        self,
        symbols: list[str],
        period: str = "2y",
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run a backtest using pure technical analysis signals.
        BUY when RSI < 35 (oversold) + MACD bullish crossover + price above SMA50.
        """
        capital = initial_capital or self.config.user.investment_amount
        engine = BacktestEngine(
            initial_capital=capital,
            exchange=self.config.market.exchange,
            max_position_pct=self.config.user.max_position_pct,
            risk_per_trade_pct=self.config.risk.risk_per_trade_pct,
        )

        # Fetch data
        price_data = {}
        for symbol in symbols:
            try:
                df = self.price_feed.get_historical(symbol, period=period, interval="1d")
                if len(df) >= 50:
                    price_data[symbol] = df
            except Exception:
                pass

        benchmark_symbol = BENCHMARK.get(self.config.market.exchange, "SPY")
        benchmark_df = self.price_feed.get_historical(
            benchmark_symbol, period=period, interval="1d"
        ) if benchmark_symbol else None

        # Generate technical signals
        all_signals = []
        for symbol, df in price_data.items():
            signals = self._generate_technical_signals(symbol, df)
            all_signals.extend(signals)

        signals_df = pd.DataFrame(all_signals) if all_signals else pd.DataFrame()
        log.info(f"Generated {len(all_signals)} technical signals")

        result = engine.run(
            price_data=price_data,
            signals=signals_df,
            benchmark_df=benchmark_df,
            strategy_name="Technical Analysis (RSI+MACD+Trend)",
        )

        self._print_summary(result)
        return result

    def _generate_technical_signals(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Generate buy signals from technical indicators."""
        close = df["close"]
        signals = []

        # Compute indicators
        sma50 = close.rolling(50).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal_line

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - 100 / (1 + rs)

        atr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1).rolling(14).mean()

        for i in range(52, len(df)):
            price = float(close.iloc[i])
            current_rsi = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50
            current_hist = float(histogram.iloc[i]) if not pd.isna(histogram.iloc[i]) else 0
            prev_hist = float(histogram.iloc[i - 1]) if not pd.isna(histogram.iloc[i - 1]) else 0
            current_sma50 = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else price
            current_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else price * 0.02

            # BUY signal: RSI oversold bounce + MACD bullish crossover + above SMA50
            macd_cross = prev_hist <= 0 and current_hist > 0
            oversold_bounce = current_rsi < 40 and current_rsi > 25
            uptrend = price > current_sma50

            if macd_cross and oversold_bounce and uptrend:
                stop_pct = min(0.08, max(0.03, (current_atr * 2) / price))
                signals.append({
                    "date": df.index[i].date(),
                    "symbol": symbol,
                    "action": "BUY",
                    "stop_loss_pct": round(stop_pct, 4),
                    "take_profit_pct": round(stop_pct * 2.5, 4),  # 2.5:1 reward:risk
                    "conviction": 7,
                })

        return signals

    def _print_summary(self, result: BacktestResult):
        """Print a formatted backtest summary."""
        print("\n" + "=" * 70)
        print(f"  BACKTEST RESULTS: {result.strategy_name}")
        print("=" * 70)
        print(f"  Period:        {result.start_date} to {result.end_date}")
        print(f"  Capital:       {result.initial_capital:,.0f} -> {result.final_capital:,.0f}")
        print(f"  Total Return:  {result.total_return_pct:+.2f}%")
        print(f"  Annual Return: {result.annualized_return_pct:+.2f}%")
        print(f"  Benchmark:     {result.benchmark_return_pct:+.2f}%")
        print(f"  Alpha:         {result.alpha:+.2f}%")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown_pct:.2f}%")
        print(f"  Volatility:    {result.volatility_annualized:.2f}%")
        print()
        print(f"  Total Trades:  {result.total_trades}")
        print(f"  Win Rate:      {result.win_rate:.1%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Avg Win:       {result.avg_win_pct:+.2f}%")
        print(f"  Avg Loss:      {result.avg_loss_pct:+.2f}%")
        print(f"  Avg Holding:   {result.avg_holding_days:.1f} days")
        print(f"  Max Consec L:  {result.max_consecutive_losses}")
        print(f"  Total Costs:   {result.total_costs:,.2f}")
        print("=" * 70 + "\n")
