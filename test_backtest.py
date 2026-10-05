import io
import json
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from backtest import (
    BacktestSettings, HistoricalCandle, build_parser, download_candles,
    generic_symbol_info, load_csv, main, prepare_symbol_info, render_csv,
    report_json, run_backtest, utc, validate_candles, parse_time,
)
from trader import StrategyConfig, build_parser as live_parser, strategy_config_from_args


EPOCH = 1_700_000_000_000
INTERVAL = 900_000
D = Decimal


def strategy(**changes):
    values = dict(
        fast_sma=2, slow_sma=3, rsi_period=2,
        buy_on_bullish_trend=False, sell_on_bearish_trend=False,
        buy_below=105, sell_above=None, buy_rsi_below=None, sell_rsi_above=None,
        stop_loss_pct=2, take_profit_pct=4, buy_rsi_min=0, buy_rsi_max=100,
        buy_crossover_lookback_candles=3, cooldown_candles=0, stop_cooldown_candles=0,
    )
    values.update(changes)
    return StrategyConfig(**values)


def settings(**changes):
    return replace(BacktestSettings(fee_pct=D(0), slippage_pct=D(0)), **changes)


def candle(index, opened=100, high=100, low=100, closed=100):
    return HistoricalCandle(
        EPOCH + index * INTERVAL, EPOCH + (index + 1) * INTERVAL - 1,
        D(str(opened)), D(str(high)), D(str(low)), D(str(closed)), D(10),
    )


def series(*bars):
    return [candle(index) for index in range(4)] + [
        candle(index + 4, *bar) for index, bar in enumerate(bars)
    ]


def filters(step="0.001", minimum="0"):
    info = generic_symbol_info("BTCUSDT")
    info["filters"] = [
        {"filterType": "LOT_SIZE", "stepSize": step, "minQty": step},
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        {"filterType": "MIN_NOTIONAL", "minNotional": minimum, "applyToMarket": True, "avgPriceMins": 0},
    ]
    return info


class SimulationTests(unittest.TestCase):
    def test_market_entry_uses_next_open_not_signal_close_or_future_close(self):
        data = series((120, 1000, 119, 1000))
        report = run_backtest(data, strategy(stop_loss_pct=0), settings(), filters())
        fill = report["fills"][0]
        self.assertEqual(fill["side"], "BUY")
        self.assertEqual(fill["average_price"], 120)
        self.assertNotEqual(fill["timestamp_utc"], fill["signal_candle_close_time_utc"])
        self.assertEqual(report["summary"]["sell_executions"], 0)

    def test_appending_future_data_does_not_change_earlier_fills(self):
        prefix = series((100, 100, 100, 100), (100, 100, 100, 100))
        first = run_backtest(prefix, strategy(), settings(), filters())
        extended = prefix + [candle(6, 100, 200, 100, 200)]
        second = run_backtest(extended, strategy(), settings(), filters())
        self.assertEqual(second["fills"][:len(first["fills"])], first["fills"])

    def test_hosted_stop_can_fill_in_entry_candle(self):
        report = run_backtest(series((100, 105, 97, 104)), strategy(), settings(), filters())
        self.assertEqual([fill["side"] for fill in report["fills"]], ["BUY", "SELL"])
        stop = report["fills"][1]
        self.assertEqual(stop["average_price"], 98)
        self.assertEqual(stop["source"], "hosted_stop_loss")
        self.assertEqual(report["summary"]["net_profit_quote"], -0.2)
        self.assertEqual(report["summary"]["max_drawdown_pct"], 0.2)

    def test_stop_gap_precedes_pending_take_profit_and_fills_at_open(self):
        report = run_backtest(
            series((100, 106, 100, 105), (90, 91, 89, 90)),
            strategy(), settings(), filters(),
        )
        exit_fill = report["fills"][1]
        self.assertEqual(exit_fill["average_price"], 90)
        self.assertEqual(exit_fill["source"], "hosted_stop_loss")
        self.assertIn("gap", exit_fill["reasons"][0])

    def test_take_profit_is_close_based_and_executes_at_following_open(self):
        data = series((100, 110, 100, 101), (101, 110, 100, 105), (104, 105, 103, 104))
        report = run_backtest(data, strategy(trailing_thresholds=((50, 0),)), settings(), filters())
        self.assertEqual(len(report["fills"]), 2)
        self.assertEqual(report["fills"][1]["average_price"], 104)
        self.assertEqual(report["fills"][1]["source"], "strategy")
        self.assertIn("take profit 4%", report["fills"][1]["reasons"])

    def test_trailing_stop_does_not_use_entry_bar_high_or_future_close(self):
        report = run_backtest(
            series((100, 103, 99, 102), (102, 102, 100, 101)),
            strategy(take_profit_pct=0), settings(), filters(),
        )
        self.assertEqual(report["fills"][1]["average_price"], 101)
        self.assertEqual(report["fills"][1]["source"], "hosted_stop_loss")
        self.assertEqual(report["summary"]["closed_trades"], 1)

    def test_process_managed_stop_waits_for_close_and_next_open(self):
        data = series((100, 101, 90, 99), (99, 100, 90, 97), (96, 97, 95, 96))
        report = run_backtest(data, strategy(), settings(hosted_stop_loss=False), filters())
        self.assertEqual(report["fills"][1]["average_price"], 96)
        self.assertEqual(report["fills"][1]["source"], "strategy")
        self.assertIn("stop loss 2%", report["fills"][1]["reasons"])

    def test_commissions_are_charged_on_both_sides_and_dust_is_valued(self):
        report = run_backtest(series((100, 100, 100, 100)), strategy(), settings(fee_pct=D("0.1"), close_at_end=True), filters())
        summary = report["summary"]
        self.assertAlmostEqual(summary["fees_quote_equivalent"], 0.0199)
        self.assertAlmostEqual(summary["net_profit_quote"], -0.0199)
        self.assertEqual(D(report["inventory"]["quantity"]), D("0.0009"))
        self.assertAlmostEqual(summary["net_profit_quote"], summary["realized_pnl_quote"] + summary["unrealized_pnl_quote"])
        self.assertEqual(summary["closed_trades"], 1)

    def test_slippage_is_adverse_on_each_side(self):
        report = run_backtest(
            series((100, 100, 100, 100)), strategy(stop_loss_pct=0),
            settings(slippage_pct=D(1), close_at_end=True), filters(),
        )
        self.assertEqual([fill["average_price"] for fill in report["fills"]], [101, 99])
        self.assertLess(report["summary"]["net_profit_quote"], 0)

    def test_open_position_is_not_silently_liquidated_at_end(self):
        report = run_backtest(series((100, 101, 100, 101)), strategy(), settings(), filters())
        self.assertEqual(report["summary"]["closed_trades"], 0)
        self.assertEqual(report["summary"]["sell_executions"], 0)
        self.assertIsNotNone(report["open_position"])
        self.assertAlmostEqual(report["summary"]["final_equity_quote"], 100.1)
        self.assertIsNone(report["summary"]["win_rate_pct"])
        json.loads(report_json(report))  # No NaN/Infinity for undefined metrics.

    def test_normal_cooldown_counts_closes_after_actual_sale(self):
        data = series((100, 104, 100, 103), *((100, 104, 100, 100),) * 5)
        report = run_backtest(data, strategy(take_profit_pct=2, cooldown_candles=3, stop_cooldown_candles=6), settings(), filters())
        buys = [fill for fill in report["fills"] if fill["side"] == "BUY"]
        self.assertEqual(len(buys), 2)
        # SELL at bar 5 open; closes of bars 5, 6, 7 must pass before BUY at bar 8 open.
        self.assertEqual(buys[1]["timestamp_utc"], utc(EPOCH + 8 * INTERVAL))
        self.assertGreaterEqual(report["summary"]["blocked_orders"]["entry_guard"], 2)

    def test_stop_cooldown_fresh_crossover_and_dust_carry_are_simulated(self):
        following = [(price, price, price, price) for price in (99, 98, 97, 96, 95)]
        data = series((100, 100, 97, 100), *following, (95, 100, 95, 100), (100, 100, 100, 100))
        report = run_backtest(
            data, strategy(take_profit_pct=0, stop_cooldown_candles=6),
            settings(fee_pct=D("0.1")), filters(),
        )
        self.assertEqual([fill["side"] for fill in report["fills"]], ["BUY", "SELL", "BUY"])
        self.assertEqual(report["open_position"]["quantity"], 0.1)
        self.assertEqual(D(report["inventory"]["residual_quantity"]), D("0.0008"))
        self.assertAlmostEqual(report["summary"]["net_profit_quote"], report["summary"]["realized_pnl_quote"] + report["summary"]["unrealized_pnl_quote"])

    def test_cooldown_expiry_does_not_allow_post_stop_continuation(self):
        data = series((100, 100, 97, 100), *((100, 100, 100, 100),) * 8)
        report = run_backtest(data, strategy(), settings(), filters())
        self.assertEqual(report["summary"]["buy_executions"], 1)
        self.assertGreater(report["summary"]["blocked_orders"]["entry_guard"], 0)

    def test_no_negative_cash_or_borrowing_with_quote_commissions(self):
        report = run_backtest(
            series((100, 100, 100, 100)), strategy(),
            settings(initial_balance=D(10), fee_pct=D(1), buy_fee_asset="quote"), filters(),
        )
        self.assertEqual(report["summary"]["buy_executions"], 0)
        self.assertEqual(report["summary"]["final_cash_quote"], 10)
        self.assertEqual(report["summary"]["blocked_orders"]["insufficient_cash"], 1)

    def test_minimum_notional_and_protective_rounding_block_small_buys(self):
        report = run_backtest(series((100, 100, 100, 100)), strategy(), settings(quote_size=D(5)), filters(minimum="5"))
        self.assertEqual(report["summary"]["buy_executions"], 0)
        self.assertIn("unprotectable_or_undersized_buy", report["summary"]["blocked_orders"])

    def test_accepted_hosted_stop_executes_below_min_notional_after_gap(self):
        report = run_backtest(series((100, 100, 100, 100), (10, 11, 9, 10)), strategy(), settings(), filters(minimum="5"))
        self.assertEqual(report["summary"]["sell_executions"], 1)
        self.assertEqual(report["fills"][1]["average_price"], 10)

    def test_warmup_is_used_only_for_signals_and_requested_end_is_exclusive(self):
        data = series((100, 100, 100, 100), (100, 100, 100, 100))
        report = run_backtest(data, strategy(), settings(), filters(), start_ms=data[4].open_time_ms, end_ms=data[5].open_time_ms)
        self.assertEqual(report["warmup_candles"], 4)
        self.assertEqual(report["evaluated_candles"], 1)
        self.assertEqual(report["summary"]["buy_executions"], 1)
        with self.assertRaisesRegex(ValueError, "warm-up"):
            run_backtest(data, strategy(), settings(), filters(), start_ms=data[2].open_time_ms)

    def test_invalid_strategy_and_costs_are_rejected(self):
        data = series((100, 100, 100, 100))
        for invalid in (settings(fee_pct=D(100)), settings(slippage_pct=D("NaN")), settings(initial_balance=D(0))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run_backtest(data, strategy(), invalid, filters())
        with self.assertRaises(ValueError):
            run_backtest(data, strategy(take_profit_pct=float("nan")), settings(), filters())


class DataAndCliTests(unittest.TestCase):
    def test_csv_roundtrip_preserves_ohlcv_precision(self):
        data = [candle(0, "100.12345678", "101.12345678", "99.12345678", "100.12345678")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text(render_csv(data))
            self.assertEqual(load_csv(path, "15m", now_ms=EPOCH + 2 * INTERVAL), data)

    def test_csv_iso_utc_dates_and_unfinished_candle_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("open_time,open,high,low,close,volume\n2023-11-14T00:00:00Z,100,101,99,100,10\n2023-11-14T00:15:00Z,100,101,99,100,10\n")
            result = load_csv(path, "15m", now_ms=parse_time("2023-11-14T00:20:00Z"))
            self.assertEqual(len(result), 1)

    def test_invalid_ohlc_gaps_duplicate_bars_and_nan_are_rejected(self):
        for data in (
            [candle(0, 100, 99, 100, 100)], [candle(0), candle(2)],
            [candle(0), candle(0)], [candle(0, closed="NaN")],
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_candles(data, "15m")

    def test_public_download_paginates_and_discards_active_bar(self):
        rows = [[EPOCH + i * INTERVAL, "100", "101", "99", "100", "10", EPOCH + (i + 1) * INTERVAL - 1] for i in range(1003)]

        class Client:
            calls = []

            def _request(self, method, path, params):
                self.calls.append((method, path, params))
                return rows[:1000] if len(self.calls) == 1 else rows[1000:]

        client = Client()
        data = download_candles(client, "BTCUSDT", "15m", EPOCH, EPOCH + 1004 * INTERVAL, now_ms=EPOCH + 1002 * INTERVAL + INTERVAL // 2)
        self.assertEqual(len(data), 1002)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(method == "GET" and path == "/api/v3/klines" for method, path, _ in client.calls))
        self.assertEqual(client.calls[1][2]["startTime"], EPOCH + 1000 * INTERVAL)
        self.assertEqual(client.calls[0][2]["limit"], 1000)

    def test_bad_pagination_response_is_not_silently_accepted(self):
        class Client:
            def _request(self, *args):
                return [[EPOCH - INTERVAL, "100", "101", "99", "100", "10", EPOCH - 1]]

        with self.assertRaisesRegex(ValueError, "old or duplicate"):
            download_candles(Client(), "BTCUSDT", "15m", EPOCH, EPOCH + INTERVAL)

    def test_offline_cli_does_not_call_binance_and_writes_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder / "prices.csv", folder / "report.json"
            source.write_text(render_csv(series((100, 100, 100, 100))))
            with patch("backtest.BinanceClient._request", side_effect=AssertionError("offline must not access Binance")), patch("sys.stdout", new=io.StringIO()):
                result = main([
                    "--csv", str(source), "--output", str(output), "--fast-sma", "2", "--slow-sma", "3", "--rsi-period", "2",
                    "--no-buy-on-bullish-trend", "--buy-below", "105", "--buy-rsi-min", "0", "--buy-rsi-max", "100",
                    "--no-sell-on-bearish-trend", "--quantity-step", "0.001", "--min-notional", "0", "--close-at-end",
                ])
            self.assertEqual(result, 0)
            report = json.loads(output.read_text())
            self.assertIn("equity_curve", report)
            self.assertEqual(report["summary"]["buy_executions"], 1)
            self.assertEqual(report["summary"]["sell_executions"], 1)
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["prices.csv", "report.json"])

    def test_saved_report_filters_reproduce_an_offline_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, snapshot, output = folder / "prices.csv", folder / "snapshot.json", folder / "repeat.json"
            data = series((100, 100, 100, 100))
            original = run_backtest(data, strategy(), settings(fee_pct=D("0.1"), slippage_pct=D("0.05")), filters())
            source.write_text(render_csv(data))
            snapshot.write_text(report_json(original))
            with patch("backtest.BinanceClient._request", side_effect=AssertionError("offline must not access Binance")), patch("sys.stdout", new=io.StringIO()):
                main([
                    "--csv", str(source), "--filters-json", str(snapshot), "--output", str(output),
                    "--fast-sma", "2", "--slow-sma", "3", "--rsi-period", "2",
                    "--no-buy-on-bullish-trend", "--buy-below", "105", "--buy-rsi-min", "0", "--buy-rsi-max", "100",
                    "--no-sell-on-bearish-trend", "--cooldown-candles", "0", "--stop-cooldown-candles", "0",
                ])
            repeated = json.loads(output.read_text())
            self.assertEqual(repeated["summary"], original["summary"])
            self.assertEqual(repeated["fills"], original["fills"])
            self.assertEqual(repeated["inventory"], original["inventory"])

    def test_live_and_backtest_use_identical_strategy_defaults(self):
        self.assertEqual(strategy_config_from_args(build_parser().parse_args([])), strategy_config_from_args(live_parser().parse_args([])))
        with self.assertRaises(SystemExit):
            with patch("sys.stderr", new=io.StringIO()):
                build_parser().parse_args(["--execute"])

    def test_filter_overrides_are_isolated_and_reported(self):
        info = filters()
        original = json.dumps(info)
        args = build_parser().parse_args(["--quantity-step", "0.01", "--price-tick", "0.1", "--min-notional", "7"])
        adjusted = prepare_symbol_info(info, args)
        self.assertEqual(json.dumps(info), original)
        self.assertIn({"filterType": "MIN_NOTIONAL", "minNotional": "7", "applyToMarket": True, "avgPriceMins": 0}, adjusted["filters"])

    def test_cli_rejects_overwriting_input_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            content = render_csv(series((100, 100, 100, 100)))
            path.write_text(content)
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                main(["--csv", str(path), "--output", str(path)])
            self.assertEqual(path.read_bytes(), content.encode())


if __name__ == "__main__":
    unittest.main()
