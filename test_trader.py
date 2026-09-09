import io
import tempfile
import unittest
import logging
import urllib.error
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path

from trader import (
    BINANCE_CLIENT_ORDER_ID,
    BinanceClient,
    BinanceError,
    Candle,
    InsufficientMarketData,
    PendingIntentError,
    Position,
    StrategyConfig,
    TradeJournalError,
    acquire_process_lock,
    build_parser,
    buy_cooldown_reason,
    configure_logging,
    decide,
    execute_cycle,
    execute_prepared_intent,
    ensure_hosted_stop,
    floor_to_step,
    initialize_trade_history,
    load_order_intent,
    load_trade_history,
    load_position,
    market_minimum_notional,
    market_sell_quantity,
    new_client_order_id,
    order_intent_path,
    operational_log_path,
    prepare_order_intent,
    protective_stop_values,
    record_trade_fill,
    clear_order_intent,
    save_position,
    simple_rsi,
    tighten_hosted_stop,
    trade_history_path,
    trade_operations_log_path,
    trailing_stop_price,
    validate_args,
)


class IntentAwareFake:
    def order_by_client_id(self, symbol, client_order_id):
        raise BinanceError("Order does not exist", code=-2013, http_status=400)


def binance_order(
    order_id,
    status,
    executed,
    quote,
    client_order_id,
    side,
    order_type,
    *,
    orig_qty=None,
    fills=None,
    symbol="BTCUSDT",
):
    result = {
        "symbol": symbol,
        "orderId": order_id,
        "clientOrderId": client_order_id,
        "side": side,
        "type": order_type,
        "status": status,
        "executedQty": str(executed),
        "cummulativeQuoteQty": str(quote),
    }
    if orig_qty is not None:
        result["origQty"] = str(orig_qty)
    if fills is not None:
        result["fills"] = fills
    return result


def config(**overrides):
    values = {
        "fast_sma": 3,
        "slow_sma": 5,
        "rsi_period": 3,
        "buy_on_bullish_trend": True,
        "sell_on_bearish_trend": True,
        "buy_below": None,
        "sell_above": None,
        "buy_rsi_below": None,
        "sell_rsi_above": None,
        "stop_loss_pct": 2,
        "take_profit_pct": 4,
    }
    values.update(overrides)
    return StrategyConfig(**values)


class IndicatorTests(unittest.TestCase):
    def test_rsi_for_only_gains(self):
        self.assertEqual(simple_rsi([1, 2, 3, 4], 3), 100)

    def test_rsi_for_flat_market(self):
        self.assertEqual(simple_rsi([2, 2, 2, 2], 3), 50)

    def test_floor_to_step(self):
        self.assertEqual(floor_to_step(1.239, "0.01"), 1.23)


class MarketDataTests(unittest.TestCase):
    def test_reports_completed_candle_shortfall_explicitly(self):
        class Client:
            def candles(self, symbol, interval, limit):
                return [Candle(100, 1_000), Candle(101, 2_000)]

        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                state_file=str(Path(directory) / "state.json"),
                symbol="BNBUSDT",
                interval="15m",
                execute=False,
                live=False,
                hosted_stop_loss=True,
                quote_size=10,
            )
            with self.assertRaisesRegex(
                InsufficientMarketData,
                r"returned 1 completed 15m candle\(s\) for BNBUSDT; 6 required",
            ):
                execute_cycle(Client(), args, config(), {})


class ProtectiveStopTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                {"filterType": "MARKET_LOT_SIZE", "stepSize": "0", "minQty": "0"},
                {
                    "filterType": "NOTIONAL",
                    "minNotional": "5.00",
                    "applyMinToMarket": True,
                    "avgPriceMins": 5,
                },
            ],
        }

    def test_rounds_stop_price_and_quantity_down(self):
        quantity, price = protective_stop_values(0.000079, 80_000.129, 2, self.info)
        self.assertEqual(quantity, 0.00007)
        self.assertEqual(price, 78_400.12)

    def test_rejects_stop_below_minimum_notional(self):
        with self.assertRaisesRegex(ValueError, "too small for a protected stop"):
            protective_stop_values(0.000062, 80_000, 2, self.info)

    def execution_args(self, state_file, quote_size):
        return Namespace(
            state_file=str(state_file),
            symbol="BTCUSDT",
            interval="15m",
            execute=True,
            live=False,
            hosted_stop_loss=True,
            quote_size=quote_size,
        )

    def test_rejects_unprotectable_buy_before_market_order(self):
        class Client(IntentAwareFake):
            buy_called = False

            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 81_000, 79_000, 79_000, 82_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size, client_order_id):
                self.buy_called = True
                raise AssertionError("market_buy must not be called")

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            with self.assertRaisesRegex(ValueError, "too small for a protected stop"):
                execute_cycle(
                    client,
                    self.execution_args(Path(directory) / "state.json", 5),
                    config(),
                    self.info,
                )
            self.assertFalse(client.buy_called)

    def test_filled_buy_places_and_persists_hosted_stop(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 81_000, 79_000, 79_000, 82_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size, client_order_id):
                return binance_order(
                    100,
                    "FILLED",
                    0.000075,
                    6,
                    client_order_id,
                    "BUY",
                    "MARKET",
                    fills=[
                        {
                            "qty": "0.000075",
                            "quoteQty": "6",
                            "commission": "0",
                            "commissionAsset": "USDT",
                        }
                    ],
                )

            def free_balance(self, asset):
                return 0.000075

            def ticker_price(self, symbol):
                return 80_000

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.stop = (quantity, stop_price)
                return binance_order(
                    12345, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS"
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            client = Client()
            execute_cycle(client, self.execution_args(path, 6), config(), self.info)
            position = load_position(path, "BTCUSDT")
            self.assertEqual(position.stop_order_id, 12345)
            self.assertEqual(position.stop_price, 78_400)
            self.assertEqual(client.stop, (0.00007, 78_400))
            history_path = trade_history_path(path, "BTCUSDT", "testnet")
            fill = load_trade_history(history_path, "BTCUSDT", "testnet")["fills"][0]
            self.assertEqual(fill["side"], "BUY")
            self.assertEqual(fill["order_id"], 100)

    def test_non_hosted_buy_rejects_quantity_that_cannot_be_sold(self):
        class Client(IntentAwareFake):
            buy_called = False

            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 81_000, 79_000, 79_000, 82_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size, client_order_id):
                self.buy_called = True
                raise AssertionError("market_buy must not be called")

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            args = self.execution_args(Path(directory) / "state.json", 5.6)
            args.hosted_stop_loss = False
            with self.assertRaisesRegex(ValueError, "below Binance's 5.00 minimum"):
                execute_cycle(client, args, config(), self.info)
            self.assertFalse(client.buy_called)

    def test_non_hosted_buy_persists_post_commission_rounded_quantity(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 81_000, 79_000, 79_000, 82_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size, client_order_id):
                return binance_order(
                    101,
                    "FILLED",
                    0.000075,
                    6,
                    client_order_id,
                    "BUY",
                    "MARKET",
                    fills=[
                        {
                            "qty": "0.000075",
                            "quoteQty": "6",
                            "commission": "0.000000075",
                            "commissionAsset": "BTC",
                        }
                    ],
                )

            def free_balance(self, asset):
                return 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            args = self.execution_args(path, 6)
            args.hosted_stop_loss = False
            execute_cycle(Client(), args, config(), self.info)
            self.assertEqual(load_position(path, "BTCUSDT").quantity, 0.00007)

    def test_initial_stop_rejection_immediately_uses_protective_market_exit(self):
        class Client(IntentAwareFake):
            protective_calls = 0

            def free_balance(self, asset):
                return 0.1

            def ticker_price(self, symbol):
                return 100

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                raise BinanceError("stop rejected")

            def market_sell(self, symbol, quantity, client_order_id):
                self.protective_calls += 1
                return binance_order(
                    102, "FILLED", 0.1, 10, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            history = trade_history_path(state, "BTCUSDT", "testnet")
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            args = self.execution_args(state, 10)
            client = Client()
            updated, protective_exit = ensure_hosted_stop(
                client,
                args,
                config(),
                self.info,
                position,
                history,
                "testnet",
                100,
            )
            self.assertTrue(protective_exit)
            self.assertIsNone(updated)
            self.assertEqual(client.protective_calls, 1)
            self.assertFalse(order_intent_path(state, "BTCUSDT", "testnet").exists())
            fills = load_trade_history(history, "BTCUSDT", "testnet")["fills"]
            self.assertEqual(fills[0]["source"], "protective_market")

    def test_ambiguous_initial_stop_does_not_submit_protective_exit(self):
        class Client(IntentAwareFake):
            protective_calls = 0

            def free_balance(self, asset):
                return 0.1

            def ticker_price(self, symbol):
                return 100

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                raise BinanceError("response lost", ambiguous=True)

            def market_sell(self, symbol, quantity, client_order_id):
                self.protective_calls += 1
                raise AssertionError("ambiguous stop must not trigger a protective sell")

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            history = trade_history_path(state, "BTCUSDT", "testnet")
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            args = self.execution_args(state, 10)
            client = Client()
            with self.assertRaises(PendingIntentError):
                ensure_hosted_stop(
                    client,
                    args,
                    config(),
                    self.info,
                    position,
                    history,
                    "testnet",
                    100,
                )
            self.assertEqual(client.protective_calls, 0)
            self.assertTrue(order_intent_path(state, "BTCUSDT", "testnet").exists())
            self.assertEqual(load_position(state, "BTCUSDT"), position)


class MarketSellFilterTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                {"filterType": "MARKET_LOT_SIZE", "stepSize": "0", "minQty": "0"},
                {
                    "filterType": "NOTIONAL",
                    "minNotional": "5.00",
                    "applyMinToMarket": True,
                    "avgPriceMins": 5,
                },
            ],
        }

    def test_falls_back_to_lot_step_and_rejects_small_notional(self):
        with self.assertRaisesRegex(ValueError, "notional 4.80 USDT"):
            market_sell_quantity(0.00006993, 79_968, self.info)
        self.assertEqual(market_sell_quantity(0.00007, 79_968, self.info), 0.00007)

    def test_notional_filter_can_be_disabled_for_market_orders(self):
        self.info["filters"][-1]["applyMinToMarket"] = False
        self.assertEqual(market_minimum_notional(self.info), (0.0, 0))
        self.assertEqual(market_sell_quantity(0.00006, 79_968, self.info), 0.00006)

    def execution_args(self, state_file, hosted_stop_loss=False):
        return Namespace(
            state_file=str(state_file),
            symbol="BTCUSDT",
            interval="15m",
            execute=True,
            live=False,
            hosted_stop_loss=hosted_stop_loss,
            quote_size=25,
        )

    def test_commission_reduced_balance_defers_sell_without_submission(self):
        class Client(IntentAwareFake):
            sell_called = False

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def average_price(self, symbol):
                return 79_968

            def free_balance(self, asset):
                return 0.00006993

            def market_sell(self, symbol, quantity, client_order_id):
                self.sell_called = True
                raise AssertionError("market_sell must not be called")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            position = Position("BTCUSDT", 0.00007, 79_918.82)
            save_position(path, position)
            client = Client()
            result = execute_cycle(client, self.execution_args(path), config(), self.info)
            self.assertEqual(result.action, "HOLD")
            self.assertIn("SELL deferred", result.reasons[0])
            self.assertFalse(client.sell_called)
            self.assertEqual(load_position(path, "BTCUSDT"), position)

    def test_unsellable_hosted_quantity_does_not_cancel_stop(self):
        class Client(IntentAwareFake):
            cancel_called = False

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "NEW",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00006,
                )

            def average_price(self, symbol):
                return 79_968

            def cancel_order(self, symbol, order_id):
                self.cancel_called = True
                raise AssertionError("cancel_order must not be called")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            position = Position("BTCUSDT", 0.00006, 79_918.82, 123, 78_320)
            save_position(path, position)
            client = Client()
            result = execute_cycle(
                client,
                self.execution_args(path, hosted_stop_loss=True),
                config(),
                self.info,
            )
            self.assertEqual(result.action, "HOLD")
            self.assertFalse(client.cancel_called)
            self.assertEqual(load_position(path, "BTCUSDT"), position)

    def test_filled_hosted_stop_is_recorded_and_clears_state(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "FILLED",
                    0.00007,
                    5.60,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00007,
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_position(path, Position("BTCUSDT", 0.00007, 79_918.82, 123, 78_320))
            result = execute_cycle(
                Client(),
                self.execution_args(path, hosted_stop_loss=True),
                config(),
                self.info,
            )
            self.assertEqual(result.action, "HOLD")
            self.assertIsNone(load_position(path, "BTCUSDT"))
            history_path = trade_history_path(path, "BTCUSDT", "testnet")
            fills = load_trade_history(history_path, "BTCUSDT", "testnet")["fills"]
            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0]["source"], "hosted_stop_loss")
            self.assertEqual(fills[0]["status"], "FILLED")

    def test_hosted_stop_journal_failure_preserves_tracked_position(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "FILLED",
                    0.00007,
                    5.60,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00007,
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            position = Position("BTCUSDT", 0.00007, 79_918.82, 123, 78_320)
            save_position(path, position)
            with patch("trader.record_trade_fill", side_effect=OSError("disk full")):
                with self.assertRaises(TradeJournalError):
                    execute_cycle(
                        Client(),
                        self.execution_args(path, hosted_stop_loss=True),
                        config(),
                        self.info,
                    )
            self.assertEqual(load_position(path, "BTCUSDT"), position)

    def test_malformed_or_mismatched_hosted_stop_status_preserves_state(self):
        responses = (
            {"status": "FILLED"},
            binance_order(
                999,
                "NEW",
                0,
                0,
                "wrong-stop",
                "SELL",
                "STOP_LOSS",
                orig_qty=0.00007,
            ),
        )
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                class Client(IntentAwareFake):
                    def closes(self, symbol, interval, limit):
                        return [
                            81_000,
                            81_000,
                            81_000,
                            81_000,
                            81_000,
                            80_000,
                            80_000,
                            90_000,
                        ]

                    def order(self, symbol, order_id):
                        return response

                path = Path(directory) / "state.json"
                position = Position("BTCUSDT", 0.00007, 79_918.82, 123, 78_320)
                save_position(path, position)
                with self.assertRaises(PendingIntentError):
                    execute_cycle(
                        Client(),
                        self.execution_args(path, hosted_stop_loss=True),
                        config(),
                        self.info,
                    )
                self.assertEqual(load_position(path, "BTCUSDT"), position)
                self.assertFalse(
                    trade_history_path(path, "BTCUSDT", "testnet").exists()
                )

    def test_partial_stop_fill_is_subtracted_before_market_sell(self):
        class Client(IntentAwareFake):
            sold_quantity = None

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "PARTIALLY_FILLED",
                    0.00002,
                    1.60,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00010,
                )

            def average_price(self, symbol):
                return 79_968

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0.00003,
                    2.40,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00010,
                )

            def free_balance(self, asset):
                return 0.00007

            def market_sell(self, symbol, quantity, client_order_id):
                self.sold_quantity = quantity
                return binance_order(
                    124, "FILLED", 0.00007, 5.60, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_position(path, Position("BTCUSDT", 0.00010, 79_918.82, 123, 78_320))
            client = Client()
            result = execute_cycle(
                client,
                self.execution_args(path, hosted_stop_loss=True),
                config(),
                self.info,
            )
            self.assertEqual(result.action, "SELL")
            self.assertEqual(client.sold_quantity, 0.00007)
            self.assertIsNone(load_position(path, "BTCUSDT"))
            history_path = trade_history_path(path, "BTCUSDT", "testnet")
            fills = load_trade_history(history_path, "BTCUSDT", "testnet")["fills"]
            self.assertEqual(len(fills), 2)
            self.assertEqual([fill["source"] for fill in fills], ["hosted_stop_loss", "strategy"])

    def test_canceled_stop_is_removed_from_state_before_balance_failure(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "NEW",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00007,
                )

            def average_price(self, symbol):
                return 79_968

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.00007,
                )

            def free_balance(self, asset):
                return 0.00006993

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_position(path, Position("BTCUSDT", 0.00007, 79_918.82, 123, 78_320))
            with self.assertRaisesRegex(BinanceError, "hosted stop was canceled"):
                execute_cycle(
                    Client(),
                    self.execution_args(path, hosted_stop_loss=True),
                    config(),
                    self.info,
                )
            position = load_position(path, "BTCUSDT")
            self.assertEqual(position.quantity, 0.00007)
            self.assertIsNone(position.stop_order_id)

    def test_zero_fill_market_sell_preserves_position(self):
        class Client(IntentAwareFake):
            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 80_000, 90_000]

            def average_price(self, symbol):
                return 80_000

            def free_balance(self, asset):
                return 0.001

            def market_sell(self, symbol, quantity, client_order_id):
                return binance_order(
                    200, "EXPIRED", 0, 0, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            position = Position("BTCUSDT", 0.001, 79_000)
            save_position(path, position)
            with self.assertRaisesRegex(BinanceError, "did not fill"):
                execute_cycle(Client(), self.execution_args(path), config(), self.info)
            self.assertEqual(load_position(path, "BTCUSDT"), position)


class TradeHistoryTests(unittest.TestCase):
    def test_initializes_empty_history_beside_state_file(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "positions" / "state.json"
            history_path = trade_history_path(state_path, "SOLUSDT", "mainnet")
            initialize_trade_history(history_path, "SOLUSDT", "mainnet")
            self.assertEqual(
                history_path,
                state_path.parent / ".trader-history-SOLUSDT-mainnet.json",
            )
            self.assertEqual(
                load_trade_history(history_path, "SOLUSDT", "mainnet"),
                {
                    "schema_version": 1,
                    "symbol": "SOLUSDT",
                    "network": "mainnet",
                    "fills": [],
                },
            )
            operations_path = trade_operations_log_path(
                state_path, "SOLUSDT", "mainnet"
            )
            self.assertTrue(operations_path.exists())
            self.assertEqual(operations_path.read_text(), "")

    def test_rejects_malformed_existing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".trader-history-SOLUSDT-mainnet.json"
            path.write_text('{"schema_version": 1, "fills": "invalid"}')
            with self.assertRaisesRegex(RuntimeError, "Invalid trade history file"):
                initialize_trade_history(path, "SOLUSDT", "mainnet")

    def test_upserts_repeated_order_status_without_duplicate_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".trader-history-SOLUSDT-mainnet.json"
            initialize_trade_history(path, "SOLUSDT", "mainnet")
            partial = {
                "orderId": 42,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.1",
                "cummulativeQuoteQty": "10",
                "updateTime": 1_700_000_000_000,
            }
            completed = {
                **partial,
                "status": "FILLED",
                "executedQty": "0.2",
                "cummulativeQuoteQty": "22",
                "updateTime": 1_700_000_001_000,
            }
            record_trade_fill(
                path,
                "SOLUSDT",
                "mainnet",
                partial,
                "SELL",
                "hosted_stop_loss",
                ("partially filled",),
            )
            record_trade_fill(
                path,
                "SOLUSDT",
                "mainnet",
                completed,
                "SELL",
                "hosted_stop_loss",
                ("filled",),
            )
            fills = load_trade_history(path, "SOLUSDT", "mainnet")["fills"]
            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0]["status"], "FILLED")
            self.assertEqual(fills[0]["quantity"], 0.2)
            self.assertEqual(fills[0]["average_price"], 110)
            operations = trade_operations_log_path(
                path, "SOLUSDT", "mainnet"
            ).read_text().splitlines()
            self.assertEqual(len(operations), 1)
            self.assertIn(
                "timestamp_gmt_minus_3=2023-11-14T19:13:21-03:00",
                operations[0],
            )
            self.assertNotIn("timestamp_utc=", operations[0])
            self.assertIn("side=SELL", operations[0])
            self.assertIn("order_id=42", operations[0])
            self.assertIn("status=FILLED", operations[0])
            self.assertIn('reasons=["filled"]', operations[0])

    def test_operations_log_contains_buy_and_sell_orders_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "SOLUSDT", "mainnet")
            initialize_trade_history(history_path, "SOLUSDT", "mainnet")
            record_trade_fill(
                history_path,
                "SOLUSDT",
                "mainnet",
                {
                    "orderId": 44,
                    "status": "FILLED",
                    "executedQty": "0.1",
                    "cummulativeQuoteQty": "10",
                },
                "BUY",
                "strategy",
                ("entry",),
            )
            record_trade_fill(
                history_path,
                "SOLUSDT",
                "mainnet",
                {
                    "orderId": 45,
                    "status": "FILLED",
                    "executedQty": "0.1",
                    "cummulativeQuoteQty": "11",
                },
                "SELL",
                "strategy",
                ("exit",),
            )
            operations = trade_operations_log_path(
                state_path, "SOLUSDT", "mainnet"
            ).read_text().splitlines()
            self.assertEqual(len(operations), 2)
            self.assertIn("side=BUY", operations[0])
            self.assertIn("order_id=44", operations[0])
            self.assertIn("side=SELL", operations[1])
            self.assertIn("order_id=45", operations[1])

    def test_rejects_cumulative_and_terminal_history_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".trader-history-SOLUSDT-mainnet.json"
            initialize_trade_history(path, "SOLUSDT", "mainnet")
            filled = {
                "orderId": 43,
                "status": "FILLED",
                "executedQty": "0.2",
                "cummulativeQuoteQty": "22",
            }
            record_trade_fill(
                path,
                "SOLUSDT",
                "mainnet",
                filled,
                "SELL",
                "hosted_stop_loss",
                ("filled",),
            )
            with self.assertRaisesRegex(RuntimeError, "cumulative fill regressed"):
                record_trade_fill(
                    path,
                    "SOLUSDT",
                    "mainnet",
                    {**filled, "executedQty": "0.1", "cummulativeQuoteQty": "11"},
                    "SELL",
                    "hosted_stop_loss",
                    ("regressed",),
                )
            with self.assertRaisesRegex(RuntimeError, "terminal status regressed"):
                record_trade_fill(
                    path,
                    "SOLUSDT",
                    "mainnet",
                    {**filled, "status": "PARTIALLY_FILLED"},
                    "SELL",
                    "hosted_stop_loss",
                    ("regressed",),
                )

    def test_cooldown_counts_unique_closed_candles_after_strategy_sell(self):
        history = {
            "fills": [
                {
                    "side": "SELL",
                    "source": "strategy",
                    "timestamp_utc": "1970-01-01T00:00:01+00:00",
                    "signal_candle_close_time_utc": "1970-01-01T00:00:01+00:00",
                }
            ]
        }
        candles = [Candle(100, timestamp) for timestamp in (1_000, 2_000, 3_000, 4_000)]
        self.assertIsNone(buy_cooldown_reason(history, candles, 3))


class OrderIntentTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MARKET_LOT_SIZE", "stepSize": "0", "minQty": "0"},
            ],
        }

    def paths(self, directory):
        state = Path(directory) / "state.json"
        return (
            state,
            trade_history_path(state, "BTCUSDT", "testnet"),
            order_intent_path(state, "BTCUSDT", "testnet"),
        )

    def test_atomic_intent_round_trip_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("signal",),
                quote_quantity=25,
                signal_candle_close_time_ms=123,
            )
            self.assertEqual(
                load_order_intent(intent_path, "BTCUSDT", "testnet"), intent
            )
            self.assertFalse(intent_path.with_suffix(".json.tmp").exists())
            clear_order_intent(intent_path)
            self.assertIsNone(load_order_intent(intent_path, "BTCUSDT", "testnet"))
            self.assertFalse(state.exists())

    def test_intent_schema_rejects_wrong_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, intent_path = self.paths(directory)
            prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("signal",),
                quote_quantity=25,
            )
            with self.assertRaisesRegex(RuntimeError, "Invalid order intent file"):
                load_order_intent(intent_path, "ETHUSDT", "testnet")

    def test_client_order_ids_are_unique_and_binance_compatible(self):
        ids = [
            new_client_order_id(kind)
            for kind in (
                "market_buy",
                "strategy_sell",
                "protective_market_sell",
                "hosted_stop",
                "hosted_stop",
                "hosted_stop",
            )
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(BINANCE_CLIENT_ORDER_ID.fullmatch(value) for value in ids))

    def test_binance_client_sends_and_queries_client_order_id(self):
        client = BinanceClient("https://example.invalid", "key", "secret")
        calls = []

        def request(method, path, params=None, *, signed=False):
            calls.append((method, path, params, signed))
            return {}

        client._request = request
        client.market_buy("BTCUSDT", 10, "buy-id")
        client.market_sell("BTCUSDT", 0.1, "sell-id")
        client.place_stop_loss("BTCUSDT", 0.1, 90, "stop-id")
        client.order_by_client_id("BTCUSDT", "lookup-id")
        self.assertEqual(calls[0][2]["newClientOrderId"], "buy-id")
        self.assertEqual(calls[1][2]["newClientOrderId"], "sell-id")
        self.assertEqual(calls[2][2]["newClientOrderId"], "stop-id")
        self.assertEqual(calls[3][2]["origClientOrderId"], "lookup-id")

    def test_crash_recovery_before_submit_uses_persisted_buy_intent(self):
        class Client(IntentAwareFake):
            submitted = []

            def market_buy(self, symbol, quote_quantity, client_order_id):
                self.submitted.append(client_order_id)
                return binance_order(
                    10,
                    "FILLED",
                    0.1,
                    10,
                    client_order_id,
                    "BUY",
                    "MARKET",
                    fills=[
                        {
                            "qty": "0.1",
                            "quoteQty": "10",
                            "commission": "0",
                            "commissionAsset": "USDT",
                        }
                    ],
                )

            def free_balance(self, asset):
                return 0.1

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("recovered",),
                quote_quantity=10,
            )
            client = Client()
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertEqual(client.submitted, [intent["client_order_id"]])
            self.assertEqual(load_position(state, "BTCUSDT"), Position("BTCUSDT", 0.1, 100))
            self.assertFalse(intent_path.exists())

    def test_accepted_response_lost_is_recovered_without_duplicate_submit(self):
        class Client:
            def order_by_client_id(self, symbol, client_order_id):
                return binance_order(
                    11, "FILLED", 0.2, 20, client_order_id, "BUY", "MARKET"
                )

            def market_buy(self, symbol, quote_quantity, client_order_id):
                raise AssertionError("accepted order must not be submitted again")

            def free_balance(self, asset):
                return 0.2

            def trades(self, symbol, order_id):
                return [
                    {
                        "orderId": order_id,
                        "qty": "0.2",
                        "quoteQty": "20",
                        "commission": "0",
                        "commissionAsset": "USDT",
                    }
                ]

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("response lost",),
                quote_quantity=20,
            )
            execute_prepared_intent(Client(), intent_path, state, history, self.info, intent)
            self.assertEqual(load_position(state, "BTCUSDT").quantity, 0.2)
            fills = load_trade_history(history, "BTCUSDT", "testnet")["fills"]
            self.assertEqual(fills[0]["order_id"], 11)
            self.assertFalse(intent_path.exists())

    def test_ambiguous_submit_is_reconciled_by_client_id_without_retry(self):
        class Client:
            lookups = 0
            submits = 0

            def order_by_client_id(self, symbol, client_order_id):
                self.lookups += 1
                if self.lookups == 1:
                    raise BinanceError("Order does not exist", code=-2013, http_status=400)
                return binance_order(
                    15, "FILLED", 0.1, 10, client_order_id, "BUY", "MARKET"
                )

            def market_buy(self, symbol, quote_quantity, client_order_id):
                self.submits += 1
                raise BinanceError("response lost", ambiguous=True)

            def free_balance(self, asset):
                return 0.1

            def trades(self, symbol, order_id):
                return [
                    {
                        "orderId": order_id,
                        "qty": "0.1",
                        "quoteQty": "10",
                        "commission": "0",
                        "commissionAsset": "USDT",
                    }
                ]

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("response lost",),
                quote_quantity=10,
            )
            client = Client()
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertEqual(client.submits, 1)
            self.assertEqual(client.lookups, 2)
            self.assertEqual(load_position(state, "BTCUSDT").quantity, 0.1)
            self.assertFalse(intent_path.exists())

    def test_ambiguous_pending_intent_blocks_strategy_evaluation(self):
        class Client(IntentAwareFake):
            candles_called = False
            submits = 0

            def market_buy(self, symbol, quote_quantity, client_order_id):
                self.submits += 1
                raise BinanceError("connection lost", ambiguous=True)

            def closes(self, symbol, interval, limit):
                self.candles_called = True
                raise AssertionError("strategy must not run while intent is unresolved")

        with tempfile.TemporaryDirectory() as directory:
            state, _, intent_path = self.paths(directory)
            prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("pending",),
                quote_quantity=10,
            )
            args = Namespace(
                state_file=str(state),
                symbol="BTCUSDT",
                interval="15m",
                execute=True,
                live=False,
                hosted_stop_loss=False,
                quote_size=10,
            )
            client = Client()
            with self.assertRaises(PendingIntentError):
                execute_cycle(client, args, config(), self.info)
            with self.assertRaises(PendingIntentError):
                execute_cycle(client, args, config(), self.info)
            self.assertFalse(client.candles_called)
            self.assertTrue(intent_path.exists())
            self.assertEqual(client.submits, 1)
            self.assertTrue(
                load_order_intent(intent_path, "BTCUSDT", "testnet")["submission_attempted"]
            )

    def test_mismatched_or_malformed_order_leaves_state_and_intent_untouched(self):
        responses = (
            {
                "symbol": "BTCUSDT",
                "orderId": 20,
                "status": "FILLED",
                "executedQty": "0.1",
                "cummulativeQuoteQty": "10",
                "clientOrderId": "wrong-id",
                "side": "SELL",
                "type": "MARKET",
            },
            {
                "status": "FILLED",
                "executedQty": "0.1",
                "cummulativeQuoteQty": "10",
            },
        )
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                state, history, intent_path = self.paths(directory)
                position = Position("BTCUSDT", 0.1, 100)
                save_position(state, position)
                intent = prepare_order_intent(
                    intent_path,
                    "BTCUSDT",
                    "testnet",
                    "strategy_sell",
                    "strategy",
                    ("exit",),
                    quantity=0.1,
                    position=position,
                )

                class Client:
                    def order_by_client_id(self, symbol, client_order_id):
                        return response

                with self.assertRaises(PendingIntentError):
                    execute_prepared_intent(
                        Client(), intent_path, state, history, self.info, intent
                    )
                self.assertEqual(load_position(state, "BTCUSDT"), position)
                self.assertTrue(intent_path.exists())
                self.assertFalse(history.exists())

    def test_malformed_successful_submit_is_query_only_afterward(self):
        class Client(IntentAwareFake):
            submits = 0

            def market_sell(self, symbol, quantity, client_order_id):
                self.submits += 1
                return {"status": "FILLED", "executedQty": "0.1"}

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "strategy_sell",
                "strategy",
                ("exit",),
                quantity=0.1,
                position=position,
            )
            client = Client()
            with self.assertRaises(PendingIntentError):
                execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            with self.assertRaises(PendingIntentError):
                execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertEqual(client.submits, 1)
            self.assertEqual(load_position(state, "BTCUSDT"), position)
            self.assertTrue(intent_path.exists())

    def test_recovered_buy_uses_order_trades_for_base_commission(self):
        class Client:
            def order_by_client_id(self, symbol, client_order_id):
                return binance_order(
                    21, "FILLED", 0.1, 10, client_order_id, "BUY", "MARKET"
                )

            def trades(self, symbol, order_id):
                return [
                    {
                        "orderId": order_id,
                        "symbol": symbol,
                        "qty": "0.1",
                        "quoteQty": "10",
                        "commission": "0.001",
                        "commissionAsset": "BTC",
                    }
                ]

            def free_balance(self, asset):
                raise AssertionError("recovery must not use unrelated account balance")

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("recover",),
                quote_quantity=10,
            )
            execute_prepared_intent(Client(), intent_path, state, history, self.info, intent)
            self.assertEqual(load_position(state, "BTCUSDT").quantity, 0.099)

    def test_active_cancel_prerequisite_leaves_state_untouched(self):
        class Client(IntentAwareFake):
            submitted = False

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "NEW",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def market_sell(self, symbol, quantity, client_order_id):
                self.submitted = True

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100, 22, 98)
            save_position(state, position)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "strategy_sell",
                "strategy",
                ("exit",),
                quantity=0.1,
                position=position,
                cancel_order_id=22,
            )
            client = Client()
            with self.assertRaises(PendingIntentError):
                execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertFalse(client.submitted)
            self.assertEqual(load_position(state, "BTCUSDT"), position)
            persisted = load_order_intent(intent_path, "BTCUSDT", "testnet")
            self.assertFalse(persisted["cancel_completed"])
            self.assertFalse(persisted["submission_attempted"])

    def test_credential_change_rejects_intent_before_lookup(self):
        class Client:
            api_key = "different-key"

            def order_by_client_id(self, symbol, client_order_id):
                raise AssertionError("credential mismatch must fail before Binance lookup")

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "market_buy",
                "strategy",
                ("buy",),
                quote_quantity=10,
            )
            with self.assertRaisesRegex(RuntimeError, "different Binance credentials"):
                execute_prepared_intent(Client(), intent_path, state, history, self.info, intent)
            self.assertTrue(intent_path.exists())

    def test_resolved_startup_intent_forces_hold_for_cycle(self):
        class Client:
            buy_called = False

            def order_by_client_id(self, symbol, client_order_id):
                return binance_order(
                    23, "FILLED", 0.1, 10, client_order_id, "SELL", "MARKET"
                )

            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 81_000, 79_000, 79_000, 82_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_quantity, client_order_id):
                self.buy_called = True
                raise AssertionError("recovery cycle must not create another strategy order")

        with tempfile.TemporaryDirectory() as directory:
            state, _, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "strategy_sell",
                "strategy",
                ("exit",),
                quantity=0.1,
                position=position,
            )
            args = Namespace(
                state_file=str(state),
                symbol="BTCUSDT",
                interval="15m",
                execute=True,
                live=False,
                hosted_stop_loss=False,
                quote_size=10,
            )
            client = Client()
            result = execute_cycle(client, args, config(cooldown_candles=0), self.info)
            self.assertEqual(result.action, "HOLD")
            self.assertIn("recovered prior order intent", result.reasons[0])
            self.assertFalse(client.buy_called)

    def test_recovered_strategy_sell_updates_history_and_cooldown(self):
        class Client:
            def order_by_client_id(self, symbol, client_order_id):
                return binance_order(
                    12, "FILLED", 0.1, 11, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "strategy_sell",
                "strategy",
                ("exit",),
                quantity=0.1,
                position=position,
                signal_candle_close_time_ms=1_000,
            )
            execute_prepared_intent(Client(), intent_path, state, history, self.info, intent)
            self.assertIsNone(load_position(state, "BTCUSDT"))
            journal = load_trade_history(history, "BTCUSDT", "testnet")
            self.assertIn("cooldown", buy_cooldown_reason(journal, [Candle(100, 2_000)], 3))

    def test_recovered_new_stop_becomes_tracked_position(self):
        class Client:
            def order_by_client_id(self, symbol, client_order_id):
                return binance_order(
                    13, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS"
                )

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100)
            save_position(state, position)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "hosted_stop",
                "hosted_stop_loss",
                ("restore",),
                quantity=0.1,
                stop_price=98,
                position=position,
            )
            execute_prepared_intent(Client(), intent_path, state, history, self.info, intent)
            self.assertEqual(
                load_position(state, "BTCUSDT"), Position("BTCUSDT", 0.1, 100, 13, 98)
            )
            self.assertFalse(intent_path.exists())

    def test_cancel_prerequisite_is_recovered_before_strategy_sell(self):
        class Client(IntentAwareFake):
            events = []

            def cancel_order(self, symbol, order_id):
                self.events.append("cancel")
                self.assert_intent_exists = intent_path.exists()
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.1

            def market_sell(self, symbol, quantity, client_order_id):
                self.events.append("sell")
                self.submitted_id = client_order_id
                return binance_order(
                    14, "FILLED", 0.1, 10, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            position = Position("BTCUSDT", 0.1, 100, 9, 98)
            save_position(state, position)
            intent = prepare_order_intent(
                intent_path,
                "BTCUSDT",
                "testnet",
                "strategy_sell",
                "strategy",
                ("exit",),
                quantity=0.1,
                position=position,
                cancel_order_id=9,
            )
            client = Client()
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertTrue(client.assert_intent_exists)
            self.assertEqual(client.events, ["cancel", "sell"])
            self.assertEqual(client.submitted_id, intent["client_order_id"])
            self.assertIsNone(load_position(state, "BTCUSDT"))


class RuntimeSafetyTests(unittest.TestCase):
    def test_default_operational_log_is_beside_state_file(self):
        self.assertEqual(
            operational_log_path(
                Path("positions/state.json"), "BTCUSDT", "testnet"
            ),
            Path("positions/trader-BTCUSDT-testnet.log"),
        )

    def test_mutating_transport_and_uncertain_http_errors_are_ambiguous(self):
        client = BinanceClient("https://example.invalid")
        errors = (
            urllib.error.HTTPError(
                "https://example.invalid",
                408,
                "timeout",
                {},
                io.BytesIO(b'{"code": -1007, "msg": "timeout"}'),
            ),
            urllib.error.HTTPError(
                "https://example.invalid",
                400,
                "unknown",
                {},
                io.BytesIO(b'{"code": -1006, "msg": "unknown"}'),
            ),
            ConnectionResetError("reset"),
        )
        for error in errors:
            with self.subTest(error=error), patch(
                "urllib.request.urlopen", side_effect=error
            ):
                with self.assertRaises(BinanceError) as raised:
                    client._request("POST", "/api/v3/order")
                self.assertTrue(raised.exception.ambiguous)

    def test_malformed_successful_post_response_is_ambiguous(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"truncated"'

        client = BinanceClient("https://example.invalid")
        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaises(BinanceError) as raised:
                client._request("POST", "/api/v3/order")
        self.assertTrue(raised.exception.ambiguous)

    def test_lookup_and_cancel_not_found_codes_are_distinct(self):
        self.assertTrue(BinanceError("missing", code=-2013).order_not_found)
        self.assertFalse(BinanceError("missing", code=-2011).order_not_found)
        self.assertTrue(BinanceError("missing", code=-2011).cancel_order_not_found)

    def test_second_process_lock_for_same_symbol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = acquire_process_lock("BTCUSDT", "mainnet")
            try:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    acquire_process_lock("BTCUSDT", "mainnet")
            finally:
                first.close()
            replacement = acquire_process_lock("BTCUSDT", "mainnet")
            replacement.close()

    def test_operational_log_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trader.log"
            configure_logging(False, str(path))
            logging.getLogger("trader").info("BTCUSDT BUY filled")
            for handler in logging.getLogger().handlers:
                handler.flush()
            self.assertIn("BTCUSDT BUY filled", path.read_text())


class TrailingStopTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "NOTIONAL", "minNotional": "5.00"},
            ],
        }

    def test_default_trailing_levels(self):
        thresholds = ((1, 0), (2, 1), (3, 2))
        self.assertEqual(trailing_stop_price(100, 100.9, 2, thresholds), 98)
        self.assertEqual(trailing_stop_price(100, 101, 2, thresholds), 100)
        self.assertEqual(trailing_stop_price(100, 102, 2, thresholds), 101)
        self.assertEqual(trailing_stop_price(100, 103, 2, thresholds), 102)

    def test_tightens_hosted_stop_and_persists_replacement(self):
        class Client(IntentAwareFake):
            def ticker_price(self, symbol):
                return 103

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.1

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.replacement = (quantity, stop_price)
                return binance_order(
                    2, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS"
                )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "BTCUSDT", "mainnet")
            initialize_trade_history(history_path, "BTCUSDT", "mainnet")
            position = Position("BTCUSDT", 0.1, 100, 1, 98)
            save_position(state_path, position)
            args = Namespace(symbol="BTCUSDT", state_file=str(state_path))
            client = Client()
            updated = tighten_hosted_stop(
                client,
                args,
                config(),
                102.1,
                self.info,
                position,
                {"orderId": 1, "executedQty": "0"},
                history_path,
                "mainnet",
            )
            self.assertEqual(client.replacement, (0.1, 101))
            self.assertEqual(updated.stop_order_id, 2)
            self.assertEqual(load_position(state_path, "BTCUSDT"), updated)

    def test_rejected_replacement_response_restores_prior_stop(self):
        class Client(IntentAwareFake):
            stop_calls = 0
            client_ids = []

            def ticker_price(self, symbol):
                return 103

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.1

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.stop_calls += 1
                self.client_ids.append(client_order_id)
                if self.stop_calls == 1:
                    return binance_order(
                        24,
                        "REJECTED",
                        0,
                        0,
                        client_order_id,
                        "SELL",
                        "STOP_LOSS",
                    )
                return binance_order(
                    25, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS"
                )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "BTCUSDT", "mainnet")
            initialize_trade_history(history_path, "BTCUSDT", "mainnet")
            position = Position("BTCUSDT", 0.1, 100, 1, 98)
            save_position(state_path, position)
            args = Namespace(symbol="BTCUSDT", state_file=str(state_path))
            client = Client()
            updated = tighten_hosted_stop(
                client,
                args,
                config(),
                102.1,
                self.info,
                position,
                {"orderId": 1, "executedQty": "0"},
                history_path,
                "mainnet",
            )
            self.assertEqual(client.stop_calls, 2)
            self.assertEqual(len(set(client.client_ids)), 2)
            self.assertEqual(updated.stop_order_id, 25)
            self.assertEqual(updated.stop_price, 98)

    def test_terminal_partial_replacement_restores_stop_for_remainder(self):
        class Client(IntentAwareFake):
            stop_calls = 0

            def ticker_price(self, symbol):
                return 103

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.06

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return binance_order(
                        26,
                        "CANCELED",
                        0.04,
                        4.04,
                        client_order_id,
                        "SELL",
                        "STOP_LOSS",
                    )
                return binance_order(
                    27, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS"
                )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "BTCUSDT", "mainnet")
            initialize_trade_history(history_path, "BTCUSDT", "mainnet")
            position = Position("BTCUSDT", 0.1, 100, 1, 98)
            save_position(state_path, position)
            args = Namespace(symbol="BTCUSDT", state_file=str(state_path))
            client = Client()
            updated = tighten_hosted_stop(
                client,
                args,
                config(),
                102.1,
                self.info,
                position,
                {"orderId": 1, "executedQty": "0"},
                history_path,
                "mainnet",
            )
            self.assertEqual(client.stop_calls, 2)
            self.assertAlmostEqual(updated.quantity, 0.06)
            self.assertEqual(updated.entry_price, 100)
            self.assertEqual(updated.stop_order_id, 27)
            self.assertEqual(updated.stop_price, 98)
            fills = load_trade_history(history_path, "BTCUSDT", "mainnet")["fills"]
            self.assertEqual(fills[0]["order_id"], 26)
            self.assertEqual(fills[0]["status"], "CANCELED")

    def test_terminal_partial_restore_sells_only_authoritative_remainder(self):
        class Client(IntentAwareFake):
            stop_calls = 0
            sold_quantity = None

            def ticker_price(self, symbol):
                return 103

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.1

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.stop_calls += 1
                if self.stop_calls == 1:
                    return binance_order(
                        28,
                        "CANCELED",
                        0.04,
                        4.04,
                        client_order_id,
                        "SELL",
                        "STOP_LOSS",
                    )
                return binance_order(
                    29,
                    "CANCELED",
                    0.02,
                    1.96,
                    client_order_id,
                    "SELL",
                    "STOP_LOSS",
                )

            def market_sell(self, symbol, quantity, client_order_id):
                self.sold_quantity = quantity
                return binance_order(
                    30,
                    "FILLED",
                    quantity,
                    quantity * 103,
                    client_order_id,
                    "SELL",
                    "MARKET",
                )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "BTCUSDT", "mainnet")
            initialize_trade_history(history_path, "BTCUSDT", "mainnet")
            position = Position("BTCUSDT", 0.1, 100, 1, 98)
            save_position(state_path, position)
            args = Namespace(symbol="BTCUSDT", state_file=str(state_path))
            client = Client()
            updated = tighten_hosted_stop(
                client,
                args,
                config(),
                102.1,
                self.info,
                position,
                {"orderId": 1, "executedQty": "0"},
                history_path,
                "mainnet",
            )
            self.assertIsNone(updated)
            self.assertAlmostEqual(client.sold_quantity, 0.04)
            self.assertIsNone(load_position(state_path, "BTCUSDT"))
            fills = load_trade_history(history_path, "BTCUSDT", "mainnet")["fills"]
            self.assertAlmostEqual(sum(fill["quantity"] for fill in fills), 0.1)

    def test_failed_replacement_and_restore_uses_protective_market_exit(self):
        class Client(IntentAwareFake):
            def ticker_price(self, symbol):
                return 103

            def cancel_order(self, symbol, order_id):
                return binance_order(
                    order_id,
                    "CANCELED",
                    0,
                    0,
                    "existing-stop",
                    "SELL",
                    "STOP_LOSS",
                    orig_qty=0.1,
                )

            def free_balance(self, asset):
                return 0.1

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                raise BinanceError("stop rejected")

            def market_sell(self, symbol, quantity, client_order_id):
                return binance_order(
                    3, "FILLED", 0.1, 10.3, client_order_id, "SELL", "MARKET"
                )

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            history_path = trade_history_path(state_path, "BTCUSDT", "mainnet")
            initialize_trade_history(history_path, "BTCUSDT", "mainnet")
            position = Position("BTCUSDT", 0.1, 100, 1, 98)
            save_position(state_path, position)
            args = Namespace(symbol="BTCUSDT", state_file=str(state_path))
            updated = tighten_hosted_stop(
                Client(),
                args,
                config(),
                102.1,
                self.info,
                position,
                {"orderId": 1, "executedQty": "0"},
                history_path,
                "mainnet",
            )
            self.assertIsNone(updated)
            self.assertIsNone(load_position(state_path, "BTCUSDT"))
            fills = load_trade_history(history_path, "BTCUSDT", "mainnet")["fills"]
            self.assertEqual(fills[0]["source"], "protective_market")


class DecisionTests(unittest.TestCase):
    def test_buys_when_all_entry_rules_match(self):
        result = decide(
            [95, 95, 97, 95, 95, 98, 96],
            config(buy_below=100, buy_rsi_below=70),
            None,
        )
        self.assertEqual(result.action, "BUY")
        self.assertIn("bullish SMA crossover confirmed", result.reasons[0])

    def test_buys_when_gap_confirms_one_candle_after_crossover(self):
        result = decide(
            [95, 95, 95, 95, 95, 95, 97, 96],
            config(),
            None,
        )
        self.assertEqual(result.action, "BUY")
        self.assertIn("1 closed candle(s) ago", result.reasons[0])

    def test_one_candle_bullish_crossover_does_not_buy(self):
        result = decide([10, 10, 10, 10, 10, 10, 13], config(), None)
        self.assertEqual(result.action, "HOLD")

    def test_unconfirmed_bullish_crossover_does_not_buy(self):
        result = decide([10, 10, 10, 10, 10, 13, 5], config(), None)
        self.assertEqual(result.action, "HOLD")

    def test_holds_when_fast_sma_was_already_above_slow_sma(self):
        result = decide([10, 10, 10, 11, 12, 13, 14], config(), None)
        self.assertEqual(result.action, "HOLD")
        self.assertIn("no bullish SMA crossover in the last 3", result.reasons[0])

    def test_low_rsi_failure_is_worded_as_a_failure(self):
        result = decide(
            [100, 100, 100, 100, 101, 99, 99],
            config(buy_on_bullish_trend=False, buy_below=200),
            None,
        )
        self.assertEqual(result.action, "HOLD")
        self.assertIn("RSI 33.33 below minimum 50", result.reasons[0])
        self.assertNotIn("RSI 33.33 >= 50", result.reasons[0])

    def test_holds_when_one_entry_rule_fails(self):
        result = decide([10, 10, 10, 10, 10, 12, 12], config(buy_below=11), None)
        self.assertEqual(result.action, "HOLD")

    def test_stop_loss_exits_a_position(self):
        position = Position("BTCUSDT", 0.01, 100)
        result = decide([105, 105, 105, 104, 103, 101, 97], config(), position)
        self.assertEqual(result.action, "SELL")
        self.assertIn("stop loss 2%", result.reasons)

    def test_bearish_crossover_exits_a_position(self):
        position = Position("BTCUSDT", 0.01, 100)
        result = decide([110, 110, 110, 110, 110, 100, 100], config(), position)
        self.assertEqual(result.action, "SELL")
        self.assertIn("bearish SMA trend", result.reasons[0])

    def test_tiny_bearish_gap_does_not_sell(self):
        position = Position("BTCUSDT", 0.01, 100)
        result = decide(
            [100, 100, 100, 100, 100, 99.99, 99.99],
            config(stop_loss_pct=0, take_profit_pct=0),
            position,
        )
        self.assertEqual(result.action, "HOLD")

    def test_unconfirmed_bearish_crossover_does_not_sell(self):
        position = Position("BTCUSDT", 0.01, 10)
        result = decide(
            [10, 10, 10, 10, 10, 9, 12],
            config(stop_loss_pct=0, take_profit_pct=0),
            position,
        )
        self.assertEqual(result.action, "HOLD")

    def test_no_entry_rules_means_hold(self):
        result = decide(
            [1, 2, 3, 4, 5, 6, 7],
            config(buy_on_bullish_trend=False),
            None,
        )
        self.assertEqual(result.action, "HOLD")


class StateTests(unittest.TestCase):
    def test_round_trip_and_remove_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            expected = Position("BTCUSDT", 0.01, 50_000)
            save_position(path, expected)
            self.assertEqual(load_position(path, "BTCUSDT"), expected)
            save_position(path, None)
            self.assertIsNone(load_position(path, "BTCUSDT"))


class ValidationTests(unittest.TestCase):
    def test_default_candle_interval_is_15_minutes(self):
        self.assertEqual(build_parser().parse_args([]).interval, "15m")

    def test_live_execution_requires_confirmation(self):
        args = Namespace(
            fast_sma=9,
            slow_sma=21,
            rsi_period=14,
            quote_size=25,
            poll_seconds=60,
            buy_rsi_below=None,
            sell_rsi_above=None,
            stop_loss_pct=2,
            take_profit_pct=4,
            min_sma_gap_pct=0.1,
            buy_crossover_lookback_candles=3,
            buy_rsi_min=50,
            buy_rsi_max=70,
            cooldown_candles=3,
            trailing_thresholds=((1, 0), (2, 1), (3, 2)),
            live=True,
            execute=True,
            confirm_live=False,
        )
        with self.assertRaisesRegex(ValueError, "--confirm-live"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
