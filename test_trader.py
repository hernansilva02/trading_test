import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from trader import (
    BinanceError,
    Position,
    StrategyConfig,
    decide,
    execute_cycle,
    floor_to_step,
    load_position,
    market_minimum_notional,
    market_sell_quantity,
    protective_stop_values,
    save_position,
    simple_rsi,
    validate_args,
)


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
            hosted_stop_loss=True,
            quote_size=quote_size,
        )

    def test_rejects_unprotectable_buy_before_market_order(self):
        class Client:
            buy_called = False

            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 79_000, 79_000, 79_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size):
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
        class Client:
            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 79_000, 79_000, 79_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size):
                return {"executedQty": "0.000075", "cummulativeQuoteQty": "6"}

            def free_balance(self, asset):
                return 0.000075

            def place_stop_loss(self, symbol, quantity, stop_price):
                self.stop = (quantity, stop_price)
                return {"orderId": 12345}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            client = Client()
            execute_cycle(client, self.execution_args(path, 6), config(), self.info)
            position = load_position(path, "BTCUSDT")
            self.assertEqual(position.stop_order_id, 12345)
            self.assertEqual(position.stop_price, 78_400)
            self.assertEqual(client.stop, (0.00007, 78_400))

    def test_non_hosted_buy_rejects_quantity_that_cannot_be_sold(self):
        class Client:
            buy_called = False

            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 79_000, 79_000, 79_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size):
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
        class Client:
            def closes(self, symbol, interval, limit):
                return [79_000, 79_000, 79_000, 79_000, 79_000, 80_000, 70_000]

            def market_buy(self, symbol, quote_size):
                return {
                    "executedQty": "0.000075",
                    "cummulativeQuoteQty": "6",
                    "fills": [
                        {
                            "commission": "0.000000075",
                            "commissionAsset": "BTC",
                        }
                    ],
                }

            def free_balance(self, asset):
                return 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            args = self.execution_args(path, 6)
            args.hosted_stop_loss = False
            execute_cycle(Client(), args, config(), self.info)
            self.assertEqual(load_position(path, "BTCUSDT").quantity, 0.00007)


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
            hosted_stop_loss=hosted_stop_loss,
            quote_size=25,
        )

    def test_commission_reduced_balance_defers_sell_without_submission(self):
        class Client:
            sell_called = False

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 90_000]

            def average_price(self, symbol):
                return 79_968

            def free_balance(self, asset):
                return 0.00006993

            def market_sell(self, symbol, quantity):
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
        class Client:
            cancel_called = False

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return {
                    "status": "NEW",
                    "orderId": order_id,
                    "origQty": "0.00006",
                    "executedQty": "0",
                }

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

    def test_partial_stop_fill_is_subtracted_before_market_sell(self):
        class Client:
            sold_quantity = None

            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return {
                    "status": "PARTIALLY_FILLED",
                    "orderId": order_id,
                    "origQty": "0.00010",
                    "executedQty": "0.00002",
                }

            def average_price(self, symbol):
                return 79_968

            def cancel_order(self, symbol, order_id):
                return {
                    "status": "CANCELED",
                    "orderId": order_id,
                    "origQty": "0.00010",
                    "executedQty": "0.00003",
                }

            def free_balance(self, asset):
                return 0.00007

            def market_sell(self, symbol, quantity):
                self.sold_quantity = quantity
                return {"executedQty": "0.00007"}

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

    def test_canceled_stop_is_removed_from_state_before_balance_failure(self):
        class Client:
            def closes(self, symbol, interval, limit):
                return [81_000, 81_000, 81_000, 81_000, 81_000, 80_000, 90_000]

            def order(self, symbol, order_id):
                return {
                    "status": "NEW",
                    "orderId": order_id,
                    "origQty": "0.00007",
                    "executedQty": "0",
                }

            def average_price(self, symbol):
                return 79_968

            def cancel_order(self, symbol, order_id):
                return {"status": "CANCELED", "orderId": order_id, "executedQty": "0"}

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


class DecisionTests(unittest.TestCase):
    def test_buys_when_all_entry_rules_match(self):
        result = decide(
            [10, 10, 10, 10, 10, 13],
            config(buy_below=14, buy_rsi_below=100),
            None,
        )
        self.assertEqual(result.action, "BUY")
        self.assertEqual(len(result.reasons), 3)

    def test_holds_when_fast_sma_was_already_above_slow_sma(self):
        result = decide([10, 10, 10, 11, 12, 13], config(), None)
        self.assertEqual(result.action, "HOLD")

    def test_holds_when_one_entry_rule_fails(self):
        result = decide([10, 10, 10, 10, 10, 12], config(buy_below=11), None)
        self.assertEqual(result.action, "HOLD")

    def test_stop_loss_exits_a_position(self):
        position = Position("BTCUSDT", 0.01, 100)
        result = decide([105, 105, 104, 103, 101, 97], config(), position)
        self.assertEqual(result.action, "SELL")
        self.assertIn("stop loss 2%", result.reasons)

    def test_bearish_crossover_exits_a_position(self):
        position = Position("BTCUSDT", 0.01, 100)
        result = decide([110, 110, 110, 110, 110, 100], config(), position)
        self.assertEqual(result.action, "SELL")
        self.assertIn("fast SMA crossed below slow SMA", result.reasons)

    def test_no_entry_rules_means_hold(self):
        result = decide(
            [1, 2, 3, 4, 5, 6],
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
            live=True,
            execute=True,
            confirm_live=False,
        )
        with self.assertRaisesRegex(ValueError, "--confirm-live"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
