import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from test_trader import IntentAwareFake, binance_order, config
from trader import (
    Candle, PendingIntentError, Position, buy_cooldown_reason, buy_entry_block_reason,
    execute_cycle, execute_prepared_intent, load_order_intent, load_position, load_trade_history,
    migrate_inventory_history, order_intent_path, prepare_order_intent, record_trade_fill,
    save_order_intent, save_position, trade_history_path, build_parser, validate_args,
)


EPOCH = 1_700_000_000_000


def utc(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def historical_fill(side, time, source="strategy", reasons=(), signal_time=None):
    fill = {"side": side, "source": source, "timestamp_utc": utc(time), "reasons": list(reasons)}
    if signal_time is not None:
        fill["signal_candle_close_time_utc"] = utc(signal_time)
    return fill


class EntryGuardTests(unittest.TestCase):
    def test_default_cooldowns_and_validation(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.cooldown_candles, args.stop_cooldown_candles), (3, 6))
        args.stop_cooldown_candles = 2
        with self.assertRaisesRegex(ValueError, "at least"):
            validate_args(args)

    def test_all_sell_sources_use_execution_time_and_unique_candle_closes(self):
        candles = [Candle(100, EPOCH + i * 1000) for i in (1, 2, 2, 3, 4, 5, 6)]
        for source, reason, required in (
            ("strategy", "take profit 4%", 3),
            ("strategy", "stop loss 2%", 6),
            ("hosted_stop_loss", "trailing stop locked in a profit", 6),
            ("protective_market", "stop placement rejected", 6),
        ):
            with self.subTest(source=source, reason=reason):
                history = {"fills": [historical_fill("SELL", EPOCH + 2500, source, (reason,), EPOCH)]}
                blocked = buy_cooldown_reason(history, candles[:5], 3, 6)
                self.assertIn(f"2/{required}", blocked)
                if required == 3:
                    self.assertIsNone(buy_cooldown_reason(history, candles, 3, 6))
                else:
                    self.assertIsNotNone(buy_cooldown_reason(history, candles, 3, 6))

    def test_later_normal_sale_does_not_shorten_protective_cooldown(self):
        history = {"fills": [
            historical_fill("SELL", EPOCH, "hosted_stop_loss"),
            historical_fill("SELL", EPOCH + 1000, reasons=("take profit 4%",)),
        ]}
        candles = [Candle(100, EPOCH + i * 1000) for i in range(1, 6)]
        self.assertIn("5/6", buy_cooldown_reason(history, candles, 3, 6))

    def test_same_or_older_buy_signal_is_blocked_even_with_no_cooldown(self):
        history = {"fills": [historical_fill("BUY", EPOCH + 100, signal_time=EPOCH)]}
        for close_time in (EPOCH - 1000, EPOCH):
            with self.subTest(close_time=close_time):
                reason = buy_entry_block_reason(history, [Candle(100, close_time)], config(cooldown_candles=0))
                self.assertIn("already used", reason)

    def test_stop_requires_fresh_crossover_even_after_cooldown(self):
        history = {"fills": [historical_fill("SELL", EPOCH, "hosted_stop_loss")]}
        candles = [Candle(price, EPOCH + (i + 1) * 1000) for i, price in enumerate([10, 10, 10, 11, 12, 13, 14])]
        self.assertIn("new bullish crossover", buy_entry_block_reason(history, candles, config()))

    def test_crossover_before_or_at_stop_cannot_rearm(self):
        candles = [Candle(price, EPOCH + (i + 1) * 1000) for i, price in enumerate([95, 95, 95, 95, 95, 95, 97, 96])]
        for stop_time in (EPOCH + 7000, EPOCH + 7500):
            history = {"fills": [historical_fill("SELL", stop_time, "protective_market")]}
            reason = buy_entry_block_reason(history, candles, config(cooldown_candles=0, stop_cooldown_candles=0))
            self.assertIn("new bullish crossover", reason)

    def test_fresh_post_stop_crossover_rearms_only_after_cooldown(self):
        history = {"fills": [historical_fill("SELL", EPOCH + 2000, "hosted_stop_loss")]}
        candles = [Candle(price, EPOCH + (i + 1) * 1000) for i, price in enumerate([95, 95, 95, 95, 95, 95, 97, 96])]
        self.assertIsNone(buy_entry_block_reason(history, candles, config()))
        self.assertIn("6/7", buy_entry_block_reason(history, candles, config(stop_cooldown_candles=7)))

    def test_restart_cycle_blocks_continuation_after_profitable_stop(self):
        class Client:
            def candles(self, symbol, interval, limit):
                return [Candle(price, EPOCH + (i + 1) * 1000) for i, price in enumerate([10, 10, 10, 11, 12, 13, 14, 15])]

            def market_buy(self, *args):
                raise AssertionError("expired cooldown cannot reuse a continuation after a stop")

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            history = trade_history_path(state, "BTCUSDT", "testnet")
            for order_id, side, amount, source in ((1, "BUY", "10", "strategy"), (2, "SELL", "11", "hosted_stop_loss")):
                order = {"orderId": order_id, "executedQty": "1", "cummulativeQuoteQty": amount, "transactTime": EPOCH - 1000 + order_id}
                record_trade_fill(history, "BTCUSDT", "testnet", order, side, source, ("entry" if side == "BUY" else "profitable trailing stop",))
            args = Namespace(state_file=str(state), symbol="BTCUSDT", interval="15m", execute=True, live=False, hosted_stop_loss=False, quote_size=10)
            decision = execute_cycle(Client(), args, config(buy_rsi_max=100), {})
            self.assertEqual(decision.action, "HOLD")
            self.assertIn("new bullish crossover", decision.reasons[0])


class InventoryClient(IntentAwareFake):
    def __init__(self, symbol="BNBUSDT"):
        self.symbol = symbol
        self.orders = {}
        self.responses = []
        self.submissions = 0

    def order_by_client_id(self, symbol, client_order_id):
        if client_order_id in self.orders:
            return self.orders[client_order_id]
        return super().order_by_client_id(symbol, client_order_id)

    def submit(self, client_order_id):
        self.submissions += 1
        result = {**self.responses.pop(0), "clientOrderId": client_order_id}
        self.orders[client_order_id] = result
        return result

    def market_buy(self, symbol, quote_quantity, client_order_id):
        return self.submit(client_order_id)

    def market_sell(self, symbol, quantity, client_order_id):
        return self.submit(client_order_id)

    def free_balance(self, asset):
        raise AssertionError("inventory ownership must never be inferred from account balance")

    def response(self, order_id, side, quantity, quote, commission="0", asset="USDT", status="FILLED"):
        return {
            **binance_order(order_id, status, quantity, quote, "pending", side, "MARKET", symbol=self.symbol, fills=[{
                "qty": quantity, "quoteQty": quote, "commission": commission, "commissionAsset": asset,
            }]),
            "transactTime": EPOCH + order_id * 1000,
        }


class InventoryLedgerTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            "baseAsset": "BNB", "quoteAsset": "USDT",
            "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"}],
        }

    def paths(self, directory):
        state = Path(directory) / "state.json"
        return state, trade_history_path(state, "BNBUSDT", "testnet"), order_intent_path(state, "BNBUSDT", "testnet")

    def execute(self, client, state, history, intent_path, response, signal_time=EPOCH):
        client.responses.append(response)
        side = response["side"]
        position = load_position(state, "BNBUSDT") if side == "SELL" else None
        intent = prepare_order_intent(
            intent_path, "BNBUSDT", "testnet", "market_buy" if side == "BUY" else "strategy_sell",
            "strategy", ("entry" if side == "BUY" else "take profit 4%",),
            quote_quantity=10 if side == "BUY" else None,
            quantity=float(response["executedQty"]) if side == "SELL" else None,
            position=position, signal_candle_close_time_ms=signal_time,
        )
        execute_prepared_intent(client, intent_path, state, history, self.info, intent)
        return intent

    def test_dust_survives_closure_and_is_combined_with_next_buy(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
            self.assertEqual(load_position(state, "BNBUSDT").quantity, 0.012)
            first = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            self.assertEqual(Decimal(first["residual_quantity"]), Decimal("0.000987"))
            self.execute(client, state, history, intent_path, client.response(2, "SELL", "0.012", "9.12", "0.00912"))
            self.assertIsNone(load_position(state, "BNBUSDT"))
            remaining = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            self.assertEqual(Decimal(remaining["quantity"]), Decimal("0.000987"))
            self.assertEqual(Decimal(remaining["cost_quote"]), Decimal(first["residual_cost_quote"]))
            # Reloaded journal/state, with no access to unrelated BNB in the account.
            restarted = InventoryClient()
            self.execute(restarted, state, history, intent_path, restarted.response(3, "BUY", "0.013", "9.75", "0.000013", "BNB"), EPOCH + 3000)
            self.assertEqual(load_position(state, "BNBUSDT").quantity, 0.013)
            ledger = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            self.assertEqual(Decimal(ledger["residual_quantity"]), Decimal("0.000974"))
            self.assertEqual(Decimal(ledger["cost_quote"]), Decimal(remaining["cost_quote"]) + Decimal("9.75"))

    def test_base_asset_sell_fee_is_not_misclassified_as_dust(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
            self.execute(client, state, history, intent_path, client.response(2, "SELL", "0.012", "9.12", "0.000012", "BNB"))
            self.assertEqual(Decimal(load_trade_history(history, "BNBUSDT", "testnet")["inventory"]["quantity"]), Decimal("0.000975"))

    def test_quote_and_third_asset_fees_preserve_base_quantity(self):
        for asset in ("USDT", "OTHER"):
            with self.subTest(asset=asset), tempfile.TemporaryDirectory() as directory:
                state, history, intent_path = self.paths(directory)
                client = InventoryClient()
                self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.01", asset))
                ledger = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
                self.assertEqual(Decimal(ledger["quantity"]), Decimal("0.013"))
                self.assertEqual(Decimal(ledger["cost_quote"]), Decimal("9.76" if asset == "USDT" else "9.75"))

    def test_decimal_execution_precision_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.123456789123456789", "9", "0.000000000123456789", "BNB"))
            ledger = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            self.assertEqual(Decimal(ledger["quantity"]), Decimal("0.123456789000000000"))

    def test_partial_buy_recovery_upserts_inventory_instead_of_adding_it_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            partial = client.response(1, "BUY", "0.006", "4.50", "0.000006", "BNB", "PARTIALLY_FILLED")
            with self.assertRaises(PendingIntentError):
                self.execute(client, state, history, intent_path, partial)
            intent = load_order_intent(intent_path, "BNBUSDT", "testnet")
            completed = client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB")
            client.orders[intent["client_order_id"]] = {**completed, "clientOrderId": intent["client_order_id"]}
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            journal = load_trade_history(history, "BNBUSDT", "testnet")
            self.assertEqual(len(journal["fills"]), 1)
            self.assertEqual(Decimal(journal["inventory"]["quantity"]), Decimal("0.012987"))
            self.assertEqual(client.submissions, 1)

    def test_restart_after_journal_write_does_not_double_credit_or_debit(self):
        for side in ("BUY", "SELL"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                state, history, intent_path = self.paths(directory)
                client = InventoryClient()
                if side == "SELL":
                    self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
                response = client.response(2, side, "0.013" if side == "BUY" else "0.012", "9.75" if side == "BUY" else "9.12", "0.000013" if side == "BUY" else "0", "BNB")
                with patch("trader.save_position", side_effect=OSError("crash after durable journal")):
                    with self.assertRaises(OSError):
                        self.execute(client, state, history, intent_path, response)
                before = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
                intent = load_order_intent(intent_path, "BNBUSDT", "testnet")
                execute_prepared_intent(client, intent_path, state, history, self.info, intent)
                self.assertEqual(load_trade_history(history, "BNBUSDT", "testnet")["inventory"], before)
                self.assertFalse(intent_path.exists())

    def test_repeated_execution_accepts_equivalent_commission_decimal_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            intent = self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.0000130", "BNB"))
            before = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            client.orders[intent["client_order_id"]]["fills"][0]["commission"] = "0.000013"
            save_order_intent(intent_path, intent)
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertEqual(load_trade_history(history, "BNBUSDT", "testnet")["inventory"], before)

    def test_recovered_sell_uses_trade_fees_and_actual_fill_time(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
            response = client.response(2, "SELL", "0.012", "9.12", "0.000012", "BNB")
            rows = response.pop("fills")
            response["transactTime"] = EPOCH + 999000  # Query/cancellation time is not the fill time.
            client.trades = lambda symbol, order_id: [{**rows[0], "symbol": symbol, "orderId": order_id, "time": EPOCH + 2000}]
            self.execute(client, state, history, intent_path, response)
            journal = load_trade_history(history, "BNBUSDT", "testnet")
            self.assertEqual(Decimal(journal["inventory"]["quantity"]), Decimal("0.000975"))
            self.assertEqual(journal["fills"][-1]["timestamp_utc"], utc(EPOCH + 2000))

    def test_cumulative_sell_updates_do_not_restart_cooldown_without_new_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
            intent = self.execute(client, state, history, intent_path, client.response(2, "SELL", "0.012", "9.12"))
            client.orders[intent["client_order_id"]]["transactTime"] = EPOCH + 999000
            save_order_intent(intent_path, intent)
            execute_prepared_intent(client, intent_path, state, history, self.info, intent)
            self.assertEqual(load_trade_history(history, "BNBUSDT", "testnet")["fills"][-1]["timestamp_utc"], utc(EPOCH + 2000))

    def test_mismatched_trade_identity_cannot_credit_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            response = client.response(1, "BUY", "0.013", "9.75")
            row = response.pop("fills")[0]
            client.trades = lambda symbol, order_id: [{**row, "orderId": order_id + 1, "symbol": symbol}]
            with self.assertRaises(PendingIntentError):
                self.execute(client, state, history, intent_path, response)
            self.assertFalse(state.exists())
            self.assertFalse(history.exists())
            self.assertTrue(intent_path.exists())

    def test_missing_sell_commissions_leave_intent_and_position_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            state, history, intent_path = self.paths(directory)
            client = InventoryClient()
            self.execute(client, state, history, intent_path, client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"))
            position = load_position(state, "BNBUSDT")
            response = client.response(2, "SELL", "0.012", "9.12")
            response.pop("fills")
            client.trades = lambda symbol, order_id: []
            with self.assertRaises(PendingIntentError):
                self.execute(client, state, history, intent_path, response)
            self.assertEqual(load_position(state, "BNBUSDT"), position)
            self.assertTrue(intent_path.exists())
            self.assertEqual(len(load_trade_history(history, "BNBUSDT", "testnet")["fills"]), 1)

    def test_legacy_dust_migration_uses_real_order_fees(self):
        with tempfile.TemporaryDirectory() as directory:
            _, history, _ = self.paths(directory)
            client = InventoryClient()
            orders = [client.response(1, "BUY", "0.013", "9.75", "0.000013", "BNB"), client.response(2, "SELL", "0.012", "9.12")]
            for order in orders:
                record_trade_fill(history, "BNBUSDT", "testnet", order, order["side"], "strategy", ("legacy",))
            client.trades = lambda symbol, order_id: [{
                **orders[order_id - 1]["fills"][0], "orderId": order_id,
                "symbol": symbol, "time": EPOCH + order_id * 1000,
            }]
            migrate_inventory_history(client, history, "BNBUSDT", "testnet", self.info, None)
            ledger = load_trade_history(history, "BNBUSDT", "testnet")["inventory"]
            self.assertEqual(Decimal(ledger["quantity"]), Decimal("0.000987"))

    def test_failed_legacy_fee_lookup_does_not_guess_or_modify_history(self):
        with tempfile.TemporaryDirectory() as directory:
            _, history, _ = self.paths(directory)
            client = InventoryClient()
            order = client.response(1, "BUY", "0.013", "9.75")
            record_trade_fill(history, "BNBUSDT", "testnet", order, "BUY", "strategy", ("legacy",))
            original = history.read_text()
            client.trades = lambda symbol, order_id: []
            with self.assertRaises(PendingIntentError):
                migrate_inventory_history(client, history, "BNBUSDT", "testnet", self.info, None)
            self.assertEqual(history.read_text(), original)


class PostEntryStopTests(unittest.TestCase):
    def test_initial_and_restored_stops_ignore_pre_buy_candle_even_after_restart(self):
        class Client(IntentAwareFake):
            buys = 0
            stops = 0

            def candles(self, symbol, interval, limit):
                prices = [79000, 79000, 81000, 79000, 79000, 82000, 80000, 70000]
                return [Candle(price, EPOCH + (index - 6) * 900000 - 1) for index, price in enumerate(prices)]

            def market_buy(self, symbol, quote_size, client_order_id):
                self.buys += 1
                return {
                    **binance_order(1, "FILLED", "0.0001", "7.6", client_order_id, "BUY", "MARKET", fills=[{
                        "qty": "0.0001", "quoteQty": "7.6", "commission": "0", "commissionAsset": "USDT",
                    }]),
                    "transactTime": EPOCH,
                }

            def free_balance(self, asset):
                return 10  # Unrelated holdings must not become part of the protected quantity.

            def ticker_price(self, symbol):
                return 76000

            def place_stop_loss(self, symbol, quantity, stop_price, client_order_id):
                self.stops += 1
                self.stop_price = stop_price
                self.stop_quantity = quantity
                return binance_order(2, "NEW", 0, 0, client_order_id, "SELL", "STOP_LOSS", orig_qty=quantity)

            def order(self, symbol, order_id):
                return binance_order(order_id, "NEW", 0, 0, "existing-stop", "SELL", "STOP_LOSS", orig_qty="0.0001")

            def market_sell(self, *args):
                raise AssertionError("a pre-entry candle must not trigger a protective or take-profit exit")

        info = {
            "baseAsset": "BTC", "quoteAsset": "USDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                {"filterType": "NOTIONAL", "minNotional": "5", "applyMinToMarket": True, "avgPriceMins": 0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            args = Namespace(symbol="BTCUSDT", interval="15m", state_file=str(state), execute=True, live=False, hosted_stop_loss=True, quote_size=10)
            client = Client()
            self.assertEqual(execute_cycle(client, args, config(), info).action, "BUY")
            self.assertEqual(client.stop_price, 74480)
            self.assertEqual(client.stop_quantity, 0.0001)
            self.assertEqual(execute_cycle(Client(), args, config(), info).action, "HOLD")
            # Model a restart after BUY reconciliation but before initial stop placement.
            save_position(state, Position("BTCUSDT", 0.0001, 76000))
            restarted = Client()
            self.assertEqual(execute_cycle(restarted, args, config(), info).action, "HOLD")
            self.assertEqual(restarted.stop_price, 74480)
            journal = load_trade_history(trade_history_path(state, "BTCUSDT", "testnet"), "BTCUSDT", "testnet")
            self.assertEqual(journal["entry_guard"]["last_buy_signal_candle_close_time_utc"], utc(EPOCH - 1))


if __name__ == "__main__":
    unittest.main()
