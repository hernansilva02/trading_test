#!/usr/bin/env python3
"""Configurable Binance Spot trading bot with safe defaults."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("trader")
TESTNET_URL = "https://testnet.binance.vision"
MAINNET_URL = "https://api.binance.com"


class BinanceError(RuntimeError):
    """Raised when Binance rejects an API request."""


@dataclass(frozen=True)
class StrategyConfig:
    fast_sma: int
    slow_sma: int
    rsi_period: int
    buy_on_bullish_trend: bool
    sell_on_bearish_trend: bool
    buy_below: float | None
    sell_above: float | None
    buy_rsi_below: float | None
    sell_rsi_above: float | None
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    entry_price: float
    stop_order_id: int | None = None
    stop_price: float | None = None


@dataclass(frozen=True)
class Decision:
    action: str
    reasons: tuple[str, ...]
    price: float
    fast_sma: float
    slow_sma: float
    rsi: float


class BinanceClient:
    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self._time_offset_ms = 0

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        values = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
            values["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            values["recvWindow"] = 5_000
            query_to_sign = urllib.parse.urlencode(values)
            values["signature"] = hmac.new(
                self.api_secret.encode(), query_to_sign.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key

        query = urllib.parse.urlencode(values)
        url = f"{self.base_url}{path}"
        data = None
        if method == "GET" and query:
            url = f"{url}?{query}"
        elif query:
            data = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("msg", body)
            except json.JSONDecodeError:
                detail = body
            raise BinanceError(f"Binance HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise BinanceError(f"Could not reach Binance: {error.reason}") from error

    def synchronize_time(self) -> None:
        response = self._request("GET", "/api/v3/time")
        self._time_offset_ms = int(response["serverTime"]) - int(time.time() * 1000)

    def closes(self, symbol: str, interval: str, limit: int) -> list[float]:
        rows = self._request(
            "GET",
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        return [float(row[4]) for row in rows]

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        response = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})
        symbols = response.get("symbols", [])
        if not symbols:
            raise BinanceError(f"Unknown symbol: {symbol}")
        return symbols[0]

    def average_price(self, symbol: str) -> float:
        response = self._request("GET", "/api/v3/avgPrice", {"symbol": symbol})
        return float(response["price"])

    def ticker_price(self, symbol: str) -> float:
        response = self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        return float(response["price"])

    def free_balance(self, asset: str) -> float:
        account = self._request("GET", "/api/v3/account", signed=True)
        for balance in account["balances"]:
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0

    def market_buy(self, symbol: str, quote_quantity: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": decimal_string(quote_quantity),
                "newOrderRespType": "FULL",
            },
            signed=True,
        )

    def market_sell(self, symbol: str, quantity: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": decimal_string(quantity),
                "newOrderRespType": "FULL",
            },
            signed=True,
        )

    def place_stop_loss(self, symbol: str, quantity: float, stop_price: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_LOSS",
                "quantity": decimal_string(quantity),
                "stopPrice": decimal_string(stop_price),
                "newOrderRespType": "RESULT",
            },
            signed=True,
        )

    def order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request(
            "GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request(
            "DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True
        )


def decimal_string(value: float) -> str:
    return format(Decimal(str(value)), "f")


def floor_to_step(value: float, step: str) -> float:
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(step)
    if decimal_step == 0:
        return value
    units = (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN)
    return float(units * decimal_step)


def simple_rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        raise ValueError(f"RSI requires at least {period + 1} prices")
    changes = [new - old for old, new in zip(closes[-period - 1 : -1], closes[-period:])]
    average_gain = sum(max(change, 0.0) for change in changes) / period
    average_loss = sum(max(-change, 0.0) for change in changes) / period
    if average_gain == 0 and average_loss == 0:
        return 50.0
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def decide(
    closes: list[float], config: StrategyConfig, position: Position | None
) -> Decision:
    required = max(config.slow_sma + 1, config.rsi_period + 1)
    if len(closes) < required:
        raise ValueError(f"Strategy requires at least {required} prices")

    price = closes[-1]
    fast = sum(closes[-config.fast_sma :]) / config.fast_sma
    slow = sum(closes[-config.slow_sma :]) / config.slow_sma
    previous_fast = sum(closes[-config.fast_sma - 1 : -1]) / config.fast_sma
    previous_slow = sum(closes[-config.slow_sma - 1 : -1]) / config.slow_sma
    bullish_crossover = previous_fast <= previous_slow and fast > slow
    bearish_crossover = previous_fast >= previous_slow and fast < slow
    rsi = simple_rsi(closes, config.rsi_period)

    if position is None:
        conditions: list[tuple[bool, str]] = []
        if config.buy_on_bullish_trend:
            conditions.append((bullish_crossover, "fast SMA crossed above slow SMA"))
        if config.buy_below is not None:
            conditions.append((price <= config.buy_below, f"price <= {config.buy_below:g}"))
        if config.buy_rsi_below is not None:
            conditions.append((rsi <= config.buy_rsi_below, f"RSI <= {config.buy_rsi_below:g}"))
        if conditions and all(matched for matched, _ in conditions):
            return Decision("BUY", tuple(reason for _, reason in conditions), price, fast, slow, rsi)
        return Decision("HOLD", ("entry conditions not met",), price, fast, slow, rsi)

    exit_reasons: list[str] = []
    if config.sell_on_bearish_trend and bearish_crossover:
        exit_reasons.append("fast SMA crossed below slow SMA")
    if config.sell_above is not None and price >= config.sell_above:
        exit_reasons.append(f"price >= {config.sell_above:g}")
    if config.sell_rsi_above is not None and rsi >= config.sell_rsi_above:
        exit_reasons.append(f"RSI >= {config.sell_rsi_above:g}")
    if config.stop_loss_pct > 0 and price <= position.entry_price * (1 - config.stop_loss_pct / 100):
        exit_reasons.append(f"stop loss {config.stop_loss_pct:g}%")
    if config.take_profit_pct > 0 and price >= position.entry_price * (1 + config.take_profit_pct / 100):
        exit_reasons.append(f"take profit {config.take_profit_pct:g}%")
    if exit_reasons:
        return Decision("SELL", tuple(exit_reasons), price, fast, slow, rsi)
    return Decision("HOLD", ("exit conditions not met",), price, fast, slow, rsi)


def load_position(path: Path, symbol: str) -> Position | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        position = Position(**data)
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as error:
        raise RuntimeError(f"Invalid state file {path}: {error}") from error
    if position.symbol != symbol:
        raise RuntimeError(
            f"State file contains {position.symbol}, but --symbol is {symbol}; use another --state-file"
        )
    return position


def save_position(path: Path, position: Position | None) -> None:
    if position is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(position), indent=2) + "\n")
    temporary.replace(path)


def filter_value(symbol_info: dict[str, Any], filter_type: str, key: str) -> str | None:
    for item in symbol_info["filters"]:
        if item["filterType"] == filter_type:
            return item.get(key)
    return None


def filter_item(symbol_info: dict[str, Any], filter_type: str) -> dict[str, Any] | None:
    for item in symbol_info["filters"]:
        if item["filterType"] == filter_type:
            return item
    return None


def minimum_market_notional(symbol_info: dict[str, Any]) -> float:
    value = filter_value(symbol_info, "NOTIONAL", "minNotional")
    if value is None:
        value = filter_value(symbol_info, "MIN_NOTIONAL", "minNotional")
    return float(value or 0)


def market_minimum_notional(symbol_info: dict[str, Any]) -> tuple[float, int]:
    rule = filter_item(symbol_info, "NOTIONAL")
    if rule is not None:
        if not rule.get("applyMinToMarket", False):
            return 0.0, 0
        return float(rule.get("minNotional", 0)), int(rule.get("avgPriceMins", 0))
    rule = filter_item(symbol_info, "MIN_NOTIONAL")
    if rule is not None and rule.get("applyToMarket", False):
        return float(rule.get("minNotional", 0)), int(rule.get("avgPriceMins", 0))
    return 0.0, 0


def market_reference_price(
    client: BinanceClient, symbol: str, symbol_info: dict[str, Any], fallback: float
) -> float:
    minimum_notional, average_minutes = market_minimum_notional(symbol_info)
    if minimum_notional <= 0:
        return fallback
    return client.average_price(symbol) if average_minutes > 0 else client.ticker_price(symbol)


def market_sell_quantity(
    quantity: float, reference_price: float, symbol_info: dict[str, Any]
) -> float:
    market_filter = filter_item(symbol_info, "MARKET_LOT_SIZE") or {}
    lot_filter = filter_item(symbol_info, "LOT_SIZE") or {}
    market_step = str(market_filter.get("stepSize", "0"))
    step = market_step if Decimal(market_step) > 0 else str(lot_filter.get("stepSize", "0"))
    rounded_quantity = floor_to_step(quantity, step)

    market_minimum = Decimal(str(market_filter.get("minQty", "0")))
    minimum_quantity = (
        market_minimum if market_minimum > 0 else Decimal(str(lot_filter.get("minQty", "0")))
    )
    if Decimal(str(rounded_quantity)) < minimum_quantity or rounded_quantity <= 0:
        raise ValueError(
            f"rounded sell quantity {rounded_quantity:g} is below Binance's minimum "
            f"{float(minimum_quantity):g}"
        )

    minimum_notional, average_minutes = market_minimum_notional(symbol_info)
    notional = Decimal(str(rounded_quantity)) * Decimal(str(reference_price))
    if minimum_notional > 0 and notional < Decimal(str(minimum_notional)):
        price_description = (
            f"{average_minutes}-minute average" if average_minutes > 0 else "current reference price"
        )
        raise ValueError(
            f"rounded sell quantity {rounded_quantity:g} {symbol_info['baseAsset']} has "
            f"{price_description} notional {float(notional):.2f} {symbol_info['quoteAsset']}, "
            f"below Binance's {minimum_notional:.2f} minimum"
        )
    return rounded_quantity


def net_base_quantity(order: dict[str, Any], base_asset: str) -> float:
    quantity = Decimal(str(order["executedQty"]))
    for fill in order.get("fills", []):
        if fill.get("commissionAsset") == base_asset:
            quantity -= Decimal(str(fill.get("commission", "0")))
    return max(0.0, float(quantity))


def protective_stop_values(
    quantity: float,
    entry_price: float,
    stop_loss_pct: float,
    symbol_info: dict[str, Any],
) -> tuple[float, float]:
    step = filter_value(symbol_info, "LOT_SIZE", "stepSize") or "0"
    tick = filter_value(symbol_info, "PRICE_FILTER", "tickSize") or "0"
    sell_quantity = floor_to_step(quantity, step)
    stop_price = floor_to_step(entry_price * (1 - stop_loss_pct / 100), tick)
    minimum_quantity = float(filter_value(symbol_info, "LOT_SIZE", "minQty") or 0)
    minimum_notional = minimum_market_notional(symbol_info)
    if sell_quantity < minimum_quantity or sell_quantity <= 0:
        raise ValueError(
            f"protective sell quantity {sell_quantity:g} is below the minimum {minimum_quantity:g}"
        )
    if sell_quantity * stop_price < minimum_notional:
        raise ValueError(
            "the requested buy is too small for a protected stop: "
            f"stop notional would be {sell_quantity * stop_price:.2f} "
            f"{symbol_info['quoteAsset']}, below Binance's {minimum_notional:.2f} minimum"
        )
    return sell_quantity, stop_price


def validate_args(args: argparse.Namespace) -> None:
    if args.fast_sma <= 0 or args.slow_sma <= 0 or args.rsi_period <= 0:
        raise ValueError("SMA windows and RSI period must be positive")
    if args.fast_sma >= args.slow_sma:
        raise ValueError("--fast-sma must be smaller than --slow-sma")
    if args.quote_size <= 0 or args.poll_seconds <= 0:
        raise ValueError("--quote-size and --poll-seconds must be positive")
    for name in ("buy_rsi_below", "sell_rsi_above"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 100")
    if not 0 <= args.stop_loss_pct < 100 or args.take_profit_pct < 0:
        raise ValueError("--stop-loss-pct must be below 100 and risk percentages cannot be negative")
    if args.live and args.execute and not args.confirm_live:
        raise ValueError("live orders require --confirm-live")


def execute_cycle(
    client: BinanceClient,
    args: argparse.Namespace,
    config: StrategyConfig,
    info: dict[str, Any],
) -> Decision:
    state_path = Path(args.state_file)
    position = load_position(state_path, args.symbol)
    # Fetch one active candle to discard and one extra closed candle for crossover detection.
    limit = max(config.slow_sma + 1, config.rsi_period + 1) + 1
    closes = client.closes(args.symbol, args.interval, limit)
    hosted_stop_filled = False
    hosted_stop_order: dict[str, Any] | None = None
    hosted_stop_executed = 0.0

    if args.execute and position is not None and position.stop_order_id is not None:
        stop_order = client.order(args.symbol, position.stop_order_id)
        hosted_stop_executed = float(stop_order.get("executedQty", 0))
        if stop_order["status"] == "FILLED" or hosted_stop_executed >= position.quantity:
            save_position(state_path, None)
            position = None
            hosted_stop_filled = True
            LOGGER.info("Hosted stop-loss order %s filled; local position cleared", stop_order["orderId"])
        elif stop_order["status"] in {"NEW", "PENDING_NEW", "PARTIALLY_FILLED"}:
            hosted_stop_order = stop_order
        else:
            remaining = max(0.0, position.quantity - hosted_stop_executed)
            position = (
                Position(position.symbol, remaining, position.entry_price) if remaining > 0 else None
            )
            save_position(state_path, position)
            if position is None:
                hosted_stop_filled = True
            LOGGER.critical(
                "Hosted stop-loss order %s is %s after filling %.8f; "
                "saved the remaining position without the inactive stop",
                stop_order["orderId"],
                stop_order["status"],
                hosted_stop_executed,
            )

    decision = decide(closes[:-1], config, position)
    if hosted_stop_filled:
        decision = Decision(
            "HOLD",
            ("hosted stop-loss filled during this cycle",),
            decision.price,
            decision.fast_sma,
            decision.slow_sma,
            decision.rsi,
        )

    sell_quantity: float | None = None
    sell_reference_price: float | None = None
    if args.execute and decision.action == "SELL":
        assert position is not None
        sell_reference_price = market_reference_price(client, args.symbol, info, decision.price)
        if hosted_stop_order is not None:
            original = float(hosted_stop_order.get("origQty", position.quantity))
            available_for_exit = min(
                max(0.0, position.quantity - hosted_stop_executed),
                max(0.0, original - hosted_stop_executed),
            )
        else:
            available_for_exit = min(position.quantity, client.free_balance(info["baseAsset"]))
        try:
            sell_quantity = market_sell_quantity(available_for_exit, sell_reference_price, info)
        except ValueError as error:
            decision = Decision(
                "HOLD",
                (f"SELL deferred: {error}; no order submitted and state was preserved",),
                decision.price,
                decision.fast_sma,
                decision.slow_sma,
                decision.rsi,
            )
    LOGGER.info(
        "%s %s price=%.8f fast_sma=%.8f slow_sma=%.8f rsi=%.2f reason=%s",
        args.symbol,
        decision.action,
        decision.price,
        decision.fast_sma,
        decision.slow_sma,
        decision.rsi,
        "; ".join(decision.reasons),
    )

    if decision.action == "HOLD" or not args.execute:
        if decision.action != "HOLD":
            LOGGER.warning("Dry run: order not submitted; add --execute to enable orders")
        return decision

    if decision.action == "BUY":
        estimated_exit_price = decision.price * (
            1 - config.stop_loss_pct / 100 if config.stop_loss_pct > 0 else 1
        )
        if args.hosted_stop_loss and config.stop_loss_pct > 0:
            # Reserve a small commission buffer before committing real funds.
            protective_stop_values(
                args.quote_size / decision.price * 0.998,
                decision.price,
                config.stop_loss_pct,
                info,
            )
        market_sell_quantity(
            args.quote_size / decision.price * 0.998,
            estimated_exit_price,
            info,
        )
        result = client.market_buy(args.symbol, args.quote_size)
        quantity = float(result["executedQty"])
        spent = float(result["cummulativeQuoteQty"])
        if quantity <= 0:
            raise BinanceError(f"Buy order did not fill: {result}")
        new_position = Position(args.symbol, quantity, spent / quantity)
        save_position(state_path, new_position)
        LOGGER.info("Bought %.8f %s at average %.8f", quantity, info["baseAsset"], new_position.entry_price)
        owned_quantity = net_base_quantity(result, info["baseAsset"])
        available = min(owned_quantity, client.free_balance(info["baseAsset"]))
        reconciled_exit_price = new_position.entry_price * (
            1 - config.stop_loss_pct / 100 if config.stop_loss_pct > 0 else 1
        )
        try:
            sellable_quantity = market_sell_quantity(available, reconciled_exit_price, info)
        except ValueError as error:
            LOGGER.critical(
                "Buy filled but the post-commission quantity cannot satisfy a future market exit; "
                "gross position remains in %s: %s",
                state_path,
                error,
            )
            raise
        new_position = Position(args.symbol, sellable_quantity, new_position.entry_price)
        save_position(state_path, new_position)
        if args.hosted_stop_loss and config.stop_loss_pct > 0:
            stop_quantity, stop_price = protective_stop_values(
                available, new_position.entry_price, config.stop_loss_pct, info
            )
            try:
                stop_order = client.place_stop_loss(args.symbol, stop_quantity, stop_price)
            except BinanceError as error:
                LOGGER.critical(
                    "Buy filled but hosted stop placement failed; position remains in %s: %s",
                    state_path,
                    error,
                )
                raise
            new_position = Position(
                args.symbol,
                stop_quantity,
                new_position.entry_price,
                int(stop_order["orderId"]),
                stop_price,
            )
            save_position(state_path, new_position)
            LOGGER.info(
                "Hosted stop-loss order %s placed: sell %.8f %s if price reaches %.8f",
                stop_order["orderId"],
                stop_quantity,
                info["baseAsset"],
                stop_price,
            )
        return decision

    assert position is not None
    remaining_before_market = position.quantity
    if position.stop_order_id is not None:
        canceled = client.cancel_order(args.symbol, position.stop_order_id)
        LOGGER.info("Canceled hosted stop-loss order %s before strategy exit", canceled["orderId"])
        canceled_executed = float(canceled.get("executedQty", hosted_stop_executed))
        remaining_before_market = max(0.0, position.quantity - canceled_executed)
        position = (
            Position(position.symbol, remaining_before_market, position.entry_price)
            if remaining_before_market > 0
            else None
        )
        save_position(state_path, position)
        if position is None:
            LOGGER.info("Hosted stop filled while cancellation was being processed; state cleared")
            return decision
        available = min(remaining_before_market, client.free_balance(info["baseAsset"]))
        assert sell_reference_price is not None
        try:
            sell_quantity = market_sell_quantity(available, sell_reference_price, info)
        except ValueError as error:
            raise BinanceError(
                f"hosted stop was canceled but the released quantity cannot be sold: {error}"
            ) from error
    assert sell_quantity is not None
    result = client.market_sell(args.symbol, sell_quantity)
    sold = float(result["executedQty"])
    remaining = max(0.0, remaining_before_market - sold)
    minimum = float(filter_value(info, "LOT_SIZE", "minQty") or 0)
    save_position(
        state_path,
        Position(position.symbol, remaining, position.entry_price) if remaining >= minimum else None,
    )
    LOGGER.info("Sold %.8f %s", sold, info["baseAsset"])
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based Binance Spot trading bot")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Candle interval (default: 1m)")
    parser.add_argument("--poll-seconds", type=float, default=60, help="Seconds between decisions")
    parser.add_argument("--once", action="store_true", help="Run one decision cycle and exit")
    parser.add_argument("--execute", action="store_true", help="Submit orders; otherwise dry-run")
    parser.add_argument("--live", action="store_true", help="Use Binance mainnet instead of Spot Testnet")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Acknowledge that --execute --live uses real funds",
    )
    parser.add_argument("--state-file", default=".trader-state.json", help="Bot-owned position state")
    parser.add_argument("--quote-size", type=float, default=25, help="Quote asset spent per buy")
    parser.add_argument("--fast-sma", type=int, default=9)
    parser.add_argument("--slow-sma", type=int, default=21)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument(
        "--buy-on-bullish-trend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require fast SMA to cross above slow SMA for entry (default: enabled)",
    )
    parser.add_argument(
        "--sell-on-bearish-trend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit when fast SMA crosses below slow SMA (default: enabled)",
    )
    parser.add_argument("--buy-below", type=float, help="Require market price at or below this value")
    parser.add_argument("--sell-above", type=float, help="Exit at or above this market price")
    parser.add_argument("--buy-rsi-below", type=float, help="Require RSI at or below this value")
    parser.add_argument("--sell-rsi-above", type=float, help="Exit when RSI is at or above this value")
    parser.add_argument("--stop-loss-pct", type=float, default=2.0, help="0 disables (default: 2)")
    parser.add_argument("--take-profit-pct", type=float, default=4.0, help="0 disables (default: 4)")
    parser.add_argument(
        "--hosted-stop-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the stop at Binance after a filled buy (default: enabled)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.symbol = args.symbol.upper()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        validate_args(args)
        config = StrategyConfig(
            fast_sma=args.fast_sma,
            slow_sma=args.slow_sma,
            rsi_period=args.rsi_period,
            buy_on_bullish_trend=args.buy_on_bullish_trend,
            sell_on_bearish_trend=args.sell_on_bearish_trend,
            buy_below=args.buy_below,
            sell_above=args.sell_above,
            buy_rsi_below=args.buy_rsi_below,
            sell_rsi_above=args.sell_rsi_above,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
        )
        client = BinanceClient(
            MAINNET_URL if args.live else TESTNET_URL,
            os.environ.get("BINANCE_API_KEY", ""),
            os.environ.get("BINANCE_API_SECRET", ""),
        )
        info = client.symbol_info(args.symbol)
        if info.get("status") != "TRADING":
            raise BinanceError(f"{args.symbol} is not currently trading")
        if args.execute:
            client.synchronize_time()

        while True:
            try:
                execute_cycle(client, args, config, info)
            except (BinanceError, RuntimeError, ValueError) as error:
                LOGGER.error("Cycle failed: %s", error)
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
    except (BinanceError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
