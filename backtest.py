#!/usr/bin/env python3
"""Historical, single-symbol Spot simulation using the live trader's strategy."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any

from trader import (
    MAINNET_URL, BinanceClient, BinanceError, Candle, Position, StrategyConfig,
    add_strategy_arguments, atomic_write_text, buy_entry_block_reason, candle_after_entry,
    decide, empty_trade_history, filter_item, filter_value, inventory_opening,
    inventory_snapshot, market_sell_quantity, protective_stop_values,
    strategy_config_from_args, trailing_stop_price, validate_strategy_args,
)


INTERVALS_MS = {
    "1s": 1000, "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
    "30m": 1800000, "1h": 3600000, "2h": 7200000, "4h": 14400000,
    "6h": 21600000, "8h": 28800000, "12h": 43200000, "1d": 86400000,
    "3d": 259200000, "1w": 604800000,
}


def decimal_value(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid decimal: {value!r}") from error
    if isinstance(value, bool) or not number.is_finite() or number < 0:
        raise ValueError("Prices, volumes, costs and filters must be finite and non-negative")
    return number


def decimal_argument(value: str) -> Decimal:
    try:
        return decimal_value(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_time(value: str) -> int:
    """Accept epoch milliseconds or an ISO-8601 time; date-only/naive times mean UTC."""
    if value.isdigit():
        return int(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid timestamp {value!r}; use ISO-8601 or epoch milliseconds") from error


def utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def round_down(value: Decimal, step: str) -> Decimal:
    increment = decimal_value(step)
    return (value / increment).to_integral_value(rounding=ROUND_DOWN) * increment if increment else value


@dataclass(frozen=True)
class HistoricalCandle:
    open_time_ms: int
    close_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def validate_candles(candles: list[HistoricalCandle], interval: str) -> None:
    if interval not in INTERVALS_MS:
        raise ValueError(f"Unsupported fixed-duration interval: {interval}")
    if not candles:
        raise ValueError("No completed historical candles in the requested range")
    duration = INTERVALS_MS[interval]
    previous = None
    for candle in candles:
        prices = (candle.open, candle.high, candle.low, candle.close)
        if (
            any(not price.is_finite() or price <= 0 for price in prices)
            or not candle.volume.is_finite() or candle.volume < 0
            or candle.low > min(candle.open, candle.close)
            or candle.high < max(candle.open, candle.close)
            or candle.low > candle.high
            or candle.open_time_ms < 0
            or candle.close_time_ms != candle.open_time_ms + duration - 1
        ):
            raise ValueError(f"Invalid OHLCV candle at {utc(candle.open_time_ms)}")
        if previous is not None and candle.open_time_ms != previous.open_time_ms + duration:
            raise ValueError("Historical candles must be ordered, unique and contiguous; missing bars can hide stops")
        previous = candle


def load_csv(path: Path, interval: str, *, now_ms: int | None = None) -> list[HistoricalCandle]:
    if interval not in INTERVALS_MS:
        raise ValueError(f"Unsupported fixed-duration interval: {interval}")
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    candles = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        if len(names) != len(reader.fieldnames or []):
            raise ValueError("CSV contains duplicate column headers")
        open_column = "open_time_ms" if "open_time_ms" in names else "open_time"
        close_column = "close_time_ms" if "close_time_ms" in names else "close_time"
        if not {open_column, "open", "high", "low", "close", "volume"} <= names:
            raise ValueError("CSV requires open_time (or open_time_ms), open, high, low, close, volume")
        for line_number, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ValueError("unexpected extra columns")
                opened = parse_time(row[open_column])
                closed = parse_time(row[close_column]) if row.get(close_column) else opened + INTERVALS_MS[interval] - 1
                candle = HistoricalCandle(
                    opened, closed,
                    *(decimal_value(row[key]) for key in ("open", "high", "low", "close", "volume")),
                )
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Invalid CSV row {line_number}: {error}") from error
            if candle.close_time_ms < now_ms:
                candles.append(candle)
    validate_candles(candles, interval)
    return candles


def download_candles(
    client: BinanceClient, symbol: str, interval: str, start_ms: int, end_ms: int,
    *, now_ms: int | None = None,
) -> list[HistoricalCandle]:
    """Public GET requests only; end is exclusive and unfinished candles are discarded."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if interval not in INTERVALS_MS or start_ms < 0 or end_ms <= start_ms:
        raise ValueError("Historical download requires a supported interval and a valid time range")
    end_ms = min(end_ms, now_ms)
    duration = INTERVALS_MS[interval]
    candles = []
    cursor = start_ms
    while cursor < end_ms:
        rows = client._request("GET", "/api/v3/klines", {
            "symbol": symbol, "interval": interval, "startTime": cursor,
            "endTime": end_ms - 1, "limit": 1000,
        })
        if not isinstance(rows, list):
            raise ValueError("Binance returned an invalid historical candle response")
        if not rows:
            break
        for row in rows:
            try:
                candle = HistoricalCandle(
                    int(row[0]), int(row[6]),
                    *(decimal_value(row[index]) for index in (1, 2, 3, 4, 5)),
                )
            except (ValueError, TypeError, IndexError, KeyError) as error:
                raise ValueError("Binance returned an invalid OHLCV row") from error
            if candle.open_time_ms < cursor:
                raise ValueError("Binance pagination returned an old or duplicate candle")
            if candle.close_time_ms < end_ms:
                candles.append(candle)
        next_cursor = int(rows[-1][0]) + duration
        if next_cursor <= cursor:
            raise ValueError("Historical pagination did not advance")
        cursor = next_cursor
    validate_candles(candles, interval)
    return candles


def render_csv(candles: list[HistoricalCandle]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"))
    for candle in candles:
        writer.writerow((
            candle.open_time_ms, candle.close_time_ms, candle.open,
            candle.high, candle.low, candle.close, candle.volume,
        ))
    return output.getvalue()


def generic_symbol_info(symbol: str) -> dict[str, Any]:
    for quote in ("FDUSD", "USDT", "USDC", "BTC", "ETH", "BNB"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return {
                "symbol": symbol, "baseAsset": symbol[:-len(quote)], "quoteAsset": quote,
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.00000001", "minQty": "0.00000001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.00000001"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5", "applyToMarket": True, "avgPriceMins": 0},
                ],
            }
    raise ValueError("Unknown quote asset for generic filters; supply --filters-json")


def prepare_symbol_info(info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(info, dict):
        raise ValueError("Symbol filters must be a JSON object")
    info = json.loads(json.dumps(info))
    if info.get("symbol", args.symbol) != args.symbol or not all(
        isinstance(info.get(key), str) and info[key] for key in ("baseAsset", "quoteAsset")
    ) or not isinstance(info.get("filters"), list):
        raise ValueError("Symbol filters must describe the selected symbol and its base/quote assets")
    if any(not isinstance(item, dict) or not isinstance(item.get("filterType"), str) for item in info["filters"]):
        raise ValueError("Each symbol filter must have a filterType")
    if len({item["filterType"] for item in info["filters"]}) != len(info["filters"]):
        raise ValueError("Symbol filters must not contain duplicate filter types")
    info["symbol"] = args.symbol
    for filter_type, key, override in (
        ("LOT_SIZE", "stepSize", args.quantity_step),
        ("LOT_SIZE", "minQty", args.min_quantity),
        ("PRICE_FILTER", "tickSize", args.price_tick),
    ):
        item = filter_item(info, filter_type)
        if item is None:
            item = {"filterType": filter_type}
            info["filters"].append(item)
        if override is not None:
            item[key] = format(override, "f")
        decimal_value(item.get(key, "0"))
    if args.quantity_step is not None:
        # An explicit precision override applies to both stop and market sell quantities.
        info["filters"] = [item for item in info["filters"] if item["filterType"] != "MARKET_LOT_SIZE"]
    if args.min_notional is not None:
        info["filters"] = [item for item in info["filters"] if item["filterType"] not in {"MIN_NOTIONAL", "NOTIONAL"}]
        info["filters"].append({
            "filterType": "MIN_NOTIONAL", "minNotional": format(args.min_notional, "f"),
            "applyToMarket": True, "avgPriceMins": 0,
        })
    for item in info["filters"]:
        for key in ("stepSize", "minQty", "tickSize", "minNotional"):
            if key in item:
                decimal_value(item[key])
        # Historical Binance avgPrice is not reconstructible from OHLC bars.
        if item["filterType"] in {"MIN_NOTIONAL", "NOTIONAL"}:
            item["avgPriceMins"] = 0
    return info


@dataclass(frozen=True)
class BacktestSettings:
    symbol: str = "BTCUSDT"
    interval: str = "15m"
    initial_balance: Decimal = Decimal("100")
    quote_size: Decimal = Decimal("10")
    fee_pct: Decimal = Decimal("0.1")
    slippage_pct: Decimal = Decimal("0.05")
    buy_fee_asset: str = "base"
    hosted_stop_loss: bool = True
    close_at_end: bool = False

    def validate(self) -> None:
        if self.interval not in INTERVALS_MS or not self.symbol:
            raise ValueError("A symbol and supported candle interval are required")
        for name in ("initial_balance", "quote_size", "fee_pct", "slippage_pct"):
            decimal_value(getattr(self, name))
        if self.initial_balance <= 0 or self.quote_size <= 0:
            raise ValueError("Initial balance and quote size must be positive")
        if self.fee_pct >= 100 or self.slippage_pct >= 100:
            raise ValueError("Fees and slippage must be below 100 percent")
        if self.buy_fee_asset not in {"base", "quote"}:
            raise ValueError("BUY commission asset must be base or quote")


def strategy_window(config: StrategyConfig) -> int:
    return max(
        config.slow_sma + config.buy_crossover_lookback_candles,
        config.rsi_period + 1, config.cooldown_candles + 1, config.stop_cooldown_candles + 1,
    )


class Simulator:
    def __init__(self, config: StrategyConfig, settings: BacktestSettings, info: dict[str, Any]):
        self.config = config
        self.settings = settings
        self.info = info
        self.cash = settings.initial_balance
        self.fee_rate = settings.fee_pct / 100
        self.slippage_rate = settings.slippage_pct / 100
        self.position: Position | None = None
        self.history = empty_trade_history(settings.symbol, "backtest")
        self.history["inventory_opening"] = inventory_opening(info, None, 0)
        self.inventory = inventory_snapshot(self.history)
        self.fills: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.current_trade: dict[str, Any] | None = None
        self.blocked = Counter()
        self.total_fees = Decimal(0)
        self.realized_pnl = Decimal(0)
        self.peak_equity = settings.initial_balance
        self.max_drawdown = Decimal(0)
        self.equity_curve: list[dict[str, Any]] = []

    def execution_price(self, reference: Decimal, side: str) -> Decimal:
        return reference * (1 + self.slippage_rate if side == "BUY" else 1 - self.slippage_rate)

    def mark(self, price: Decimal) -> Decimal:
        return self.cash + Decimal(self.inventory["quantity"]) * price

    def record_fill(
        self, side: str, quantity: Decimal, price: Decimal, fee: Decimal, fee_asset: str,
        execution_time: int, source: str, reasons: tuple[str, ...], signal: Candle | None,
    ) -> dict[str, Any]:
        quote = quantity * price
        fill = {
            "order_id": len(self.fills) + 1, "side": side, "status": "FILLED", "source": source,
            "timestamp_utc": utc(execution_time), "quantity": float(quantity),
            "quote_quantity": float(quote), "average_price": float(price),
            "executed_quantity": format(quantity, "f"), "executed_quote_quantity": format(quote, "f"),
            "commissions": {fee_asset: format(fee, "f")}, "reasons": list(reasons),
        }
        if signal is not None:
            fill["signal_candle_close_time_utc"] = utc(signal.close_time_ms)
        self.history["fills"].append(fill)
        self.inventory = inventory_snapshot(self.history)
        self.fills.append(fill)
        return fill

    def buy(self, candle: HistoricalCandle, signal: Candle, reasons: tuple[str, ...]) -> None:
        price = self.execution_price(candle.open, "BUY")
        gross_quantity = round_down(
            self.settings.quote_size / price,
            filter_value(self.info, "LOT_SIZE", "stepSize") or "0",
        )
        spent = gross_quantity * price
        base_fee = gross_quantity * self.fee_rate if self.settings.buy_fee_asset == "base" else Decimal(0)
        quote_fee = spent * self.fee_rate if self.settings.buy_fee_asset == "quote" else Decimal(0)
        if spent + quote_fee > self.cash:
            self.blocked["insufficient_cash"] += 1
            return
        prospective = Decimal(self.inventory["quantity"]) + gross_quantity - base_fee
        stop_price = float(price) * (1 - self.config.stop_loss_pct / 100)
        try:
            # Match the live trader's protective quantity buffer and minimum-exit checks.
            buffered = float(self.settings.quote_size / price * Decimal("0.998"))
            if self.settings.hosted_stop_loss and self.config.stop_loss_pct > 0:
                protective_stop_values(buffered, float(price), self.config.stop_loss_pct, self.info)
            market_sell_quantity(buffered, stop_price, self.info)
            market_sell_quantity(float(prospective), stop_price, self.info)
            if self.settings.hosted_stop_loss and self.config.stop_loss_pct > 0:
                protective_stop_values(float(prospective), float(price), self.config.stop_loss_pct, self.info)
            if gross_quantity <= 0:
                raise ValueError("buy quantity rounded to zero")
        except ValueError:
            self.blocked["unprotectable_or_undersized_buy"] += 1
            return
        self.cash -= spent + quote_fee
        self.total_fees += base_fee * price + quote_fee
        fee = base_fee if self.settings.buy_fee_asset == "base" else quote_fee
        asset = self.info["baseAsset"] if self.settings.buy_fee_asset == "base" else self.info["quoteAsset"]
        self.record_fill(
            "BUY", gross_quantity, price, fee, asset, candle.open_time_ms,
            "strategy", reasons, signal,
        )
        owned = Decimal(self.inventory["quantity"])
        sellable = round_down(owned, filter_value(self.info, "LOT_SIZE", "stepSize") or "0")
        self.position = Position(self.settings.symbol, float(sellable), float(price))
        self.current_trade = {
            "entry_time_utc": utc(candle.open_time_ms), "entry_price": float(price),
            "quantity": float(sellable), "realized_pnl_quote": Decimal(0),
        }
        if self.settings.hosted_stop_loss and self.config.stop_loss_pct > 0:
            _, stop = protective_stop_values(float(sellable), float(price), self.config.stop_loss_pct, self.info)
            self.position = replace(self.position, stop_price=stop)

    def sell(
        self, reference: Decimal, execution_time: int, source: str, reasons: tuple[str, ...],
        signal: Candle | None = None, *, hosted: bool = False,
    ) -> bool:
        if self.position is None:
            return False
        available = min(Decimal(str(self.position.quantity)), Decimal(self.inventory["quantity"]))
        try:
            # An accepted hosted stop is not subject to a new min-notional gate at trigger time.
            quantity = available if hosted else Decimal(str(
                market_sell_quantity(float(available), float(reference), self.info)
            ))
            if quantity <= 0:
                raise ValueError("no sellable quantity")
        except ValueError:
            self.blocked["undersized_sell"] += 1
            return False
        price = self.execution_price(reference, "SELL")
        proceeds = quantity * price
        fee = proceeds * self.fee_rate
        cost = Decimal(self.inventory["cost_quote"]) * quantity / Decimal(self.inventory["quantity"])
        pnl = proceeds - fee - cost
        original_quantity = Decimal(str(self.position.quantity))
        self.cash += proceeds - fee
        self.total_fees += fee
        self.realized_pnl += pnl
        fill = self.record_fill(
            "SELL", quantity, price, fee, self.info["quoteAsset"],
            execution_time, source, reasons, signal,
        )
        fill["allocated_cost_quote"] = float(cost)
        fill["realized_pnl_quote"] = float(pnl)
        if self.current_trade is not None:
            self.current_trade["realized_pnl_quote"] += pnl
        remaining = min(original_quantity - quantity, Decimal(self.inventory["quantity"]))
        minimum = decimal_value(filter_value(self.info, "LOT_SIZE", "minQty") or "0")
        if remaining <= 0 or remaining < minimum:
            if self.current_trade is not None:
                self.trades.append({
                    **self.current_trade, "exit_time_utc": utc(execution_time),
                    "exit_price": float(price), "source": source, "reasons": list(reasons),
                    "realized_pnl_quote": float(self.current_trade["realized_pnl_quote"]),
                })
            self.position = None
            self.current_trade = None
        else:
            self.position = replace(self.position, quantity=float(remaining))
        return True

    def tighten_stop(self, signal: Candle) -> None:
        if self.position is None or self.position.stop_price is None:
            return
        target = trailing_stop_price(
            self.position.entry_price, signal.close,
            self.config.stop_loss_pct, self.config.trailing_thresholds,
        )
        target = round_down(Decimal(str(target)), filter_value(self.info, "PRICE_FILTER", "tickSize") or "0")
        if target > Decimal(str(self.position.stop_price)) and Decimal(str(signal.close)) > target:
            self.position = replace(self.position, stop_price=float(target))

    def sample_equity(self, price: Decimal, timestamp: int) -> None:
        equity = self.mark(price)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.equity_curve.append({
            "timestamp_utc": utc(timestamp), "equity_quote": float(equity),
            "cash_quote": float(self.cash), "inventory_quantity": self.inventory["quantity"],
            "drawdown_pct": float(drawdown),
        })


def buy_and_hold(settings: BacktestSettings, info: dict[str, Any], first: Decimal, last: Decimal) -> float | None:
    rate = settings.fee_pct / 100
    price = first * (1 + settings.slippage_pct / 100)
    budget = settings.initial_balance / (1 + rate) if settings.buy_fee_asset == "quote" else settings.initial_balance
    gross = round_down(budget / price, filter_value(info, "LOT_SIZE", "stepSize") or "0")
    try:
        market_sell_quantity(float(gross), float(price), info)
    except ValueError:
        return None
    cash = settings.initial_balance - gross * price * (1 + rate if settings.buy_fee_asset == "quote" else 1)
    owned = gross * (1 - rate if settings.buy_fee_asset == "base" else 1)
    if settings.close_at_end:
        try:
            quantity = Decimal(str(market_sell_quantity(float(owned), float(last), info)))
        except ValueError:
            quantity = Decimal(0)
        cash += quantity * last * (1 - settings.slippage_pct / 100) * (1 - rate)
        owned -= quantity
    return float((cash + owned * last - settings.initial_balance) / settings.initial_balance * 100)


def run_backtest(
    candles: list[HistoricalCandle], config: StrategyConfig, settings: BacktestSettings,
    info: dict[str, Any], *, start_ms: int | None = None, end_ms: int | None = None,
) -> dict[str, Any]:
    settings.validate()
    if info.get("symbol", settings.symbol) != settings.symbol:
        raise ValueError("Backtest symbol and symbol filters do not match")
    validate_strategy_args(argparse.Namespace(**asdict(config), quote_size=float(settings.quote_size)))
    if any(isinstance(value, float) and not math.isfinite(value) for value in asdict(config).values()):
        raise ValueError("Strategy numeric parameters must be finite")
    validate_candles(candles, settings.interval)
    required = max(config.slow_sma + 1, config.rsi_period + 1)
    requested_index = next(
        (i for i, candle in enumerate(candles) if start_ms is None or candle.open_time_ms >= start_ms),
        len(candles),
    )
    if start_ms is not None and requested_index < required:
        raise ValueError(f"At least {required} completed warm-up candles are required before --start")
    first_index = max(required, requested_index)
    evaluation = [
        i for i in range(first_index, len(candles))
        if end_ms is None or candles[i].close_time_ms < end_ms
    ]
    if not evaluation:
        raise ValueError("Not enough candles after warm-up for the requested backtest period")
    simulator = Simulator(config, settings, info)
    first = candles[evaluation[0]]
    simulator.sample_equity(first.open, first.open_time_ms)
    window = strategy_window(config)
    for index in evaluation:
        candle = candles[index]
        closed = [Candle(float(row.close), row.close_time_ms) for row in candles[max(0, index - window):index]]
        signal = closed[-1]
        decision = decide([row.close for row in closed], config, simulator.position)
        if simulator.position is not None and not candle_after_entry(simulator.history, signal.close_time_ms):
            decision = replace(decision, action="HOLD")
        if decision.action == "BUY":
            reason = buy_entry_block_reason(simulator.history, closed, config)
            if reason is not None:
                simulator.blocked["entry_guard"] += 1
                decision = replace(decision, action="HOLD")
        if decision.action == "HOLD":
            simulator.tighten_stop(signal)
        # A hosted stop that gaps through its trigger precedes a queued strategy sell.
        stop = simulator.position.stop_price if simulator.position is not None else None
        gap_exit = stop is not None and candle.open <= Decimal(str(stop))
        if gap_exit:
            simulator.sell(
                candle.open, candle.open_time_ms, "hosted_stop_loss",
                ("hosted stop-loss order filled after a gap",), hosted=True,
            )
        elif decision.action == "BUY":
            simulator.buy(candle, signal, decision.reasons)
        elif decision.action == "SELL":
            simulator.sell(candle.open, candle.open_time_ms, "strategy", decision.reasons, signal)
        stop = simulator.position.stop_price if simulator.position is not None else None
        if stop is not None and candle.open <= Decimal(str(stop)):
            simulator.sell(
                candle.open, candle.open_time_ms, "protective_market",
                ("initial hosted stop would trigger immediately",), hosted=True,
            )
        elif stop is not None and candle.low <= Decimal(str(stop)):
            # OHLC cannot locate the touch within the bar; anchor cooldown at its close.
            simulator.sell(
                Decimal(str(stop)), candle.close_time_ms, "hosted_stop_loss",
                ("hosted stop-loss order filled within candle",), hosted=True,
            )
        simulator.sample_equity(candle.close, candle.close_time_ms)
    last = candles[evaluation[-1]]
    if settings.close_at_end and simulator.position is not None:
        simulator.sell(last.close, last.close_time_ms, "strategy", ("explicit end-of-backtest liquidation",))
        simulator.sample_equity(last.close, last.close_time_ms)
    final_equity = simulator.mark(last.close)
    profit = sum((
        Decimal(str(trade["realized_pnl_quote"])) for trade in simulator.trades
        if trade["realized_pnl_quote"] > 0
    ), Decimal(0))
    loss = -sum((
        Decimal(str(trade["realized_pnl_quote"])) for trade in simulator.trades
        if trade["realized_pnl_quote"] < 0
    ), Decimal(0))
    wins = sum(trade["realized_pnl_quote"] > 0 for trade in simulator.trades)
    unrealized = Decimal(simulator.inventory["quantity"]) * last.close - Decimal(simulator.inventory["cost_quote"])
    return {
        "report_schema_version": 1, "symbol": settings.symbol, "interval": settings.interval,
        "period_start_utc": utc(first.open_time_ms), "period_end_utc": utc(last.close_time_ms),
        "warmup_candles": evaluation[0], "evaluated_candles": len(evaluation),
        "strategy": asdict(config),
        "simulation": {**asdict(settings), "symbol_info": info},
        "assumptions": [
            "One decision per completed candle; market execution at the next candle open.",
            "Hosted stops use candle lows; gaps fill at the open, with adverse slippage.",
            "Intrabar stop timestamps are conservatively assigned to the candle close.",
            "Take-profit and trailing changes use closes, never intrabar highs.",
            "Trailing changes are effective before the next open; no order/API latency is modeled.",
            "Market orders are assumed fully filled; API failures and partial liquidity fills are not modeled.",
            "Fixed commission/slippage and static filters; min-notional uses execution price, not Binance avgPrice.",
            "Base-asset BUY fees are converted to quote at their execution price in the fee total.",
            "Equity and drawdown are sampled at candle closes; open inventory/dust is marked to the last close.",
        ],
        "summary": {
            "initial_balance_quote": float(settings.initial_balance), "final_equity_quote": float(final_equity),
            "final_cash_quote": float(simulator.cash), "net_profit_quote": float(final_equity - settings.initial_balance),
            "net_return_pct": float((final_equity / settings.initial_balance - 1) * 100),
            "realized_pnl_quote": float(simulator.realized_pnl), "unrealized_pnl_quote": float(unrealized),
            "fees_quote_equivalent": float(simulator.total_fees), "max_drawdown_pct": float(simulator.max_drawdown),
            "closed_trades": len(simulator.trades), "buy_executions": sum(fill["side"] == "BUY" for fill in simulator.fills),
            "sell_executions": sum(fill["side"] == "SELL" for fill in simulator.fills),
            "win_rate_pct": wins / len(simulator.trades) * 100 if simulator.trades else None,
            "profit_factor": float(profit / loss) if loss else None,
            "winning_pnl_quote": float(profit), "losing_pnl_quote": float(loss),
            "buy_and_hold_return_pct": buy_and_hold(settings, info, first.open, last.close),
            "blocked_orders": dict(simulator.blocked),
        },
        "open_position": asdict(simulator.position) if simulator.position else None,
        "inventory": simulator.inventory, "trades": simulator.trades,
        "fills": simulator.fills, "equity_curve": simulator.equity_curve,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest the Spot trader using historical Binance OHLCV or an offline CSV")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", choices=tuple(INTERVALS_MS), default="15m")
    parser.add_argument("--start", help="UTC start date/time, inclusive; required for downloads")
    parser.add_argument("--end", help="UTC end date/time, exclusive; required for downloads")
    parser.add_argument("--csv", type=Path, help="Read local OHLCV candles without a network request")
    parser.add_argument("--save-csv", type=Path, help="Save loaded/downloaded candles including warm-up")
    parser.add_argument("--filters-json", type=Path, help="Static symbol info, exchangeInfo JSON, or a previous backtest report")
    parser.add_argument("--quantity-step", type=decimal_argument, help="Override quantity rounding step")
    parser.add_argument("--price-tick", type=decimal_argument, help="Override stop-price rounding tick")
    parser.add_argument("--min-quantity", type=decimal_argument, help="Override LOT_SIZE minQty")
    parser.add_argument("--min-notional", type=decimal_argument, help="Override minimum quote notional")
    parser.add_argument("--initial-balance", type=decimal_argument, default=Decimal("100"))
    parser.add_argument(
        "--fee-pct", type=decimal_argument, default=Decimal("0.1"),
        help="Commission per execution in percent (default: 0.1)",
    )
    parser.add_argument(
        "--slippage-pct", type=decimal_argument, default=Decimal("0.05"),
        help="Adverse price adjustment per execution in percent (default: 0.05)",
    )
    parser.add_argument("--buy-fee-asset", choices=("base", "quote"), default="base")
    parser.add_argument("--close-at-end", action="store_true", help="Explicitly sell an open position at the final close")
    parser.add_argument("--output", type=Path, help="Write the complete JSON report, including trades and equity curve")
    add_strategy_arguments(parser)
    parser.set_defaults(quote_size=10)
    return parser


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(
        report, indent=2, allow_nan=False,
        default=lambda value: format(value, "f") if isinstance(value, Decimal) else str(value),
    ) + "\n"


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    quote = report["simulation"]["symbol_info"]["quoteAsset"]
    print(f"Backtest {report['symbol']} {report['interval']}: {report['period_start_utc']} -> {report['period_end_utc']}")
    print(f"Equity: {summary['initial_balance_quote']:.4f} -> {summary['final_equity_quote']:.4f} {quote}")
    print(f"Net P/L: {summary['net_profit_quote']:+.4f} {quote} ({summary['net_return_pct']:+.3f}%)")
    print(f"Fees: {summary['fees_quote_equivalent']:.4f} {quote}; max close-to-close drawdown: {summary['max_drawdown_pct']:.3f}%")
    win_rate = f"{summary['win_rate_pct']:.2f}%" if summary["win_rate_pct"] is not None else "n/a"
    if summary["profit_factor"] is not None:
        profit_factor = f"{summary['profit_factor']:.3f}"
    else:
        profit_factor = "infinite (no losing closed trades)" if summary["winning_pnl_quote"] > 0 else "n/a"
    print(f"Closed trades: {summary['closed_trades']}; win rate: {win_rate}; net profit factor: {profit_factor}")
    benchmark = summary["buy_and_hold_return_pct"]
    print(f"Buy & hold (all initial capital): {benchmark:+.3f}%" if benchmark is not None else "Buy & hold: below symbol minimum")
    print(f"Remaining inventory: {report['inventory']['quantity']}; fractional dust: {report['inventory']['residual_quantity']}")
    print(f"Blocked orders: {summary['blocked_orders']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.symbol = args.symbol.upper()
    try:
        validate_strategy_args(args)
        config = strategy_config_from_args(args)
        settings = BacktestSettings(
            symbol=args.symbol, interval=args.interval, initial_balance=args.initial_balance,
            quote_size=Decimal(str(args.quote_size)), fee_pct=args.fee_pct, slippage_pct=args.slippage_pct,
            buy_fee_asset=args.buy_fee_asset, hosted_stop_loss=args.hosted_stop_loss,
            close_at_end=args.close_at_end,
        )
        settings.validate()
        start = parse_time(args.start) if args.start else None
        end = parse_time(args.end) if args.end else None
        if start is not None and end is not None and end <= start:
            raise ValueError("--end must be after --start")
        inputs = [path.resolve() for path in (args.csv, args.filters_json) if path is not None]
        outputs = [path.resolve() for path in (args.output, args.save_csv) if path is not None]
        if len(set(outputs)) != len(outputs) or any(path in inputs for path in outputs):
            raise ValueError("Output files must be distinct and must not overwrite input data")
        client = BinanceClient(MAINNET_URL)
        if args.csv:
            candles = load_csv(args.csv, args.interval)
            info = {} if args.filters_json else generic_symbol_info(args.symbol)
            source = "csv"
            filter_source = "generic CSV defaults"
        else:
            if start is None or end is None:
                raise ValueError("Public downloads require --start and --end; alternatively supply --csv")
            warmup_start = max(0, start - (strategy_window(config) + 1) * INTERVALS_MS[args.interval])
            candles = download_candles(client, args.symbol, args.interval, warmup_start, end)
            info = {} if args.filters_json else client.symbol_info(args.symbol)
            source = "Binance public mainnet klines"
            filter_source = "current Binance symbol filters (not historical)"
        if args.filters_json:
            info = json.loads(args.filters_json.read_text())
            if isinstance(info, dict) and isinstance(info.get("simulation"), dict):
                info = info["simulation"].get("symbol_info", {})
            if isinstance(info, dict) and "symbols" in info:
                if not isinstance(info["symbols"], list) or any(not isinstance(item, dict) for item in info["symbols"]):
                    raise ValueError("exchangeInfo symbols must be an array of symbol objects")
                info = next((item for item in info["symbols"] if item.get("symbol") == args.symbol), {})
            filter_source = str(args.filters_json)
        info = prepare_symbol_info(info, args)
        report = run_backtest(candles, config, settings, info, start_ms=start, end_ms=end)
        report["data_source"] = source
        report["filter_source"] = filter_source
        report["requested_start_utc"] = utc(start) if start is not None else None
        report["requested_end_utc"] = utc(end) if end is not None else None
        if args.save_csv:
            atomic_write_text(args.save_csv, render_csv(candles))
        if args.output:
            atomic_write_text(args.output, report_json(report))
        print_summary(report)
        if args.output:
            print(f"Full report: {args.output}")
        return 0
    except (ValueError, OSError, RuntimeError, BinanceError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
