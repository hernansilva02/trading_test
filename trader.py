#!/usr/bin/env python3
"""Configurable Binance Spot trading bot with safe defaults."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import http.client
import json
import logging
import logging.handlers
import math
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("trader")
TESTNET_URL = "https://testnet.binance.vision"
MAINNET_URL = "https://api.binance.com"
HISTORY_SCHEMA_VERSION = 1
INTENT_SCHEMA_VERSION = 2
BINANCE_CLIENT_ORDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,36}$")
OPERATIONS_TIMEZONE = timezone(timedelta(hours=-3))
TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH"}
ORDER_STATUSES = TERMINAL_ORDER_STATUSES | {
    "NEW",
    "PENDING_NEW",
    "PARTIALLY_FILLED",
    "PENDING_CANCEL",
}
INACTIVE_ORDER_STATUSES = {"CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "FILLED"}
SIGNAL_COLORS = {
    "HOLD": "\033[34m",
    "BUY": "\033[31m",
    "SELL": "\033[32m",
}
SIGNAL_PATTERN = re.compile(r"\b(?:HOLD|BUY|SELL)\b")


class SignalColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return SIGNAL_PATTERN.sub(
            lambda match: f"{SIGNAL_COLORS[match.group(0)]}{match.group(0)}\033[0m",
            rendered,
        )


class BinanceError(RuntimeError):
    """Raised when Binance rejects an API request."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        http_status: int | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.ambiguous = ambiguous

    @property
    def order_not_found(self) -> bool:
        return self.code == -2013

    @property
    def cancel_order_not_found(self) -> bool:
        return self.code == -2011


class InsufficientMarketData(BinanceError):
    """Raised while Binance has too few completed candles for the strategy."""


class PendingIntentError(RuntimeError):
    """Raised when an order intent cannot yet be resolved safely."""


class TradeJournalError(RuntimeError):
    """Raised when an executed strategy fill cannot be persisted."""


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
    min_sma_gap_pct: float = 0.05
    buy_crossover_lookback_candles: int = 5
    buy_rsi_min: float = 50.0
    buy_rsi_max: float = 70.0
    cooldown_candles: int = 3
    stop_cooldown_candles: int = 6
    trailing_thresholds: tuple[tuple[float, float], ...] = (
        (1.0, 0.0),
        (2.0, 1.0),
        (3.0, 2.0),
    )


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


@dataclass(frozen=True)
class Candle:
    close: float
    close_time_ms: int


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
                try:
                    return json.loads(response.read().decode())
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BinanceError(
                        "Binance returned a malformed successful response",
                        ambiguous=method != "GET",
                    ) from error
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            try:
                payload = json.loads(body)
                detail = payload.get("msg", body) if isinstance(payload, dict) else body
                binance_code = payload.get("code") if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                detail = body
                binance_code = None
            raise BinanceError(
                f"Binance HTTP {error.code}: {detail}",
                code=binance_code if isinstance(binance_code, int) else None,
                http_status=error.code,
                ambiguous=(
                    error.code == 408
                    or error.code >= 500
                    or binance_code in {-1006, -1007}
                ),
            ) from error
        except urllib.error.URLError as error:
            raise BinanceError(
                f"Could not reach Binance: {error.reason}", ambiguous=True
            ) from error
        except TimeoutError as error:
            raise BinanceError("Binance request timed out", ambiguous=True) from error
        except (OSError, http.client.HTTPException) as error:
            raise BinanceError(f"Binance connection failed: {error}", ambiguous=True) from error

    def synchronize_time(self) -> None:
        response = self._request("GET", "/api/v3/time")
        self._time_offset_ms = int(response["serverTime"]) - int(time.time() * 1000)

    def candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        rows = self._request(
            "GET",
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        return [Candle(float(row[4]), int(row[6])) for row in rows]

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

    def market_buy(
        self, symbol: str, quote_quantity: float, client_order_id: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": decimal_string(quote_quantity),
                "newOrderRespType": "FULL",
                "newClientOrderId": client_order_id,
            },
            signed=True,
        )

    def market_sell(
        self, symbol: str, quantity: float, client_order_id: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": decimal_string(quantity),
                "newOrderRespType": "FULL",
                "newClientOrderId": client_order_id,
            },
            signed=True,
        )

    def place_stop_loss(
        self, symbol: str, quantity: float, stop_price: float, client_order_id: str
    ) -> dict[str, Any]:
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
                "newClientOrderId": client_order_id,
            },
            signed=True,
        )

    def order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request(
            "GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True
        )

    def order_by_client_id(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v3/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True,
        )

    def trades(self, symbol: str, order_id: int) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/api/v3/myTrades",
            {"symbol": symbol, "orderId": order_id, "limit": 1000},
            signed=True,
        )

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request(
            "DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True
        )


def decimal_string(value: float) -> str:
    return format(Decimal(str(value)), "f")


def floor_to_step(value: float | Decimal, step: str) -> float:
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(step)
    if decimal_step == 0:
        return value
    units = (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN)
    return float(units * decimal_step)


def remaining_quantity(original: float, executed: float) -> float:
    return float(max(Decimal(0), Decimal(str(original)) - Decimal(str(executed))))


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


def latest_bullish_crossover_age(closes: list[float], config: StrategyConfig) -> int | None:
    available_crossovers = min(
        config.buy_crossover_lookback_candles,
        len(closes) - config.slow_sma,
    )
    for age in range(available_crossovers):
        end = len(closes) - age
        candidate_fast = sum(closes[end - config.fast_sma : end]) / config.fast_sma
        candidate_slow = sum(closes[end - config.slow_sma : end]) / config.slow_sma
        candidate_previous_fast = (
            sum(closes[end - config.fast_sma - 1 : end - 1]) / config.fast_sma
        )
        candidate_previous_slow = (
            sum(closes[end - config.slow_sma - 1 : end - 1]) / config.slow_sma
        )
        if candidate_previous_fast <= candidate_previous_slow and candidate_fast > candidate_slow:
            return age
    return None


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
    fast_slope = fast - previous_fast
    slow_slope = slow - previous_slow
    bullish_gap_pct = (fast - slow) / slow * 100
    bearish_gap_pct = (slow - fast) / slow * 100
    bullish_crossover_age = latest_bullish_crossover_age(closes, config)
    bullish_crossover = bullish_crossover_age is not None
    strong_bearish_trend = (
        fast < slow
        and slow_slope < 0
        and bearish_gap_pct >= config.min_sma_gap_pct
    )
    rsi = simple_rsi(closes, config.rsi_period)

    if position is None:
        conditions: list[tuple[bool, str, str]] = []
        if config.buy_on_bullish_trend:
            conditions.extend(
                (
                    (
                        fast > slow,
                        (
                            "bullish SMA trend confirmed (fast SMA above slow SMA"
                            + (
                                f", crossover {bullish_crossover_age} closed candle(s) ago)"
                                if bullish_crossover
                                else "; continuation entry allowed)"
                            )
                        ),
                        f"fast SMA {fast:.8f} is not above slow SMA {slow:.8f}",
                    ),
                    (
                        slow_slope > 0,
                        "slow SMA rising",
                        f"slow SMA not rising (slope={slow_slope:.8f})",
                    ),
                    (
                        fast_slope > 0,
                        "fast SMA rising",
                        f"fast SMA not rising (slope={fast_slope:.8f})",
                    ),
                    (
                        bullish_gap_pct >= config.min_sma_gap_pct,
                        f"bullish SMA gap {bullish_gap_pct:.3f}% >= {config.min_sma_gap_pct:g}%",
                        f"bullish SMA gap {bullish_gap_pct:.3f}% below minimum "
                        f"{config.min_sma_gap_pct:g}%",
                    ),
                )
            )
        if config.buy_below is not None:
            conditions.append(
                (
                    price <= config.buy_below,
                    f"price {price:.8f} <= {config.buy_below:g}",
                    f"price {price:.8f} above buy limit {config.buy_below:g}",
                )
            )
        if config.buy_rsi_below is not None:
            conditions.append(
                (
                    rsi <= config.buy_rsi_below,
                    f"RSI {rsi:.2f} <= {config.buy_rsi_below:g}",
                    f"RSI {rsi:.2f} above configured limit {config.buy_rsi_below:g}",
                )
            )
        entry_enabled = bool(conditions)
        if entry_enabled:
            conditions.extend(
                (
                    (
                        rsi >= config.buy_rsi_min,
                        f"RSI {rsi:.2f} >= {config.buy_rsi_min:g}",
                        f"RSI {rsi:.2f} below minimum {config.buy_rsi_min:g}",
                    ),
                    (
                        rsi <= config.buy_rsi_max,
                        f"RSI {rsi:.2f} <= {config.buy_rsi_max:g}",
                        f"RSI {rsi:.2f} above maximum {config.buy_rsi_max:g}",
                    ),
                )
            )
        if conditions and all(matched for matched, _, _ in conditions):
            return Decision(
                "BUY",
                tuple(success for _, success, _ in conditions),
                price,
                fast,
                slow,
                rsi,
            )
        if not entry_enabled:
            reasons = ("BUY blocked: no entry rule is enabled",)
        else:
            failed = [failure for matched, _, failure in conditions if not matched]
            reasons = ("BUY blocked: " + "; ".join(failed),)
        return Decision("HOLD", reasons, price, fast, slow, rsi)

    exit_reasons: list[str] = []
    if config.sell_on_bearish_trend and strong_bearish_trend:
        exit_reasons.append(
            f"bearish SMA trend: gap {bearish_gap_pct:.3f}% >= "
            f"{config.min_sma_gap_pct:g}% and slow SMA falling"
        )
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
    return Decision(
        "HOLD",
        (
            f"SELL blocked: bearish gap={max(0.0, bearish_gap_pct):.3f}%, "
            f"slow slope={slow_slope:.8f}, return={(price / position.entry_price - 1) * 100:.3f}%",
        ),
        price,
        fast,
        slow,
        rsi,
    )


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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_delete(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    fsync_directory(path.parent)


def save_position(path: Path, position: Position | None) -> None:
    if position is None:
        atomic_delete(path)
        return
    atomic_write_text(path, json.dumps(asdict(position), indent=2) + "\n")


def order_intent_path(state_path: Path, symbol: str, network: str) -> Path:
    return state_path.parent / f".trader-intent-{symbol}-{network}.json"


def credential_fingerprint(api_key: str | None = None) -> str:
    value = os.environ.get("BINANCE_API_KEY", "") if api_key is None else api_key
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def client_credential_fingerprint(client: BinanceClient) -> str:
    return credential_fingerprint(getattr(client, "api_key", None))


def new_client_order_id(kind: str) -> str:
    code = {
        "market_buy": "B",
        "strategy_sell": "S",
        "protective_market_sell": "P",
        "hosted_stop": "H",
    }[kind]
    timestamp = format(time.time_ns(), "x")
    return f"trd-{code}-{timestamp}-{secrets.token_hex(6)}"[:36]


def validate_order_intent(
    intent: Any, path: Path, symbol: str, network: str
) -> dict[str, Any]:
    required = {
        "schema_version",
        "symbol",
        "network",
        "client_order_id",
        "kind",
        "side",
        "order_type",
        "source",
        "reasons",
        "quantity",
        "quote_quantity",
        "stop_price",
        "position_quantity",
        "entry_price",
        "prior_stop_price",
        "signal_candle_close_time_ms",
        "cancel_order_id",
        "cancel_completed",
        "submission_attempted",
        "credential_fingerprint",
    }
    if not isinstance(intent, dict) or set(intent) != required:
        raise RuntimeError(f"Invalid order intent file {path}: invalid fields")
    if (
        intent["schema_version"] != INTENT_SCHEMA_VERSION
        or intent["symbol"] != symbol
        or intent["network"] != network
        or intent["kind"]
        not in {"market_buy", "strategy_sell", "protective_market_sell", "hosted_stop"}
        or intent["side"] not in {"BUY", "SELL"}
        or intent["order_type"] not in {"MARKET", "STOP_LOSS"}
        or intent["source"] not in {"strategy", "hosted_stop_loss", "protective_market"}
        or not isinstance(intent["client_order_id"], str)
        or BINANCE_CLIENT_ORDER_ID.fullmatch(intent["client_order_id"]) is None
        or not isinstance(intent["reasons"], list)
        or not all(isinstance(reason, str) for reason in intent["reasons"])
        or not isinstance(intent["cancel_completed"], bool)
        or not isinstance(intent["submission_attempted"], bool)
        or not isinstance(intent["credential_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{16}", intent["credential_fingerprint"]) is None
    ):
        raise RuntimeError(f"Invalid order intent file {path}: invalid values")
    for key in (
        "quantity",
        "quote_quantity",
        "stop_price",
        "position_quantity",
        "entry_price",
        "prior_stop_price",
    ):
        value = intent[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RuntimeError(f"Invalid order intent file {path}: invalid {key}")
    for key in ("signal_candle_close_time_ms", "cancel_order_id"):
        value = intent[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise RuntimeError(f"Invalid order intent file {path}: invalid {key}")
    kind = intent["kind"]
    if (
        (kind == "market_buy" and (intent["side"], intent["order_type"]) != ("BUY", "MARKET"))
        or (
            kind in {"strategy_sell", "protective_market_sell"}
            and (intent["side"], intent["order_type"]) != ("SELL", "MARKET")
        )
        or (kind == "hosted_stop" and (intent["side"], intent["order_type"]) != ("SELL", "STOP_LOSS"))
        or (kind == "market_buy" and intent["quote_quantity"] is None)
        or (kind != "market_buy" and intent["quantity"] is None)
        or (kind == "hosted_stop" and intent["stop_price"] is None)
        or (kind != "market_buy" and (intent["position_quantity"] is None or intent["entry_price"] is None))
        or (intent["cancel_completed"] and intent["cancel_order_id"] is None)
        or (kind == "market_buy" and intent["source"] != "strategy")
        or (kind == "strategy_sell" and intent["source"] != "strategy")
        or (kind == "protective_market_sell" and intent["source"] != "protective_market")
        or (kind == "hosted_stop" and intent["source"] != "hosted_stop_loss")
        or (kind == "market_buy" and intent["quantity"] is not None)
        or (kind != "market_buy" and intent["quote_quantity"] is not None)
        or (kind in {"market_buy", "protective_market_sell"} and intent["cancel_order_id"] is not None)
    ):
        raise RuntimeError(f"Invalid order intent file {path}: inconsistent order parameters")
    return intent


def load_order_intent(path: Path, symbol: str, network: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        intent = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid order intent file {path}: {error}") from error
    return validate_order_intent(intent, path, symbol, network)


def save_order_intent(path: Path, intent: dict[str, Any]) -> None:
    validate_order_intent(intent, path, intent["symbol"], intent["network"])
    atomic_write_text(path, json.dumps(intent, indent=2, sort_keys=True) + "\n")


def clear_order_intent(path: Path) -> None:
    atomic_delete(path)


def process_lock_path(symbol: str, network: str) -> Path:
    runtime_directory = Path("/tmp") / f"binance-trader-locks-{os.getuid()}"
    return runtime_directory / f"{symbol}-{network}.lock"


def acquire_process_lock(symbol: str, network: str) -> Any:
    path = process_lock_path(symbol, network)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"another trader is already running for {symbol} on {network} ({path})"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def cancel_order_reconciled(
    client: BinanceClient, symbol: str, order_id: int
) -> dict[str, Any]:
    try:
        return client.cancel_order(symbol, order_id)
    except BinanceError as cancel_error:
        if not cancel_error.ambiguous and not cancel_error.cancel_order_not_found:
            raise
        order = client.order(symbol, order_id)
        if order.get("status") in INACTIVE_ORDER_STATUSES:
            return order
        raise cancel_error


def trade_history_path(state_path: Path, symbol: str, network: str) -> Path:
    return state_path.parent / f".trader-history-{symbol}-{network}.json"


def operational_log_path(state_path: Path, symbol: str, network: str) -> Path:
    return state_path.parent / f"trader-{symbol}-{network}.log"


def trade_operations_log_path(state_path: Path, symbol: str, network: str) -> Path:
    return state_path.parent / f"trader-operations-{symbol}-{network}.log"


def empty_trade_history(symbol: str, network: str) -> dict[str, Any]:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "symbol": symbol,
        "network": network,
        "fills": [],
    }


def accounting_decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise RuntimeError("Inventory contains an invalid decimal") from error
    if isinstance(value, bool) or not number.is_finite() or number < 0:
        raise RuntimeError("Inventory quantities and costs must be finite and non-negative")
    return number


def inventory_snapshot(history: dict[str, Any]) -> dict[str, str]:
    """Replay cumulative order records, so recovery cannot credit the same dust twice."""
    opening = history["inventory_opening"]
    quantity = accounting_decimal(opening["quantity"])
    cost = accounting_decimal(opening["cost_quote"])
    base_asset = opening["base_asset"]
    quote_asset = opening["quote_asset"]
    for fill in history["fills"][opening["start_index"] :]:
        fees = fill["commissions"]
        executed = accounting_decimal(fill.get("executed_quantity", fill["quantity"]))
        base_fee = accounting_decimal(fees.get(base_asset, "0"))
        if fill["side"] == "BUY":
            acquired = executed - base_fee
            if acquired <= 0:
                raise RuntimeError("BUY commission consumed the acquired inventory")
            quantity += acquired
            quote = accounting_decimal(fill.get("executed_quote_quantity", fill["quote_quantity"]))
            cost += quote + accounting_decimal(fees.get(quote_asset, "0"))
        else:
            consumed = executed + base_fee
            if consumed > quantity:
                raise RuntimeError("SELL exceeds the bot-owned inventory ledger")
            consumed = min(consumed, quantity)
            cost = cost * (quantity - consumed) / quantity if quantity else Decimal(0)
            quantity -= consumed
    step = accounting_decimal(opening["step_size"])
    tradable = (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step if step else quantity
    residual = quantity - tradable
    return {
        "quantity": format(quantity, "f"),
        "cost_quote": format(cost, "f"),
        "residual_quantity": format(residual, "f"),
        "residual_cost_quote": format(cost * residual / quantity if quantity else Decimal(0), "f"),
    }


def inventory_opening(info: dict[str, Any], position: Position | None, start_index: int) -> dict:
    quantity = Decimal(str(position.quantity)) if position is not None else Decimal(0)
    return {
        "base_asset": info["baseAsset"],
        "quote_asset": info["quoteAsset"],
        "step_size": filter_value(info, "LOT_SIZE", "stepSize") or "0",
        "quantity": format(quantity, "f"),
        "cost_quote": format(
            quantity * Decimal(str(position.entry_price)) if position else Decimal(0), "f"
        ),
        "start_index": start_index,
    }


def inventory_sellable_quantity(history: dict[str, Any], info: dict[str, Any]) -> float:
    return floor_to_step(
        Decimal(history["inventory"]["quantity"]),
        filter_value(info, "LOT_SIZE", "stepSize") or "0",
    )


def load_trade_history(path: Path, symbol: str, network: str) -> dict[str, Any]:
    if not path.exists():
        return empty_trade_history(symbol, network)
    try:
        history = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid trade history file {path}: {error}") from error
    if not isinstance(history, dict):
        raise RuntimeError(f"Invalid trade history file {path}: expected a JSON object")
    if history.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise RuntimeError(
            f"Invalid trade history file {path}: unsupported schema version"
        )
    if history.get("symbol") != symbol or history.get("network") != network:
        raise RuntimeError(
            f"Invalid trade history file {path}: expected {symbol} on {network}"
        )
    fills = history.get("fills")
    if not isinstance(fills, list):
        raise RuntimeError(f"Invalid trade history file {path}: fills must be an array")
    required = {
        "timestamp_utc",
        "order_id",
        "side",
        "source",
        "status",
        "quantity",
        "quote_quantity",
        "average_price",
        "reasons",
    }
    seen_order_ids: set[int] = set()
    for index, fill in enumerate(fills):
        if not isinstance(fill, dict) or not required <= fill.keys():
            raise RuntimeError(f"Invalid trade history file {path}: invalid fill at index {index}")
        numeric = (fill["quantity"], fill["quote_quantity"], fill["average_price"])
        if (
            isinstance(fill["order_id"], bool)
            or not isinstance(fill["order_id"], int)
            or fill["order_id"] <= 0
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 0
                for value in numeric
            )
            or fill["side"] not in {"BUY", "SELL"}
            or fill["source"] not in {"strategy", "hosted_stop_loss", "protective_market"}
            or not isinstance(fill["status"], str)
            or fill["status"] not in ORDER_STATUSES
            or not isinstance(fill["timestamp_utc"], str)
            or not isinstance(fill["reasons"], list)
            or not all(isinstance(reason, str) for reason in fill["reasons"])
            or (
                "signal_candle_close_time_utc" in fill
                and not isinstance(fill["signal_candle_close_time_utc"], str)
            )
        ):
            raise RuntimeError(f"Invalid trade history file {path}: invalid fill at index {index}")
        if fill["order_id"] in seen_order_ids:
            raise RuntimeError(f"Invalid trade history file {path}: duplicate order ID")
        seen_order_ids.add(fill["order_id"])
        if "commissions" in fill:
            fees = fill["commissions"]
            if not isinstance(fees, dict) or not all(
                isinstance(asset, str) and asset for asset in fees
            ):
                raise RuntimeError(f"Invalid trade history file {path}: invalid commissions")
            for amount in fees.values():
                accounting_decimal(amount)
            for exact_key, numeric_key in (
                ("executed_quantity", "quantity"),
                ("executed_quote_quantity", "quote_quantity"),
            ):
                if exact_key in fill and not math.isclose(
                    float(accounting_decimal(fill[exact_key])), fill[numeric_key],
                    rel_tol=1e-12, abs_tol=1e-15,
                ):
                    raise RuntimeError(
                        f"Invalid trade history file {path}: inconsistent exact execution quantities"
                    )
    if "inventory_opening" in history:
        opening = history["inventory_opening"]
        if (
            not isinstance(opening, dict)
            or set(opening) != {
                "base_asset", "quote_asset", "step_size", "quantity", "cost_quote", "start_index"
            }
            or not all(
                isinstance(opening[key], str) and opening[key]
                for key in ("base_asset", "quote_asset")
            )
            or isinstance(opening["start_index"], bool)
            or not isinstance(opening["start_index"], int)
            or not 0 <= opening["start_index"] <= len(fills)
            or any("commissions" not in fill for fill in fills[opening["start_index"] :])
        ):
            raise RuntimeError(f"Invalid trade history file {path}: invalid inventory opening")
        snapshot = inventory_snapshot(history)
        if history.get("inventory") != snapshot:
            raise RuntimeError(f"Invalid trade history file {path}: inconsistent inventory ledger")
    return history


def render_trade_operations_log(history: dict[str, Any]) -> str:
    lines = []
    for fill in history["fills"]:
        try:
            timestamp = datetime.fromisoformat(fill["timestamp_utc"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("Trade history contains an invalid UTC timestamp") from error
        if timestamp.tzinfo is None:
            raise RuntimeError("Trade history UTC timestamp is missing a timezone")
        lines.append(
            "\t".join(
                (
                    f"timestamp_gmt_minus_3={timestamp.astimezone(OPERATIONS_TIMEZONE).isoformat()}",
                    f"side={fill['side']}",
                    f"order_id={fill['order_id']}",
                    f"source={fill['source']}",
                    f"status={fill['status']}",
                    f"quantity={fill['quantity']:.12g}",
                    f"quote_quantity={fill['quote_quantity']:.12g}",
                    f"average_price={fill['average_price']:.12g}",
                    "reasons=" + json.dumps(fill["reasons"], ensure_ascii=True),
                )
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def save_trade_history(path: Path, history: dict[str, Any]) -> None:
    if "inventory_opening" in history:
        history["inventory"] = inventory_snapshot(history)
    if history["fills"]:
        history["entry_guard"] = entry_guard(history)
    atomic_write_text(path, json.dumps(history, indent=2, sort_keys=True) + "\n")
    operations_path = trade_operations_log_path(
        path, history["symbol"], history["network"]
    )
    atomic_write_text(operations_path, render_trade_operations_log(history))


def initialize_trade_history(path: Path, symbol: str, network: str) -> None:
    history = load_trade_history(path, symbol, network)
    # Rewriting atomically verifies the journal is writable before an order can be submitted.
    save_trade_history(path, history)


def order_timestamp_utc(order: dict[str, Any]) -> str:
    for key in ("transactTime", "updateTime", "time"):
        try:
            milliseconds = float(order[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(milliseconds) and milliseconds > 0:
            return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def record_trade_fill(
    path: Path,
    symbol: str,
    network: str,
    order: dict[str, Any],
    side: str,
    source: str,
    reasons: tuple[str, ...],
    signal_candle_close_time_ms: int | None = None,
    *,
    commissions: dict[str, str] | None = None,
    info: dict[str, Any] | None = None,
    position: Position | None = None,
) -> None:
    try:
        order_id = int(order["orderId"])
        quantity = float(order["executedQty"])
        quote_quantity = float(order["cummulativeQuoteQty"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Binance fill is missing valid execution fields") from error
    if quantity <= 0:
        return
    if order_id <= 0 or not all(
        math.isfinite(value) and value > 0 for value in (quantity, quote_quantity)
    ):
        raise RuntimeError("Binance fill quantities must be positive and finite")
    fill = {
        "timestamp_utc": order_timestamp_utc(order),
        "order_id": order_id,
        "side": side,
        "source": source,
        "status": str(order.get("status", "FILLED")),
        "quantity": quantity,
        "quote_quantity": quote_quantity,
        "average_price": quote_quantity / quantity,
        "reasons": list(reasons),
    }
    if signal_candle_close_time_ms is not None:
        fill["signal_candle_close_time_utc"] = datetime.fromtimestamp(
            signal_candle_close_time_ms / 1000, timezone.utc
        ).isoformat()
    history = load_trade_history(path, symbol, network)
    if commissions is not None:
        fill["commissions"] = commissions
        fill["executed_quantity"] = format(accounting_decimal(order["executedQty"]), "f")
        fill["executed_quote_quantity"] = format(accounting_decimal(order["cummulativeQuoteQty"]), "f")
        if "inventory_opening" not in history:
            if info is None:
                raise RuntimeError("Inventory accounting requires symbol filters")
            # A legacy position is an explicit ownership checkpoint, never an account balance.
            start_index = next(
                (index for index, previous in enumerate(history["fills"])
                 if previous["order_id"] == order_id),
                len(history["fills"]),
            )
            history["inventory_opening"] = inventory_opening(info, position, start_index)
        if info is not None:
            opening = history["inventory_opening"]
            if opening["base_asset"] != info["baseAsset"] or opening["quote_asset"] != info["quoteAsset"]:
                raise RuntimeError("Inventory ledger asset identity mismatch")
            opening["step_size"] = filter_value(info, "LOT_SIZE", "stepSize") or "0"
    for index, existing in enumerate(history["fills"]):
        if existing["order_id"] == order_id:
            if existing["side"] != side or existing["source"] != source:
                raise RuntimeError(f"Trade history order {order_id} conflicts with a prior fill")
            if quantity < float(existing["quantity"]) or quote_quantity < float(
                existing["quote_quantity"]
            ):
                raise RuntimeError(
                    f"Trade history order {order_id} cumulative fill regressed"
                )
            if (
                existing["status"] in TERMINAL_ORDER_STATUSES
                and fill["status"] != existing["status"]
            ):
                raise RuntimeError(
                    f"Trade history order {order_id} terminal status regressed"
                )
            same_quantity = accounting_decimal(order["executedQty"]) == accounting_decimal(
                existing.get("executed_quantity", existing["quantity"])
            )
            if "commissions" in existing and commissions is not None:
                assets = existing["commissions"].keys() | commissions.keys()
                if any(
                    accounting_decimal(commissions.get(asset, "0"))
                    < accounting_decimal(existing["commissions"].get(asset, "0"))
                    for asset in assets
                ):
                    raise RuntimeError(f"Trade history order {order_id} cumulative commission regressed")
                if same_quantity and any(
                    accounting_decimal(commissions.get(asset, "0"))
                    != accounting_decimal(existing["commissions"].get(asset, "0"))
                    for asset in assets
                ):
                    raise RuntimeError(f"Trade history order {order_id} commissions changed")
            if same_quantity:
                if accounting_decimal(order["cummulativeQuoteQty"]) != accounting_decimal(
                    existing.get("executed_quote_quantity", existing["quote_quantity"])
                ):
                    raise RuntimeError(f"Trade history order {order_id} quote changed without a new fill")
                fill["timestamp_utc"] = existing["timestamp_utc"]
                for key in ("commissions", "executed_quantity", "executed_quote_quantity"):
                    if key in existing:
                        fill[key] = existing[key]
            if "signal_candle_close_time_utc" in existing:
                fill["signal_candle_close_time_utc"] = existing["signal_candle_close_time_utc"]
            history["fills"][index] = fill
            break
    else:
        history["fills"].append(fill)
    save_trade_history(path, history)


def record_trade_fill_safely(
    path: Path,
    symbol: str,
    network: str,
    order: dict[str, Any],
    side: str,
    source: str,
    reasons: tuple[str, ...],
    signal_candle_close_time_ms: int | None = None,
) -> None:
    try:
        record_trade_fill(
            path,
            symbol,
            network,
            order,
            side,
            source,
            reasons,
            signal_candle_close_time_ms,
        )
    except (OSError, RuntimeError) as error:
        # A fill cannot be rolled back; preserve trading safeguards and make the audit failure loud.
        LOGGER.critical("Could not record executed fill in %s: %s", path, error)


def record_trade_fill_required(
    path: Path,
    symbol: str,
    network: str,
    order: dict[str, Any],
    side: str,
    reasons: tuple[str, ...],
    signal_candle_close_time_ms: int,
) -> None:
    try:
        record_trade_fill(
            path,
            symbol,
            network,
            order,
            side,
            "strategy",
            reasons,
            signal_candle_close_time_ms,
        )
    except (OSError, RuntimeError) as error:
        raise TradeJournalError(
            f"executed {side} could not be recorded in {path}; trading stopped: {error}"
        ) from error


def record_and_log_hosted_fill(
    path: Path,
    symbol: str,
    network: str,
    order: dict[str, Any],
    reasons: tuple[str, ...],
    entry_price: float,
    quote_asset: str,
    *,
    client: BinanceClient,
    info: dict[str, Any],
    position: Position,
) -> None:
    order_id = int(order["orderId"])
    quantity = float(order.get("executedQty", 0))
    quote_quantity = float(order.get("cummulativeQuoteQty", 0))
    previous_quantity = 0.0
    previous_quote = 0.0
    try:
        history = load_trade_history(path, symbol, network)
        previous = next(
            (fill for fill in history["fills"] if fill["order_id"] == order_id), None
        )
        if previous is not None:
            previous_quantity = float(previous["quantity"])
            previous_quote = float(previous["quote_quantity"])
        record_accounted_fill(
            client,
            path,
            symbol,
            network,
            order,
            "SELL",
            "hosted_stop_loss",
            reasons,
            info,
            position,
        )
    except PendingIntentError:
        raise
    except (OSError, RuntimeError) as error:
        raise TradeJournalError(
            f"hosted stop fill could not be recorded in {path}; trading stopped: {error}"
        ) from error
    delta_quantity = quantity - previous_quantity
    delta_quote = quote_quantity - previous_quote
    if delta_quantity <= 0 or delta_quote <= 0:
        return
    average_price = delta_quote / delta_quantity
    gross_pnl = (average_price - entry_price) * delta_quantity
    LOGGER.info(
        "%s SELL hosted_stop order_id=%s status=%s quantity=%.8f average=%.8f "
        "gross_pnl=%.8f %s reason=%s",
        symbol,
        order_id,
        order.get("status", "FILLED"),
        delta_quantity,
        average_price,
        gross_pnl,
        quote_asset,
        "; ".join(reasons),
    )


def timestamp_ms(timestamp: str) -> int:
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            raise ValueError("missing timezone")
        return int(parsed.timestamp() * 1000)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Trade history contains an invalid execution timestamp") from error


def is_protective_sell(fill: dict[str, Any]) -> bool:
    return fill["side"] == "SELL" and (
        fill["source"] in {"hosted_stop_loss", "protective_market"}
        or any(
            reason.lower().startswith(("stop loss", "stop-loss", "trailing stop"))
            for reason in fill.get("reasons", [])
        )
    )


def entry_guard(history: dict[str, Any]) -> dict[str, Any]:
    buys = [fill for fill in history["fills"] if fill["side"] == "BUY"]
    stops = [fill for fill in history["fills"] if is_protective_sell(fill)]
    latest_buy = max(buys, key=lambda fill: timestamp_ms(fill["timestamp_utc"]), default=None)
    latest_stop = max(stops, key=lambda fill: timestamp_ms(fill["timestamp_utc"]), default=None)
    return {
        "last_buy_signal_candle_close_time_utc": max(
            (fill.get("signal_candle_close_time_utc", fill["timestamp_utc"]) for fill in buys),
            key=timestamp_ms,
            default=None,
        ),
        "last_buy_execution_time_utc": latest_buy["timestamp_utc"] if latest_buy else None,
        "last_stop_execution_time_utc": latest_stop["timestamp_utc"] if latest_stop else None,
        "requires_rearm": latest_stop is not None and (
            latest_buy is None
            or timestamp_ms(latest_stop["timestamp_utc"]) >= timestamp_ms(latest_buy["timestamp_utc"])
        ),
    }


def buy_cooldown_reason(
    history: dict[str, Any], closed_candles: list[Candle], cooldown_candles: int,
    stop_cooldown_candles: int = 6,
) -> str | None:
    # Check all sales: a later normal exit must not shorten a still-active stop cooldown.
    for fill in sorted(
        history["fills"], key=lambda fill: timestamp_ms(fill["timestamp_utc"]), reverse=True
    ):
        if fill["side"] != "SELL":
            continue
        protective = is_protective_sell(fill)
        required = max(cooldown_candles, stop_cooldown_candles) if protective else cooldown_candles
        elapsed = len({
            candle.close_time_ms for candle in closed_candles
            if candle.close_time_ms > timestamp_ms(fill["timestamp_utc"])
        })
        if elapsed < required:
            return (
                f"BUY blocked by post-{'stop' if protective else 'sell'} cooldown: "
                f"{elapsed}/{required} closed candles elapsed"
            )
    return None


def buy_entry_block_reason(
    history: dict[str, Any], closed_candles: list[Candle], config: StrategyConfig
) -> str | None:
    guard = entry_guard(history)
    latest_close = closed_candles[-1].close_time_ms
    previous_signal = guard["last_buy_signal_candle_close_time_utc"]
    if previous_signal is not None and latest_close <= timestamp_ms(previous_signal):
        return "BUY blocked: this candle/signal was already used by a prior BUY"
    cooldown = buy_cooldown_reason(
        history, closed_candles, config.cooldown_candles, config.stop_cooldown_candles
    )
    if cooldown is not None:
        return cooldown
    if not guard["requires_rearm"]:
        return None
    stop_time = timestamp_ms(guard["last_stop_execution_time_utc"])
    age = latest_bullish_crossover_age([candle.close for candle in closed_candles], config)
    if latest_close <= stop_time or age is None or closed_candles[-1 - age].close_time_ms <= stop_time:
        return "BUY blocked: post-stop rearm requires a new bullish crossover after the stop"
    return None


def candle_after_entry(history: dict[str, Any], close_time_ms: int | None) -> bool:
    entry_time = entry_guard(history)["last_buy_execution_time_utc"]
    return close_time_ms is not None and (entry_time is None or close_time_ms > timestamp_ms(entry_time))


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
    tick = filter_value(symbol_info, "PRICE_FILTER", "tickSize") or "0"
    stop_price = floor_to_step(entry_price * (1 - stop_loss_pct / 100), tick)
    return stop_order_values(quantity, stop_price, symbol_info)


def stop_order_values(
    quantity: float, stop_price: float, symbol_info: dict[str, Any]
) -> tuple[float, float]:
    step = filter_value(symbol_info, "LOT_SIZE", "stepSize") or "0"
    tick = filter_value(symbol_info, "PRICE_FILTER", "tickSize") or "0"
    sell_quantity = floor_to_step(quantity, step)
    stop_price = floor_to_step(stop_price, tick)
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


def trailing_stop_price(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float,
    thresholds: tuple[tuple[float, float], ...],
) -> float:
    stop_price = entry_price * (1 - stop_loss_pct / 100)
    profit_pct = (current_price / entry_price - 1) * 100
    for trigger_pct, protected_profit_pct in thresholds:
        if profit_pct >= trigger_pct:
            stop_price = max(stop_price, entry_price * (1 + protected_profit_pct / 100))
    return stop_price


def parse_trailing_thresholds(value: str) -> tuple[tuple[float, float], ...]:
    try:
        thresholds = tuple(
            (float(trigger), float(protected))
            for item in value.split(",")
            for trigger, protected in [item.split(":", 1)]
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated TRIGGER:PROTECTED pairs, for example 1:0,2:1,3:2"
        ) from error
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one trailing threshold is required")
    return thresholds


def strategy_config_from_args(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(**{field.name: getattr(args, field.name) for field in fields(StrategyConfig)})


def validate_strategy_args(args: argparse.Namespace) -> None:
    if args.fast_sma <= 0 or args.slow_sma <= 0 or args.rsi_period <= 0:
        raise ValueError("SMA windows and RSI period must be positive")
    if args.fast_sma >= args.slow_sma:
        raise ValueError("--fast-sma must be smaller than --slow-sma")
    if not math.isfinite(args.quote_size) or args.quote_size <= 0:
        raise ValueError("--quote-size must be finite and positive")
    for name in ("buy_rsi_below", "sell_rsi_above"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 100")
    if not 0 <= args.stop_loss_pct < 100 or args.take_profit_pct < 0:
        raise ValueError("--stop-loss-pct must be below 100 and risk percentages cannot be negative")
    if not math.isfinite(args.min_sma_gap_pct) or args.min_sma_gap_pct < 0:
        raise ValueError("--min-sma-gap-pct must be finite and non-negative")
    if args.buy_crossover_lookback_candles <= 0:
        raise ValueError("--buy-crossover-lookback-candles must be positive")
    if not 0 <= args.buy_rsi_min <= args.buy_rsi_max <= 100:
        raise ValueError("buy RSI range must satisfy 0 <= --buy-rsi-min <= --buy-rsi-max <= 100")
    if args.cooldown_candles < 0:
        raise ValueError("--cooldown-candles cannot be negative")
    if getattr(args, "stop_cooldown_candles", 6) < args.cooldown_candles:
        raise ValueError("--stop-cooldown-candles must be at least --cooldown-candles")
    previous_trigger = -math.inf
    previous_protection = -math.inf
    for trigger, protection in args.trailing_thresholds:
        if (
            not math.isfinite(trigger)
            or not math.isfinite(protection)
            or trigger <= 0
            or protection < 0
            or protection >= trigger
            or trigger <= previous_trigger
            or protection < previous_protection
        ):
            raise ValueError(
                "--trailing-thresholds require increasing positive triggers and non-decreasing "
                "protections satisfying 0 <= PROTECTED < TRIGGER"
            )
        previous_trigger = trigger
        previous_protection = protection


def validate_args(args: argparse.Namespace) -> None:
    validate_strategy_args(args)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be finite and positive")
    if args.live and args.execute and not args.confirm_live:
        raise ValueError("live orders require --confirm-live")


def prepare_order_intent(
    path: Path,
    symbol: str,
    network: str,
    kind: str,
    source: str,
    reasons: tuple[str, ...],
    *,
    quantity: float | None = None,
    quote_quantity: float | None = None,
    stop_price: float | None = None,
    position: Position | None = None,
    signal_candle_close_time_ms: int | None = None,
    cancel_order_id: int | None = None,
) -> dict[str, Any]:
    if load_order_intent(path, symbol, network) is not None:
        raise RuntimeError(f"Cannot prepare another order while an intent exists in {path}")
    side = "BUY" if kind == "market_buy" else "SELL"
    intent = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "symbol": symbol,
        "network": network,
        "client_order_id": new_client_order_id(kind),
        "kind": kind,
        "side": side,
        "order_type": "STOP_LOSS" if kind == "hosted_stop" else "MARKET",
        "source": source,
        "reasons": list(reasons),
        "quantity": quantity,
        "quote_quantity": quote_quantity,
        "stop_price": stop_price,
        "position_quantity": position.quantity if position is not None else None,
        "entry_price": position.entry_price if position is not None else None,
        "prior_stop_price": position.stop_price if position is not None else None,
        "signal_candle_close_time_ms": signal_candle_close_time_ms,
        "cancel_order_id": cancel_order_id,
        "cancel_completed": False,
        "submission_attempted": False,
        "credential_fingerprint": credential_fingerprint(),
    }
    save_order_intent(path, intent)
    LOGGER.info(
        "%s intent prepared client_order_id=%s kind=%s",
        symbol,
        intent["client_order_id"],
        kind,
    )
    return intent


def query_intended_order(
    client: BinanceClient, intent: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        return client.order_by_client_id(intent["symbol"], intent["client_order_id"])
    except BinanceError as error:
        if error.order_not_found:
            return None
        LOGGER.warning(
            "%s intent pending client_order_id=%s: lookup failed: %s",
            intent["symbol"],
            intent["client_order_id"],
            error,
        )
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} lookup is unresolved"
        ) from error


def validate_cancel_result(
    canceled: Any, intent: dict[str, Any], cancel_order_id: int
) -> tuple[float, float]:
    try:
        order_id = canceled["orderId"]
        status = canceled["status"]
        executed = float(canceled["executedQty"])
        quote = float(canceled.get("cummulativeQuoteQty", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} received an invalid cancel response"
        ) from error
    original_quantity = float(intent["position_quantity"])
    try:
        reported_original = float(canceled["origQty"])
    except (KeyError, TypeError, ValueError) as error:
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} received invalid cancel quantities"
        ) from error
    if (
        isinstance(order_id, bool)
        or not isinstance(order_id, int)
        or order_id != cancel_order_id
        or not isinstance(status, str)
        or status not in INACTIVE_ORDER_STATUSES
        or not all(math.isfinite(value) and value >= 0 for value in (executed, quote))
        or (executed > 0 and quote <= 0)
        or (executed == 0 and quote != 0)
        or not math.isfinite(reported_original)
        or reported_original <= 0
        or not math.isclose(
            reported_original, original_quantity, rel_tol=1e-9, abs_tol=1e-12
        )
        or executed > reported_original
        or (
            status == "FILLED"
            and not math.isclose(
                executed, reported_original, rel_tol=1e-9, abs_tol=1e-12
            )
        )
        or canceled.get("symbol") != intent["symbol"]
        or canceled.get("side") != "SELL"
        or canceled.get("type") != "STOP_LOSS"
    ):
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} received an inconsistent cancel response"
        )
    return executed, quote


def finish_cancel_prerequisite(
    client: BinanceClient,
    intent_path: Path,
    intent: dict[str, Any],
    state_path: Path,
    history_path: Path,
    info: dict[str, Any],
) -> bool:
    cancel_order_id = intent["cancel_order_id"]
    if cancel_order_id is None or intent["cancel_completed"]:
        return True
    try:
        canceled = cancel_order_reconciled(client, intent["symbol"], cancel_order_id)
    except BinanceError as error:
        LOGGER.warning(
            "%s intent pending client_order_id=%s: cancel prerequisite unresolved: %s",
            intent["symbol"],
            intent["client_order_id"],
            error,
        )
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} cancel prerequisite is unresolved"
        ) from error

    try:
        canceled_quantity, _ = validate_cancel_result(canceled, intent, cancel_order_id)
    except PendingIntentError:
        LOGGER.warning(
            "%s intent pending client_order_id=%s: invalid cancel prerequisite response",
            intent["symbol"],
            intent["client_order_id"],
        )
        raise
    initial_quantity = float(intent["position_quantity"])
    remaining = remaining_quantity(initial_quantity, canceled_quantity)
    updated_position = (
        Position(
            intent["symbol"],
            remaining,
            float(intent["entry_price"]),
            None,
            intent["prior_stop_price"],
        )
        if remaining > 0
        else None
    )
    if canceled_quantity > 0:
        try:
            journal = record_accounted_fill(
                client,
                history_path,
                intent["symbol"],
                intent["network"],
                canceled,
                "SELL",
                "hosted_stop_loss",
                ("hosted stop-loss partially filled during cancellation prerequisite",),
                info,
                Position(intent["symbol"], initial_quantity, float(intent["entry_price"])),
            )
            remaining = min(remaining, inventory_sellable_quantity(journal, info))
            updated_position = (
                Position(intent["symbol"], remaining, float(intent["entry_price"]), None, intent["prior_stop_price"])
                if remaining > 0 else None
            )
        except PendingIntentError:
            raise
        except (OSError, RuntimeError) as error:
            raise TradeJournalError(
                f"hosted stop fill could not be recorded in {history_path}; "
                f"trading stopped: {error}"
            ) from error
    save_position(state_path, updated_position)

    if remaining <= 0:
        clear_order_intent(intent_path)
        LOGGER.info(
            "%s intent reconciled/recovered client_order_id=%s: cancel prerequisite filled position",
            intent["symbol"],
            intent["client_order_id"],
        )
        return False

    available = min(remaining, client.free_balance(info["baseAsset"]))
    try:
        if intent["kind"] == "hosted_stop":
            quantity, stop_price = stop_order_values(
                available, float(intent["stop_price"]), info
            )
            intent["stop_price"] = stop_price
        else:
            reference = market_reference_price(
                client, intent["symbol"], info, float(intent["entry_price"])
            )
            quantity = market_sell_quantity(available, reference, info)
    except ValueError as error:
        clear_order_intent(intent_path)
        LOGGER.error(
            "%s intent failed client_order_id=%s after cancel prerequisite: %s",
            intent["symbol"],
            intent["client_order_id"],
            error,
        )
        return False
    intent["quantity"] = quantity
    intent["position_quantity"] = remaining
    intent["cancel_completed"] = True
    save_order_intent(intent_path, intent)
    return True


def validate_reconciled_order(
    intent: dict[str, Any], order: Any
) -> tuple[int, str, float, float]:
    try:
        order_id = order["orderId"]
        status = order["status"]
        executed = float(order["executedQty"])
        quote = float(order["cummulativeQuoteQty"])
    except (KeyError, TypeError, ValueError) as error:
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} returned malformed order data"
        ) from error
    expected_quantity = (
        float(intent["quote_quantity"])
        if intent["kind"] == "market_buy"
        else float(intent["quantity"])
    )
    actual_quantity = quote if intent["kind"] == "market_buy" else executed
    if (
        not isinstance(order, dict)
        or isinstance(order_id, bool)
        or not isinstance(order_id, int)
        or order_id <= 0
        or not isinstance(status, str)
        or status not in ORDER_STATUSES
        or not all(math.isfinite(value) and value >= 0 for value in (executed, quote))
        or (executed > 0 and quote <= 0)
        or (executed == 0 and quote != 0)
        or actual_quantity > expected_quantity + max(1e-12, expected_quantity * 1e-10)
        or order.get("symbol") != intent["symbol"]
        or order.get("clientOrderId") != intent["client_order_id"]
        or order.get("side") != intent["side"]
        or order.get("type") != intent["order_type"]
        or (
            intent["kind"] == "hosted_stop"
            and status == "FILLED"
            and not math.isclose(
                executed, expected_quantity, rel_tol=1e-9, abs_tol=1e-12
            )
        )
    ):
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} returned inconsistent order data"
        )
    return order_id, status, executed, quote


def validate_hosted_stop_order(
    order: Any, position: Position
) -> tuple[str, float, float]:
    try:
        order_id = order["orderId"]
        status = order["status"]
        original = float(order["origQty"])
        executed = float(order["executedQty"])
        quote = float(order["cummulativeQuoteQty"])
    except (KeyError, TypeError, ValueError) as error:
        raise PendingIntentError(
            f"hosted stop {position.stop_order_id} returned malformed order data"
        ) from error
    if (
        not isinstance(order, dict)
        or isinstance(order_id, bool)
        or not isinstance(order_id, int)
        or order_id != position.stop_order_id
        or not isinstance(status, str)
        or status not in ORDER_STATUSES
        or order.get("symbol") != position.symbol
        or order.get("side") != "SELL"
        or order.get("type") != "STOP_LOSS"
        or not all(
            math.isfinite(value) and value >= 0
            for value in (original, executed, quote)
        )
        or original <= 0
        or not math.isclose(
            original, position.quantity, rel_tol=1e-9, abs_tol=1e-12
        )
        or executed > original
        or (executed > 0 and quote <= 0)
        or (executed == 0 and quote != 0)
        or (
            status == "FILLED"
            and not math.isclose(executed, original, rel_tol=1e-9, abs_tol=1e-12)
        )
    ):
        raise PendingIntentError(
            f"hosted stop {position.stop_order_id} returned inconsistent order data"
        )
    return status, executed, original


def execution_commissions(
    client: BinanceClient, symbol: str, order: dict[str, Any]
) -> tuple[dict[str, str], int | None, str, str]:
    """Verify a complete execution breakdown before crediting bot-owned inventory."""
    rows = order.get("fills")
    from_trades = rows is None
    if from_trades:
        try:
            rows = client.trades(symbol, int(order["orderId"]))
        except (AttributeError, BinanceError, OSError, RuntimeError, ValueError) as error:
            raise PendingIntentError(f"Order {order['orderId']} execution commissions are unavailable") from error
    if not isinstance(rows, list) or not rows:
        raise PendingIntentError(f"Order {order['orderId']} execution trades are incomplete")
    total_quantity = Decimal(0)
    total_quote = Decimal(0)
    fees: dict[str, Decimal] = {}
    execution_times = []
    seen_trade_ids = set()
    try:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid fill")
            if from_trades and (
                row.get("orderId") != order["orderId"]
                or row.get("symbol", symbol) != symbol
                or ("isBuyer" in row and row["isBuyer"] != (order.get("side") == "BUY"))
            ):
                raise ValueError("trade identity mismatch")
            if "id" in row:
                if row["id"] in seen_trade_ids:
                    raise ValueError("duplicate trade")
                seen_trade_ids.add(row["id"])
            quantity = accounting_decimal(row["qty"])
            quote = (
                accounting_decimal(row["quoteQty"]) if "quoteQty" in row
                else quantity * accounting_decimal(row["price"])
            )
            commission = accounting_decimal(row["commission"])
            asset = row["commissionAsset"]
            if quantity <= 0 or quote <= 0 or not isinstance(asset, str) or not asset:
                raise ValueError("invalid execution")
            total_quantity += quantity
            total_quote += quote
            fees[asset] = fees.get(asset, Decimal(0)) + commission
            if "time" in row:
                execution_time = int(row["time"])
                if execution_time <= 0:
                    raise ValueError("invalid execution time")
                execution_times.append(execution_time)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise PendingIntentError(f"Order {order['orderId']} execution commissions are inconsistent") from error
    if not all(
        math.isclose(float(actual), float(order[field]), rel_tol=1e-9, abs_tol=1e-12)
        for actual, field in (
            (total_quantity, "executedQty"), (total_quote, "cummulativeQuoteQty")
        )
    ):
        raise PendingIntentError(f"Order {order['orderId']} execution trade totals are inconsistent")
    return (
        {asset: format(amount, "f") for asset, amount in fees.items()},
        max(execution_times, default=None),
        format(total_quantity, "f"),
        format(total_quote, "f"),
    )


def record_accounted_fill(
    client: BinanceClient, path: Path, symbol: str, network: str,
    order: dict[str, Any], side: str, source: str, reasons: tuple[str, ...],
    info: dict[str, Any], position: Position | None = None,
    signal_candle_close_time_ms: int | None = None,
) -> dict[str, Any]:
    fees, execution_time, exact_quantity, exact_quote = execution_commissions(client, symbol, order)
    history = load_trade_history(path, symbol, network)
    if "inventory_opening" not in history and history["fills"] and history["fills"][0]["side"] == "BUY":
        migrate_inventory_history(client, path, symbol, network, info, position)
    if execution_time is not None:
        order = {**order, "transactTime": execution_time}
    order = {**order, "executedQty": exact_quantity, "cummulativeQuoteQty": exact_quote}
    record_trade_fill(
        path, symbol, network, order, side, source, reasons, signal_candle_close_time_ms,
        commissions=fees, info=info, position=position,
    )
    return load_trade_history(path, symbol, network)


def migrate_inventory_history(
    client: BinanceClient, path: Path, symbol: str, network: str,
    info: dict[str, Any], position: Position | None,
) -> None:
    """Recover legacy dust only from identified bot orders and their actual commissions."""
    history = load_trade_history(path, symbol, network)
    if "inventory_opening" in history:
        return
    fills = history["fills"]
    if fills and fills[0]["side"] != "BUY":
        # An incomplete journal cannot prove historical dust ownership.
        LOGGER.warning("Inventory begins at the existing bot position; earlier dust is not provable")
        history["inventory_opening"] = inventory_opening(info, position, len(fills))
    else:
        for fill in fills:
            fees, execution_time, exact_quantity, exact_quote = execution_commissions(
                client, symbol, {
                    "orderId": fill["order_id"], "side": fill["side"],
                    "executedQty": fill["quantity"], "cummulativeQuoteQty": fill["quote_quantity"],
                },
            )
            fill["commissions"] = fees
            fill["executed_quantity"] = exact_quantity
            fill["executed_quote_quantity"] = exact_quote
            if execution_time is not None:
                fill["timestamp_utc"] = datetime.fromtimestamp(execution_time / 1000, timezone.utc).isoformat()
        history["inventory_opening"] = inventory_opening(info, position if not fills else None, 0)
    save_trade_history(path, history)
    LOGGER.info("Bot-owned inventory ledger initialized: %s", history["inventory"])


def reconcile_order_intent(
    client: BinanceClient,
    intent_path: Path,
    intent: dict[str, Any],
    order: dict[str, Any],
    state_path: Path,
    history_path: Path,
    info: dict[str, Any],
    *,
    recovered: bool,
) -> dict[str, Any] | None:
    order_id, status, executed, quote = validate_reconciled_order(intent, order)

    terminal = status in TERMINAL_ORDER_STATUSES
    kind = intent["kind"]
    if kind == "market_buy":
        if executed > 0 and quote > 0:
            journal = record_accounted_fill(
                client,
                history_path,
                intent["symbol"],
                intent["network"],
                order,
                "BUY",
                intent["source"],
                tuple(intent["reasons"]),
                info,
                signal_candle_close_time_ms=intent["signal_candle_close_time_ms"],
            )
            owned = Decimal(journal["inventory"]["quantity"])
            quantity = floor_to_step(owned, filter_value(info, "LOT_SIZE", "stepSize") or "0")
            if quantity <= 0:
                quantity = float(owned)
            save_position(
                state_path,
                Position(intent["symbol"], quantity, quote / executed),
            )
        elif terminal:
            save_position(state_path, None)
    elif kind in {"strategy_sell", "protective_market_sell"}:
        initial_quantity = float(intent["position_quantity"])
        remaining = remaining_quantity(initial_quantity, executed)
        minimum = float(filter_value(info, "LOT_SIZE", "minQty") or 0)
        updated = (
            Position(
                intent["symbol"],
                remaining,
                float(intent["entry_price"]),
                None,
                intent["prior_stop_price"],
            )
            if remaining >= minimum and remaining > 0
            else None
        )
        if executed > 0 and quote > 0:
            journal = record_accounted_fill(
                client,
                history_path,
                intent["symbol"],
                intent["network"],
                order,
                "SELL",
                intent["source"],
                tuple(intent["reasons"]),
                info,
                Position(intent["symbol"], initial_quantity, float(intent["entry_price"])),
                intent["signal_candle_close_time_ms"],
            )
            remaining = min(remaining, inventory_sellable_quantity(journal, info))
            updated = (
                Position(intent["symbol"], remaining, float(intent["entry_price"]), None, intent["prior_stop_price"])
                if remaining >= minimum and remaining > 0 else None
            )
        save_position(state_path, updated)
    else:
        initial_quantity = float(intent["position_quantity"])
        if status in {"NEW", "PENDING_NEW", "PARTIALLY_FILLED"}:
            if executed > 0 and quote > 0:
                try:
                    record_accounted_fill(
                        client,
                        history_path,
                        intent["symbol"],
                        intent["network"],
                        order,
                        "SELL",
                        "hosted_stop_loss",
                        tuple(intent["reasons"]),
                        info,
                        Position(intent["symbol"], initial_quantity, float(intent["entry_price"])),
                    )
                except PendingIntentError:
                    raise
                except (OSError, RuntimeError) as error:
                    raise TradeJournalError(
                        f"hosted stop fill could not be recorded in {history_path}; "
                        f"trading stopped: {error}"
                    ) from error
            save_position(
                state_path,
                Position(
                    intent["symbol"],
                    initial_quantity,
                    float(intent["entry_price"]),
                    order_id,
                    float(intent["stop_price"]),
                ),
            )
            clear_order_intent(intent_path)
            LOGGER.info(
                "%s intent %s client_order_id=%s status=%s",
                intent["symbol"],
                "reconciled/recovered" if recovered else "reconciled",
                intent["client_order_id"],
                status,
            )
            return order
        if not terminal:
            LOGGER.warning(
                "%s intent pending client_order_id=%s status=%s",
                intent["symbol"],
                intent["client_order_id"],
                status,
            )
            raise PendingIntentError(
                f"order intent {intent['client_order_id']} remains {status}"
            )
        if executed == 0:
            clear_order_intent(intent_path)
            LOGGER.error(
                "%s intent failed client_order_id=%s status=%s with zero fill",
                intent["symbol"],
                intent["client_order_id"],
                status,
            )
            return None
        remaining = remaining_quantity(initial_quantity, executed)
        minimum = float(filter_value(info, "LOT_SIZE", "minQty") or 0)
        if executed > 0 and quote > 0:
            try:
                journal = record_accounted_fill(
                    client,
                    history_path,
                    intent["symbol"],
                    intent["network"],
                    order,
                    "SELL",
                    "hosted_stop_loss",
                    tuple(intent["reasons"]),
                    info,
                    Position(intent["symbol"], initial_quantity, float(intent["entry_price"])),
                )
                remaining = min(remaining, inventory_sellable_quantity(journal, info))
            except PendingIntentError:
                raise
            except (OSError, RuntimeError) as error:
                raise TradeJournalError(
                    f"hosted stop fill could not be recorded in {history_path}; "
                    f"trading stopped: {error}"
                ) from error
        save_position(
            state_path,
            Position(
                intent["symbol"],
                remaining,
                float(intent["entry_price"]),
                None,
                intent["prior_stop_price"],
            )
            if remaining >= minimum and remaining > 0
            else None,
        )
        if remaining > 0:
            clear_order_intent(intent_path)
            LOGGER.error(
                "%s intent failed client_order_id=%s status=%s after partial execution; "
                "remaining position requires protection",
                intent["symbol"],
                intent["client_order_id"],
                status,
            )
            return None

    if not terminal:
        LOGGER.warning(
            "%s intent pending client_order_id=%s status=%s",
            intent["symbol"],
            intent["client_order_id"],
            status,
        )
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} remains {status}"
        )

    clear_order_intent(intent_path)
    if executed == 0:
        LOGGER.error(
            "%s intent failed client_order_id=%s status=%s with zero fill",
            intent["symbol"],
            intent["client_order_id"],
            status,
        )
    else:
        LOGGER.info(
            "%s intent %s client_order_id=%s status=%s",
            intent["symbol"],
            "reconciled/recovered" if recovered else "reconciled",
            intent["client_order_id"],
            status,
        )
    return order


def resolve_order_intent(
    client: BinanceClient,
    intent_path: Path,
    state_path: Path,
    history_path: Path,
    symbol: str,
    network: str,
    info: dict[str, Any],
    *,
    allow_submit: bool = True,
) -> dict[str, Any] | None:
    intent = load_order_intent(intent_path, symbol, network)
    if intent is None:
        return None
    current_fingerprint = client_credential_fingerprint(client)
    if intent["credential_fingerprint"] != current_fingerprint:
        raise RuntimeError(
            f"Order intent {intent['client_order_id']} belongs to different Binance credentials"
        )
    order = query_intended_order(client, intent)
    if order is not None:
        return reconcile_order_intent(
            client,
            intent_path,
            intent,
            order,
            state_path,
            history_path,
            info,
            recovered=True,
        )
    if intent["submission_attempted"]:
        LOGGER.warning(
            "%s intent pending client_order_id=%s: prior submission may have reached Binance",
            symbol,
            intent["client_order_id"],
        )
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} was attempted and is query-only"
        )
    if not allow_submit:
        LOGGER.warning(
            "%s intent pending client_order_id=%s: execution is disabled",
            symbol,
            intent["client_order_id"],
        )
        raise PendingIntentError(
            f"order intent {intent['client_order_id']} is unresolved while execution is disabled"
        )
    if intent["cancel_order_id"] is not None:
        if not finish_cancel_prerequisite(
            client, intent_path, intent, state_path, history_path, info
        ):
            return None
        intent = load_order_intent(intent_path, symbol, network)
        if intent is None:
            return None
        order = query_intended_order(client, intent)
        if order is not None:
            return reconcile_order_intent(
                client,
                intent_path,
                intent,
                order,
                state_path,
                history_path,
                info,
                recovered=True,
            )
    intent["submission_attempted"] = True
    save_order_intent(intent_path, intent)
    LOGGER.info(
        "%s intent submitted client_order_id=%s kind=%s",
        symbol,
        intent["client_order_id"],
        intent["kind"],
    )
    try:
        if intent["kind"] == "market_buy":
            order = client.market_buy(
                symbol, float(intent["quote_quantity"]), intent["client_order_id"]
            )
        elif intent["kind"] in {"strategy_sell", "protective_market_sell"}:
            order = client.market_sell(
                symbol, float(intent["quantity"]), intent["client_order_id"]
            )
        else:
            order = client.place_stop_loss(
                symbol,
                float(intent["quantity"]),
                float(intent["stop_price"]),
                intent["client_order_id"],
            )
    except BinanceError as error:
        if not error.ambiguous:
            clear_order_intent(intent_path)
            LOGGER.error(
                "%s intent failed client_order_id=%s: %s",
                symbol,
                intent["client_order_id"],
                error,
            )
            raise
        order = query_intended_order(client, intent)
        if order is None:
            LOGGER.warning(
                "%s intent pending client_order_id=%s after ambiguous submission",
                symbol,
                intent["client_order_id"],
            )
            raise PendingIntentError(
                f"order intent {intent['client_order_id']} submission is unresolved"
            ) from error
        return reconcile_order_intent(
            client,
            intent_path,
            intent,
            order,
            state_path,
            history_path,
            info,
            recovered=True,
        )
    return reconcile_order_intent(
        client,
        intent_path,
        intent,
        order,
        state_path,
        history_path,
        info,
        recovered=False,
    )


def execute_prepared_intent(
    client: BinanceClient,
    intent_path: Path,
    state_path: Path,
    history_path: Path,
    info: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any] | None:
    return resolve_order_intent(
        client,
        intent_path,
        state_path,
        history_path,
        intent["symbol"],
        intent["network"],
        info,
    )


def execute_protective_market_exit(
    client: BinanceClient,
    args: argparse.Namespace,
    info: dict[str, Any],
    position: Position,
    history_path: Path,
    network: str,
    reason: str,
) -> Position | None:
    state_path = Path(args.state_file)
    intent_path = order_intent_path(state_path, args.symbol, network)
    reference_price = client.ticker_price(args.symbol)
    available = min(position.quantity, client.free_balance(info["baseAsset"]))
    journal = load_trade_history(history_path, args.symbol, network)
    if "inventory" in journal:
        available = min(available, float(journal["inventory"]["quantity"]))
    quantity = market_sell_quantity(available, reference_price, info)
    intent = prepare_order_intent(
        intent_path,
        args.symbol,
        network,
        "protective_market_sell",
        "protective_market",
        (reason,),
        quantity=quantity,
        position=position,
    )
    result = execute_prepared_intent(
        client, intent_path, state_path, history_path, info, intent
    )
    if result is None:
        return load_position(state_path, args.symbol)
    sold = float(result["executedQty"])
    received = float(result["cummulativeQuoteQty"])
    if not all(math.isfinite(value) and value > 0 for value in (sold, received)):
        raise BinanceError(f"Protective market sell did not fill: {result}")
    average_price = received / sold
    gross_pnl = (average_price - position.entry_price) * sold
    LOGGER.critical(
        "%s SELL protective_market order_id=%s quantity=%.8f average=%.8f "
        "gross_pnl=%.8f %s reason=%s",
        args.symbol,
        result["orderId"],
        sold,
        average_price,
        gross_pnl,
        info["quoteAsset"],
        reason,
    )
    return load_position(state_path, args.symbol)


def ensure_hosted_stop(
    client: BinanceClient,
    args: argparse.Namespace,
    config: StrategyConfig,
    info: dict[str, Any],
    position: Position,
    history_path: Path,
    network: str,
    closed_price: float,
    *,
    closed_time_ms: int | None = None,
    initial: bool = False,
) -> tuple[Position | None, bool]:
    if position.stop_order_id is not None or not args.hosted_stop_loss or config.stop_loss_pct <= 0:
        return position, False
    history = load_trade_history(history_path, args.symbol, network)
    eligible_price = (
        closed_price if not initial and candle_after_entry(history, closed_time_ms)
        else position.entry_price
    )
    desired_stop = trailing_stop_price(
        position.entry_price, eligible_price, config.stop_loss_pct, config.trailing_thresholds
    )
    desired_stop = max(desired_stop, position.stop_price or 0.0)
    available = min(position.quantity, client.free_balance(info["baseAsset"]))
    if "inventory" in history:
        available = min(available, float(history["inventory"]["quantity"]))
    stop_quantity, desired_stop = stop_order_values(available, desired_stop, info)
    live_price = client.ticker_price(args.symbol)
    if live_price <= desired_stop:
        updated = execute_protective_market_exit(
            client,
            args,
            info,
            position,
            history_path,
            network,
            f"missing hosted stop with live price {live_price:.8f} at/below {desired_stop:.8f}",
        )
        return updated, True
    state_path = Path(args.state_file)
    intent_path = order_intent_path(state_path, args.symbol, network)
    intent = prepare_order_intent(
        intent_path,
        args.symbol,
        network,
        "hosted_stop",
        "hosted_stop_loss",
        ("place or restore hosted stop-loss",),
        quantity=stop_quantity,
        stop_price=desired_stop,
        position=position,
    )
    try:
        stop_order = execute_prepared_intent(
            client, intent_path, state_path, history_path, info, intent
        )
    except BinanceError as error:
        if load_order_intent(intent_path, args.symbol, network) is not None:
            raise
        updated = load_position(state_path, args.symbol)
        if updated is None:
            return None, True
        protected = execute_protective_market_exit(
            client,
            args,
            info,
            updated,
            history_path,
            network,
            f"hosted stop placement was definitively rejected: {error}",
        )
        return protected, True
    updated = load_position(state_path, args.symbol)
    if stop_order is None:
        if updated is None:
            return None, True
        protected = execute_protective_market_exit(
            client,
            args,
            info,
            updated,
            history_path,
            network,
            "hosted stop placement terminated without protecting the remaining position",
        )
        return protected, True
    if updated is None:
        return updated, updated is None
    LOGGER.warning(
        "%s restored missing hosted stop order %s at %.8f",
        args.symbol,
        stop_order["orderId"],
        desired_stop,
    )
    return updated, False


def tighten_hosted_stop(
    client: BinanceClient,
    args: argparse.Namespace,
    config: StrategyConfig,
    closed_price: float,
    info: dict[str, Any],
    position: Position,
    stop_order: dict[str, Any],
    history_path: Path,
    network: str,
    *,
    closed_time_ms: int | None = None,
) -> Position | None:
    if closed_time_ms is not None and not candle_after_entry(
        load_trade_history(history_path, args.symbol, network), closed_time_ms
    ):
        return position
    profit_pct = (closed_price / position.entry_price - 1) * 100
    if not any(profit_pct >= trigger for trigger, _ in config.trailing_thresholds):
        return position
    desired_price = trailing_stop_price(
        position.entry_price,
        closed_price,
        config.stop_loss_pct,
        config.trailing_thresholds,
    )
    _, desired_price = stop_order_values(position.quantity, desired_price, info)
    current_stop = position.stop_price or 0.0
    if desired_price <= current_stop:
        return position

    live_price = client.ticker_price(args.symbol)
    if live_price <= desired_price:
        LOGGER.warning(
            "%s HOLD trailing stop not changed: live price %.8f is at/below target %.8f",
            args.symbol,
            live_price,
            desired_price,
        )
        return position

    state_path = Path(args.state_file)
    intent_path = order_intent_path(state_path, args.symbol, network)
    intent = prepare_order_intent(
        intent_path,
        args.symbol,
        network,
        "hosted_stop",
        "hosted_stop_loss",
        ("replace hosted stop-loss with tighter trailing stop",),
        quantity=position.quantity,
        stop_price=desired_price,
        position=position,
        cancel_order_id=position.stop_order_id,
    )
    try:
        replacement = execute_prepared_intent(
            client, intent_path, state_path, history_path, info, intent
        )
    except BinanceError as replacement_error:
        if load_order_intent(intent_path, args.symbol, network) is not None:
            raise PendingIntentError(
                f"trailing stop intent {intent['client_order_id']} remains unresolved"
            ) from replacement_error
        replacement = None
        LOGGER.error("Trailing stop replacement was rejected: %s", replacement_error)
    updated = load_position(state_path, args.symbol)
    if replacement is None:
        if updated is None:
            return None
        if current_stop > 0 and client.ticker_price(args.symbol) > current_stop:
            try:
                available = min(updated.quantity, client.free_balance(info["baseAsset"]))
                restored_quantity, restored_price = stop_order_values(
                    available, current_stop, info
                )
                restore_intent = prepare_order_intent(
                    intent_path,
                    args.symbol,
                    network,
                    "hosted_stop",
                    "hosted_stop_loss",
                    ("restore prior stop after trailing replacement failed",),
                    quantity=restored_quantity,
                    stop_price=restored_price,
                    position=updated,
                )
                restored = execute_prepared_intent(
                    client,
                    intent_path,
                    state_path,
                    history_path,
                    info,
                    restore_intent,
                )
                if restored is None:
                    raise BinanceError("Prior hosted stop restoration failed with no fill")
            except (BinanceError, ValueError) as restore_error:
                remaining_position = load_position(state_path, args.symbol)
                if remaining_position is None:
                    return None
                return execute_protective_market_exit(
                    client,
                    args,
                    info,
                    remaining_position,
                    history_path,
                    network,
                    "trailing stop replacement and prior-stop restoration failed: "
                    f"{restore_error}",
                )
            restored_position = load_position(state_path, args.symbol)
            LOGGER.warning(
                "%s HOLD tighter stop %.8f was rejected; restored stop %.8f",
                args.symbol,
                desired_price,
                restored_price,
            )
            return restored_position
        return execute_protective_market_exit(
            client,
            args,
            info,
            updated,
            history_path,
            network,
            "trailing stop replacement failed and prior stop cannot be restored",
        )
    updated = load_position(state_path, args.symbol)
    protected_pct = (desired_price / position.entry_price - 1) * 100
    LOGGER.info(
        "%s TRAIL stop moved %.8f -> %.8f, protecting %.3f%%",
        args.symbol,
        current_stop,
        desired_price,
        protected_pct,
    )
    return updated


def execute_cycle(
    client: BinanceClient,
    args: argparse.Namespace,
    config: StrategyConfig,
    info: dict[str, Any],
) -> Decision:
    state_path = Path(args.state_file)
    network = "mainnet" if args.live else "testnet"
    history_path = trade_history_path(state_path, args.symbol, network)
    intent_path = order_intent_path(state_path, args.symbol, network)
    intent_existed_at_start = load_order_intent(intent_path, args.symbol, network) is not None
    resolve_order_intent(
        client,
        intent_path,
        state_path,
        history_path,
        args.symbol,
        network,
        info,
        allow_submit=args.execute,
    )
    position = load_position(state_path, args.symbol)
    # Fetch one active candle to discard; every signal and trailing update uses only closed candles.
    limit = (
        max(
            config.slow_sma + config.buy_crossover_lookback_candles,
            config.rsi_period + 1,
            config.cooldown_candles + 1,
            config.stop_cooldown_candles + 1,
        )
        + 1
    )
    if hasattr(client, "candles"):
        candles = client.candles(args.symbol, args.interval, limit)
    else:
        prices = client.closes(args.symbol, args.interval, limit)
        candles = [Candle(close, index) for index, close in enumerate(prices)]
    closed_candles = candles[:-1]
    closes = [candle.close for candle in closed_candles]
    if not closed_candles:
        raise BinanceError("Binance returned no closed candles")
    latest_close_time_ms = closed_candles[-1].close_time_ms
    hosted_stop_filled = False
    hosted_stop_order: dict[str, Any] | None = None
    hosted_stop_executed = 0.0

    if args.execute and position is not None and position.stop_order_id is None:
        position, hosted_stop_filled = ensure_hosted_stop(
            client,
            args,
            config,
            info,
            position,
            history_path,
            network,
            closes[-1],
            closed_time_ms=latest_close_time_ms,
        )

    if args.execute and position is not None and position.stop_order_id is not None:
        stop_order = client.order(args.symbol, position.stop_order_id)
        stop_status, hosted_stop_executed, _ = validate_hosted_stop_order(
            stop_order, position
        )
        if hosted_stop_executed > 0:
            stop_reason = (
                "hosted stop-loss order filled"
                if stop_status == "FILLED"
                else "hosted stop-loss order partially filled"
            )
            record_and_log_hosted_fill(
                history_path,
                args.symbol,
                network,
                stop_order,
                (stop_reason,),
                position.entry_price,
                info["quoteAsset"],
                client=client,
                info=info,
                position=position,
            )
        if stop_status == "FILLED" or hosted_stop_executed >= position.quantity:
            save_position(state_path, None)
            position = None
            hosted_stop_filled = True
            LOGGER.info("Hosted stop-loss order %s filled; local position cleared", stop_order["orderId"])
        elif stop_status in {"NEW", "PENDING_NEW", "PARTIALLY_FILLED"}:
            hosted_stop_order = stop_order
        else:
            remaining = remaining_quantity(position.quantity, hosted_stop_executed)
            journal = load_trade_history(history_path, args.symbol, network)
            if "inventory" in journal:
                remaining = min(remaining, inventory_sellable_quantity(journal, info))
            position = (
                Position(
                    position.symbol,
                    remaining,
                    position.entry_price,
                    None,
                    position.stop_price,
                )
                if remaining > 0
                else None
            )
            save_position(state_path, position)
            if position is None:
                hosted_stop_filled = True
            LOGGER.critical(
                "Hosted stop-loss order %s is %s after filling %.8f; "
                "saved the remaining position without the inactive stop",
                stop_order["orderId"],
                stop_status,
                hosted_stop_executed,
            )
            if position is not None:
                position, protective_exit = ensure_hosted_stop(
                    client,
                    args,
                    config,
                    info,
                    position,
                    history_path,
                    network,
                    closes[-1],
                    closed_time_ms=latest_close_time_ms,
                )
                hosted_stop_filled = hosted_stop_filled or protective_exit
                if position is not None and position.stop_order_id is not None:
                    hosted_stop_order = {
                        "orderId": position.stop_order_id,
                        "status": "NEW",
                        "origQty": decimal_string(position.quantity),
                        "executedQty": "0",
                    }

    required_prices = max(config.slow_sma + 1, config.rsi_period + 1)
    if len(closes) < required_prices:
        raise InsufficientMarketData(
            f"Waiting for market history: Binance returned {len(closes)} completed "
            f"{args.interval} candle(s) for {args.symbol}; {required_prices} required"
        )

    decision = decide(closes, config, position)
    history = load_trade_history(history_path, args.symbol, network)
    if intent_existed_at_start:
        decision = Decision(
            "HOLD",
            ("recovered prior order intent; strategy deferred for this cycle",),
            decision.price,
            decision.fast_sma,
            decision.slow_sma,
            decision.rsi,
        )
    elif hosted_stop_filled:
        decision = Decision(
            "HOLD",
            ("hosted stop-loss filled during this cycle",),
            decision.price,
            decision.fast_sma,
            decision.slow_sma,
            decision.rsi,
        )
    elif position is not None and not candle_after_entry(history, latest_close_time_ms):
        decision = Decision(
            "HOLD", ("waiting for the first closed candle after the BUY execution",),
            decision.price, decision.fast_sma, decision.slow_sma, decision.rsi,
        )

    if decision.action == "BUY":
        cooldown_reason = buy_entry_block_reason(history, closed_candles, config)
        if cooldown_reason is not None:
            decision = Decision(
                "HOLD",
                (cooldown_reason,),
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
        if "inventory" in history:
            available_for_exit = min(available_for_exit, float(history["inventory"]["quantity"]))
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
        if (
            args.execute
            and position is not None
            and hosted_stop_order is not None
            and args.hosted_stop_loss
            and config.stop_loss_pct > 0
        ):
            tighten_hosted_stop(
                client,
                args,
                config,
                decision.price,
                info,
                position,
                hosted_stop_order,
                history_path,
                network,
                closed_time_ms=latest_close_time_ms,
            )
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
        intent = prepare_order_intent(
            intent_path,
            args.symbol,
            network,
            "market_buy",
            "strategy",
            decision.reasons,
            quote_quantity=args.quote_size,
            signal_candle_close_time_ms=latest_close_time_ms,
        )
        result = execute_prepared_intent(
            client, intent_path, state_path, history_path, info, intent
        )
        if result is None:
            return decision
        quantity = float(result["executedQty"])
        spent = float(result["cummulativeQuoteQty"])
        if not all(math.isfinite(value) and value > 0 for value in (quantity, spent)):
            raise BinanceError(f"Buy order did not fill: {result}")
        new_position = load_position(state_path, args.symbol)
        if new_position is None:
            raise RuntimeError("filled BUY was reconciled without a local position")
        LOGGER.info(
            "%s BUY filled order_id=%s quantity=%.8f %s average=%.8f spent=%.8f %s reason=%s",
            args.symbol,
            result["orderId"],
            quantity,
            info["baseAsset"],
            new_position.entry_price,
            spent,
            info["quoteAsset"],
            "; ".join(decision.reasons),
        )
        available = min(new_position.quantity, client.free_balance(info["baseAsset"]))
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
            try:
                new_position, _ = ensure_hosted_stop(
                    client,
                    args,
                    config,
                    info,
                    new_position,
                    history_path,
                    network,
                    decision.price,
                    closed_time_ms=latest_close_time_ms,
                    initial=True,
                )
            except BinanceError as error:
                LOGGER.critical(
                    "Buy filled but hosted stop placement failed; position remains in %s: %s",
                    state_path,
                    error,
                )
                raise
            if new_position is None:
                return decision
            LOGGER.info(
                "Hosted stop-loss order %s placed: sell %.8f %s if price reaches %.8f",
                new_position.stop_order_id,
                new_position.quantity,
                info["baseAsset"],
                new_position.stop_price,
            )
        return decision

    assert position is not None
    entry_price = position.entry_price
    assert sell_quantity is not None
    intent = prepare_order_intent(
        intent_path,
        args.symbol,
        network,
        "strategy_sell",
        "strategy",
        decision.reasons,
        quantity=sell_quantity,
        position=position,
        signal_candle_close_time_ms=latest_close_time_ms,
        cancel_order_id=position.stop_order_id,
    )
    result = execute_prepared_intent(
        client, intent_path, state_path, history_path, info, intent
    )
    if result is None:
        position = load_position(state_path, args.symbol)
        if position is not None and position.stop_order_id is None:
            try:
                position, protective_exit = ensure_hosted_stop(
                    client,
                    args,
                    config,
                    info,
                    position,
                    history_path,
                    network,
                    decision.price,
                    closed_time_ms=latest_close_time_ms,
                )
            except (BinanceError, ValueError) as error:
                raise BinanceError(
                    "hosted stop was canceled but the changed position could not be sold "
                    f"or protected: {error}"
                ) from error
            if protective_exit:
                return decision
        return Decision(
            "HOLD",
            ("SELL canceled because its cancellation prerequisite consumed or changed the position",),
            decision.price,
            decision.fast_sma,
            decision.slow_sma,
            decision.rsi,
        )
    sold = float(result["executedQty"])
    received = float(result["cummulativeQuoteQty"])
    if not all(math.isfinite(value) and value > 0 for value in (sold, received)):
        raise BinanceError(f"Sell order did not fill: {result}")
    average_price = received / sold
    gross_pnl = (average_price - entry_price) * sold
    gross_pnl_pct = (average_price / entry_price - 1) * 100
    LOGGER.info(
        "%s SELL filled order_id=%s quantity=%.8f %s average=%.8f gross_pnl=%.8f %s "
        "(%.3f%%) reason=%s",
        args.symbol,
        result["orderId"],
        sold,
        info["baseAsset"],
        average_price,
        gross_pnl,
        info["quoteAsset"],
        gross_pnl_pct,
        "; ".join(decision.reasons),
    )
    return decision


def add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quote-size", type=float, default=25, help="Quote asset spent per buy")
    parser.add_argument("--fast-sma", type=int, default=9)
    parser.add_argument("--slow-sma", type=int, default=21)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument(
        "--buy-on-bullish-trend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a rising bullish SMA trend (default: enabled)",
    )
    parser.add_argument(
        "--sell-on-bearish-trend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit on a strong bearish SMA trend with a falling slow SMA (default: enabled)",
    )
    parser.add_argument("--buy-below", type=float, help="Require market price at or below this value")
    parser.add_argument("--sell-above", type=float, help="Exit at or above this market price")
    parser.add_argument("--buy-rsi-below", type=float, help="Require RSI at or below this value")
    parser.add_argument("--sell-rsi-above", type=float, help="Exit when RSI is at or above this value")
    parser.add_argument("--stop-loss-pct", type=float, default=2.0, help="0 disables (default: 2)")
    parser.add_argument("--take-profit-pct", type=float, default=4.0, help="0 disables (default: 4)")
    parser.add_argument(
        "--min-sma-gap-pct",
        type=float,
        default=0.1,
        help="Minimum bullish/bearish SMA separation in percent (default: 0.10)",
    )
    parser.add_argument(
        "--buy-crossover-lookback-candles",
        type=int,
        default=3,
        help="Closed candles allowed for SMA crossover confirmation (default: 3)",
    )
    parser.add_argument("--buy-rsi-min", type=float, default=50.0, help="Minimum entry RSI")
    parser.add_argument("--buy-rsi-max", type=float, default=70.0, help="Maximum entry RSI")
    parser.add_argument(
        "--cooldown-candles",
        type=int,
        default=3,
        help="Closed candles after any normal sell execution before another BUY (default: 3)",
    )
    parser.add_argument(
        "--stop-cooldown-candles", type=int, default=6,
        help="Closed candles after a stop/protective sell; also requires a fresh bullish crossover (default: 6)",
    )
    parser.add_argument(
        "--trailing-thresholds",
        type=parse_trailing_thresholds,
        default=parse_trailing_thresholds("1:0,2:1,3:2"),
        metavar="TRIGGER:PROTECTED,...",
        help="Profit trigger and protected-profit percentages (default: 1:0,2:1,3:2)",
    )
    parser.add_argument(
        "--hosted-stop-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the stop at Binance after a filled buy (default: enabled)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based Binance Spot trading bot")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="15m", help="Candle interval (default: 15m)")
    parser.add_argument("--poll-seconds", type=float, default=60, help="Seconds between decisions")
    parser.add_argument("--once", action="store_true", help="Run one decision cycle and exit")
    parser.add_argument("--execute", action="store_true", help="Submit orders; otherwise dry-run")
    parser.add_argument("--live", action="store_true", help="Use Binance mainnet instead of Spot Testnet")
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Acknowledge that --execute --live uses real funds",
    )
    parser.add_argument("--state-file", default=".trader-state.json", help="Bot-owned position state")
    add_strategy_arguments(parser)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        help="Operational log path (default: trader-<SYMBOL>-<network>.log beside state)",
    )
    return parser


def configure_logging(verbose: bool, log_file: str | None) -> None:
    log_format = "%(asctime)s %(levelname)s %(message)s"
    plain_formatter = logging.Formatter(log_format)
    stream_handler = logging.StreamHandler()
    if getattr(stream_handler.stream, "isatty", lambda: False)():
        stream_handler.setFormatter(SignalColorFormatter(log_format))
    else:
        stream_handler.setFormatter(plain_formatter)
    handlers: list[logging.Handler] = [stream_handler]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(plain_formatter)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.symbol = args.symbol.upper()
    network = "mainnet" if args.live else "testnet"
    if args.log_file is None:
        args.log_file = str(
            operational_log_path(Path(args.state_file), args.symbol, network)
        )
    lock_handle = None
    try:
        configure_logging(args.verbose, args.log_file)
        validate_args(args)
        lock_handle = acquire_process_lock(args.symbol, network)
        LOGGER.info("Process lock acquired for %s on %s", args.symbol, network)
        history_path = trade_history_path(Path(args.state_file), args.symbol, network)
        initialize_trade_history(history_path, args.symbol, network)
        LOGGER.info("Trade history: %s", history_path)
        LOGGER.info(
            "BUY/SELL operations log: %s",
            trade_operations_log_path(Path(args.state_file), args.symbol, network),
        )
        LOGGER.info("Operational log: %s", args.log_file)
        config = strategy_config_from_args(args)
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
            pending = load_order_intent(
                order_intent_path(Path(args.state_file), args.symbol, network), args.symbol, network
            )
            if pending is None:
                migrate_inventory_history(
                    client, history_path, args.symbol, network, info,
                    load_position(Path(args.state_file), args.symbol),
                )

        while True:
            try:
                execute_cycle(client, args, config, info)
            except TradeJournalError as error:
                LOGGER.critical("%s", error)
                return 1
            except InsufficientMarketData as error:
                LOGGER.warning("%s", error)
                if args.once:
                    return 1
            except (BinanceError, OSError, RuntimeError, ValueError) as error:
                LOGGER.error("Cycle failed: %s", error)
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
    except (BinanceError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    finally:
        if lock_handle is not None:
            lock_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
