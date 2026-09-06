"""Read-only Binance market data and deterministic indicator calculations."""

from __future__ import annotations

import json
import math
import statistics
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAINNET_URL = "https://api.binance.com"
TESTNET_URL = "https://testnet.binance.vision"


class MarketDataError(RuntimeError):
    """Raised when market data cannot be loaded or validated."""


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_binance_row(cls, row: list[Any]) -> "Candle":
        if len(row) < 7:
            raise MarketDataError("Binance returned an invalid kline")
        try:
            candle = cls(
                open_time_ms=int(row[0]),
                close_time_ms=int(row[6]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        except (TypeError, ValueError) as error:
            raise MarketDataError(f"Binance returned a malformed kline: {error}") from error
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(math.isfinite(value) for value in values):
            raise MarketDataError("Binance returned a non-finite kline value")
        if any(value <= 0 for value in values[:4]) or candle.volume < 0:
            raise MarketDataError("Binance returned invalid price or volume data")
        if not candle.low <= candle.open <= candle.high or not candle.low <= candle.close <= candle.high:
            raise MarketDataError("Binance returned an internally inconsistent kline")
        if candle.open_time_ms < 0 or candle.close_time_ms <= candle.open_time_ms:
            raise MarketDataError("Binance returned invalid kline timestamps")
        return candle

    def agent_dict(self) -> dict[str, Any]:
        return {
            "open_time_utc": datetime.fromtimestamp(
                self.open_time_ms / 1000, tz=timezone.utc
            ).isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BinanceMarketClient:
    """Small read-only client. This class has no order methods or credentials."""

    def __init__(self, base_url: str = MAINNET_URL, timeout: float = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def closed_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        query = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "limit": min(limit + 1, 1000)}
        )
        request = urllib.request.Request(
            f"{self.base_url}/api/v3/klines?{query}",
            headers={"User-Agent": "multi-agent-trader/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                rows = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("msg", body)
            except json.JSONDecodeError:
                detail = body
            raise MarketDataError(f"Binance HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise MarketDataError(f"Could not reach Binance: {reason}") from error
        except json.JSONDecodeError as error:
            raise MarketDataError("Binance returned invalid JSON") from error

        if not isinstance(rows, list):
            raise MarketDataError("Binance returned an unexpected kline response")
        candles = [Candle.from_binance_row(row) for row in rows]
        # The latest Binance kline is the active interval. Always discard it instead
        # of comparing with the local clock, which may be skewed across a boundary.
        completed = candles[:-1]
        if len(completed) < limit:
            raise MarketDataError(
                f"Binance returned only {len(completed)} completed candles; {limit} required"
            )
        return completed[-limit:]


def simple_rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        raise ValueError(f"RSI requires at least {period + 1} closing prices")
    changes = [new - old for old, new in zip(closes[-period - 1 : -1], closes[-period:])]
    average_gain = sum(max(change, 0.0) for change in changes) / period
    average_loss = sum(max(-change, 0.0) for change in changes) / period
    if average_gain == 0 and average_loss == 0:
        return 50.0
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def percentage_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new / old - 1) * 100


def load_position_context(
    symbol: str,
    position_file: Path | None = None,
    quantity: float | None = None,
    entry_price: float | None = None,
) -> dict[str, Any]:
    """Load the simple position format used by the original trader."""
    if position_file is not None and (quantity is not None or entry_price is not None):
        raise ValueError("use either --position-file or manual position values, not both")
    if (quantity is None) != (entry_price is None):
        raise ValueError("--position-quantity and --entry-price must be supplied together")

    data: dict[str, Any] | None = None
    if position_file is not None:
        if not position_file.exists():
            raise MarketDataError(f"Position file does not exist: {position_file}")
        try:
            loaded = json.loads(position_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise MarketDataError(f"Cannot read position file {position_file}: {error}") from error
        if not isinstance(loaded, dict):
            raise MarketDataError(f"Position file {position_file} must contain a JSON object")
        data = loaded
    elif quantity is not None and entry_price is not None:
        data = {"symbol": symbol, "quantity": quantity, "entry_price": entry_price}

    if data is None:
        return {"is_open": False, "symbol": symbol, "quantity": 0.0, "entry_price": 0.0}
    if str(data.get("symbol", "")).upper() != symbol:
        raise MarketDataError(
            f"Position contains {data.get('symbol')}, but the requested symbol is {symbol}"
        )
    try:
        parsed_quantity = float(data["quantity"])
        parsed_entry = float(data["entry_price"])
    except (KeyError, TypeError, ValueError) as error:
        raise MarketDataError("Position must contain numeric quantity and entry_price") from error
    if not all(math.isfinite(value) and value > 0 for value in (parsed_quantity, parsed_entry)):
        raise MarketDataError("Position quantity and entry_price must be positive and finite")
    return {
        "is_open": True,
        "symbol": symbol,
        "quantity": parsed_quantity,
        "entry_price": parsed_entry,
    }


def build_market_snapshot(
    symbol: str,
    interval: str,
    candles: list[Candle],
    position: dict[str, Any],
    *,
    fast_sma: int,
    slow_sma: int,
    rsi_period: int,
    volume_window: int,
    quote_size: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    agent_candles: int,
) -> dict[str, Any]:
    required = max(slow_sma, rsi_period + 1, volume_window, 21)
    if len(candles) < required:
        raise ValueError(f"at least {required} completed candles are required")
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    returns = [percentage_change(old, new) for old, new in zip(closes[-21:-1], closes[-20:])]
    average_volume = sum(volumes[-volume_window:]) / volume_window
    current_volume = volumes[-1]
    current_price = closes[-1]
    metrics = {
        "price": current_price,
        "change_1_candle_pct": percentage_change(closes[-2], current_price),
        "change_5_candles_pct": percentage_change(closes[-6], current_price),
        "change_20_candles_pct": percentage_change(closes[-21], current_price),
        "sma_fast": sum(closes[-fast_sma:]) / fast_sma,
        "sma_slow": sum(closes[-slow_sma:]) / slow_sma,
        "rsi": simple_rsi(closes, rsi_period),
        "current_volume": current_volume,
        "average_volume": average_volume,
        "volume_ratio": current_volume / average_volume if average_volume else 0.0,
        "return_volatility_20_pct": statistics.pstdev(returns),
        "high_low_range_20_pct": percentage_change(
            min(candle.low for candle in candles[-20:]),
            max(candle.high for candle in candles[-20:]),
        ),
    }
    if position["is_open"]:
        metrics["position_unrealized_pct"] = percentage_change(
            float(position["entry_price"]), current_price
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "completed_candle_count": len(candles),
        "as_of_utc": datetime.fromtimestamp(
            candles[-1].close_time_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "indicator_parameters": {
            "fast_sma_period": fast_sma,
            "slow_sma_period": slow_sma,
            "rsi_period": rsi_period,
            "rsi_method": "simple average of gains/losses over the most recent period",
            "volume_window": volume_window,
        },
        "risk_parameters": {
            "planned_quote_size": quote_size,
            "configured_stop_loss_pct": stop_loss_pct,
            "configured_take_profit_pct": take_profit_pct,
        },
        "position": position,
        "metrics": metrics,
        "recent_completed_candles": [
            candle.agent_dict() for candle in candles[-agent_candles:]
        ],
    }


def build_agent_projection(snapshot: dict[str, Any], max_quote_size: float) -> dict[str, Any]:
    """Project the auditable snapshot into the smallest useful model context."""
    recent = snapshot["recent_completed_candles"]
    closes = [float(candle["close"]) for candle in recent]
    volumes = [float(candle["volume"]) for candle in recent[-10:]]
    first_close = closes[0]
    average_volume = sum(volumes) / len(volumes) if volumes else 0
    metrics = snapshot["metrics"]
    position = snapshot["position"]

    close_path = [round(percentage_change(first_close, close), 4) for close in closes]
    volume_path = [round(volume / average_volume, 3) if average_volume else 0 for volume in volumes]
    sma_spread = percentage_change(float(metrics["sma_slow"]), float(metrics["sma_fast"]))
    model_metrics = [
        float(f"{metrics['price']:.8g}"),
        round(metrics["change_1_candle_pct"], 4),
        round(metrics["change_5_candles_pct"], 4),
        round(metrics["change_20_candles_pct"], 4),
        float(f"{metrics['sma_fast']:.8g}"),
        float(f"{metrics['sma_slow']:.8g}"),
        round(sma_spread, 4),
        round(metrics["rsi"], 2),
        round(metrics["return_volatility_20_pct"], 4),
        round(metrics["high_low_range_20_pct"], 4),
    ]
    model_position = [0, 0, 0, 0]
    if position["is_open"]:
        model_position = [
            1,
            float(f"{position['quantity']:.8g}"),
            float(f"{position['entry_price']:.8g}"),
            round(metrics["position_unrealized_pct"], 4),
        ]
    risk = snapshot["risk_parameters"]
    return {
        "s": snapshot["symbol"],
        "i": snapshot["interval"],
        "cp": close_path,
        "vr": volume_path,
        "m": model_metrics,
        "p": model_position,
        "cfg": [
            risk["planned_quote_size"],
            max_quote_size,
            risk["configured_stop_loss_pct"],
            risk["configured_take_profit_pct"],
        ],
    }
