"""Command-line entry point for the multi-agent trading analysis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agents import (
    AgentError,
    OpenAIResponsesClient,
    apply_decision_guardrails,
    hold_decision,
    protective_exit_decision,
    run_agent_pipeline,
)
from .market import (
    MAINNET_URL,
    TESTNET_URL,
    BinanceMarketClient,
    MarketDataError,
    build_agent_projection,
    build_market_snapshot,
    load_position_context,
)


LOGGER = logging.getLogger("multi_agent_trader")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Get a token-efficient guarded BUY/SELL/HOLD decision from GPT-5.6 Sol"
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance Spot symbol")
    parser.add_argument("--interval", default="15m", help="Binance candle interval")
    parser.add_argument("--candle-limit", type=int, default=100)
    parser.add_argument("--agent-candles", type=int, default=21)
    parser.add_argument("--fast-sma", type=int, default=9)
    parser.add_argument("--slow-sma", type=int, default=21)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--quote-size", type=float, default=25)
    parser.add_argument("--max-quote-size", type=float, default=25)
    parser.add_argument("--stop-loss-pct", type=float, default=2)
    parser.add_argument("--take-profit-pct", type=float, default=4)
    parser.add_argument("--position-file", type=Path, help="Original trader JSON state file")
    parser.add_argument("--position-quantity", type=float, help="Manual open-position quantity")
    parser.add_argument("--entry-price", type=float, help="Manual open-position entry price")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
        help="Default OpenAI model (default: OPENAI_MODEL or gpt-5.6-sol)",
    )
    parser.add_argument("--specialist-model", help="Override the combined specialist model")
    parser.add_argument("--decision-model", help="Override the final decision agent model")
    parser.add_argument("--api-retries", type=int, default=0, help="Retries per OpenAI call")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--testnet-data",
        action="store_true",
        help="Read Binance Spot Testnet candles instead of read-only mainnet candles",
    )
    parser.add_argument("--output", type=Path, help="Also atomically write the JSON report here")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    periods = (args.fast_sma, args.slow_sma, args.rsi_period, args.volume_window)
    if any(period <= 0 for period in periods):
        raise ValueError("indicator periods must be positive")
    if args.fast_sma >= args.slow_sma:
        raise ValueError("--fast-sma must be smaller than --slow-sma")
    required = max(args.slow_sma, args.rsi_period + 1, args.volume_window, 21)
    if not required <= args.candle_limit <= 999:
        raise ValueError(f"--candle-limit must be between {required} and 999")
    if not 1 <= args.agent_candles <= args.candle_limit:
        raise ValueError("--agent-candles must be between 1 and --candle-limit")
    numeric_values = (
        args.quote_size,
        args.max_quote_size,
        args.stop_loss_pct,
        args.take_profit_pct,
        args.min_confidence,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("numeric arguments must be finite")
    if args.quote_size <= 0:
        raise ValueError("--quote-size must be positive")
    if args.max_quote_size <= 0:
        raise ValueError("--max-quote-size must be positive")
    if not 0 <= args.stop_loss_pct < 100 or args.take_profit_pct < 0:
        raise ValueError("risk percentages must be non-negative and stop loss must be below 100")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")
    if args.position_file and (args.position_quantity is not None or args.entry_price is not None):
        raise ValueError("use either --position-file or manual position values, not both")
    if (args.position_quantity is None) != (args.entry_price is None):
        raise ValueError("--position-quantity and --entry-price must be supplied together")
    if not 0 <= args.api_retries <= 5:
        raise ValueError("--api-retries must be between 0 and 5")


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        validate_args(args)
        symbol = args.symbol.upper()
        position = load_position_context(
            symbol,
            args.position_file,
            args.position_quantity,
            args.entry_price,
        )
        market_url = TESTNET_URL if args.testnet_data else MAINNET_URL
        LOGGER.info("Loading %s completed %s candles for %s", args.candle_limit, args.interval, symbol)
        candles = BinanceMarketClient(market_url).closed_candles(
            symbol, args.interval, args.candle_limit
        )
        snapshot = build_market_snapshot(
            symbol,
            args.interval,
            candles,
            position,
            fast_sma=args.fast_sma,
            slow_sma=args.slow_sma,
            rsi_period=args.rsi_period,
            volume_window=args.volume_window,
            quote_size=args.quote_size,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            agent_candles=args.agent_candles,
        )
        agent_input = build_agent_projection(snapshot, args.max_quote_size)
        default_model = args.model
        models = {
            "specialist": args.specialist_model or default_model,
            "decision": args.decision_model or default_model,
        }
        client: OpenAIResponsesClient | None = None
        protective_exit = protective_exit_decision(
            position,
            snapshot["metrics"]["price"],
            args.stop_loss_pct,
            args.take_profit_pct,
        )
        if protective_exit is not None:
            pipeline = {
                "specialists": None,
                "raw_decision": None,
                "decision": protective_exit,
                "decision_source": "deterministic_exit",
                "logical_api_calls": 0,
            }
            effective = protective_exit
            guardrails = [protective_exit["summary"]]
        elif not position["is_open"] and args.quote_size > args.max_quote_size:
            effective = hold_decision(
                f"Planned quote size {args.quote_size:g} exceeds local limit "
                f"{args.max_quote_size:g}",
                "GUARDRAIL",
            )
            pipeline = {
                "specialists": None,
                "raw_decision": None,
                "decision": effective,
                "decision_source": "deterministic_guardrail",
                "logical_api_calls": 0,
            }
            guardrails = [effective["summary"]]
        else:
            LOGGER.info("Running combined specialist analysis, then final decision if needed")
            client = OpenAIResponsesClient(
                os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                retries=args.api_retries,
            )
            pipeline = run_agent_pipeline(client, agent_input, models, position["is_open"])
            if pipeline["raw_decision"] is None:
                effective = pipeline["decision"]
                guardrails = [effective["summary"]]
            else:
                effective, guardrails = apply_decision_guardrails(
                    pipeline["raw_decision"],
                    pipeline["specialists"]["risk"],
                    position,
                    args.quote_size,
                    args.max_quote_size,
                    args.min_confidence,
                    snapshot["metrics"]["price"],
                    args.stop_loss_pct,
                    args.take_profit_pct,
                )
                if guardrails:
                    pipeline["decision_source"] = "deterministic_guardrail"
        report = {
            "report_schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "ANALYSIS_ONLY_NO_ORDER_EXECUTION",
            "models": models,
            "logical_api_calls": pipeline["logical_api_calls"],
            "physical_http_requests": client.physical_requests if client else 0,
            "token_usage": client.usage_summary() if client else {"requests": [], "totals": {}},
            "decision_source": pipeline["decision_source"],
            "market_snapshot": snapshot,
            "specialist_reports": pipeline["specialists"],
            "raw_decision": pipeline["raw_decision"],
            "decision": effective,
            "guardrail_reasons": guardrails,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            write_report(args.output, report)
            LOGGER.info("Wrote report to %s", args.output)
        LOGGER.info(
            "Final signal: %s (confidence %.2f); no order was submitted",
            effective["action"],
            effective["confidence"],
        )
        return 0
    except (AgentError, MarketDataError, OSError, ValueError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
        return 130
    return 2


if __name__ == "__main__":
    sys.exit(main())
