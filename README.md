# Binance Spot Rule-Based Trader

A dependency-free Python bot that evaluates configurable SMA, RSI, price, stop-loss, and take-profit rules. It uses Binance Spot Testnet by default and runs in dry-run mode unless order execution is explicitly enabled.

This is example software, not financial advice. Test strategies carefully and use API keys that have withdrawals disabled.

## Requirements

- Python 3.10 or newer
- A Binance Spot Testnet account and API key for test orders

Create test credentials at [testnet.binance.vision](https://testnet.binance.vision/), then export them without putting secrets in source code:

```bash
export BINANCE_API_KEY="your-testnet-key"
export BINANCE_API_SECRET="your-testnet-secret"
```

## Run It

Inspect one decision without placing an order:

```bash
python3 trader.py --symbol BTCUSDT --once
```

Run continuously on Spot Testnet and allow orders:

```bash
python3 trader.py \
  --symbol BTCUSDT \
  --quote-size 25 \
  --fast-sma 9 \
  --slow-sma 21 \
  --buy-rsi-below 40 \
  --sell-rsi-above 70 \
  --stop-loss-pct 2 \
  --take-profit-pct 4 \
  --execute
```

Entry rules are combined with **AND**. For example, the command above buys only when the fast SMA crosses from at or below the slow SMA to above it and RSI is at or below 40. Exit rules are combined with **OR**, so a bearish SMA crossover, RSI, price, stop-loss, or take-profit condition can cause a sale.

Signals use completed candles only; the currently forming candle is ignored.

Use `python3 trader.py --help` to see every flag. Examples:

```bash
# Price-only entry; disable the default SMA entry rule
python3 trader.py --symbol ETHUSDT --no-buy-on-bullish-trend --buy-below 1800 --once

# Disable crossover-based exits and use only risk/price exit rules
python3 trader.py --no-sell-on-bearish-trend --sell-above 75000 --once
```

## Safety Behavior

- Dry-run is the default. `--execute` is required to submit an order.
- Testnet is the default. Mainnet additionally requires `--live --confirm-live`.
- The bot records its own filled position in `.trader-state.json` and sells no more than that amount.
- By default, a configured stop-loss is submitted to Binance after a filled buy and remains active when the script exits. Its order ID is saved in the state file. Use `--no-hosted-stop-loss` only if process-managed stops are intentional.
- Before buying, the bot verifies that the rounded sell quantity at the stop price will satisfy Binance's minimum notional. Small buys can be rejected without placing an order.
- Every filled buy is reconciled with the available balance so base-asset commission is excluded from the saved sellable quantity.
- Before a market exit, the bot checks the rounded quantity against Binance's applicable market notional using its configured average-price window. An undersized exit is deferred as `HOLD` without submitting a rejected order; state is preserved, and an active hosted stop is not canceled unless the precheck passes.
- Only one bot process should use a given state file. Use a separate `--state-file` for each symbol or strategy.
- A state file is written only after an executed buy. Dry-run decisions do not simulate holdings.
- Hosted stop orders execute at market after the trigger. Their final fill price can be lower than the trigger during a rapid move.

Real-fund execution, after testing, requires mainnet API credentials:

```bash
python3 trader.py --symbol BTCUSDT --execute --live --confirm-live
```

## Tests

```bash
python3 -m unittest -v
```

## Multi-Agent Analyzer

The separate [`multi_agent_trader`](multi_agent_trader/README.md) package uses one compact OpenAI
call for trend, technical, and risk specialist classifications, followed by a minimal final
decision call. It produces a guarded `BUY`/`SELL`/`HOLD` signal and cannot submit Binance orders.
