# Binance Spot Rule-Based Trader

A dependency-free Python bot that evaluates configurable SMA, RSI, price, stop-loss, and take-profit rules. It uses 15-minute candles and Binance Spot Testnet by default, and runs in dry-run mode unless order execution is explicitly enabled.

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
  --min-sma-gap-pct 0.10 \
  --buy-crossover-lookback-candles 3 \
  --buy-rsi-min 50 \
  --buy-rsi-max 70 \
  --sell-rsi-above 75 \
  --stop-loss-pct 2 \
  --take-profit-pct 4 \
  --cooldown-candles 3 \
  --trailing-thresholds 1:0,2:1,3:2 \
  --log-file trader-BTCUSDT-testnet.log \
  --execute
```

Entry rules are combined with **AND**. A normal bullish crossover arms an entry for three completed candles by default, allowing time for the required 0.10% SMA gap to form. The latest candle must also have rising fast and slow SMAs, a faster fast-SMA slope, and RSI from 50 through 70. Configure the confirmation window with `--buy-crossover-lookback-candles`. A strategy sell starts a three-completed-candle entry cooldown.

Exit rules are combined with **OR**. The SMA exit requires the fast SMA below the slow SMA, a falling slow SMA, and the configured minimum bearish gap, so tiny crosses are ignored. Price, RSI, stop-loss, and take-profit exits remain independent.

Signals and trailing-stop changes use completed candles only; the currently forming candle is ignored. The initial hosted stop is 2% below entry by default. With `--trailing-thresholds 1:0,2:1,3:2`, a completed-candle gain of 1% moves it to breakeven, 2% protects 1%, and 3% protects 2%.

Use `python3 trader.py --help` to see every flag. Examples:

```bash
# Price-triggered entry with the default RSI filter; disable the SMA entry rule
python3 trader.py --symbol ETHUSDT --no-buy-on-bullish-trend --buy-below 1800 --once

# Disable SMA-trend exits and use only risk/price exit rules
python3 trader.py --no-sell-on-bearish-trend --sell-above 75000 --once
```

## Safety Behavior

- Dry-run is the default. `--execute` is required to submit an order.
- Testnet is the default. Mainnet additionally requires `--live --confirm-live`.
- The bot records its own filled position in `.trader-state.json` and sells no more than that amount.
- At startup, the bot creates a persistent fill journal beside the state file. Its name includes the symbol and network, for example `.trader-history-SOLUSDT-mainnet.json`.
- Before every Binance order or hosted-stop replacement cancellation, the bot durably writes one per-symbol/network intent file beside the state file, for example `.trader-intent-SOLUSDT-mainnet.json`. State, history, and intent updates use atomic replacement with file and directory synchronization.
- Every BUY, strategy or protective SELL, and initial, restored, or trailing hosted stop has a unique Binance `clientOrderId`. On restart the bot queries that ID before submission, reconciles an already accepted order without duplicating it, or submits the same persisted ID only after Binance confirms it is absent.
- Before the first order POST, the intent is durably marked as attempted. If that attempt has an ambiguous timeout, disconnect, malformed response, Binance 408/5xx, or `-1006`/`-1007` result, recovery becomes query-only: a client-ID not-found result remains pending and requires operator investigation rather than automatic resubmission.
- Pending intents are bound to a fingerprint of the API key and cannot be replayed after credentials change. No strategy is evaluated while an intent is unresolved, and a successfully recovered intent forces `HOLD` for that cycle after required position protection is restored.
- Order, cancellation, fill, and active hosted-stop responses are identity- and quantity-checked before local mutation. A definitively rejected or terminally incomplete hosted stop immediately restores protection or triggers the durable protective market exit; an ambiguous placement remains pending without a second sell.
- Decisions and executed BUY/SELL movements are always appended to `trader-<SYMBOL>-<network>.log` beside the state file, including average prices, reasons, trailing changes, and gross realized P/L. `--log-file` overrides that path. Logs retain three 5 MB backups.
- A separate `trader-operations-<SYMBOL>-<network>.log` contains only executed BUY/SELL orders with fixed GMT-3 (`-03:00`) timestamps. It is regenerated atomically from the UTC JSON fill history, so recovered and partial orders are updated without duplicate entries.
- Every instance acquires a per-user runtime lock keyed by symbol and network; changing API keys cannot bypass locking for shared state.
- By default, a configured stop-loss is submitted to Binance after a filled buy and remains active when the script exits. Its order ID is saved in the state file. Use `--no-hosted-stop-loss` only if process-managed stops are intentional.
- Before buying, the bot verifies that the rounded sell quantity at the stop price will satisfy Binance's minimum notional. Small buys can be rejected without placing an order.
- Every filled buy is reconciled with the available balance so base-asset commission is excluded from the saved sellable quantity.
- Before a market exit, the bot checks the rounded quantity against Binance's applicable market notional using its configured average-price window. An undersized exit is deferred as `HOLD` without submitting a rejected order; state is preserved, and an active hosted stop is not canceled unless the precheck passes.
- Only one bot process per user can use a symbol/network combination. Testnet and mainnet use separate locks.
- A state file is written only after an executed buy. Dry-run decisions do not simulate holdings.
- Trade history survives position closure and contains executed market buys, strategy sells, hosted stop-loss fills, and fail-closed protective market exits. Dry-run signals and `HOLD` decisions are not recorded.
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
