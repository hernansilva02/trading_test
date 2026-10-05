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
  --stop-cooldown-candles 6 \
  --trailing-thresholds 1:0,2:1,3:2 \
  --log-file trader-BTCUSDT-testnet.log \
  --execute
```

Entry rules are combined with **AND**. Entries require the fast SMA above the slow SMA, both SMAs rising, the configured minimum gap (0.10% by default), and RSI from 50 through 70. Continuation entries are allowed during normal operation. Each BUY signal candle can be used only once, including across position closure and restarts.

Every executed SELL starts an entry cooldown, counted from its actual execution time using unique completed-candle closes. Normal exits use `--cooldown-candles` (default: 3). Stop-loss, hosted trailing-stop, and protective market exits use `--stop-cooldown-candles` (default: 6, at least the normal cooldown), even if the exit made a profit. After a protective exit, another BUY also requires a bullish crossover **after** that exit, within the confirmation window configured by `--buy-crossover-lookback-candles` (default: 3). Waiting out the cooldown alone does not permit a continuation entry. Partial fills count as executions; merely canceling or replacing an unfilled stop does not.

Exit rules are combined with **OR**. The SMA exit requires the fast SMA below the slow SMA, a falling slow SMA, and the configured minimum bearish gap, so tiny crosses are ignored. Price, RSI, stop-loss, and take-profit exits remain independent.

Signals and trailing-stop changes use completed candles only; the currently forming candle is ignored. The initial hosted stop is 2% below the actual BUY execution price by default. Pre-entry candles cannot tighten that stop or trigger a strategy exit, including after a restart. With `--trailing-thresholds 1:0,2:1,3:2`, a post-entry completed-candle gain of 1% moves it to breakeven, 2% protects 1%, and 3% protects 2%.

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
- The fill journal also contains a Decimal-based bot-owned inventory ledger (`inventory_opening` and `inventory`), including total quantity, remaining quote cost, fractional `residual_quantity`, and its cost. Rounding for an order does not discard ownership: the next BUY combines its net acquisition with the bot's remaining inventory. Account balances only limit availability; unrelated holdings are never credited to the ledger.
- BUY and SELL commissions are recorded by asset. Base-asset commissions reduce owned inventory; quote-asset BUY commissions enter its cost basis. Third-asset fees remain recorded in their native units. The ledger cost basis is separate from the latest execution price used by strategy stops.
- Inventory and entry guards are saved atomically with cumulative order records. Recovery replays/upserts each order, including partial fills, without crediting or consuming dust twice. Missing or inconsistent execution breakdowns leave reconciliation pending instead of assuming zero commissions.
- On the first execution-enabled startup, legacy journals beginning with a BUY are migrated by retrieving actual executions/commissions for those identified bot orders. An unavailable or inconsistent breakdown stops migration without guessing ownership. For incomplete journals beginning with a SELL, only the existing bot position is adopted as an explicit ownership checkpoint; earlier dust cannot be proven from that journal. Dry-run does not query private trade executions.
- Before a market exit, the bot checks the rounded quantity against Binance's applicable market notional using its configured average-price window. An undersized exit is deferred as `HOLD` without submitting a rejected order; state is preserved, and an active hosted stop is not canceled unless the precheck passes.
- Only one bot process per user can use a symbol/network combination. Testnet and mainnet use separate locks.
- A state file is written only after an executed buy. Dry-run decisions do not simulate holdings.
- Trade history survives position closure and contains executed market buys, strategy sells, hosted stop-loss fills, and fail-closed protective market exits. Dry-run signals and `HOLD` decisions are not recorded.
- Hosted stop orders execute at market after the trigger. Their final fill price can be lower than the trigger during a rapid move.

Real-fund execution, after testing, requires mainnet API credentials:

```bash
python3 trader.py --symbol BTCUSDT --execute --live --confirm-live
```

## Backtesting

[`backtest.py`](backtest.py) simulates one Spot symbol using the same strategy arguments,
SMA/RSI decision function, cooldowns, post-stop rearm, and dust accounting as `trader.py`.
It uses only public Binance data, or an offline CSV, and starts with a separate simulated
cash balance. It never submits orders or reads/writes the running bot's state or journal.

Download a month of completed candles and evaluate the default strategy with 100 USDT
and 10 USDT per entry:

```bash
python3 backtest.py \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2026-09-01 \
  --end 2026-10-01 \
  --initial-balance 100 \
  --quote-size 10 \
  --fee-pct 0.1 \
  --slippage-pct 0.05 \
  --save-csv reports/BTCUSDT-2026-09.csv \
  --output reports/BTCUSDT-2026-09.json
```

Dates are UTC; the start is inclusive and the end is exclusive. Downloads paginate
through Binance's 1,000-candle pages and include indicator warm-up before the requested
start. Incomplete candles are discarded. Duplicate, unordered, missing, or invalid OHLCV
bars are rejected so missing data cannot silently hide a stop. Monthly `1M` candles are
not supported; `--interval` accepts fixed-duration Binance intervals.

Repeat the run offline with the saved filter snapshot:

```bash
python3 backtest.py \
  --symbol BTCUSDT --interval 15m \
  --csv reports/BTCUSDT-2026-09.csv \
  --filters-json reports/BTCUSDT-2026-09.json \
  --start 2026-09-01 --end 2026-10-01 \
  --initial-balance 100 --quote-size 10 \
  --fee-pct 0.1 --slippage-pct 0.05 \
  --output reports/BTCUSDT-2026-09-offline.json
```

All strategy flags are shared with the live trader. To evaluate a particular launcher,
pass its SMA, RSI, stop, take-profit, cooldown, and trailing values to the backtester.
Strategy defaults remain SMA 9/21, RSI 50–70, 2% stop, 4% target, and 3/6 candle cooldowns.
The backtest defaults to a 100-unit quote balance, 10 per BUY, 0.1% fee per execution,
0.05% adverse slippage per execution, base-asset BUY commission, and quote-asset SELL
commission. `--buy-fee-asset quote` models BUY fees paid in quote instead.

### Simulation and report

- Signals use completed candles. Market BUY/SELL signals execute at the **next candle's
  open**, with slippage; the simulator never buys retrospectively at its signal's close.
- Hosted stops can execute inside a candle. A gap through a stop fills at the opening
  price rather than the unavailable stop price. An intrabar touch fills at the stop price,
  with adverse slippage; its timestamp is conservatively assigned to the candle close
  because OHLCV cannot identify the exact touch time.
- Take-profit uses closing prices, as in the live trader. Intrabar highs do not trigger
  take-profit or advance trailing. A trailing update from a completed close becomes active
  for the next candle; the initial stop uses the actual simulated BUY price.
- Commissions, cash limits, quantity/tick rounding, protective minimum notional, and
  bot-owned residual inventory are included. Dust remains part of equity and is combined
  with a subsequent BUY. Cash cannot be borrowed.
- By default, an open position is marked to the final close, including its remaining
  inventory. `--close-at-end` requests an explicit final market sale and charges exit
  costs; an exit below the symbol's minimum remains deferred.
- The console summary and optional JSON report include final equity/cash, net return,
  realized/unrealized P/L, commission cost, close-sampled maximum drawdown, closed trades,
  win rate, net profit factor, blocked orders, remaining inventory/dust, complete fills,
  trade records, and the equity curve. Undefined win rate/profit factor use JSON `null`.
- Buy-and-hold invests **all initial capital**, with the same fee/slippage assumptions;
  it is a fully invested benchmark, while the bot may use only a small part of its cash.

Downloads use the **current** public symbol filters, which may differ from historical
filters. `--filters-json` accepts a Binance symbol object, an `exchangeInfo` object, or
the filter snapshot in a prior backtest report. CSV-only runs without a snapshot use
generic defaults: quantity/tick step `0.00000001` and minimum notional `5` quote units.
Use `--quantity-step`, `--price-tick`, `--min-quantity`, and `--min-notional` to specify
the rules appropriate to the symbol/period. For example, an offline BNB simulation can
use `--quantity-step 0.001 --price-tick 0.01 --min-notional 5` if those are its applicable
rules. The JSON report records the actual rules and assumptions used.

This is a bar-based model: it assumes complete market fills and no API/order latency,
and uses execution price for minimum notional rather than Binance's historical average
price window. Drawdown is sampled at candle closes, not every market tick. Consequently,
it is a strategy evaluation rather than an exact reconstruction of live executions.

Custom CSV files require:

```csv
open_time,open,high,low,close,volume
2026-09-01T00:00:00Z,80000,80100,79900,80050,120
2026-09-01T00:15:00Z,80050,80200,80000,80150,130
```

Include enough preceding rows for warm-up: at least `max(slow_sma + 1, rsi_period + 1)`
before an explicit `--start`. Without `--start`, those initial rows are used only for
warm-up. Timestamps can be ISO-8601 or epoch **milliseconds**; `open_time_ms` and optional
`close_time`/`close_time_ms` columns are also accepted. If omitted, close time is derived
from the selected interval. `--save-csv` writes the full loaded series including warm-up.

## Tests

```bash
python3 -m unittest -v
```

## Multi-Agent Analyzer

The separate [`multi_agent_trader`](multi_agent_trader/README.md) package uses one compact OpenAI
call for trend, technical, and risk specialist classifications, followed by a minimal final
decision call. It produces a guarded `BUY`/`SELL`/`HOLD` signal and cannot submit Binance orders.
