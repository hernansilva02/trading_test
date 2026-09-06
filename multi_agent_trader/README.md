# Token-Efficient Multi-Agent Trading Analyzer

A read-only Binance Spot analyzer optimized for GPT-5.6 Sol token usage. It uses two OpenAI
Responses API calls during a normal cycle:

1. A combined specialist call produces independent trend, RSI/SMA/volume, and risk classifications.
2. A final decision call receives only those compact classifications and returns `BUY`, `SELL`, or
   `HOLD`.

The specialist roles share one model call, so they are not independent model executions. This
reduces context duplication while preserving separate structured specialist results.

## Safety Boundary

This program **does not place orders** and does not accept Binance API credentials. It reads public
market data and emits an analysis signal. Connecting an LLM output directly to real-fund execution
requires additional order reconciliation, exposure limits, and operational safeguards.

Candle data, indicator values, risk parameters, and supplied position context are sent to OpenAI.
Responses use `store: false`.

## Token Controls

The API receives a compact projection instead of the complete local report:

- Close prices are normalized into a percentage path.
- Only ten normalized volume ratios are sent.
- Repeated candle field names and timestamps are omitted.
- Specialist and decision outputs use compact enums, then expand locally into readable reports.
- GPT-5.6 Sol uses standard mode with `reasoning.effort: none` and `text.verbosity: low`.
- Output limits are 256 tokens for the specialist call and 96 tokens for the decision call.
- Prompt caching uses explicit mode with no breakpoints. This prevents cache writes for short,
  changing prompts that cannot reach GPT-5.6's 1,024-token cache threshold.
- API retries default to zero to avoid potentially billable duplicate requests.

The full market snapshot remains in the local JSON report for auditing but is not sent to OpenAI.

## Call Shortcuts

Normal analysis uses two logical API calls. Some decisions need fewer:

- A risk veto while flat skips the final decision call, using one call.
- A configured stop-loss or take-profit exit uses zero calls.
- A planned quote size above the local maximum while flat uses zero calls.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with GPT-5.6 Sol and Responses API access
- Network access to OpenAI and Binance

Set the key in the environment rather than writing it to a file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Run One Analysis

From `/home/hernan/trading`:

```bash
python3 -m multi_agent_trader \
  --symbol BTCUSDT \
  --interval 15m \
  --quote-size 25 \
  --max-quote-size 25 \
  --stop-loss-pct 2 \
  --take-profit-pct 4
```

`gpt-5.6-sol` is the default model. Public Binance mainnet candles are used by default because the
program is read-only. Add `--testnet-data` to analyze Spot Testnet candles instead.

The report includes:

- `report_schema_version: 2`
- Logical API calls and physical HTTP requests
- Per-request and total input, output, cache, and reasoning token usage
- Full local market snapshot
- Expanded specialist classifications
- Raw and guarded decisions
- Decision source and guardrail reasons

Write the report atomically to a file:

```bash
python3 -m multi_agent_trader --symbol BTCUSDT --output reports/BTCUSDT.json
```

## Position Context

Without position information, the system assumes the account is flat and prevents `SELL` signals.
If `--position-file` is supplied, that exact file must exist.

Use an original trader state file:

```bash
python3 -m multi_agent_trader \
  --symbol BTCUSDT \
  --position-file .trader-state-BTCUSDT-mainnet.json
```

Or provide a position manually:

```bash
python3 -m multi_agent_trader \
  --symbol BTCUSDT \
  --position-quantity 0.001 \
  --entry-price 60000
```

Do not combine `--position-file` with manual position arguments.

## Model Overrides

Use one model for both calls or override either stage:

```bash
python3 -m multi_agent_trader \
  --model gpt-5.6-sol \
  --specialist-model gpt-5.6-sol \
  --decision-model gpt-5.6-sol
```

Retries are disabled by default. To permit one retry for transient API errors:

```bash
python3 -m multi_agent_trader --api-retries 1
```

A retry can cause an additional billable request if OpenAI processed the original request before
the connection failed.

## Guardrails

The local code changes a model action to `HOLD` when:

- Decision confidence is below `--min-confidence`, default `0.65`.
- The decision is `SELL` but no position was supplied.
- The decision is `BUY` but a position is already open.
- The risk specialist vetoes opening a position.
- `--quote-size` exceeds `--max-quote-size`.

For an open position, configured stop-loss and take-profit thresholds are evaluated locally before
calling OpenAI. A breached threshold deterministically produces `SELL`. For an accepted `BUY`, the
effective report uses the locally configured percentages.

The reported RSI uses a simple average of gains and losses over the latest period; it is not
Wilder-smoothed RSI. An LLM decision remains probabilistic and can be wrong.

## Tests

Tests use fake agents and do not call Binance or OpenAI:

```bash
python3 -m unittest -v multi_agent_trader.test_multi_agent_trader
```
