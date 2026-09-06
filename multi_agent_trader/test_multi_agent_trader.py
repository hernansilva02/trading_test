import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from multi_agent_trader.agents import (
    DECISION_AGENT,
    SPECIALIST_AGENT,
    SPECIALIST_SCHEMA,
    AgentError,
    OpenAIResponsesClient,
    apply_decision_guardrails,
    protective_exit_decision,
    run_agent_pipeline,
    validate_schema,
)
from multi_agent_trader.main import validate_args
from multi_agent_trader.market import (
    Candle,
    build_agent_projection,
    build_market_snapshot,
    load_position_context,
    simple_rsi,
)


def candles(count=30, start=100.0, step=1.0):
    result = []
    for index in range(count):
        close = start + index * step
        result.append(
            Candle(
                open_time_ms=index * 60_000,
                close_time_ms=(index + 1) * 60_000 - 1,
                open=close - 0.25,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=100 + index,
            )
        )
    return result


def flat_position():
    return {"is_open": False, "symbol": "BTCUSDT", "quantity": 0.0, "entry_price": 0.0}


def open_position():
    return {"is_open": True, "symbol": "BTCUSDT", "quantity": 1.0, "entry_price": 100.0}


def snapshot(position=None):
    return build_market_snapshot(
        "BTCUSDT",
        "15m",
        candles(),
        position or flat_position(),
        fast_sma=3,
        slow_sma=5,
        rsi_period=3,
        volume_window=5,
        quote_size=25,
        stop_loss_pct=2,
        take_profit_pct=40,
        agent_candles=21,
    )


def specialist_wire(risk_allowed=True):
    return {
        "t": {"d": "U", "s": "M", "c": 75},
        "x": {"s": "B", "r": "N", "m": "B", "v": "E", "c": 70},
        "r": {"l": "M", "a": risk_allowed, "c": 70, "f": "VOL"},
    }


class IndicatorTests(unittest.TestCase):
    def test_rsi_only_gains(self):
        self.assertEqual(simple_rsi([1, 2, 3, 4], 3), 100)

    def test_snapshot_contains_deterministic_indicators(self):
        result = snapshot()
        self.assertEqual(result["metrics"]["price"], 129)
        self.assertEqual(result["metrics"]["sma_fast"], 128)
        self.assertEqual(result["metrics"]["sma_slow"], 127)
        self.assertEqual(result["metrics"]["rsi"], 100)
        self.assertEqual(len(result["recent_completed_candles"]), 21)

    def test_agent_projection_is_compact_and_omits_repeated_candle_fields(self):
        projection = build_agent_projection(snapshot(), 25)
        rendered = json.dumps(projection, separators=(",", ":"))
        self.assertLess(len(rendered), 1_000)
        self.assertNotIn("open_time_utc", rendered)
        self.assertNotIn("recent_completed_candles", rendered)
        self.assertEqual(len(projection["cp"]), 21)
        self.assertEqual(len(projection["vr"]), 10)


class PositionTests(unittest.TestCase):
    def test_loads_original_trader_state_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "symbol": "BTCUSDT",
                        "quantity": 0.01,
                        "entry_price": 50_000,
                        "stop_order_id": 123,
                        "stop_price": 49_000,
                    }
                )
            )
            result = load_position_context("BTCUSDT", path)
        self.assertTrue(result["is_open"])
        self.assertEqual(result["quantity"], 0.01)
        self.assertEqual(result["entry_price"], 50_000)

    def test_missing_requested_position_file_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            load_position_context("BTCUSDT", Path("/definitely/not/here.json"))


class StructuredOutputTests(unittest.TestCase):
    def response(self, data, usage=None):
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(data)}],
                }
            ],
            "usage": usage or {},
        }

    def test_extracts_and_validates_compact_specialists(self):
        result = OpenAIResponsesClient._extract_json(
            self.response(specialist_wire()), "market_committee"
        )
        validate_schema(result, SPECIALIST_SCHEMA, "market_committee")
        self.assertEqual(result["t"]["d"], "U")

    def test_rejects_invalid_nested_confidence(self):
        invalid = specialist_wire()
        invalid["t"]["c"] = 101
        with self.assertRaisesRegex(AgentError, "at most 100"):
            validate_schema(invalid, SPECIALIST_SCHEMA, "market_committee")

    def test_payload_uses_gpt_56_token_controls(self):
        payload = OpenAIResponsesClient.build_payload(
            "gpt-5.6-sol", SPECIALIST_AGENT, {"m": [1, 2, 3]}
        )
        self.assertEqual(payload["reasoning"], {"mode": "standard", "effort": "none"})
        self.assertEqual(payload["text"]["verbosity"], "low")
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["prompt_cache_options"], {"mode": "explicit"})
        self.assertFalse(payload["store"])
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))

    def test_records_reasoning_and_cache_usage(self):
        client = OpenAIResponsesClient("test-key")
        client._record_usage(
            "market_committee",
            self.response(
                {},
                {
                    "input_tokens": 500,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 80,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 580,
                },
            ),
        )
        totals = client.usage_summary()["totals"]
        self.assertEqual(totals["input_tokens"], 500)
        self.assertEqual(totals["output_tokens"], 80)
        self.assertEqual(totals["reasoning_tokens"], 0)


class PipelineTests(unittest.TestCase):
    class FakeClient:
        def __init__(self, risk_allowed=True):
            self.risk_allowed = risk_allowed
            self.calls = []

        def complete(self, model, spec, input_data):
            self.calls.append((spec.name, input_data))
            if spec.name == SPECIALIST_AGENT.name:
                return specialist_wire(self.risk_allowed)
            return {"a": "B", "c": 72, "b": "ALIGN"}

    def test_normal_flow_uses_exactly_two_calls(self):
        client = self.FakeClient()
        models = {"specialist": "gpt-5.6-sol", "decision": "gpt-5.6-sol"}
        projection = build_agent_projection(snapshot(), 25)
        result = run_agent_pipeline(client, projection, models, False)
        self.assertEqual([call[0] for call in client.calls], ["market_committee", "trade_decision"])
        self.assertEqual(result["logical_api_calls"], 2)
        self.assertEqual(result["raw_decision"]["action"], "BUY")
        final_input = client.calls[1][1]
        self.assertEqual(set(final_input), {"p", "s"})
        self.assertNotIn("cp", json.dumps(final_input))

    def test_flat_risk_veto_skips_final_call(self):
        client = self.FakeClient(risk_allowed=False)
        models = {"specialist": "gpt-5.6-sol", "decision": "gpt-5.6-sol"}
        result = run_agent_pipeline(client, build_agent_projection(snapshot(), 25), models, False)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["logical_api_calls"], 1)
        self.assertIsNone(result["raw_decision"])
        self.assertEqual(result["decision"]["action"], "HOLD")

    def test_risk_veto_does_not_skip_exit_decision_for_open_position(self):
        client = self.FakeClient(risk_allowed=False)
        models = {"specialist": "gpt-5.6-sol", "decision": "gpt-5.6-sol"}
        result = run_agent_pipeline(
            client, build_agent_projection(snapshot(open_position()), 25), models, True
        )
        self.assertEqual(len(client.calls), 2)


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.decision = {
            "action": "BUY",
            "confidence": 0.8,
            "basis": "ALIGN",
            "summary": "BUY based on aligned specialist signals",
            "stop_loss_pct": 0,
            "take_profit_pct": 0,
        }
        self.risk = {"trade_allowed": True}

    def apply(self, position=None, price=100):
        return apply_decision_guardrails(
            self.decision,
            self.risk,
            position or flat_position(),
            25,
            25,
            0.65,
            price,
            2,
            40,
        )

    def test_allows_valid_buy_and_applies_local_protection(self):
        effective, reasons = self.apply()
        self.assertEqual(effective["action"], "BUY")
        self.assertEqual(effective["stop_loss_pct"], 2)
        self.assertEqual(effective["take_profit_pct"], 40)
        self.assertEqual(reasons, [])

    def test_risk_veto_changes_buy_to_hold(self):
        self.risk["trade_allowed"] = False
        effective, reasons = self.apply()
        self.assertEqual(effective["action"], "HOLD")
        self.assertIn("risk specialist vetoed", reasons[0])

    def test_sell_without_position_changes_to_hold(self):
        self.decision["action"] = "SELL"
        effective, reasons = self.apply()
        self.assertEqual(effective["action"], "HOLD")
        self.assertIn("without an open position", reasons[0])

    def test_stop_loss_can_skip_all_model_calls(self):
        decision = protective_exit_decision(open_position(), 97, 2, 40)
        self.assertEqual(decision["action"], "SELL")
        self.assertEqual(decision["basis"], "STOP_LOSS")


class ArgumentTests(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "fast_sma": 9,
            "slow_sma": 21,
            "rsi_period": 14,
            "volume_window": 20,
            "candle_limit": 100,
            "agent_candles": 21,
            "quote_size": 25,
            "max_quote_size": 25,
            "stop_loss_pct": 2,
            "take_profit_pct": 4,
            "min_confidence": 0.65,
            "position_file": None,
            "position_quantity": None,
            "entry_price": None,
            "api_retries": 0,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_rejects_non_finite_numbers(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_args(self.args(quote_size=float("nan")))

    def test_rejects_excessive_retry_count(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 5"):
            validate_args(self.args(api_retries=6))


if __name__ == "__main__":
    unittest.main()
