"""Token-efficient OpenAI specialist and decision orchestration."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class AgentError(RuntimeError):
    """Raised when an agent request fails or returns an invalid result."""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    instructions: str
    schema: dict[str, Any]
    max_output_tokens: int


# Wire keys and enum values are intentionally short. Results are expanded to
# descriptive names before they are written to the local report.
SPECIALIST_SCHEMA = {
    "type": "object",
    "properties": {
        "t": {
            "type": "object",
            "properties": {
                "d": {"type": "string", "enum": ["U", "D", "S"]},
                "s": {"type": "string", "enum": ["S", "M", "W"]},
                "c": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["d", "s", "c"],
            "additionalProperties": False,
        },
        "x": {
            "type": "object",
            "properties": {
                "s": {"type": "string", "enum": ["B", "R", "N"]},
                "r": {"type": "string", "enum": ["O", "N", "B"]},
                "m": {"type": "string", "enum": ["B", "R", "M"]},
                "v": {"type": "string", "enum": ["E", "A", "C"]},
                "c": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["s", "r", "m", "v", "c"],
            "additionalProperties": False,
        },
        "r": {
            "type": "object",
            "properties": {
                "l": {"type": "string", "enum": ["L", "M", "H", "X"]},
                "a": {"type": "boolean"},
                "c": {"type": "number", "minimum": 0, "maximum": 100},
                "f": {
                    "type": "string",
                    "enum": ["NONE", "VOL", "RANGE", "SIZE", "STOP", "DATA"],
                },
            },
            "required": ["l", "a", "c", "f"],
            "additionalProperties": False,
        },
    },
    "required": ["t", "x", "r"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "string", "enum": ["B", "S", "H"]},
        "c": {"type": "number", "minimum": 0, "maximum": 100},
        "b": {
            "type": "string",
            "enum": ["ALIGN", "TREND", "TECH", "RISK", "MIXED", "WEAK"],
        },
    },
    "required": ["a", "c", "b"],
    "additionalProperties": False,
}

SPECIALIST_AGENT = AgentSpec(
    "market_committee",
    """Classify completed-candle numeric JSON as three specialist roles. Input legend:
cp=close path % from first point; vr=recent volume/mean ratios; m=[price,change1,change5,
change20,fastSMA,slowSMA,SMAspread%,RSI,volatility%,range%]; p=[open,qty,entry,
unrealized%]; cfg=[planned quote,max quote,stop%,target%]. Output: t trend d U/D/S and
strength S/M/W; x technical signal B=bullish,R=bearish,N=neutral, RSI O/N/B,
SMA B/R/M, volume E/A/C; r risk L/M/H/X, entry allowed, confidence, main flag.
Use only supplied data. Assess each role independently. No prose or external data.""",
    SPECIALIST_SCHEMA,
    256,
)

DECISION_AGENT = AgentSpec(
    "trade_decision",
    """Choose one action from compact specialist JSON. p=1 means position open.
Action B=buy,S=sell,H=hold. Never sell when p=0 or buy when p=1. Risk veto blocks
buy, not sell. Prefer hold for mixed or weak evidence. Return confidence and basis only.""",
    DECISION_SCHEMA,
    96,
)


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60,
        retries: int = 0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.physical_requests = 0
        self.usage_records: list[dict[str, Any]] = []

    def complete(self, model: str, spec: AgentSpec, input_data: dict[str, Any]) -> dict[str, Any]:
        payload = self.build_payload(model, spec, input_data)
        response = self._post(payload)
        self._record_usage(spec.name, response)
        result = self._extract_json(response, spec.name)
        validate_schema(result, spec.schema, spec.name)
        return result

    @staticmethod
    def build_payload(model: str, spec: AgentSpec, input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": model,
            "store": False,
            "reasoning": {"mode": "standard", "effort": "none"},
            "input": [
                {"role": "developer", "content": spec.instructions},
                {"role": "user", "content": json.dumps(input_data, separators=(",", ":"))},
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": spec.name,
                    "schema": spec.schema,
                    "strict": True,
                },
            },
            "max_output_tokens": spec.max_output_tokens,
            # Short, changing market prompts cannot reach GPT-5.6's 1,024-token
            # cache threshold. Explicit mode avoids paying to write useless entries.
            "prompt_cache_options": {"mode": "explicit"},
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "multi-agent-trader/2.0",
                },
                method="POST",
            )
            self.physical_requests += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode())
                if not isinstance(parsed, dict):
                    raise AgentError("OpenAI returned an unexpected response")
                return parsed
            except urllib.error.HTTPError as error:
                body = error.read().decode(errors="replace")
                try:
                    detail = json.loads(body).get("error", {}).get("message", body)
                except json.JSONDecodeError:
                    detail = body
                if error.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self.retries:
                    raise AgentError(f"OpenAI HTTP {error.code}: {detail}") from error
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else None
                except ValueError:
                    delay = None
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.retries:
                    reason = getattr(error, "reason", error)
                    raise AgentError(f"Could not reach OpenAI: {reason}") from error
                delay = None
            except json.JSONDecodeError as error:
                raise AgentError("OpenAI returned invalid JSON") from error
            time.sleep(delay if delay is not None else 2**attempt + random.uniform(0, 0.5))
        raise AssertionError("unreachable")

    def _record_usage(self, agent_name: str, response: dict[str, Any]) -> None:
        usage = response.get("usage", {})
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        self.usage_records.append(
            {
                "agent": agent_name,
                "input_tokens": int(usage.get("input_tokens", 0)),
                "cached_tokens": int(input_details.get("cached_tokens", 0)),
                "cache_write_tokens": int(input_details.get("cache_write_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "reasoning_tokens": int(output_details.get("reasoning_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        keys = (
            "input_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
        return {
            "requests": self.usage_records,
            "totals": {key: sum(record[key] for record in self.usage_records) for key in keys},
        }

    @staticmethod
    def _extract_json(response: dict[str, Any], agent_name: str) -> dict[str, Any]:
        status = response.get("status")
        if status == "incomplete":
            reason = response.get("incomplete_details", {}).get("reason", "unknown reason")
            raise AgentError(f"{agent_name} response was incomplete: {reason}")
        if status != "completed":
            error = response.get("error")
            detail = error.get("message") if isinstance(error, dict) else status or "unknown status"
            raise AgentError(f"{agent_name} response did not complete: {detail}")
        texts: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "refusal":
                    raise AgentError(f"{agent_name} refused: {content.get('refusal', 'no reason')}")
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        if not texts:
            raise AgentError(f"{agent_name} returned no text output")
        try:
            parsed = json.loads("".join(texts))
        except json.JSONDecodeError as error:
            raise AgentError(f"{agent_name} returned invalid structured JSON") from error
        if not isinstance(parsed, dict):
            raise AgentError(f"{agent_name} output must be a JSON object")
        return parsed


def validate_schema(value: Any, schema: dict[str, Any], path: str = "result") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise AgentError(f"{path} must be an object")
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise AgentError(f"{path} is missing fields: {', '.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise AgentError(f"{path} has unexpected fields: {', '.join(extra)}")
        for key, child_schema in properties.items():
            if key in value:
                validate_schema(value[key], child_schema, f"{path}.{key}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise AgentError(f"{path} must be an array")
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], f"{path}[{index}]")
        return
    if expected == "string":
        if not isinstance(value, str):
            raise AgentError(f"{path} must be a string")
        if "enum" in schema and value not in schema["enum"]:
            raise AgentError(f"{path} has invalid value {value!r}")
        return
    if expected == "boolean":
        if not isinstance(value, bool):
            raise AgentError(f"{path} must be a boolean")
        return
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AgentError(f"{path} must be a finite number")
        if "minimum" in schema and value < schema["minimum"]:
            raise AgentError(f"{path} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise AgentError(f"{path} must be at most {schema['maximum']}")
        return
    raise AgentError(f"{path} uses unsupported schema type {expected!r}")


def expand_specialists(wire: dict[str, Any]) -> dict[str, Any]:
    trend_direction = {"U": "UP", "D": "DOWN", "S": "SIDEWAYS"}
    trend_strength = {"S": "STRONG", "M": "MODERATE", "W": "WEAK"}
    signals = {"B": "BULLISH", "R": "BEARISH", "N": "NEUTRAL"}
    rsi_states = {"O": "OVERSOLD", "N": "NEUTRAL", "B": "OVERBOUGHT"}
    sma_states = {"B": "BULLISH_ALIGNMENT", "R": "BEARISH_ALIGNMENT", "M": "MIXED"}
    volume_states = {"E": "EXPANDING", "A": "AVERAGE", "C": "CONTRACTING"}
    risk_levels = {"L": "LOW", "M": "MEDIUM", "H": "HIGH", "X": "EXTREME"}
    risk_flags = {
        "NONE": "none",
        "VOL": "volatility",
        "RANGE": "price_range",
        "SIZE": "position_size",
        "STOP": "protection",
        "DATA": "data_uncertainty",
    }
    return {
        "trend": {
            "direction": trend_direction[wire["t"]["d"]],
            "strength": trend_strength[wire["t"]["s"]],
            "confidence": wire["t"]["c"] / 100,
        },
        "technical": {
            "signal": signals[wire["x"]["s"]],
            "rsi_state": rsi_states[wire["x"]["r"]],
            "sma_state": sma_states[wire["x"]["m"]],
            "volume_state": volume_states[wire["x"]["v"]],
            "confidence": wire["x"]["c"] / 100,
        },
        "risk": {
            "risk_level": risk_levels[wire["r"]["l"]],
            "trade_allowed": wire["r"]["a"],
            "confidence": wire["r"]["c"] / 100,
            "primary_concern": risk_flags[wire["r"]["f"]],
        },
    }


def expand_decision(wire: dict[str, Any]) -> dict[str, Any]:
    actions = {"B": "BUY", "S": "SELL", "H": "HOLD"}
    bases = {
        "ALIGN": "aligned specialist signals",
        "TREND": "trend signal",
        "TECH": "technical signal",
        "RISK": "risk constraint",
        "MIXED": "mixed specialist signals",
        "WEAK": "weak evidence",
    }
    action = actions[wire["a"]]
    basis = bases[wire["b"]]
    return {
        "action": action,
        "confidence": wire["c"] / 100,
        "basis": wire["b"],
        "summary": f"{action} based on {basis}",
        "stop_loss_pct": 0,
        "take_profit_pct": 0,
    }


def hold_decision(summary: str, basis: str = "RISK") -> dict[str, Any]:
    return {
        "action": "HOLD",
        "confidence": 1.0,
        "basis": basis,
        "summary": summary,
        "stop_loss_pct": 0,
        "take_profit_pct": 0,
    }


def protective_exit_decision(
    position: dict[str, Any], current_price: float, stop_loss_pct: float, take_profit_pct: float
) -> dict[str, Any] | None:
    if not position["is_open"]:
        return None
    entry_price = float(position["entry_price"])
    if stop_loss_pct > 0 and current_price <= entry_price * (1 - stop_loss_pct / 100):
        return {
            "action": "SELL",
            "confidence": 1.0,
            "basis": "STOP_LOSS",
            "summary": "Deterministic exit: configured stop loss was breached",
            "stop_loss_pct": 0,
            "take_profit_pct": 0,
        }
    if take_profit_pct > 0 and current_price >= entry_price * (1 + take_profit_pct / 100):
        return {
            "action": "SELL",
            "confidence": 1.0,
            "basis": "TAKE_PROFIT",
            "summary": "Deterministic exit: configured take profit was reached",
            "stop_loss_pct": 0,
            "take_profit_pct": 0,
        }
    return None


def run_agent_pipeline(
    client: OpenAIResponsesClient,
    agent_input: dict[str, Any],
    models: dict[str, str],
    position_open: bool,
) -> dict[str, Any]:
    specialist_wire = client.complete(models["specialist"], SPECIALIST_AGENT, agent_input)
    specialists = expand_specialists(specialist_wire)
    if not position_open and not specialists["risk"]["trade_allowed"]:
        return {
            "specialists": specialists,
            "raw_decision": None,
            "decision": hold_decision("Risk specialist vetoed opening a position"),
            "decision_source": "deterministic_guardrail",
            "logical_api_calls": 1,
        }

    final_wire = client.complete(
        models["decision"],
        DECISION_AGENT,
        {"p": 1 if position_open else 0, "s": specialist_wire},
    )
    decision = expand_decision(final_wire)
    return {
        "specialists": specialists,
        "raw_decision": decision,
        "decision": decision,
        "decision_source": "model_guarded",
        "logical_api_calls": 2,
    }


def apply_decision_guardrails(
    decision: dict[str, Any],
    risk: dict[str, Any],
    position: dict[str, Any],
    quote_size: float,
    max_quote_size: float,
    min_confidence: float,
    current_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[dict[str, Any], list[str]]:
    protective_exit = protective_exit_decision(
        position, current_price, stop_loss_pct, take_profit_pct
    )
    if protective_exit is not None:
        return protective_exit, [protective_exit["summary"]]

    effective = dict(decision)
    reasons: list[str] = []
    action = effective["action"]
    if action != "HOLD" and effective["confidence"] < min_confidence:
        reasons.append(
            f"confidence {effective['confidence']:.2f} is below required {min_confidence:.2f}"
        )
    if action == "BUY" and position["is_open"]:
        reasons.append("BUY is invalid while a position is already open")
    if action == "SELL" and not position["is_open"]:
        reasons.append("SELL is invalid without an open position")
    if action == "BUY" and not risk["trade_allowed"]:
        reasons.append("risk specialist vetoed opening a position")
    if action == "BUY" and quote_size > max_quote_size:
        reasons.append(f"planned quote size {quote_size:g} exceeds local limit {max_quote_size:g}")
    if reasons:
        effective = hold_decision(
            "Guardrails changed the model decision to HOLD: " + "; ".join(reasons),
            "GUARDRAIL",
        )
    elif effective["action"] == "BUY":
        effective["stop_loss_pct"] = stop_loss_pct
        effective["take_profit_pct"] = take_profit_pct
    else:
        effective["stop_loss_pct"] = 0
        effective["take_profit_pct"] = 0
    return effective, reasons
