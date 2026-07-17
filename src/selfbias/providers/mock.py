"""Deterministic mock provider.

Returns canned-but-plausible responses derived purely from the request payload, so:

* identical requests always return identical responses (cache-friendly, test-stable);
* no network, no keys, no vendor SDKs - the whole Phase 0 pipeline runs on this;
* it powers the dashboard's keyless "demo mode".

Behaviour is driven by the request's ``response_schema`` when present (structured
output - judging/probe calls), otherwise it emits free text sized to ``max_tokens``
(generation calls). The mock is deliberately *unbiased*: it does not know or use author
identity, so an end-to-end mock run yields valid rows with metrics near their nulls.
Bias signal for the dashboard demo is produced separately in ``selfbias.synthetic``.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from ..schemas import LLMRequest, LLMResponse, Provider, Usage, stable_hash
from ..tokens import approx_chars, approx_tokens
from .base import Provider as BaseProvider

_LOREM = (
    "the model writes measured prose about its subject with steady clauses and "
    "concrete nouns considered carefully before each deliberate sentence unfolds "
    "toward a quiet and orderly conclusion that satisfies the given constraints "
).split()


class MockProvider(BaseProvider):
    """A stateless, deterministic provider used for Phase 0 and all tests."""

    @property
    def name(self) -> str:
        return Provider.mock.value

    def _rng(self, request: LLMRequest) -> random.Random:
        return random.Random(stable_hash(request.cache_payload(), length=16))

    def _call(self, request: LLMRequest) -> LLMResponse:
        rng = self._rng(request)
        input_tokens = sum(approx_tokens(m.content) for m in request.messages)

        if request.response_schema is not None:
            parsed = _fill_schema(request.response_schema, rng)
            text = json.dumps(parsed, separators=(",", ":"))
            usage = Usage(input_tokens=input_tokens, output_tokens=approx_tokens(text))
            return LLMResponse(
                provider=Provider.mock,
                model=request.model,
                text=text,
                usage=usage,
                raw_response={"mock": True, "parsed": parsed},
                parsed=parsed,
            )

        # Free-text generation. If the prompt states a target length ("approximately N
        # tokens"), aim for it within a tight band so the length-compliance loop
        # converges immediately; otherwise fall back to a fraction of max_tokens.
        target = _requested_length(request)
        if target is not None:
            target = max(1, int(target * rng.uniform(0.92, 1.05)))
        else:
            target = max(1, int(request.max_tokens * rng.uniform(0.85, 1.0)))
        text = _lorem(target, rng)
        usage = Usage(input_tokens=input_tokens, output_tokens=approx_tokens(text))
        return LLMResponse(
            provider=Provider.mock,
            model=request.model,
            text=text,
            usage=usage,
            raw_response={"mock": True, "chars": len(text)},
        )


_LEN_RE = re.compile(r"approximately (\d+) tokens")


def _requested_length(request: LLMRequest) -> int | None:
    """Read the target token count our generation prompt states, if present."""

    for m in request.messages:
        match = _LEN_RE.search(m.content)
        if match:
            return int(match.group(1))
    return None


def _lorem(target_tokens: int, rng: random.Random) -> str:
    """Emit deterministic filler of roughly ``target_tokens`` tokens."""

    target_chars = approx_chars(target_tokens)
    words: list[str] = []
    length = 0
    i = rng.randrange(len(_LOREM))
    while length < target_chars:
        w = _LOREM[i % len(_LOREM)]
        words.append(w)
        length += len(w) + 1
        i += 1
    return " ".join(words)


def _fill_schema(schema: dict[str, Any], rng: random.Random) -> Any:
    """Produce a deterministic value satisfying a (subset of) JSON Schema.

    Supports object/array/integer/number/boolean/string plus ``enum`` and integer
    ``minimum``/``maximum`` - enough for every structured call the pipeline issues.
    """

    if "enum" in schema:
        return rng.choice(schema["enum"])

    typ = schema.get("type")
    if typ == "object":
        props: dict[str, Any] = schema.get("properties", {})
        return {k: _fill_schema(v, rng) for k, v in props.items()}
    if typ == "array":
        items = schema.get("items", {"type": "string"})
        n = schema.get("_mock_len", 3)
        return [_fill_schema(items, rng) for _ in range(n)]
    if typ == "integer":
        lo = schema.get("minimum", 0)
        hi = schema.get("maximum", 5)
        return rng.randint(int(lo), int(hi))
    if typ == "number":
        lo = schema.get("minimum", 0.0)
        hi = schema.get("maximum", 1.0)
        return round(rng.uniform(float(lo), float(hi)), 4)
    if typ == "boolean":
        return rng.random() < 0.5
    # string / unknown
    return rng.choice(["yes", "no", "unsure"])
