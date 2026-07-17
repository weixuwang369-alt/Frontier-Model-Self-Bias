"""Provider-agnostic structured output: instruct JSON, then parse + validate it.

Judging/probe calls need a small structured verdict. Rather than depend on each
provider's native JSON-schema mode (Anthropic ``output_config``, OpenAI
``response_format``) - which differ, reject some of our schema keys, and aren't
supported by every open-model endpoint - we use one universal contract:

1. :func:`json_instruction` appends an explicit "return ONLY this JSON" spec to the prompt.
2. :func:`parse_structured` extracts and validates the JSON from the model's text.

If a real model returns malformed or incomplete JSON, ``parse_structured`` returns
``None`` so the pipeline can **fail loud** (flag + skip the row) instead of silently
coercing it to a default verdict. The mock always returns valid JSON, so Phase 0 paths
never hit the failure branch.
"""

from __future__ import annotations

import json
from typing import Any

# Schema keys that are hints for us / the mock, not part of the JSON contract.
_INTERNAL_KEYS = {"_mock_len"}


def _field_hint(name: str, spec: dict[str, Any]) -> str:
    if "enum" in spec:
        allowed = ", ".join(json.dumps(v) for v in spec["enum"])
        return f'"{name}": one of {allowed}'
    typ = spec.get("type")
    if typ == "number":
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if lo is not None and hi is not None:
            return f'"{name}": a number between {lo} and {hi}'
        return f'"{name}": a number'
    if typ == "integer":
        return f'"{name}": an integer'
    if typ == "boolean":
        return f'"{name}": true or false'
    if typ == "array":
        n = spec.get("_mock_len")
        items = spec.get("items", {})
        itype = items.get("type", "value")
        count = f"exactly {n} " if n else ""
        return f'"{name}": a JSON array of {count}{itype} values'
    return f'"{name}": a {typ or "value"}'


def json_instruction(schema: dict[str, Any]) -> str:
    """A compact 'return ONLY this JSON' instruction derived from ``schema``."""

    props: dict[str, Any] = schema.get("properties", {})
    lines = [f"- {_field_hint(name, spec)}" for name, spec in props.items()]
    return (
        "Return ONLY a single JSON object and nothing else - no prose, no markdown "
        "fences. It must have exactly these fields:\n" + "\n".join(lines)
    )


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from model output.

    Handles ```json fences and leading/trailing prose by scanning for the first
    balanced ``{...}`` and json-loading it.
    """

    if not text:
        return None
    s = text.strip()
    # Fast path: whole string is JSON.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced brace span.
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # malformed; try next '{'
        start = s.find("{", start + 1)
    return None


def parse_structured(text: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Extract + validate structured output. Returns None on any failure.

    Validation checks that every required field is present. Returning None signals a
    parse/validation failure the caller must treat as fail-loud, not coerce to defaults.
    """

    data = extract_json(text)
    if data is None:
        return None
    required = schema.get("required") or list(schema.get("properties", {}).keys())
    for key in required:
        if key in _INTERNAL_KEYS:
            continue
        if key not in data:
            return None
    return data
