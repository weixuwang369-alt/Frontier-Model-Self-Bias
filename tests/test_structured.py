"""Provider-agnostic structured output parsing (Phase 0.5, W2)."""

from __future__ import annotations

from selfbias.prompts import PWC_SCHEMA, rb_schema
from selfbias.structured import extract_json, json_instruction, parse_structured


def test_extract_json_plain():
    assert extract_json('{"verdict": 1, "confidence": 0.9}') == {"verdict": 1, "confidence": 0.9}


def test_extract_json_with_fences_and_prose():
    text = 'Sure!\n```json\n{"verdict": -1, "confidence": 0.5}\n```\nDone.'
    assert extract_json(text) == {"verdict": -1, "confidence": 0.5}


def test_extract_json_embedded_object():
    text = 'The answer is {"choice": 0, "confidence": 0.7} based on style.'
    assert extract_json(text) == {"choice": 0, "confidence": 0.7}


def test_extract_json_malformed_returns_none():
    assert extract_json("no json here") is None
    assert extract_json("{verdict: 1}") is None  # invalid JSON (unquoted key)
    assert extract_json("") is None


def test_parse_structured_requires_all_fields():
    ok = parse_structured('{"verdict": 0, "confidence": 0.4}', PWC_SCHEMA)
    assert ok == {"verdict": 0, "confidence": 0.4}
    # Missing 'confidence' → fail-loud None.
    assert parse_structured('{"verdict": 0}', PWC_SCHEMA) is None


def test_json_instruction_mentions_fields():
    instr = json_instruction(PWC_SCHEMA)
    assert "verdict" in instr and "confidence" in instr
    rb_instr = json_instruction(rb_schema(4))
    assert "exactly 4" in rb_instr  # array length hint for the rubric verdicts
