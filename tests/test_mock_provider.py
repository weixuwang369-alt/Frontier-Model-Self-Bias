from __future__ import annotations

from selfbias.prompts import PWC_SCHEMA, rb_schema
from selfbias.providers.mock import MockProvider
from selfbias.schemas import LLMRequest, Message, Provider
from selfbias.tokens import approx_tokens


def _req(**kw):
    base = dict(
        provider=Provider.mock,
        model="mock-model",
        messages=(Message(role="user", content="write something"),),
        temperature=0.7,
        max_tokens=80,
        seed=5,
    )
    base.update(kw)
    return LLMRequest(**base)


def test_mock_is_deterministic():
    p = MockProvider()
    a = p.generate(_req())
    b = p.generate(_req())
    assert a.text == b.text
    assert a.usage.output_tokens == b.usage.output_tokens


def test_mock_free_text_respects_max_tokens_roughly():
    p = MockProvider()
    resp = p.generate(_req(max_tokens=120))
    n = approx_tokens(resp.text)
    # Mock aims for 85-100% of max_tokens.
    assert 0.7 * 120 <= n <= 120


def test_mock_fills_pwc_schema():
    p = MockProvider()
    resp = p.generate(_req(response_schema=PWC_SCHEMA))
    assert resp.parsed is not None
    assert resp.parsed["verdict"] in (-1, 0, 1)
    assert 0.0 <= resp.parsed["confidence"] <= 1.0


def test_mock_fills_rubric_schema_with_correct_length():
    p = MockProvider()
    resp = p.generate(_req(response_schema=rb_schema(4)))
    assert len(resp.parsed["satisfied"]) == 4
    assert all(isinstance(b, bool) for b in resp.parsed["satisfied"])
