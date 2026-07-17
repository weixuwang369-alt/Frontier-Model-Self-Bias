"""Deterministic, dependency-free token approximation.

Phase 0 never calls real tokenizers (no vendor SDKs required). We approximate tokens as
~4 characters/token, which is close enough for cost *estimates* and for sizing mock
generations. Real token counts always come from provider ``usage`` on live calls and are
persisted verbatim; this helper is only for planning and the mock provider.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def approx_chars(n_tokens: int) -> int:
    return max(1, n_tokens * CHARS_PER_TOKEN)
