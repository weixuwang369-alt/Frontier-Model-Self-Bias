"""Programmatic constraint checkers (objective-arm ground truth).

Each :class:`~selfbias.schemas.Constraint` names a checker here. Checkers are pure and
local - they are the free, exact ground truth for the verifiable instruction-following
domain. ``check(constraint, text) -> bool`` returns whether ``text`` satisfies it.
"""

from __future__ import annotations

import re

from .schemas import Constraint


def exact_sentence_count(text: str, count: int) -> bool:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences) == count


def must_include(text: str, substring: str) -> bool:
    return substring.lower() in text.lower()


def max_words(text: str, max_words: int) -> bool:
    return len(text.split()) <= max_words


_REGISTRY = {
    "exact_sentence_count": exact_sentence_count,
    "must_include": must_include,
    "max_words": max_words,
}


def check(constraint: Constraint, text: str) -> bool:
    fn = _REGISTRY.get(constraint.checker)
    if fn is None:
        raise KeyError(f"unknown checker: {constraint.checker}")
    return bool(fn(text, **constraint.params))
