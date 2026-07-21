"""LLM-generated prompt mode (Phase 0.5, W4).

When ``prompts.source == llm_generated``, a roster model drafts a fresh prompt set at run
time (one structured call per domain returning a list of prompt strings). The pipeline
turns those into tasks via :func:`selfbias.tasks.make_task`, so they get the same
ground-truth scaffolding as built-in or Excel prompts.

Drafting is a real, cost-guarded, cache-first call (temperature 0 + fixed seed → stable
and reproducible across resumes). The request builder lives here; the pipeline owns the
execution and the dedupe/backfill so counts always match the estimate.
"""

from __future__ import annotations

from .config import DomainConfig, RosterModel
from .schemas import LLMRequest, Message, ReferenceSource
from .structured import json_instruction


def prompts_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "_mock_len": max(1, n),
            }
        },
        "required": ["prompts"],
    }


def promptgen_request(domain: DomainConfig, model: RosterModel, n: int, seed: int) -> LLMRequest:
    schema = prompts_schema(n)
    if domain.reference == ReferenceSource.programmatic:
        kind = (
            "objective instruction-following tasks with a single, programmatically "
            "checkable constraint each (e.g. an exact sentence count, a required word, "
            "or a word limit)"
        )
    else:
        kind = "open-ended tasks suitable for subjective, rubric-based evaluation"
    system = (
        "You design diverse, self-contained evaluation task prompts for a research study "
        "on LLM judging. Each prompt must stand alone and be answerable without extra "
        "context."
    )
    user = (
        f"Write {n} diverse {kind} for the '{domain.name}' domain. "
        f"Make them varied in topic and phrasing.\n\n{json_instruction(schema)}"
    )
    return LLMRequest(
        provider=model.provider,
        model=model.model,
        messages=(Message(role="system", content=system), Message(role="user", content=user)),
        temperature=0.0,  # deterministic + cacheable so resumes reproduce the same set
        max_tokens=n * 48 + 64,
        seed=seed,
        response_schema=schema,
        purpose="promptgen",
    )
