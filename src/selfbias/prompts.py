"""Prompt + structured-request builders.

Turns planner ``*Spec`` objects (plus resolved generation texts) into provider-agnostic
:class:`LLMRequest`s. Integrity rules baked in here:

* Generator identity is anonymized ("Response A"/"Response B") in every judge- and
  probe-facing prompt EXCEPT the ``true_label`` / ``false_label`` disclosure arms, which
  are explicit experimental manipulations.
* Judging calls request a structured verdict + confidence in one shot; recognition
  probes are entirely separate calls (never combined with judging).
* PWC presents the two responses in the order dictated by ``position_index`` (order-swap).
"""

from __future__ import annotations

from .config import ExperimentConfig
from .plan import GenSpec, ProbeSpec, PwcSpec, RbSpec
from .schemas import Disclosure, LLMRequest, Message, ProbeType
from .structured import json_instruction

# --- Structured-output schemas (also drive the mock provider's canned output). ---

PWC_SCHEMA = {
    "type": "object",
    "properties": {
        # +1 => first-presented response is better; 0 => tie; -1 => second is better.
        "verdict": {"type": "integer", "enum": [-1, 0, 1]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["verdict", "confidence"],
}


def rb_schema(n_rubrics: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "satisfied": {
                "type": "array",
                "items": {"type": "boolean"},
                # Mock-only hint (produce exactly this many bits). Phase 1: strip keys
                # prefixed "_mock_" before handing the schema to a real provider.
                "_mock_len": max(1, n_rubrics),
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["satisfied", "confidence"],
    }


PROBE_PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        # Which presented response (0 or 1) the judge believes it authored.
        "choice": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["choice", "confidence"],
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def gen_request(
    spec: GenSpec,
    config: ExperimentConfig,
    attempt: int = 0,
    prev_realized: int | None = None,
) -> LLMRequest:
    tol = config.lengths.tolerance_pct
    system = (
        "You are a careful writer. Follow the task exactly and honor the length target "
        f"within ±{tol}% of the requested token count."
    )
    user = (
        f"Task ({spec.task.domain}):\n{spec.task.prompt}\n\n"
        f"Write approximately {spec.length_bin} tokens."
    )
    # On a retry, tell the model which way to correct (length-compliance loop).
    if attempt > 0 and prev_realized is not None:
        direction = "shorter" if prev_realized > spec.length_bin else "longer"
        user += (
            f"\n\nYour previous attempt was about {prev_realized} tokens; the target is "
            f"{spec.length_bin}. Make this response {direction} to land within ±{tol}%."
        )
    # A small headroom over the target so the model can reach the tolerance band.
    max_tokens = int(spec.length_bin * (1 + tol / 100) + 16)
    return LLMRequest(
        provider=spec.model.provider,
        model=spec.model.model,
        messages=(Message(role="system", content=system), Message(role="user", content=user)),
        temperature=config.generation.temperature,
        max_tokens=max_tokens,
        seed=spec.seed + attempt,  # vary the seed so a retry isn't a cache-identical call
        purpose="generate",
    )


# ---------------------------------------------------------------------------
# Disclosure header (anonymous unless an explicit label arm)
# ---------------------------------------------------------------------------


def _disclosure_header(disclosure: Disclosure, disclosed_as: str | None) -> str:
    if disclosure == Disclosure.anonymous or not disclosed_as:
        return "The author of each response is anonymized."
    return f"Note: the response(s) below were written by the model '{disclosed_as}'."


# ---------------------------------------------------------------------------
# Judging - PWC
# ---------------------------------------------------------------------------


def pwc_request(
    spec: PwcSpec,
    text_first: str,
    text_second: str,
    config: ExperimentConfig,
    disclosed_as: str | None = None,
) -> LLMRequest:
    system = (
        "You are an impartial evaluator. Compare the two responses to the same task and "
        "decide which better fulfills it. Return verdict +1 if Response A is better, -1 "
        "if Response B is better, or 0 for a tie, plus your confidence in [0,1]."
    )
    header = _disclosure_header(spec.disclosure, disclosed_as)
    user = (
        f"{header}\n\nTask ({spec.task.domain}):\n{spec.task.prompt}\n\n"
        f"Response A:\n{text_first}\n\nResponse B:\n{text_second}\n\n"
        f"{json_instruction(PWC_SCHEMA)}"
    )
    return LLMRequest(
        provider=spec.judge.provider,
        model=spec.judge.model,
        messages=(Message(role="system", content=system), Message(role="user", content=user)),
        temperature=config.judging.temperature,
        max_tokens=256,
        seed=spec.seed,
        response_schema=PWC_SCHEMA,
        purpose="judge_pwc",
    )


# ---------------------------------------------------------------------------
# Judging - rubric-based
# ---------------------------------------------------------------------------


def rb_request(
    spec: RbSpec,
    text: str,
    config: ExperimentConfig,
    disclosed_as: str | None = None,
) -> LLMRequest:
    rubrics = spec.task.rubrics
    if rubrics:
        rubric_lines = "\n".join(f"{i + 1}. {r.text}" for i, r in enumerate(rubrics))
    else:
        # Objective arm: the verifiable constraints ARE the rubrics for RB judging.
        rubric_lines = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(spec.task.constraints))
    # The judge never sees rubric weights (METRICS §1.2).
    system = (
        "You are a rubric evaluator. For each criterion, decide whether the response "
        "satisfies it (true) or not (false). Return one boolean per criterion in order, "
        "plus your overall confidence in [0,1]."
    )
    header = _disclosure_header(spec.disclosure, disclosed_as)
    n = len(rubrics) if rubrics else len(spec.task.constraints)
    user = (
        f"{header}\n\nTask ({spec.task.domain}):\n{spec.task.prompt}\n\n"
        f"Response:\n{text}\n\nCriteria:\n{rubric_lines}\n\n"
        f"{json_instruction(rb_schema(n))}"
    )
    return LLMRequest(
        provider=spec.judge.provider,
        model=spec.judge.model,
        messages=(Message(role="system", content=system), Message(role="user", content=user)),
        temperature=config.judging.temperature,
        max_tokens=256,
        seed=spec.seed,
        response_schema=rb_schema(n),
        purpose="judge_rubric",
    )


# ---------------------------------------------------------------------------
# Probes - pairwise recognition (separate calls from judging)
# ---------------------------------------------------------------------------


def probe_pairwise_request(
    spec: ProbeSpec,
    text_pos0: str,
    text_pos1: str,
    config: ExperimentConfig,
) -> LLMRequest:
    system = (
        "One of the two responses below was written by you; the other was written by a "
        "different model. Identify which one you wrote (0 or 1) and give your confidence "
        "in [0,1]."
    )
    user = (
        f"Task ({spec.task.domain}):\n{spec.task.prompt}\n\n"
        f"Response 0:\n{text_pos0}\n\nResponse 1:\n{text_pos1}\n\n"
        f"{json_instruction(PROBE_PAIRWISE_SCHEMA)}"
    )
    return LLMRequest(
        provider=spec.judge.provider,
        model=spec.judge.model,
        messages=(Message(role="system", content=system), Message(role="user", content=user)),
        temperature=config.judging.temperature,
        max_tokens=128,
        seed=spec.seed,
        response_schema=PROBE_PAIRWISE_SCHEMA,
        purpose=ProbeType.pairwise_recognition.value,
    )
