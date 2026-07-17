"""Pydantic schemas for every persisted artifact and every config object.

All data rows are append-only JSONL on disk with deterministic IDs (see
``docs/ARCHITECTURE.md``). Corrections are new rows carrying ``supersedes``; rows are
never mutated or deleted. Any change to these schemas needs a migration note in
``docs/DECISIONS.md``.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Deterministic ID helper
# ---------------------------------------------------------------------------


def stable_hash(*parts: Any, length: int = 16) -> str:
    """SHA-256 over a canonical JSON encoding of ``parts``; first ``length`` hex chars.

    Used for all deterministic artifact IDs and cache keys so that identical logical
    work maps to identical ids/keys across runs (idempotency + cache-first).
    """

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Enums (kept wide so later phases are config changes, not refactors)
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Adapter *kind* - which client speaks to the endpoint. The set is small on
    purpose; ``openai_compatible`` (with a configurable ``base_url``) covers essentially
    any OpenAI-style API - local Ollama/vLLM, OpenRouter, Together, Groq, and open models
    like Qwen/Llama - without new code."""

    anthropic = "anthropic"
    google = "google"
    openai = "openai"
    openai_compatible = "openai_compatible"
    mock = "mock"


# NOTE: model "family" is a free-form label (e.g. "anthropic", "qwen", "llama"), not an
# enum - the roster is open-ended, and family only needs to group siblings for HSPP-R_fam.


class ReferenceSource(str, Enum):
    programmatic = "programmatic"
    ensemble = "ensemble"
    human = "human"  # FUTURE milestone 1D - schema support only


class Paradigm(str, Enum):
    pwc = "pwc"
    rubric = "rubric"
    da = "da"  # FUTURE - enum present so adding it later is config + template only


class Disclosure(str, Enum):
    anonymous = "anonymous"
    true_label = "true_label"
    false_label = "false_label"


class RubricPolarity(str, Enum):
    positive = "positive"
    negative = "negative"


class ProbeType(str, Enum):
    pairwise_recognition = "pairwise_recognition"
    single_recognition = "single_recognition"


class RunStatus(str, Enum):
    created = "created"
    running = "running"
    halted_budget = "halted_budget"
    paused = "paused"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Provider request / response (provider-agnostic transport objects)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Provider-agnostic request. Every field that changes the model output is part of
    the cache key; see :func:`cache_key`."""

    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None
    # Optional JSON schema the provider should coerce output into (structured output).
    response_schema: dict[str, Any] | None = None
    # Free-form provider-agnostic tag for logging/debug; NOT part of the cache key.
    purpose: str | None = Field(default=None, exclude=True)

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, v: Any) -> Any:
        if isinstance(v, list):
            return tuple(v)
        return v

    def cache_payload(self) -> dict[str, Any]:
        """Canonical dict used for the cache key. Excludes ``purpose`` (debug only)."""

        return {
            "provider": self.provider.value,
            "model": self.model,
            "messages": [m.model_dump() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "response_schema": self.response_schema,
        }


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Parsed provider response plus the full raw body (persisted, never discarded)."""

    provider: Provider
    model: str
    text: str
    usage: Usage = Field(default_factory=Usage)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    # True when the value was served from the response cache (cost $0).
    cache_hit: bool = False
    # Optional structured payload parsed out of ``text`` (e.g. verdict json).
    parsed: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Task / rubric artifacts
# ---------------------------------------------------------------------------


class Constraint(BaseModel):
    """A programmatically verifiable instruction-following constraint (objective arm)."""

    constraint_id: str
    text: str
    # Name of the checker in src/selfbias/tasks that verifies this constraint.
    checker: str
    params: dict[str, Any] = Field(default_factory=dict)


class Rubric(BaseModel):
    rubric_id: str
    text: str
    polarity: RubricPolarity
    weight: float = 1.0
    n_tokens: int = 0  # rubric length in tokens - a Pombal factor variable


class Task(BaseModel):
    task_id: str
    domain: str
    prompt: str
    reference_source: ReferenceSource
    constraints: list[Constraint] = Field(default_factory=list)
    rubrics: list[Rubric] = Field(default_factory=list)
    supersedes: str | None = None

    @staticmethod
    def make_id(domain: str, prompt: str) -> str:
        return stable_hash(domain, prompt)

    @model_validator(mode="after")
    def _check_reference_shape(self) -> Task:
        if self.reference_source == ReferenceSource.programmatic and not self.constraints:
            raise ValueError("programmatic reference tasks require at least one constraint")
        return self


# ---------------------------------------------------------------------------
# Generation artifacts
# ---------------------------------------------------------------------------


class Generation(BaseModel):
    gen_id: str
    task_id: str
    model: str
    family: str
    target_tokens: int
    realized_tokens: int
    text: str
    seed: int
    # For a truncation-series item, the gen_id it was truncated from; else None.
    truncation_of: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    # Reserved for milestone 2A (perplexity/familiarity proxy). Nullable by design.
    familiarity_scores: dict[str, float] | None = None
    supersedes: str | None = None

    @staticmethod
    def make_id(task_id: str, model: str, length_bin: int, seed: int) -> str:
        return stable_hash(task_id, model, length_bin, seed)


# ---------------------------------------------------------------------------
# Judgment artifacts
# ---------------------------------------------------------------------------


class PerRubricVerdict(BaseModel):
    rubric_id: str
    satisfied: bool  # b_J(G, x, k) == +1


class Judgment(BaseModel):
    judg_id: str
    judge_model: str
    judge_family: str
    paradigm: Paradigm
    disclosure: Disclosure
    disclosed_as: str | None = None  # model named to the judge (true_/false_label arms)
    # 1 gen for rubric-based / single, 2 *ordered* gens for PWC.
    subject_gen_ids: list[str]
    # PWC verdict in {+1, 0, -1} = (first wins / tie / second wins) BEFORE resolution.
    # Rubric-based leaves this None and populates per_rubric instead.
    verdict: int | None = None
    per_rubric: list[PerRubricVerdict] = Field(default_factory=list)
    confidence: float | None = None  # c in [0,1]
    position_index: int = 0  # 0 or 1 - which ordering this call represents (PWC)
    seed: int
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    supersedes: str | None = None

    @staticmethod
    def make_id(
        judge_model: str,
        paradigm: str,
        disclosure: str,
        subject_gen_ids: list[str],
        position_index: int,
        seed: int,
    ) -> str:
        return stable_hash(
            judge_model, paradigm, disclosure, list(subject_gen_ids), position_index, seed
        )


# ---------------------------------------------------------------------------
# Probe artifacts (recognition; separate calls from judging)
# ---------------------------------------------------------------------------


class Probe(BaseModel):
    probe_id: str
    judge_model: str
    judge_family: str
    probe_type: ProbeType
    # Which length series the probed texts come from - the RQ2 analysis (METRICS §5.2)
    # reports controlled-length and truncation-series recognition separately.
    series: Literal["controlled", "truncation"]
    subject_gen_ids: list[str]
    # pairwise: index (0/1) the judge picked as its own; single: "yes"/"no".
    answer: str
    confidence: float | None = None
    correct: bool | None = None
    seed: int
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    supersedes: str | None = None

    @staticmethod
    def make_id(judge_model: str, probe_type: str, subject_gen_ids: list[str], seed: int) -> str:
        return stable_hash(judge_model, probe_type, list(subject_gen_ids), seed)


# ---------------------------------------------------------------------------
# Run manifest (resumability + cost accounting + frozen config snapshot)
# ---------------------------------------------------------------------------


class StageCost(BaseModel):
    calls_planned: int = 0
    calls_done: int = 0
    cache_hits: int = 0
    # Structured judge/probe calls whose output couldn't be parsed/validated. The call
    # still cost budget (it was made); the row is skipped rather than coerced.
    parse_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RunManifest(BaseModel):
    run_id: str
    run_name: str
    config_snapshot: dict[str, Any]  # frozen copy of the ExperimentConfig
    seed: int
    budget_usd: float
    status: RunStatus = RunStatus.created
    # Per-stage planned/actuals (keys: curate/generate/judge/probe/...).
    stages: dict[str, StageCost] = Field(default_factory=dict)
    # ids of completed calls, per stage, for skip-if-done resumability.
    completed_call_ids: dict[str, list[str]] = Field(default_factory=dict)
    cost_by_provider: dict[str, float] = Field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(self.cost_by_provider.values()), 6)

    def is_done(self, stage: str, call_id: str) -> bool:
        return call_id in self.completed_call_ids.get(stage, [])
