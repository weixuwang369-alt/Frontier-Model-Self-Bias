"""Curate stage - build/validate task sets per domain (no API calls).

Phase 0 synthesizes deterministic placeholder tasks so the pipeline is exercisable
end-to-end on the mock provider. The *shape* is production-real: objective tasks carry
programmatically checkable :class:`Constraint`s; subjective tasks carry authored
:class:`Rubric`s with polarity and token length (Pombal factor variables). Real task
curation (IFEval-style constraints, LLM-drafted+human-reviewed rubrics) lands in
Phase 1+ and only changes this module's contents, not its interface.
"""

from __future__ import annotations

import warnings

from .config import DomainConfig, ExperimentConfig
from .schemas import (
    Constraint,
    ReferenceSource,
    Rubric,
    RubricPolarity,
    Task,
    stable_hash,
)
from .tokens import approx_tokens

# Distinct seed prompts per subjective domain (reference = ensemble/majority vote).
# n_prompts beyond the pool size falls back to "(variant N)" clones - see the fake-power
# warning in curate_domain; a real run should cap n_prompts at the pool size.
_SEED_PROMPTS: dict[str, list[str]] = {
    "open_qa_summarization": [
        "Summarize the main causes of the 1929 financial crash for a general reader.",
        "Explain the trade-offs between solar and wind power for home energy.",
        "Summarize how vaccines train the immune system, briefly.",
        "Explain what causes ocean tides, in plain language.",
        "Summarize the plot and central theme of a classic hero's-journey story.",
        "Explain the difference between weather and climate for a curious teenager.",
        "Summarize why sleep matters for memory and learning.",
        "Explain how a bill becomes law, at a high level.",
        "Summarize the core idea of supply and demand in economics.",
        "Explain what an API is to someone non-technical.",
    ],
    "creative_writing": [
        "Write a short scene: a lighthouse keeper receives an unexpected letter.",
        "Write a vignette about the last bookstore in a rain-soaked city.",
        "Write a brief story where a clock runs backwards for one person only.",
        "Write a small myth explaining why the sea is salt.",
        "Write a short scene: two strangers share an umbrella at a bus stop.",
        "Write a vignette about a gardener who grows a plant no one can name.",
        "Write a brief story about a map that redraws itself overnight.",
        "Write a short fable in which the moon borrows light and forgets to return it.",
        "Write a scene: an old musician plays to an empty subway platform.",
        "Write a brief story about a town where it has not rained in a decade.",
    ],
}

# Objective domain: each seed pairs a prompt with a checker that MATCHES its stated
# instruction, so the programmatic ground truth is sound (the constraint text is what the
# judge scores as a rubric; the checker is the free reference verdict).
_VERIFIABLE_SEEDS: list[tuple[str, str, str, dict]] = [
    (
        "Write a short product blurb for a reusable water bottle in exactly three sentences.",
        "The response is exactly three sentences.",
        "exact_sentence_count",
        {"count": 3},
    ),
    (
        "Explain how to reset a home Wi-Fi router, and include the word 'firmware'.",
        "The response includes the word 'firmware'.",
        "must_include",
        {"substring": "firmware"},
    ),
    (
        "Summarize how photosynthesis works, in at most 40 words.",
        "The response is at most 40 words.",
        "max_words",
        {"max_words": 40},
    ),
    (
        "Give directions from a train station to the town museum in exactly two sentences.",
        "The response is exactly two sentences.",
        "exact_sentence_count",
        {"count": 2},
    ),
    (
        "Describe what powers a living cell, and include the word 'mitochondria'.",
        "The response includes the word 'mitochondria'.",
        "must_include",
        {"substring": "mitochondria"},
    ),
    (
        "Define machine learning for a beginner in at most 25 words.",
        "The response is at most 25 words.",
        "max_words",
        {"max_words": 25},
    ),
    (
        "Describe the water cycle in exactly four sentences.",
        "The response is exactly four sentences.",
        "exact_sentence_count",
        {"count": 4},
    ),
    (
        "Explain why most leaves are green, and include the word 'chlorophyll'.",
        "The response includes the word 'chlorophyll'.",
        "must_include",
        {"substring": "chlorophyll"},
    ),
    (
        "Summarize how a search engine ranks pages, in at most 60 words.",
        "The response is at most 60 words.",
        "max_words",
        {"max_words": 60},
    ),
    (
        "Explain how a recipe is like a computer program, and include the word 'algorithm'.",
        "The response includes the word 'algorithm'.",
        "must_include",
        {"substring": "algorithm"},
    ),
]

_RUBRIC_TEMPLATES: list[tuple[str, RubricPolarity]] = [
    ("The response stays on topic and answers the prompt.", RubricPolarity.positive),
    ("The response is well-structured and coherent.", RubricPolarity.positive),
    ("The response is free of factual errors.", RubricPolarity.positive),
    ("The response avoids needless repetition.", RubricPolarity.negative),
    ("The response avoids vague filler and cliché.", RubricPolarity.negative),
    ("The response does not contradict itself.", RubricPolarity.negative),
]


def _constraint_for(prompt: str, idx: int) -> Constraint:
    """Attach one deterministic, checkable constraint to an objective task."""

    kind = idx % 3
    if kind == 0:
        return Constraint(
            constraint_id=stable_hash(prompt, "n_sentences"),
            text="Respond in exactly three sentences.",
            checker="exact_sentence_count",
            params={"count": 3},
        )
    if kind == 1:
        return Constraint(
            constraint_id=stable_hash(prompt, "must_include"),
            text="Include the word 'firmware'.",
            checker="must_include",
            params={"substring": "firmware"},
        )
    return Constraint(
        constraint_id=stable_hash(prompt, "max_words"),
        text="Use at most 40 words.",
        checker="max_words",
        params={"max_words": 40},
    )


def _rubrics_for(prompt: str, domain: DomainConfig) -> list[Rubric]:
    spec = domain.rubrics
    lo = spec.per_task_min if spec else 4
    hi = spec.per_task_max if spec else 6
    n = max(lo, min(hi, len(_RUBRIC_TEMPLATES)))
    rubrics: list[Rubric] = []
    for k in range(n):
        text, polarity = _RUBRIC_TEMPLATES[k % len(_RUBRIC_TEMPLATES)]
        rubrics.append(
            Rubric(
                rubric_id=stable_hash(prompt, "rubric", k),
                text=text,
                polarity=polarity,
                weight=1.0,
                n_tokens=approx_tokens(text),
            )
        )
    return rubrics


def make_task(
    domain: DomainConfig, prompt: str, idx: int, constraint: Constraint | None = None
) -> Task:
    """Build one Task from a prompt, attaching the right ground-truth scaffolding.

    Shared by the built-in curator, the Excel importer's default path, and the
    LLM-generated prompt mode so every prompt source produces identically-shaped tasks.
    ``constraint`` supplies a matched objective checker; without it a programmatic task
    falls back to the generic index-based constraint.
    """

    task_id = Task.make_id(domain.name, prompt)
    if domain.reference == ReferenceSource.programmatic:
        return Task(
            task_id=task_id,
            domain=domain.name,
            prompt=prompt,
            reference_source=ReferenceSource.programmatic,
            constraints=[constraint or _constraint_for(prompt, idx)],
        )
    return Task(
        task_id=task_id,
        domain=domain.name,
        prompt=prompt,
        reference_source=domain.reference,
        rubrics=_rubrics_for(prompt, domain),
    )


def _distinct_pool_size(domain: DomainConfig) -> int:
    if domain.reference == ReferenceSource.programmatic:
        return len(_VERIFIABLE_SEEDS)
    return len(_SEED_PROMPTS.get(domain.name, [])) or 4


def curate_domain(domain: DomainConfig) -> list[Task]:
    pool = _distinct_pool_size(domain)
    if domain.n_prompts > pool:
        warnings.warn(
            f"{domain.name}: n_prompts={domain.n_prompts} exceeds {pool} distinct built-in "
            "prompts; the extras are '(variant N)' clones that inflate prompt-level "
            "bootstrap confidence. Cap n_prompts at the pool size, or supply an Excel "
            "library for a properly powered run.",
            stacklevel=2,
        )

    tasks: list[Task] = []
    # Objective arm: prompts paired with a matching checker.
    if domain.reference == ReferenceSource.programmatic:
        for i in range(domain.n_prompts):
            base, ctext, checker, params = _VERIFIABLE_SEEDS[i % len(_VERIFIABLE_SEEDS)]
            prompt = base if i < len(_VERIFIABLE_SEEDS) else f"{base} (variant {i // pool + 1})"
            constraint = Constraint(
                constraint_id=stable_hash(prompt, checker, str(params)),
                text=ctext,
                checker=checker,
                params=params,
            )
            tasks.append(make_task(domain, prompt, i, constraint=constraint))
        return tasks

    # Subjective arm: distinct prompt strings.
    seeds = _SEED_PROMPTS.get(domain.name) or [
        f"{domain.name}: complete task variant {i}." for i in range(4)
    ]
    for i in range(domain.n_prompts):
        base = seeds[i % len(seeds)]
        prompt = base if i < len(seeds) else f"{base} (variant {i // len(seeds) + 1})"
        tasks.append(make_task(domain, prompt, i))
    return tasks


def curate(config: ExperimentConfig) -> list[Task]:
    """Built-in task set, across config domains. Deterministic and idempotent."""

    tasks: list[Task] = []
    for domain in config.domains:
        tasks.extend(curate_domain(domain))
    return tasks


def generated_count(config: ExperimentConfig, domain: DomainConfig) -> int:
    """How many prompts the llm_generated mode drafts for a domain."""

    return config.prompts.n_per_domain or domain.n_prompts


def placeholder_task(domain: DomainConfig, idx: int) -> Task:
    """A deterministic stand-in for one to-be-generated prompt (count-accurate).

    Used by ``estimate`` and manifest planning for the llm_generated source, where the
    real prompt text isn't known until the run drafts it. Downstream call counts depend
    only on the task count and rubric/constraint shape, so this makes the estimate exact.
    """

    return make_task(domain, f"[{domain.name}] generated task prompt #{idx + 1}", idx)


def placeholder_llm_tasks(config: ExperimentConfig) -> list[Task]:
    tasks: list[Task] = []
    for domain in config.domains:
        for i in range(generated_count(config, domain)):
            tasks.append(placeholder_task(domain, i))
    return tasks


def build_tasks(config: ExperimentConfig) -> list[Task]:
    """Resolve tasks from the configured prompt source (builtin / excel / llm_generated).

    Entry point for CLI/dashboard/estimate. ``curate`` is the builtin case. For
    ``llm_generated`` this returns count-accurate PLACEHOLDERS - the pipeline replaces
    them with real drafted prompts at run time (see ``Pipeline._llm_generated_tasks``).
    """

    source = config.prompts.source
    if source == "builtin":
        return curate(config)
    if source == "excel":
        from .excel import import_tasks  # local import avoids a config<->excel cycle

        return import_tasks(config.prompts.excel_path, config)
    if source == "llm_generated":
        return placeholder_llm_tasks(config)
    raise ValueError(f"unknown prompts.source: {source}")
