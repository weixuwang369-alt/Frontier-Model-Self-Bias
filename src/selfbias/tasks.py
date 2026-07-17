"""Curate stage - build/validate task sets per domain (no API calls).

Phase 0 synthesizes deterministic placeholder tasks so the pipeline is exercisable
end-to-end on the mock provider. The *shape* is production-real: objective tasks carry
programmatically checkable :class:`Constraint`s; subjective tasks carry authored
:class:`Rubric`s with polarity and token length (Pombal factor variables). Real task
curation (IFEval-style constraints, LLM-drafted+human-reviewed rubrics) lands in
Phase 1+ and only changes this module's contents, not its interface.
"""

from __future__ import annotations

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

# A small, deterministic seed pool per domain. Cycled + indexed to reach n_prompts.
_SEED_PROMPTS: dict[str, list[str]] = {
    "verifiable_if": [
        "Write a short product blurb. Constraint: exactly three sentences.",
        "List steps to reset a router. Constraint: include the word 'firmware'.",
        "Describe a sunset. Constraint: do not use the letter 'e'.",
        "Summarize photosynthesis. Constraint: at most 40 words.",
    ],
    "open_qa_summarization": [
        "Summarize the causes of the 1929 financial crash for a general reader.",
        "Answer: what are the trade-offs between solar and wind power?",
        "Summarize the plot and themes of a hero's-journey story.",
        "Explain how vaccines train the immune system, briefly.",
    ],
    "creative_writing": [
        "Write a short scene: a lighthouse keeper receives an unexpected letter.",
        "Write a vignette about the last bookstore in a rain-soaked city.",
        "Write a brief story where a clock runs backwards for one person only.",
        "Write a small myth explaining why the sea is salt.",
    ],
}

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


def make_task(domain: DomainConfig, prompt: str, idx: int) -> Task:
    """Build one Task from a prompt, attaching the right ground-truth scaffolding.

    Shared by the built-in curator, the Excel importer's default path, and the
    LLM-generated prompt mode so every prompt source produces identically-shaped tasks.
    """

    task_id = Task.make_id(domain.name, prompt)
    if domain.reference == ReferenceSource.programmatic:
        return Task(
            task_id=task_id,
            domain=domain.name,
            prompt=prompt,
            reference_source=ReferenceSource.programmatic,
            constraints=[_constraint_for(prompt, idx)],
        )
    return Task(
        task_id=task_id,
        domain=domain.name,
        prompt=prompt,
        reference_source=domain.reference,
        rubrics=_rubrics_for(prompt, domain),
    )


def curate_domain(domain: DomainConfig) -> list[Task]:
    seeds = _SEED_PROMPTS.get(domain.name)
    if not seeds:
        # Unknown domain name still works: generate generic prompts deterministically.
        seeds = [f"{domain.name}: complete task variant {i}." for i in range(4)]

    tasks: list[Task] = []
    for i in range(domain.n_prompts):
        base = seeds[i % len(seeds)]
        # Make each prompt unique+deterministic across the requested count.
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
