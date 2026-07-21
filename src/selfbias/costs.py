"""Dry-run cost estimator + live cost accounting.

Estimate = call inventory (from :mod:`selfbias.plan`) × per-call token estimates ×
per-model pricing (from ``config/pricing.yaml``). Every run must show this before any
API call (a hard cost guardrail). Live accounting converts real ``usage`` into USD
with the same pricing so estimate-vs-actual is an apples-to-apples comparison.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import ExperimentConfig, Pricing
from .plan import (
    iter_gen_specs,
    iter_probe_specs,
    iter_pwc_specs,
    iter_rb_specs,
)
from .schemas import Task, Usage


def cost_usd(usage: Usage, model: str, pricing: Pricing) -> float:
    price, _ = pricing.price_for(model)
    return (
        usage.input_tokens / 1_000_000 * price.input_per_mtok
        + usage.output_tokens / 1_000_000 * price.output_per_mtok
    )


@dataclass
class StageEstimate:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostEstimate:
    by_stage: dict[str, StageEstimate] = field(default_factory=dict)
    by_provider: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, StageEstimate] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)

    @property
    def total_calls(self) -> int:
        return sum(s.calls for s in self.by_stage.values())

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.by_stage.values()), 4)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.by_stage.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.by_stage.values())


def _accumulate(
    est: CostEstimate,
    stage: str,
    model: str,
    provider: str,
    est_in: int,
    est_out: int,
    pricing: Pricing,
) -> None:
    price, found = pricing.price_for(model)
    if not found and not model.startswith("mock"):
        est.unpriced_models.add(model)
    c = est_in / 1_000_000 * price.input_per_mtok + est_out / 1_000_000 * price.output_per_mtok

    s = est.by_stage.setdefault(stage, StageEstimate())
    s.calls += 1
    s.input_tokens += est_in
    s.output_tokens += est_out
    s.cost_usd += c

    m = est.by_model.setdefault(model, StageEstimate())
    m.calls += 1
    m.input_tokens += est_in
    m.output_tokens += est_out
    m.cost_usd += c

    est.by_provider[provider] = est.by_provider.get(provider, 0.0) + c


def estimate_cost(config: ExperimentConfig, tasks: list[Task], pricing: Pricing) -> CostEstimate:
    """Dry-run estimate for the full run. No API calls, no disk writes."""

    est = CostEstimate()

    # promptgen - one drafting call per domain (llm_generated source only).
    if config.prompts.source == "llm_generated":
        gen = config.prompt_generator()
        for domain in config.domains:
            n = config.prompts.n_per_domain or domain.n_prompts
            _accumulate(est, "promptgen", gen.model, gen.provider.value, 80, n * 48, pricing)

    # generate - billed to the generating model.
    for spec in iter_gen_specs(config, tasks):
        ei, eo = spec.est_tokens()
        _accumulate(est, "generate", spec.model.model, spec.model.provider.value, ei, eo, pricing)

    # judge PWC / RB - billed to the judge model.
    for spec in iter_pwc_specs(config, tasks):
        ei, eo = spec.est_tokens()
        _accumulate(est, "judge_pwc", spec.judge.model, spec.judge.provider.value, ei, eo, pricing)
    for spec in iter_rb_specs(config, tasks):
        ei, eo = spec.est_tokens()
        _accumulate(
            est, "judge_rubric", spec.judge.model, spec.judge.provider.value, ei, eo, pricing
        )

    # probe - billed to the probing judge model.
    for spec in iter_probe_specs(config, tasks):
        ei, eo = spec.est_tokens()
        _accumulate(est, "probe", spec.judge.model, spec.judge.provider.value, ei, eo, pricing)

    # Round provider tallies for display.
    est.by_provider = {k: round(v, 4) for k, v in est.by_provider.items()}
    return est


def counts_by_stage(config: ExperimentConfig, tasks: list[Task]) -> dict[str, int]:
    """Call counts only (cheap; used for planned inventory in the manifest)."""

    counts: dict[str, int] = defaultdict(int)
    if config.prompts.source == "llm_generated":
        counts["promptgen"] = len(config.domains)
    for _ in iter_gen_specs(config, tasks):
        counts["generate"] += 1
    for _ in iter_pwc_specs(config, tasks):
        counts["judge_pwc"] += 1
    for _ in iter_rb_specs(config, tasks):
        counts["judge_rubric"] += 1
    for _ in iter_probe_specs(config, tasks):
        counts["probe"] += 1
    return dict(counts)
