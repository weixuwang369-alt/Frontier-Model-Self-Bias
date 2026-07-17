"""Call-inventory planner.

The single source of truth for *what calls a run makes*. Both ``estimate`` (dry-run
cost) and ``run`` (execution) iterate the same generators here, so the estimate can
never silently diverge from what actually executes.

Each generator yields a lightweight ``*Spec`` describing one unit of work. Specs carry
enough to (a) estimate token usage before any call and (b) be turned into a concrete
:class:`~selfbias.schemas.LLMRequest` at execution time. Combinatorics follow
``docs/METRICS.md`` and ``docs/RESEARCH_PLAN.md`` §4.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .config import ExperimentConfig, RosterModel
from .schemas import Disclosure, Paradigm, Task, stable_hash

# --- Token-estimate overheads (approximate; live usage always recorded verbatim). ---
GEN_INSTR_OVERHEAD = 120  # system + length-control instruction around the prompt
JUDGE_INSTR_OVERHEAD = 200  # rubric-free judging scaffolding + response-format ask
RUBRIC_LINE_OVERHEAD = 8  # per-rubric wrapping in the judge prompt
VERDICT_OUT = 40  # structured PWC verdict + confidence
RB_OUT_PER_RUBRIC = 6  # per-rubric satisfied bit in structured output
RB_OUT_OVERHEAD = 20
PROBE_INSTR_OVERHEAD = 120
PROBE_OUT = 24


def seed_int(base_seed: int, *parts) -> int:
    """Deterministic 63-bit seed from the run seed + descriptor parts."""

    return int(stable_hash(base_seed, *parts, length=15), 16)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenSpec:
    task: Task
    model: RosterModel
    length_bin: int
    seed: int

    def est_tokens(self) -> tuple[int, int]:
        from .tokens import approx_tokens

        est_in = approx_tokens(self.task.prompt) + GEN_INSTR_OVERHEAD
        est_out = self.length_bin
        return est_in, est_out


@dataclass(frozen=True)
class PwcSpec:
    task: Task
    length_bin: int
    judge: RosterModel
    model_a: RosterModel
    model_b: RosterModel
    position_index: int  # 0 => (a,b), 1 => (b,a)
    disclosure: Disclosure
    seed: int

    def est_tokens(self) -> tuple[int, int]:
        est_in = 2 * self.length_bin + JUDGE_INSTR_OVERHEAD
        return est_in, VERDICT_OUT


@dataclass(frozen=True)
class RbSpec:
    task: Task
    length_bin: int
    judge: RosterModel
    model: RosterModel
    disclosure: Disclosure
    seed: int

    def est_tokens(self) -> tuple[int, int]:
        rubric_tokens = sum(r.n_tokens + RUBRIC_LINE_OVERHEAD for r in self.task.rubrics)
        n_rubrics = max(1, len(self.task.rubrics))
        est_in = self.length_bin + rubric_tokens + JUDGE_INSTR_OVERHEAD
        est_out = RB_OUT_OVERHEAD + RB_OUT_PER_RUBRIC * n_rubrics
        return est_in, est_out


@dataclass(frozen=True)
class ProbeSpec:
    task: Task
    length_bin: int
    judge: RosterModel
    other: RosterModel
    truncation: bool  # controlled-length vs truncation series
    seed: int

    def est_tokens(self) -> tuple[int, int]:
        est_in = 2 * self.length_bin + PROBE_INSTR_OVERHEAD
        return est_in, PROBE_OUT


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def iter_gen_specs(config: ExperimentConfig, tasks: list[Task]):
    for task in tasks:
        for model in config.roster:
            for length_bin in config.lengths.target_bins_tokens:
                yield GenSpec(
                    task=task,
                    model=model,
                    length_bin=length_bin,
                    seed=seed_int(config.run.seed, "gen", task.task_id, model.model, length_bin),
                )


def iter_pwc_specs(config: ExperimentConfig, tasks: list[Task]):
    if Paradigm.pwc not in config.judging.paradigms:
        return
    roster = config.roster
    for task in tasks:
        for length_bin in config.lengths.target_bins_tokens:
            for model_a, model_b in combinations(roster, 2):  # unordered cross-gen pairs
                for judge in roster:
                    for disclosure in config.judging.disclosure_arms:
                        for position_index in (0, 1):  # order-swap (non-negotiable)
                            yield PwcSpec(
                                task=task,
                                length_bin=length_bin,
                                judge=judge,
                                model_a=model_a,
                                model_b=model_b,
                                position_index=position_index,
                                disclosure=disclosure,
                                seed=seed_int(
                                    config.run.seed,
                                    "pwc",
                                    task.task_id,
                                    length_bin,
                                    judge.model,
                                    model_a.model,
                                    model_b.model,
                                    position_index,
                                    disclosure.value,
                                ),
                            )


def iter_rb_specs(config: ExperimentConfig, tasks: list[Task]):
    if Paradigm.rubric not in config.judging.paradigms:
        return
    roster = config.roster
    for task in tasks:
        for length_bin in config.lengths.target_bins_tokens:
            for model in roster:
                for judge in roster:
                    for disclosure in config.judging.disclosure_arms:
                        yield RbSpec(
                            task=task,
                            length_bin=length_bin,
                            judge=judge,
                            model=model,
                            disclosure=disclosure,
                            seed=seed_int(
                                config.run.seed,
                                "rb",
                                task.task_id,
                                length_bin,
                                judge.model,
                                model.model,
                                disclosure.value,
                            ),
                        )


def iter_probe_specs(config: ExperimentConfig, tasks: list[Task]):
    if not config.probes.pairwise_recognition:
        return
    roster = config.roster
    series = [False]
    if config.probes.on_truncation_series and config.lengths.truncation_series:
        series.append(True)
    for task in tasks:
        for length_bin in config.lengths.target_bins_tokens:
            for judge in roster:
                for other in roster:
                    if other.model == judge.model:
                        continue  # pairwise recognition pairs judge's own vs another's
                    for truncation in series:
                        yield ProbeSpec(
                            task=task,
                            length_bin=length_bin,
                            judge=judge,
                            other=other,
                            truncation=truncation,
                            seed=seed_int(
                                config.run.seed,
                                "probe",
                                task.task_id,
                                length_bin,
                                judge.model,
                                other.model,
                                truncation,
                            ),
                        )
