"""Pipeline runner: curate -> generate -> judge -> probe.

Every stage is idempotent, resumable, and cache-first. Before each *paid* call the
runner checks the budget cap and halts (resumably) if the next call would cross it.
Cache hits cost $0 and are tallied. On the mock provider the whole thing runs offline at
$0 - this is the Phase 0 end-to-end path.

Metrics/analysis are intentionally NOT run here; they recompute from disk (see
``docs/ARCHITECTURE.md`` stage 7) and the dashboard demo uses synthetic data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .cache import ResponseCache, cache_key
from .config import ExperimentConfig, Pricing, Settings, load_pricing
from .costs import cost_usd
from .manifest import ManifestStore, load_or_init, record_call, would_exceed_budget
from .plan import (
    ProbeSpec,
    PwcSpec,
    RbSpec,
    iter_gen_specs,
    iter_probe_specs,
    iter_pwc_specs,
    iter_rb_specs,
    seed_int,
)
from .prompts import (
    gen_request,
    probe_pairwise_request,
    pwc_request,
    rb_request,
)
from .providers import build_providers
from .schemas import (
    Generation,
    Judgment,
    LLMResponse,
    PerRubricVerdict,
    Probe,
    ProbeType,
    RunManifest,
    RunStatus,
    StageCost,
    Task,
    Usage,
    stable_hash,
)
from .storage import DataPaths, JsonlStore, default_data_paths
from .structured import parse_structured
from .tasks import build_tasks
from .tokens import approx_chars, approx_tokens

ProgressFn = Callable[[str, int, int], None]


class BudgetHalt(Exception):
    """Raised internally when the next paid call would exceed the budget cap."""


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    manifest: RunManifest
    n_tasks: int = 0
    n_generations: int = 0
    n_judgments: int = 0
    n_probes: int = 0
    halted: bool = False
    messages: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        data_root: str | Path = "data",
        pricing_path: str | Path = "config/pricing.yaml",
        settings: Settings | None = None,
        pricing: Pricing | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.config = config
        self.paths: DataPaths = default_data_paths(data_root).ensure()
        self.settings = settings or Settings()
        self.pricing = pricing or load_pricing(pricing_path)
        self.progress = progress or (lambda stage, done, total: None)

        self.cache = ResponseCache(self.paths.cache)
        # Providers are keyed per model string (each roster entry may carry its own
        # endpoint/key), so calls look up by request.model.
        self.providers = build_providers(config.roster, self.settings, self.cache)

        self.manifest_store = ManifestStore(self.paths.manifests)
        self.tasks_store = JsonlStore(self.paths.tasks / "tasks.jsonl")
        self.gen_store = JsonlStore(self.paths.generations / "generations.jsonl")
        self.judg_store = JsonlStore(self.paths.judgments / "judgments.jsonl")
        self.probe_store = JsonlStore(self.paths.probes / "probes.jsonl")

        # Populated during the run; also reloaded from disk on resume.
        self._gens: dict[tuple[str, str, int], Generation] = {}  # (task, model, bin)
        self._trunc: dict[tuple[str, str, int], Generation] = {}
        # This run's expected generation ids (include seed). The generations store is
        # append-only and shared across runs; when reloading we index ONLY rows that
        # belong to THIS run so a different-seed run can't shadow generation and cause
        # silent text reuse. Populated at the start of _generate().
        self._run_gen_ids: set[str] = set()
        # Existing row ids on disk - makes judge/probe idempotent even if the manifest
        # is lost (append-only integrity: never write a duplicate row).
        self._existing_judg_ids: set[str] = set()
        self._existing_probe_ids: set[str] = set()

    # -- call execution (cache-first + budget guard) ------------------------

    def _execute(
        self,
        request,
        stage: str,
        call_id: str,
        est_in: int,
        manifest: RunManifest,
    ) -> LLMResponse:
        provider = self.providers[request.model]
        key = cache_key(request)
        is_hit = self.cache.has(key)
        if not is_hit:
            # Budget guard projects output at max_tokens (the hard ceiling the provider
            # can return), not the softer display estimate, so a single call can never
            # push recorded actuals over the cap. est_out is used only for the estimate.
            projected = cost_usd(
                Usage(input_tokens=est_in, output_tokens=request.max_tokens),
                request.model,
                self.pricing,
            )
            if would_exceed_budget(manifest, projected):
                raise BudgetHalt(stage)
        response = provider.generate(request)
        actual = (
            0.0 if response.cache_hit else cost_usd(response.usage, request.model, self.pricing)
        )
        record_call(
            manifest,
            stage,
            call_id,
            request.provider.value,
            actual,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.cache_hit,
        )
        return response

    # -- stages -------------------------------------------------------------

    def _curate(self, tasks: list[Task]) -> None:
        existing = self.tasks_store.ids("task_id")
        new = [t for t in tasks if t.task_id not in existing]
        self.tasks_store.append_many(new)

    def _load_existing_gens(self) -> None:
        # Only load rows belonging to THIS run (see self._run_gen_ids). Controlled rows
        # are matched by gen_id; truncation rows by the gen_id they derive from.
        for g in self.gen_store.read_all(Generation):
            if g.truncation_of is None:
                if g.gen_id in self._run_gen_ids:
                    self._gens[(g.task_id, g.model, g.target_tokens)] = g
            elif g.truncation_of in self._run_gen_ids:
                self._trunc[(g.task_id, g.model, g.target_tokens)] = g

    def _within_band(self, realized: int, target: int) -> bool:
        tol = self.config.lengths.tolerance_pct / 100.0
        return abs(realized - target) <= target * tol

    def _generate_one(self, spec, gen_id, manifest):
        """Generate with a bounded length-compliance retry loop.

        Retries (with a corrective nudge + varied seed) while realized length is outside
        the ±tolerance band, up to ``lengths.max_retries``. Each attempt is a real,
        separately-cached call. The final attempt's text is kept regardless. The mock
        provider sizes output into the band, so it typically converges on the first try.
        """

        max_retries = self.config.lengths.max_retries
        resp = None
        realized = 0
        prev = None
        for attempt in range(max_retries + 1):
            req = gen_request(spec, self.config, attempt=attempt, prev_realized=prev)
            ei = spec.est_tokens()[0]
            resp = self._execute(req, "generate", gen_id, ei, manifest)
            realized = resp.usage.output_tokens or approx_tokens(resp.text)
            if self._within_band(realized, spec.length_bin):
                break
            prev = realized
        return resp, realized

    def _parse_structured_or_flag(self, resp, request, stage, manifest) -> dict | None:
        """Parse a structured response; on failure flag it and return None (fail-loud).

        The mock always returns valid JSON, so this never flags in Phase 0. Against real
        providers a malformed/incomplete response is counted and the row is skipped -
        never coerced to a default verdict.
        """

        parsed = parse_structured(resp.text, request.response_schema)
        if parsed is None:
            manifest.stages.setdefault(stage, StageCost()).parse_failures += 1
        return parsed

    def _generate(self, manifest: RunManifest) -> int:
        specs = list(iter_gen_specs(self.config, self.tasks))
        self._run_gen_ids = {
            Generation.make_id(s.task.task_id, s.model.model, s.length_bin, s.seed) for s in specs
        }
        self._load_existing_gens()
        total = len(specs)
        for i, spec in enumerate(specs):
            gen_id = Generation.make_id(
                spec.task.task_id, spec.model.model, spec.length_bin, spec.seed
            )
            key3 = (spec.task.task_id, spec.model.model, spec.length_bin)
            if key3 in self._gens or manifest.is_done("generate", gen_id):
                self.progress("generate", i + 1, total)
                continue
            resp, realized = self._generate_one(spec, gen_id, manifest)
            gen = Generation(
                gen_id=gen_id,
                task_id=spec.task.task_id,
                model=spec.model.model,
                family=spec.model.family,
                target_tokens=spec.length_bin,
                realized_tokens=realized,
                text=resp.text,
                seed=spec.seed,
                raw_response=resp.raw_response if self.config.generation.persist_raw else {},
                usage=resp.usage,
            )
            self.gen_store.append(gen)
            self._gens[key3] = gen
            self.progress("generate", i + 1, total)
        return total

    def _derive_truncation_series(self) -> None:
        """Truncate the longest generation per (task, model) to each smaller bin.

        Truncated texts feed recognition/attribution only - never quality judging
        (Decision 6C). No API calls, no cost.
        """

        if not self.config.lengths.truncation_series:
            return
        bins = self.config.lengths.target_bins_tokens
        existing = set(self._trunc.keys())
        by_pair: dict[tuple[str, str], Generation] = {}
        for (task_id, model, _bin), gen in self._gens.items():
            cur = by_pair.get((task_id, model))
            if cur is None or gen.realized_tokens > cur.realized_tokens:
                by_pair[(task_id, model)] = gen
        new_rows: list[Generation] = []
        for (task_id, model), longest in by_pair.items():
            for b in bins:
                if approx_chars(b) >= len(longest.text):
                    continue  # bin not shorter than the source; nothing to truncate
                key3 = (task_id, model, b)
                if key3 in existing:
                    continue
                text = longest.text[: approx_chars(b)]
                trunc_id = stable_hash(longest.gen_id, "trunc", b)
                row = Generation(
                    gen_id=trunc_id,
                    task_id=task_id,
                    model=model,
                    family=longest.family,
                    target_tokens=b,
                    realized_tokens=approx_tokens(text),
                    text=text,
                    seed=longest.seed,
                    truncation_of=longest.gen_id,
                )
                new_rows.append(row)
                self._trunc[key3] = row
        self.gen_store.append_many(new_rows)

    def _disclosed_as(self, spec, true_model: str) -> str | None:
        from .schemas import Disclosure

        if spec.disclosure == Disclosure.anonymous:
            return None
        if spec.disclosure == Disclosure.true_label:
            return true_model
        # false_label: name a different roster model, chosen deterministically & balanced.
        others = [m.model for m in self.config.roster if m.model != true_model]
        idx = int(stable_hash(spec.seed, "false_label", length=8), 16) % len(others)
        return others[idx]

    def _judge(self, manifest: RunManifest) -> int:
        done = 0
        # PWC
        pwc = list(iter_pwc_specs(self.config, self.tasks))
        rb = list(iter_rb_specs(self.config, self.tasks))
        total = len(pwc) + len(rb)
        for i, spec in enumerate(pwc):
            done += self._judge_pwc(spec, manifest)
            self.progress("judge_pwc", i + 1, len(pwc))
        for i, spec in enumerate(rb):
            done += self._judge_rb(spec, manifest)
            self.progress("judge_rubric", i + 1, len(rb))
        return total

    def _judge_pwc(self, spec: PwcSpec, manifest: RunManifest) -> int:
        gen_a = self._gens.get((spec.task.task_id, spec.model_a.model, spec.length_bin))
        gen_b = self._gens.get((spec.task.task_id, spec.model_b.model, spec.length_bin))
        if gen_a is None or gen_b is None:
            return 0
        # position_index selects the presentation order (order-swap).
        if spec.position_index == 0:
            first, second = gen_a, gen_b
        else:
            first, second = gen_b, gen_a
        judg_id = Judgment.make_id(
            spec.judge.model,
            "pwc",
            spec.disclosure.value,
            [first.gen_id, second.gen_id],
            spec.position_index,
            spec.seed,
        )
        if judg_id in self._existing_judg_ids or manifest.is_done("judge_pwc", judg_id):
            return 0
        # In label arms, the disclosed identity refers to the FIRST-presented response.
        disclosed_as = self._disclosed_as(spec, first.model)
        req = pwc_request(spec, first.text, second.text, self.config, disclosed_as)
        ei = spec.est_tokens()[0]
        resp = self._execute(req, "judge_pwc", judg_id, ei, manifest)
        parsed = self._parse_structured_or_flag(resp, req, "judge_pwc", manifest)
        if parsed is None:
            return 0  # malformed output flagged; skip rather than record a fake tie
        judgment = Judgment(
            judg_id=judg_id,
            judge_model=spec.judge.model,
            judge_family=spec.judge.family,
            paradigm="pwc",
            disclosure=spec.disclosure,
            disclosed_as=disclosed_as,
            subject_gen_ids=[first.gen_id, second.gen_id],
            verdict=int(parsed.get("verdict", 0)),
            confidence=_clip01(parsed.get("confidence")),
            position_index=spec.position_index,
            seed=spec.seed,
            raw_response=resp.raw_response,
            usage=resp.usage,
        )
        self.judg_store.append(judgment)
        self._existing_judg_ids.add(judg_id)
        return 1

    def _judge_rb(self, spec: RbSpec, manifest: RunManifest) -> int:
        gen = self._gens.get((spec.task.task_id, spec.model.model, spec.length_bin))
        if gen is None:
            return 0
        judg_id = Judgment.make_id(
            spec.judge.model, "rubric", spec.disclosure.value, [gen.gen_id], 0, spec.seed
        )
        if judg_id in self._existing_judg_ids or manifest.is_done("judge_rubric", judg_id):
            return 0
        disclosed_as = self._disclosed_as(spec, gen.model)
        req = rb_request(spec, gen.text, self.config, disclosed_as)
        ei = spec.est_tokens()[0]
        resp = self._execute(req, "judge_rubric", judg_id, ei, manifest)
        parsed = self._parse_structured_or_flag(resp, req, "judge_rubric", manifest)
        if parsed is None:
            return 0
        satisfied = parsed.get("satisfied", [])
        criteria = spec.task.rubrics or spec.task.constraints
        per_rubric = []
        for idx, crit in enumerate(criteria):
            crit_id = getattr(crit, "rubric_id", None) or crit.constraint_id
            val = bool(satisfied[idx]) if idx < len(satisfied) else False
            per_rubric.append(PerRubricVerdict(rubric_id=crit_id, satisfied=val))
        judgment = Judgment(
            judg_id=judg_id,
            judge_model=spec.judge.model,
            judge_family=spec.judge.family,
            paradigm="rubric",
            disclosure=spec.disclosure,
            disclosed_as=disclosed_as,
            subject_gen_ids=[gen.gen_id],
            per_rubric=per_rubric,
            confidence=_clip01(parsed.get("confidence")),
            position_index=0,
            seed=spec.seed,
            raw_response=resp.raw_response,
            usage=resp.usage,
        )
        self.judg_store.append(judgment)
        self._existing_judg_ids.add(judg_id)
        return 1

    def _probe(self, manifest: RunManifest) -> int:
        specs = list(iter_probe_specs(self.config, self.tasks))
        total = len(specs)
        for i, spec in enumerate(specs):
            self._probe_pairwise(spec, manifest)
            self.progress("probe", i + 1, total)
        return total

    def _probe_pairwise(self, spec: ProbeSpec, manifest: RunManifest) -> int:
        # Draw both texts from the SAME series. For the truncation series a bin at or
        # above the source length has no truncated text - there is nothing to truncate,
        # so we skip rather than fall back to controlled text (which would write an
        # ambiguous, duplicate row). Controlled probes always have their gens.
        pool = self._trunc if spec.truncation else self._gens
        own_gen = pool.get((spec.task.task_id, spec.judge.model, spec.length_bin))
        other_gen = pool.get((spec.task.task_id, spec.other.model, spec.length_bin))
        if own_gen is None or other_gen is None:
            return 0
        series = "truncation" if spec.truncation else "controlled"
        # Deterministic presentation order; record which slot holds the judge's own text.
        own_pos = int(stable_hash(spec.seed, "own_pos", length=8), 16) % 2
        text_pos0, text_pos1 = (
            (own_gen.text, other_gen.text) if own_pos == 0 else (other_gen.text, own_gen.text)
        )
        probe_id = Probe.make_id(
            spec.judge.model,
            ProbeType.pairwise_recognition.value,
            [own_gen.gen_id, other_gen.gen_id, series],
            spec.seed,
        )
        if probe_id in self._existing_probe_ids or manifest.is_done("probe", probe_id):
            return 0
        req = probe_pairwise_request(spec, text_pos0, text_pos1, self.config)
        ei = spec.est_tokens()[0]
        resp = self._execute(req, "probe", probe_id, ei, manifest)
        parsed = self._parse_structured_or_flag(resp, req, "probe", manifest)
        if parsed is None:
            return 0
        choice = int(parsed.get("choice", 0))
        probe = Probe(
            probe_id=probe_id,
            judge_model=spec.judge.model,
            judge_family=spec.judge.family,
            probe_type=ProbeType.pairwise_recognition,
            series=series,
            subject_gen_ids=[own_gen.gen_id, other_gen.gen_id],
            answer=str(choice),
            confidence=_clip01(parsed.get("confidence")),
            correct=(choice == own_pos),
            seed=spec.seed,
            raw_response=resp.raw_response,
            usage=resp.usage,
        )
        self.probe_store.append(probe)
        self._existing_probe_ids.add(probe_id)
        return 1

    # -- llm-generated prompts ----------------------------------------------

    def _llm_generated_tasks(self, manifest: RunManifest) -> list[Task]:
        """Draft a fresh prompt set via the generator model, one call per domain.

        Model output is deduped and then backfilled with the same deterministic
        placeholders `estimate` used, so the task count always equals the planned count
        (and equals the estimate exactly if the model returns nothing usable).
        """

        from .promptgen import promptgen_request
        from .tasks import generated_count, make_task, placeholder_task

        model = self.config.prompt_generator()
        tasks: list[Task] = []
        for domain in self.config.domains:
            n = generated_count(self.config, domain)
            seed = seed_int(self.config.run.seed, "promptgen", domain.name)
            req = promptgen_request(domain, model, n, seed)
            call_id = stable_hash("promptgen", domain.name, model.model, n, seed)
            est_in = sum(approx_tokens(m.content) for m in req.messages)
            resp = self._execute(req, "promptgen", call_id, est_in, manifest)

            parsed = parse_structured(resp.text, req.response_schema)
            prompts: list[str] = []
            seen: set[str] = set()
            for p in (parsed or {}).get("prompts", []):
                p = str(p).strip()
                if p and p not in seen:
                    seen.add(p)
                    prompts.append(p)
            # Backfill deterministically to the planned count.
            while len(prompts) < n:
                fill = placeholder_task(domain, len(prompts)).prompt
                if fill in seen:
                    fill = f"{fill} ({len(prompts)})"
                seen.add(fill)
                prompts.append(fill)

            for idx, prompt in enumerate(prompts[:n]):
                tasks.append(make_task(domain, prompt, idx))
        return tasks

    # -- orchestration ------------------------------------------------------

    def run(self, resume: bool = False) -> RunResult:
        self.tasks = build_tasks(self.config)
        existed = self.manifest_store.exists(self.config.run_id())
        manifest = load_or_init(self.manifest_store, self.config, self.tasks)
        prior_status = manifest.status

        result = RunResult(run_id=manifest.run_id, status=RunStatus.running, manifest=manifest)
        # Make the run/resume distinction meaningful. Execution is idempotent either way
        # (id-based skips), so these only add clarity, never change data.
        if resume and not existed:
            result.messages.append(
                "No prior manifest for this config - 'resume' is starting a fresh run."
            )
        if not resume and existed and prior_status == RunStatus.completed:
            result.messages.append(
                "This config already completed (same run_id). Re-running is a no-op "
                "(idempotent, cache-first); change run.name to start a distinct run."
            )

        manifest.status = RunStatus.running
        self.manifest_store.save(manifest)
        result.n_tasks = len(self.tasks)
        # Load existing row ids so judge/probe never write duplicates on resume.
        self._existing_judg_ids = self.judg_store.ids("judg_id")
        self._existing_probe_ids = self.probe_store.ids("probe_id")
        try:
            # llm_generated: replace the count-accurate placeholders with real drafted
            # prompts (cost-guarded, cache-first). Planned counts already match.
            if self.config.prompts.source == "llm_generated":
                self.tasks = self._llm_generated_tasks(manifest)
                result.n_tasks = len(self.tasks)
            self._curate(self.tasks)
            self._generate(manifest)
            self._derive_truncation_series()
            self.manifest_store.save(manifest)
            self._judge(manifest)
            self.manifest_store.save(manifest)
            self._probe(manifest)
            manifest.status = RunStatus.completed
        except BudgetHalt as halt:
            manifest.status = RunStatus.halted_budget
            result.halted = True
            result.messages.append(
                f"Budget cap ${manifest.budget_usd:.2f} reached during '{halt}'. "
                f"Run is resumable: rerun to continue."
            )
        finally:
            self.manifest_store.save(manifest)

        # Tally row counts from disk (authoritative).
        result.n_generations = sum(1 for _ in self.gen_store.read_raw())
        result.n_judgments = sum(1 for _ in self.judg_store.read_raw())
        result.n_probes = sum(1 for _ in self.probe_store.read_raw())
        result.status = manifest.status
        result.manifest = manifest
        return result


def _clip01(v) -> float | None:
    if v is None:
        return None
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None
