"""Run manifests - resumability + live cost accounting.

A manifest is the durable record of a run: a frozen config snapshot, the planned call
inventory, which calls have completed (for skip-if-done resumability), and running cost
by provider vs the hard budget cap. It is written after every batch so a killed run
resumes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ExperimentConfig
from .costs import counts_by_stage
from .schemas import RunManifest, RunStatus, StageCost, Task


class ManifestStore:
    def __init__(self, manifests_dir: str | Path) -> None:
        self.dir = Path(manifests_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def exists(self, run_id: str) -> bool:
        return self.path(run_id).exists()

    def load(self, run_id: str) -> RunManifest:
        with self.path(run_id).open("r", encoding="utf-8") as fh:
            return RunManifest.model_validate(json.load(fh))

    def save(self, manifest: RunManifest) -> None:
        tmp = self.path(manifest.run_id).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(manifest.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)
        tmp.replace(self.path(manifest.run_id))

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))


def init_manifest(config: ExperimentConfig, tasks: list[Task]) -> RunManifest:
    """Fresh manifest with the planned inventory filled in per stage."""

    planned = counts_by_stage(config, tasks)
    stages = {stage: StageCost(calls_planned=n) for stage, n in planned.items()}
    return RunManifest(
        run_id=config.run_id(),
        run_name=config.run.name,
        config_snapshot=config.model_dump(mode="json"),
        seed=config.run.seed,
        budget_usd=config.run.budget_usd,
        status=RunStatus.created,
        stages=stages,
        completed_call_ids={stage: [] for stage in planned},
        cost_by_provider={},
    )


def load_or_init(store: ManifestStore, config: ExperimentConfig, tasks: list[Task]) -> RunManifest:
    run_id = config.run_id()
    if store.exists(run_id):
        return store.load(run_id)
    manifest = init_manifest(config, tasks)
    store.save(manifest)
    return manifest


def record_call(
    manifest: RunManifest,
    stage: str,
    call_id: str,
    provider: str,
    cost: float,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
) -> None:
    """Update manifest counters for one executed (or cache-hit) call. In-memory only;
    caller persists via :meth:`ManifestStore.save`."""

    stage_cost = manifest.stages.setdefault(stage, StageCost())
    stage_cost.calls_done += 1
    stage_cost.input_tokens += input_tokens
    stage_cost.output_tokens += output_tokens
    if cache_hit:
        stage_cost.cache_hits += 1
    else:
        stage_cost.cost_usd = round(stage_cost.cost_usd + cost, 6)
        manifest.cost_by_provider[provider] = round(
            manifest.cost_by_provider.get(provider, 0.0) + cost, 6
        )
    manifest.completed_call_ids.setdefault(stage, []).append(call_id)


def would_exceed_budget(manifest: RunManifest, next_cost: float) -> bool:
    return manifest.total_cost_usd + next_cost > manifest.budget_usd
