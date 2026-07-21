"""On-disk layout + append-only JSONL store.

Data conventions: datasets are append-only JSONL under ``data/``; rows are
never mutated or deleted; corrections are new rows with a ``supersedes`` field. Raw API
responses are persisted in full so metrics are recomputable from disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations under a data root. Created on demand.

    When ``run_id`` is set, a run's own artifacts (tasks, generations, judgments, probes,
    features) live under ``root/runs/<run_id>/`` so runs never commingle on disk. The
    cache, manifests, reports, and the seed prompt library stay shared at the root - the
    cache is content-addressed (cross-run reuse is a feature) and the others are already
    keyed by run id or are read-only input.
    """

    root: Path
    run_id: str | None = None

    @property
    def _run_root(self) -> Path:
        return self.root / "runs" / self.run_id if self.run_id else self.root

    @property
    def tasks(self) -> Path:
        return self._run_root / "tasks"

    @property
    def seeds(self) -> Path:
        return self.root / "tasks" / "seeds"

    @property
    def generations(self) -> Path:
        return self._run_root / "generations"

    @property
    def judgments(self) -> Path:
        return self._run_root / "judgments"

    @property
    def probes(self) -> Path:
        return self._run_root / "probes"

    @property
    def features(self) -> Path:
        return self._run_root / "features"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> DataPaths:
        for p in (
            self.tasks,
            self.seeds,
            self.generations,
            self.judgments,
            self.probes,
            self.features,
            self.manifests,
            self.cache,
            self.reports,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_data_paths(root: str | Path = "data", run_id: str | None = None) -> DataPaths:
    return DataPaths(Path(root), run_id=run_id)


class JsonlStore:
    """Append-only JSONL for one artifact type at ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: BaseModel) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def append_many(self, rows: list[BaseModel]) -> None:
        if not rows:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def read_raw(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read_all(self, model_cls: type[T]) -> list[T]:
        return [model_cls.model_validate(d) for d in self.read_raw()]

    def ids(self, id_field: str) -> set[str]:
        return {d[id_field] for d in self.read_raw() if id_field in d}
