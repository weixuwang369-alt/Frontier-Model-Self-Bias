from __future__ import annotations

from selfbias.costs import counts_by_stage
from selfbias.manifest import (
    ManifestStore,
    init_manifest,
    record_call,
    would_exceed_budget,
)
from selfbias.tasks import curate


def test_init_manifest_records_planned_inventory(mock_config):
    tasks = curate(mock_config)
    m = init_manifest(mock_config, tasks)
    planned = counts_by_stage(mock_config, tasks)
    for stage, n in planned.items():
        assert m.stages[stage].calls_planned == n
    assert m.status.value == "created"
    assert m.budget_usd == mock_config.run.budget_usd


def test_record_call_updates_counters_and_skips_cache_cost(mock_config):
    tasks = curate(mock_config)
    m = init_manifest(mock_config, tasks)
    record_call(m, "generate", "id1", "anthropic", 0.5, 100, 50, cache_hit=False)
    assert m.stages["generate"].calls_done == 1
    assert m.total_cost_usd == 0.5
    # Cache hit adds no cost but still counts.
    record_call(m, "generate", "id2", "anthropic", 0.5, 100, 50, cache_hit=True)
    assert m.stages["generate"].cache_hits == 1
    assert m.total_cost_usd == 0.5
    assert m.is_done("generate", "id1") and m.is_done("generate", "id2")


def test_would_exceed_budget(mock_config):
    tasks = curate(mock_config)
    m = init_manifest(mock_config, tasks)
    m.budget_usd = 1.0
    assert would_exceed_budget(m, 1.5) is True
    assert would_exceed_budget(m, 0.5) is False


def test_manifest_store_roundtrip(mock_config, tmp_path):
    tasks = curate(mock_config)
    store = ManifestStore(tmp_path / "manifests")
    m = init_manifest(mock_config, tasks)
    store.save(m)
    assert store.exists(m.run_id)
    loaded = store.load(m.run_id)
    assert loaded.run_id == m.run_id
    assert loaded.stages.keys() == m.stages.keys()
    assert m.run_id in store.list_runs()
