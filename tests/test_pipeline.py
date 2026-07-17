from __future__ import annotations

from math import comb

from selfbias.config import ExperimentConfig
from selfbias.manifest import ManifestStore
from selfbias.pipeline import Pipeline
from selfbias.schemas import (
    Disclosure,
    Generation,
    Judgment,
    Paradigm,
    Probe,
    RunStatus,
)
from selfbias.storage import JsonlStore, default_data_paths
from selfbias.tasks import curate


def _pipeline(config, tmp_path):
    return Pipeline(
        config,
        data_root=str(tmp_path / "data"),
        pricing_path="config/pricing.yaml",
    )


def test_end_to_end_mock_run(mock_config, tmp_path):
    pipe = _pipeline(mock_config, tmp_path)
    result = pipe.run()

    assert result.status == RunStatus.completed
    assert result.manifest.total_cost_usd == 0.0

    tasks = curate(mock_config)
    n_tasks = len(tasks)
    n_models = len(mock_config.roster)
    n_bins = len(mock_config.lengths.target_bins_tokens)

    # Controlled generations exactly account for the design; truncations add more.
    controlled = n_tasks * n_models * n_bins
    assert result.n_generations >= controlled
    assert result.n_judgments == (
        n_tasks * n_bins * comb(n_models, 2) * 2 * n_models  # pwc
        + n_tasks * n_bins * n_models * n_models  # rb
    )
    # Controlled-length probes always cover every (task, bin, judge, other) cell; the
    # truncation series adds probes only for bins shorter than the longest generation,
    # so total is between the controlled count and twice it.
    controlled_probes = n_tasks * n_bins * n_models * (n_models - 1)
    assert controlled_probes <= result.n_probes <= 2 * controlled_probes


def test_rerun_is_idempotent(mock_config, tmp_path):
    pipe1 = _pipeline(mock_config, tmp_path)
    r1 = pipe1.run()
    pipe2 = _pipeline(mock_config, tmp_path)
    r2 = pipe2.run()
    assert (r1.n_generations, r1.n_judgments, r1.n_probes) == (
        r2.n_generations,
        r2.n_judgments,
        r2.n_probes,
    )


def test_rows_are_valid_and_anonymized(mock_config, tmp_path):
    pipe = _pipeline(mock_config, tmp_path)
    pipe.run()
    paths = default_data_paths(tmp_path / "data")

    gens = JsonlStore(paths.generations / "generations.jsonl").read_all(Generation)
    assert gens and all(g.realized_tokens > 0 for g in gens)

    judgments = JsonlStore(paths.judgments / "judgments.jsonl").read_all(Judgment)
    # Anonymous arm: no disclosed identity leaks into the record.
    assert all(j.disclosure == Disclosure.anonymous for j in judgments)
    assert all(j.disclosed_as is None for j in judgments)
    # PWC has two ordered subjects; RB has one and per-rubric verdicts.
    pwc = [j for j in judgments if j.paradigm == Paradigm.pwc]
    rb = [j for j in judgments if j.paradigm == Paradigm.rubric]
    assert all(len(j.subject_gen_ids) == 2 for j in pwc)
    assert all(len(j.subject_gen_ids) == 1 and j.per_rubric for j in rb)

    probes = JsonlStore(paths.probes / "probes.jsonl").read_all(Probe)
    assert probes and all(p.correct in (True, False) for p in probes)


def test_order_swap_produces_both_orderings(mock_config, tmp_path):
    pipe = _pipeline(mock_config, tmp_path)
    pipe.run()
    paths = default_data_paths(tmp_path / "data")
    judgments = JsonlStore(paths.judgments / "judgments.jsonl").read_all(Judgment)
    pwc = [j for j in judgments if j.paradigm == Paradigm.pwc]
    positions = {j.position_index for j in pwc}
    assert positions == {0, 1}


def test_probe_series_is_recorded(mock_config, tmp_path):
    pipe = _pipeline(mock_config, tmp_path)
    pipe.run()
    paths = default_data_paths(tmp_path / "data")
    probes = JsonlStore(paths.probes / "probes.jsonl").read_all(Probe)
    series = {p.series for p in probes}
    # Both the controlled-length and truncation series are probed and labelled.
    assert series == {"controlled", "truncation"}


def test_shared_data_root_isolated_by_seed(config_dict, tmp_path):
    # Two configs identical except run.seed, sharing one data_root, must NOT reuse each
    # other's generations (Finding A: generation identity includes the seed).
    cfg_a = ExperimentConfig.model_validate(
        {**config_dict, "run": {**config_dict["run"], "name": "A", "seed": 111}}
    )
    cfg_b = ExperimentConfig.model_validate(
        {**config_dict, "run": {**config_dict["run"], "name": "B", "seed": 222}}
    )
    root = str(tmp_path / "data")

    ra = Pipeline(cfg_a, data_root=root, pricing_path="config/pricing.yaml").run()
    rb = Pipeline(cfg_b, data_root=root, pricing_path="config/pricing.yaml").run()

    tasks = curate(cfg_a)
    controlled = len(tasks) * len(cfg_a.roster) * len(cfg_a.lengths.target_bins_tokens)
    # Run B actually generated its own texts (did not skip via A's rows).
    assert ra.manifest.stages["generate"].calls_done == controlled
    assert rb.manifest.stages["generate"].calls_done == controlled

    # The shared file holds both runs' distinct controlled generations.
    gens = JsonlStore(default_data_paths(root).generations / "generations.jsonl").read_all(
        Generation
    )
    controlled_ids = {g.gen_id for g in gens if g.truncation_of is None}
    assert len(controlled_ids) == 2 * controlled


def test_budget_cap_halts_resumably(config_dict, tmp_path):
    # Mock provider (offline) but a priced model string + tiny budget => halt on first call.
    cfg = dict(config_dict)
    cfg["run"] = dict(config_dict["run"], budget_usd=0.000001, name="halt-test")
    cfg["roster"] = [
        {"slot": "a", "provider": "mock", "model": "expensive-a", "family": "anthropic"},
        {"slot": "b", "provider": "mock", "model": "expensive-b", "family": "openai"},
    ]
    config = ExperimentConfig.model_validate(cfg)

    pipe = Pipeline(config, data_root=str(tmp_path / "data"), pricing_path="config/pricing.yaml")
    result = pipe.run()
    assert result.halted is True
    assert result.status == RunStatus.halted_budget

    store = ManifestStore(default_data_paths(tmp_path / "data").manifests)
    m = store.load(config.run_id())
    assert m.status == RunStatus.halted_budget
    assert m.total_cost_usd <= config.run.budget_usd
