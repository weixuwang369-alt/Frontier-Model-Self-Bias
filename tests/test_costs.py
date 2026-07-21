from __future__ import annotations

from math import comb

from selfbias.config import ExperimentConfig, ModelPrice, Pricing
from selfbias.costs import cost_usd, counts_by_stage, estimate_cost
from selfbias.schemas import Usage
from selfbias.tasks import curate


def test_counts_match_design_combinatorics(mock_config):
    tasks = curate(mock_config)
    n_tasks = len(tasks)
    n_models = len(mock_config.roster)
    n_bins = len(mock_config.lengths.target_bins_tokens)
    counts = counts_by_stage(mock_config, tasks)

    assert counts["generate"] == n_tasks * n_models * n_bins
    # PWC: unordered cross-gen pairs x order-swap x judges x arms
    assert counts["judge_pwc"] == n_tasks * n_bins * comb(n_models, 2) * 2 * n_models * 1
    # RB: one call per (gen x judge x arm)
    assert counts["judge_rubric"] == n_tasks * n_bins * n_models * n_models * 1
    # Probe: judge x other(!=judge) x (controlled + truncation series)
    assert counts["probe"] == n_tasks * n_bins * n_models * (n_models - 1) * 2


def test_cost_usd_matches_pricing():
    pricing = Pricing(default=ModelPrice(input_per_mtok=10.0, output_per_mtok=30.0))
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(usage, "anything", pricing) == 40.0


def test_estimate_totals_are_internally_consistent(mock_config, pricing):
    tasks = curate(mock_config)
    est = estimate_cost(mock_config, tasks, pricing)
    # Sum of stage costs equals reported total.
    assert round(sum(s.cost_usd for s in est.by_stage.values()), 4) == est.total_cost_usd
    # All-mock roster is free and never flagged unpriced.
    assert est.total_cost_usd == 0.0
    assert est.unpriced_models == set()


def test_priced_estimate_is_exact_for_one_model(config_dict, pricing):
    # One priced model + one free model (min-2 roster). Judging/probes off so the priced
    # model has exactly one generate call → exact arithmetic on its per-model estimate.
    cfg = dict(config_dict)
    cfg["roster"] = [
        {"slot": "x", "provider": "mock", "model": "priced-x", "family": "anthropic"},
        {"slot": "free", "provider": "mock", "model": "mock-free", "family": "other"},
    ]
    cfg["domains"] = [{"name": "verifiable_if", "n_prompts": 1, "reference": "programmatic"}]
    cfg["lengths"] = dict(config_dict["lengths"], target_bins_tokens=[100])
    cfg["judging"] = dict(config_dict["judging"], paradigms=[])  # no judging calls
    cfg["probes"] = {
        "pairwise_recognition": False,
        "single_recognition": False,
        "on_truncation_series": False,
    }
    config = ExperimentConfig.model_validate(cfg)

    priced = Pricing(
        default=ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0),
        models={"priced-x": ModelPrice(input_per_mtok=1000.0, output_per_mtok=2000.0)},
    )
    tasks = curate(config)
    est = estimate_cost(config, tasks, priced)
    m = est.by_model["priced-x"]
    assert m.calls == 1
    expected = m.input_tokens / 1e6 * 1000.0 + m.output_tokens / 1e6 * 2000.0
    assert abs(m.cost_usd - expected) < 1e-9
