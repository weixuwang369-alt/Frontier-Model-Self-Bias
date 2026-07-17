"""Typer CLI: estimate / run / resume / status.

``estimate`` always precedes spend: it prints a per-stage, per-provider dry-run cost from
the config + pricing and makes no API calls. ``run`` shows that estimate and requires an
explicit confirmation before executing (cache-first, budget-capped, resumable).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_experiment_config, load_pricing
from .costs import estimate_cost
from .manifest import ManifestStore
from .pipeline import Pipeline
from .storage import default_data_paths
from .tasks import build_tasks

app = typer.Typer(add_completion=False, help="Self-preference bias research pipeline.")
console = Console()

DEFAULT_PRICING = "config/pricing.yaml"
DEFAULT_DATA = "data"


def _load(config_path: Path):
    config = load_experiment_config(config_path)
    tasks = build_tasks(config)
    return config, tasks


def _estimate_table(config_path: Path, pricing_path: str):
    config, tasks = _load(config_path)
    pricing = load_pricing(pricing_path)
    est = estimate_cost(config, tasks, pricing)

    table = Table(title=f"Dry-run cost estimate - run '{config.run.name}'")
    table.add_column("Stage", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Input tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")
    for stage in ("promptgen", "generate", "judge_pwc", "judge_rubric", "probe"):
        if stage in est.by_stage:
            s = est.by_stage[stage]
            table.add_row(
                stage,
                f"{s.calls:,}",
                f"{s.input_tokens:,}",
                f"{s.output_tokens:,}",
                f"${s.cost_usd:,.4f}",
            )
    table.add_section()
    table.add_row(
        "TOTAL",
        f"{est.total_calls:,}",
        f"{est.total_input_tokens:,}",
        f"{est.total_output_tokens:,}",
        f"${est.total_cost_usd:,.4f}",
    )
    console.print(table)

    if est.by_provider:
        pt = Table(title="By provider")
        pt.add_column("Provider", style="cyan")
        pt.add_column("Cost (USD)", justify="right", style="green")
        for prov, cost in sorted(est.by_provider.items()):
            pt.add_row(prov, f"${cost:,.4f}")
        console.print(pt)

    console.print(
        f"Budget cap: [bold]${config.run.budget_usd:,.2f}[/bold]  "
        f"Estimated: [bold]${est.total_cost_usd:,.4f}[/bold]  "
        f"Tasks: {len(tasks)}  Models: {len(config.roster)}  "
        f"Bins: {len(config.lengths.target_bins_tokens)}"
    )
    if est.unpriced_models:
        console.print(
            "[yellow]WARNING[/yellow]: no pricing.yaml entry for "
            f"{sorted(est.unpriced_models)} - used the default price. Update pricing.yaml."
        )
    if est.total_cost_usd > config.run.budget_usd:
        console.print(
            "[red]Estimate exceeds the budget cap[/red] - the run will halt (resumable) "
            "before finishing. Raise budget_usd or reduce the design."
        )
    return config, est


@app.command()
def estimate(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    pricing: str = typer.Option(DEFAULT_PRICING, help="pricing.yaml path"),
):
    """Print a dry-run cost estimate. Makes no API calls."""

    _estimate_table(config_path, pricing)


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    pricing: str = typer.Option(DEFAULT_PRICING, help="pricing.yaml path"),
    data_root: str = typer.Option(DEFAULT_DATA, help="data directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
):
    """Estimate, confirm, then execute the pipeline (cache-first, budget-capped)."""

    config, _ = _estimate_table(config_path, pricing)
    if not yes:
        proceed = typer.confirm("Proceed with the run?", default=False)
        if not proceed:
            console.print("Aborted. No API calls made.")
            raise typer.Exit(code=0)

    _execute(config_path, pricing, data_root, resume=False)


@app.command()
def resume(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    pricing: str = typer.Option(DEFAULT_PRICING, help="pricing.yaml path"),
    data_root: str = typer.Option(DEFAULT_DATA, help="data directory"),
):
    """Resume a halted/partial run. Completed calls are skipped; cache makes them free."""

    _execute(config_path, pricing, data_root, resume=True)


def _execute(config_path: Path, pricing: str, data_root: str, resume: bool):
    config = load_experiment_config(config_path)
    _last = {"stage": None}

    def progress(stage: str, done: int, total: int):
        if stage != _last["stage"]:
            console.print(f"[cyan]{stage}[/cyan]: {total:,} calls planned")
            _last["stage"] = stage
        if done == total or done % 500 == 0:
            console.print(f"  {stage}: {done:,}/{total:,}")

    pipe = Pipeline(config, data_root=data_root, pricing_path=pricing, progress=progress)
    result = pipe.run(resume=resume)

    console.print()
    console.print(f"[bold]Run {result.run_id}[/bold] - status: [bold]{result.status.value}[/bold]")
    console.print(
        f"tasks={result.n_tasks}  generations={result.n_generations}  "
        f"judgments={result.n_judgments}  probes={result.n_probes}"
    )
    m = result.manifest
    console.print(
        f"Spend: [green]${m.total_cost_usd:,.4f}[/green] / cap ${m.budget_usd:,.2f}  "
        f"(cache hits: {sum(s.cache_hits for s in m.stages.values()):,})"
    )
    for msg in result.messages:
        console.print(f"[yellow]{msg}[/yellow]")


@app.command()
def status(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    data_root: str = typer.Option(DEFAULT_DATA, help="data directory"),
):
    """Show manifest status for the run defined by a config (stages, spend, cache)."""

    config = load_experiment_config(config_path)
    store = ManifestStore(default_data_paths(data_root).manifests)
    run_id = config.run_id()
    if not store.exists(run_id):
        console.print(f"No manifest for run '{config.run.name}' ({run_id}). Not started.")
        raise typer.Exit(code=0)
    m = store.load(run_id)

    table = Table(title=f"Run '{m.run_name}' ({m.run_id}) - {m.status.value}")
    table.add_column("Stage", style="cyan")
    table.add_column("Done/Planned", justify="right")
    table.add_column("Cache hits", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")
    for stage, s in m.stages.items():
        table.add_row(
            stage,
            f"{s.calls_done:,}/{s.calls_planned:,}",
            f"{s.cache_hits:,}",
            f"${s.cost_usd:,.4f}",
        )
    console.print(table)
    console.print(
        f"Total spend: [green]${m.total_cost_usd:,.4f}[/green] / cap ${m.budget_usd:,.2f}"
    )


@app.command()
def analyze(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    data_root: str = typer.Option(DEFAULT_DATA, help="data directory"),
    n_boot: int = typer.Option(1000, help="bootstrap iterations (>=1000 recommended)"),
):
    """Compute metrics from a run's data (curves, onset L*, regression). No API calls."""

    from .analysis import analyze as run_analysis

    config = load_experiment_config(config_path)
    console.print(f"Analyzing run '{config.run.name}' (bootstrap B={n_boot:,})…")
    report = run_analysis(config, data_root=data_root, n_boot=n_boot)

    ot = Table(title="RQ2 onset length L* (first bin whose 95% CI clears the null)")
    ot.add_column("Curve", style="cyan")
    ot.add_column("Null", justify="right")
    ot.add_column("Onset L*", justify="right", style="green")
    for key in ("recognition_accuracy", "attribution_f1", "hspp_r_self"):
        onset = report["onsets"].get(key)
        ot.add_row(key, f"{report['nulls'][key]:.3f}", str(onset) if onset else "- none")
    console.print(ot)

    ct = Table(title="Length-sweep curves (point [95% CI])")
    ct.add_column("bin", justify="right")
    for key in ("recognition_accuracy", "attribution_f1", "hspp_r_self"):
        ct.add_column(key)
    curves = report["curves"]
    for i, b in enumerate(report["length_bins"]):
        cells = [str(b)]
        for key in ("recognition_accuracy", "attribution_f1", "hspp_r_self"):
            pt = curves[key][i]
            p, lo, hi = pt["point"], pt["lo"], pt["hi"]
            cells.append(
                f"{p:.2f} [{lo:.2f},{hi:.2f}]" if p is not None and lo is not None else "-"
            )
        ct.add_row(*cells)
    console.print(ct)

    reg = report["regression"]
    if reg.get("status") == "ok":
        rt = Table(title=f"Mechanism regression v1 (odds ratios; n={reg['n_obs']:,})")
        rt.add_column("Term", style="cyan")
        rt.add_column("Odds ratio", justify="right")
        rt.add_column("95% CI")
        rt.add_column("p", justify="right")
        for term, v in reg["terms"].items():
            rt.add_row(
                term,
                f"{v['odds_ratio']:.3f}",
                f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]",
                f"{v['p_value']:.3f}",
            )
        console.print(rt)
    else:
        console.print(f"[yellow]Regression: {reg.get('status')}[/yellow]")

    console.print(f"Saved report → [green]{report['_path']}[/green]")


@app.command()
def check(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
):
    """Test that each roster model's keys/endpoint work - one tiny real call per model."""

    from .check import check_models

    config = load_experiment_config(config_path)
    console.print(
        f"Testing {len({m.model for m in config.roster})} model(s) with one tiny call each…"
    )
    results = check_models(config)

    table = Table(title="Connection test")
    table.add_column("Model", style="cyan")
    table.add_column("Provider")
    table.add_column("Key env")
    table.add_column("Result")
    table.add_column("Latency", justify="right")
    for r in results:
        if r.ok:
            status = "[green]✓ ok[/green]"
        elif r.skipped:
            status = "[yellow]- no key[/yellow]"
        else:
            status = f"[red]✗ {r.error}[/red]"
        latency = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "-"
        table.add_row(r.model, r.provider, r.key_env or "-", status, latency)
    console.print(table)

    failures = [r for r in results if not r.ok and not r.skipped]
    missing = [r for r in results if r.skipped]
    if failures:
        console.print(
            f"[red]{len(failures)} model(s) failed.[/red] Fix keys/endpoints before a run."
        )
        raise typer.Exit(code=1)
    if missing:
        console.print(
            f"[yellow]{len(missing)} model(s) have no key.[/yellow] Add them to `.env` to use them."
        )
    else:
        console.print("[green]All models reachable.[/green]")


@app.command()
def template(
    out_path: Path = typer.Argument("prompts.xlsx", help="Where to write the starter file"),
):
    """Write a starter Excel/CSV prompt library you can fill in and point a config at."""

    from .excel import write_template

    written = write_template(out_path)
    console.print(f"Wrote prompt-library template → [green]{written}[/green]")
    console.print(
        "Fill it in, then set in your config:\n"
        "  [cyan]prompts:\n    source: excel\n    excel_path: "
        f"{written}[/cyan]"
    )


@app.command("import-check")
def import_check(
    excel_path: Path = typer.Argument(..., exists=True, help="Prompt library (.xlsx/.csv)"),
):
    """Validate a prompt library and summarize what it parses to (no run)."""

    from .excel import import_tasks

    tasks = import_tasks(excel_path)
    from collections import Counter

    by_domain = Counter(t.domain for t in tasks)
    obj = sum(1 for t in tasks if t.reference_source.value == "programmatic")
    console.print(f"[green]OK[/green] - {len(tasks)} tasks parsed.")
    console.print(f"  objective (programmatic): {obj}   subjective (ensemble): {len(tasks) - obj}")
    for d, n in by_domain.items():
        console.print(f"  domain '{d}': {n}")


@app.command()
def export(
    config_path: Path = typer.Argument(..., exists=True, help="Experiment YAML"),
    out_path: Path = typer.Argument("results.xlsx", help="Output .xlsx or .csv"),
    data_root: str = typer.Option(DEFAULT_DATA, help="data directory"),
):
    """Export a run's generations/judgments/probes to Excel (or CSVs)."""

    from .excel import export_results

    written = export_results(data_root, out_path)
    console.print(f"Exported run data → [green]{written}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
