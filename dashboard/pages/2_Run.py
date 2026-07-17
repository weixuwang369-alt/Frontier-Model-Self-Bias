"""Run - dry-run estimate, explicit confirm, execution with live progress.

Real sweeps are never launched implicitly: this page shows the estimate and key status,
and only executes on an explicit button press with a typed confirmation. Providers
without keys block a real launch ("keys missing"); all-mock configs run offline at $0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "dashboard")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ui  # noqa: E402

from selfbias.config import Settings, load_experiment_config, load_pricing  # noqa: E402
from selfbias.costs import estimate_cost  # noqa: E402
from selfbias.pipeline import Pipeline  # noqa: E402
from selfbias.schemas import Provider as ProviderEnum  # noqa: E402
from selfbias.tasks import build_tasks  # noqa: E402

st.set_page_config(page_title="Run | SelfBias", layout="wide")
ui.apply_theme()
ui.eyebrow("Run")
st.title("Run a configured experiment")

CONFIG_DIR = ROOT / "config"
RUNS_DIR = CONFIG_DIR / "runs"


def _configs() -> dict[str, Path]:
    out = {p.name: p for p in sorted(CONFIG_DIR.glob("experiment*.yaml"))}
    out.update({f"runs/{p.name}": p for p in sorted(RUNS_DIR.glob("*.yaml"))})
    return out


configs = _configs()
if not configs:
    st.info("No configs found. Create one on the Configure page.")
    st.stop()

choice = st.selectbox("Config", list(configs.keys()))
config = load_experiment_config(configs[choice])
pricing = load_pricing(CONFIG_DIR / "pricing.yaml")
tasks = build_tasks(config)
est = estimate_cost(config, tasks, pricing)

ui.section(
    "What this run will cost",
    "An estimate before anything runs. It never spends money on its own.",
    "Estimated as the planned number of model calls times expected tokens times each "
    "model's price. The table below breaks it down by stage: writing, judging, and the "
    "self-recognition probes.",
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model calls", f"{est.total_calls:,}")
c2.metric("Estimated cost", f"${est.total_cost_usd:,.2f}")
c3.metric("Spending cap", f"${config.run.budget_usd:,.2f}")
c4.metric("Providers", ", ".join(sorted({m.provider.value for m in config.roster})))
st.dataframe(
    [
        {
            "stage": stage,
            "calls": s.calls,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "cost_usd": round(s.cost_usd, 4),
        }
        for stage, s in est.by_stage.items()
    ],
    use_container_width=True,
    hide_index=True,
)

# --- Key gating (roster-aware) ------------------------------------------------
settings = Settings()
status = settings.key_status(config.roster)
missing = [s for s in status if s["required"] and not s["present"]]
all_mock = all(m.provider == ProviderEnum.mock for m in config.roster)

ui.section(
    "Start the run",
    "Here's which models are ready to go. You can test your keys, then start when you're set.",
    "A model is ready when its key is found, or when it needs none (local endpoints and the "
    "offline mock). The run only starts after you type the run name to confirm.",
)
st.dataframe(
    [
        {
            "model": s["model"],
            "provider": s["provider"],
            "key name": s["key_env"] or "none",
            "status": "ready"
            if s["present"]
            else ("key missing" if s["required"] else "no key needed"),
        }
        for s in status
    ],
    use_container_width=True,
    hide_index=True,
)
if missing:
    st.warning(
        "These models are missing a key: "
        + ", ".join(f"{s['model']} ({s['key_env']})" for s in missing)
        + ". Add them to your .env file, or pick a config that runs offline."
    )
elif all_mock:
    st.info("This config runs entirely offline at no cost. Safe to start.")
else:
    st.success("All the keys you need are in place.")

if st.button("Test my keys", help="Makes one tiny real call per model to confirm the keys work."):
    from selfbias.check import check_models  # noqa: E402

    with st.spinner("Checking each model..."):
        results = check_models(config, settings)
    st.dataframe(
        [
            {
                "model": r.model,
                "provider": r.provider,
                "result": "ok" if r.ok else ("no key" if r.skipped else f"failed: {r.error}"),
                "latency_ms": r.latency_ms if r.latency_ms is not None else "n/a",
            }
            for r in results
        ],
        use_container_width=True,
        hide_index=True,
    )

confirm = st.text_input(
    "Type the run name to confirm you want to start",
    placeholder=config.run.name,
    help="A safety step so a real run never starts by accident.",
)
launch_disabled = bool(missing) or confirm.strip() != config.run.name

if st.button("Start the run", type="primary", disabled=launch_disabled):
    prog = st.progress(0.0, text="starting")
    status_box = st.empty()

    state = {"stage": "", "done": 0, "total": 1}

    def _progress(stage: str, done: int, total: int):
        state.update(stage=stage, done=done, total=max(1, total))
        prog.progress(min(1.0, done / max(1, total)), text=f"{stage}: {done:,}/{total:,}")

    pipe = Pipeline(
        config,
        data_root=str(ROOT / "data"),
        pricing_path=str(CONFIG_DIR / "pricing.yaml"),
        progress=_progress,
    )
    with st.spinner("Running..."):
        result = pipe.run()
    prog.progress(1.0, text="done")
    m = result.manifest
    status_box.success(
        f"Done ({result.status.value}). Wrote {result.n_generations} pieces of writing, "
        f"{result.n_judgments} judgments, and {result.n_probes} recognition checks. "
        f"Spent ${m.total_cost_usd:.4f} of your ${m.budget_usd:.2f} cap."
    )
    for msg in result.messages:
        st.warning(msg)
