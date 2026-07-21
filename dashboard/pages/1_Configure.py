"""Configure - form-based editing + validation of an experiment YAML.

Loads a base config, exposes the most-edited knobs as widgets, validates by constructing
the real :class:`ExperimentConfig` (same schema the pipeline uses), previews the dry-run
cost, and writes to ``config/runs/<name>.yaml``. No experiment logic lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "dashboard")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ui  # noqa: E402

from selfbias.config import ExperimentConfig, load_pricing  # noqa: E402
from selfbias.costs import estimate_cost  # noqa: E402
from selfbias.tasks import build_tasks  # noqa: E402

st.set_page_config(page_title="Configure | SelfBias", layout="wide")
ui.apply_theme()
ui.eyebrow("Configure")
st.title("Configure an experiment")

CONFIG_DIR = ROOT / "config"
RUNS_DIR = CONFIG_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _base_configs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(CONFIG_DIR.glob("experiment*.yaml")):
        out[p.name] = p
    for p in sorted(RUNS_DIR.glob("*.yaml")):
        out[f"runs/{p.name}"] = p
    return out


bases = _base_configs()
choice = st.selectbox("Start from", list(bases.keys()))
raw = bases[choice].read_text(encoding="utf-8")
doc = yaml.safe_load(raw)

st.subheader("Basics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    doc["run"]["name"] = st.text_input(
        "Name for this run", value=doc["run"]["name"], help="A label so you can find it later."
    )
with c2:
    doc["run"]["seed"] = st.number_input(
        "Seed",
        value=int(doc["run"]["seed"]),
        step=1,
        help="A number that makes runs repeatable: the same seed gives the same choices.",
    )
with c3:
    doc["run"]["budget_usd"] = st.number_input(
        "Spending cap (USD)",
        value=float(doc["run"]["budget_usd"]),
        min_value=0.0,
        step=25.0,
        help="A hard limit. The run pauses itself before it would spend past this.",
    )
with c4:
    doc["run"]["phase"] = st.number_input(
        "Phase",
        value=int(doc["run"].get("phase", 0)),
        min_value=0,
        max_value=3,
        step=1,
        help="Which stage of the research plan this run belongs to. Informational.",
    )

bins_str = st.text_input(
    "Text lengths to test",
    value=",".join(str(b) for b in doc["lengths"]["target_bins_tokens"]),
    help="Target lengths (in tokens) the models write at, so you can see how the effect "
    "changes with length. Comma-separated whole numbers.",
)
try:
    doc["lengths"]["target_bins_tokens"] = [
        int(x) for x in bins_str.replace(" ", "").split(",") if x
    ]
except ValueError:
    st.error("Lengths need to be whole numbers, separated by commas.")

ui.section(
    "The models to compare",
    "The AI models that will write and grade. Add at least two.",
    "The model string is the only place a vendor name appears and is the identity used "
    "everywhere in the pipeline, so the strings must be unique. Edit the raw config below "
    "to add, remove, or repoint models (including local ones via a base_url).",
)
st.dataframe(doc["roster"], use_container_width=True, hide_index=True)

with st.expander("Full config (advanced)", expanded=False):
    edited = st.text_area("experiment.yaml", value=yaml.safe_dump(doc, sort_keys=False), height=360)
    try:
        doc = yaml.safe_load(edited)
    except yaml.YAMLError as exc:
        st.error(f"YAML parse error: {exc}")

st.divider()
colv, cols = st.columns(2)
config: ExperimentConfig | None = None
with colv:
    if st.button("Validate", type="primary", use_container_width=True):
        try:
            config = ExperimentConfig.model_validate(doc)
            st.success("Valid configuration.")
        except Exception as exc:  # noqa: BLE001 - surface validation errors to the user
            st.error(f"Invalid configuration:\n\n{exc}")

# Always attempt a silent validation so we can show the estimate preview.
if config is None:
    try:
        config = ExperimentConfig.model_validate(doc)
    except Exception:  # noqa: BLE001
        config = None

if config is not None:
    try:
        pricing = load_pricing(CONFIG_DIR / "pricing.yaml")
        tasks = build_tasks(config)
        est = estimate_cost(config, tasks, pricing)
        ui.section(
            "What this run would cost",
            "An estimate before anything runs. Nothing here spends money.",
            "Estimated as the planned number of model calls times expected tokens times "
            "each model's price from pricing.yaml. Actual cost is tracked live during a run.",
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model calls", f"{est.total_calls:,}")
        m2.metric("Estimated cost", f"${est.total_cost_usd:,.2f}")
        m3.metric("Spending cap", f"${config.run.budget_usd:,.2f}")
        m4.metric("Prompts", f"{len(tasks):,}")
        if est.total_cost_usd > config.run.budget_usd:
            st.warning(
                "This would cost more than your cap, so the run would pause partway. "
                "Raise the cap or trim the setup. You can resume a paused run later."
            )
        st.bar_chart(
            {stage: s.cost_usd for stage, s in est.by_stage.items()},
        )
    except Exception as exc:  # noqa: BLE001
        st.info(f"Estimate unavailable: {exc}")

with cols:
    if st.button("Save to config/runs/", use_container_width=True):
        if config is None:
            st.error("Fix validation errors before saving.")
        else:
            out = RUNS_DIR / f"{config.run.name}.yaml"
            out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            st.success(f"Saved {out.relative_to(ROOT)}")
