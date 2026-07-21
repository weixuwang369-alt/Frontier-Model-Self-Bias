"""Streamlit entry point - a thin client over ``src/selfbias`` (no experiment logic here).

Three areas live in ``pages/``: Configure, Run, Results. This landing page shows project
orientation and the API-key status. The app starts and renders with keys missing,
degrading gracefully (Results runs fully on synthetic demo data; Run shows a clear
keys-missing state for real providers).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ui  # noqa: E402

from selfbias.config import Settings  # noqa: E402
from selfbias.schemas import Provider  # noqa: E402

st.set_page_config(page_title="SelfBias", layout="wide")
ui.apply_theme()

ui.eyebrow("SelfBias")
st.title("Do AI models play favorites when they grade each other?")
ui.subtitle(
    "This app measures whether a model tends to prefer its own writing when it acts as a "
    "judge. Set up an experiment, keep an eye on the cost, and explore the results, all in "
    "one place."
)
st.write("")

settings = Settings()
keys = {
    "ANTHROPIC_API_KEY": bool(settings.resolve_key(Provider.anthropic)),
    "GOOGLE_API_KEY": bool(settings.resolve_key(Provider.google)),
    "OPENAI_API_KEY": bool(settings.resolve_key(Provider.openai)),
}

ui.section(
    "Your API keys",
    "Which providers you're set up to use. You only need keys for the models you actually "
    "want to run.",
    "These are the three common key names read from .env. A model can point at its own key "
    "with api_key_env in the config (e.g. OPENROUTER_API_KEY), and local endpoints like "
    "Ollama or vLLM often need no key. The Run page checks exactly the keys a given config "
    "needs.",
)
cols = st.columns(3)
for col, (env_name, present) in zip(cols, keys.items(), strict=True):
    col.metric(env_name, "ready" if present else "not set")

if not any(keys.values()):
    st.info(
        "No keys found yet, and that's fine. You can explore the whole app and see example "
        "results without any. Add keys to a file named .env whenever you want to run real "
        "models."
    )

st.write("")
ui.eyebrow("What's here")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.subheader("Configure")
    st.caption("Pick your models and prompts, and see what a run would cost before it runs.")
with c2.container(border=True):
    st.subheader("Run")
    st.caption("Check your keys, confirm the cost, and watch the experiment go, step by step.")
with c3.container(border=True):
    st.subheader("Results")
    st.caption("Charts of who favors whom, how it changes with length, and your analyzed runs.")

st.divider()
st.caption("Nothing runs or spends money unless you start it yourself.")
