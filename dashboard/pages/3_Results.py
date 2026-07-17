"""Results - bias heatmaps, the length-sweep curves, breakdowns, and real analyzed runs.

Runs fully without keys on the synthetic demo dataset (``selfbias.synthetic``), which
plants a literature-shaped signal so every chart renders (clearly labelled as
illustrative). If a run has been analyzed (``selfbias analyze``), its saved report is
selectable at the bottom for real curves, onsets, and the mechanism regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "src"), str(ROOT / "dashboard")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ui  # noqa: E402

from selfbias.synthetic import ROSTER, build_demo_dataset  # noqa: E402

st.set_page_config(page_title="Results | SelfBias", layout="wide")
ui.apply_theme()
ui.eyebrow("Results")
st.title("Who favors whom")
st.info(
    "These are example results with made-up numbers, so you can see what the charts look "
    "like before running anything. To see your own, run an experiment and analyze it. Your "
    "results then show up at the bottom of this page."
)

data = build_demo_dataset()

# --- headline stat tiles ------------------------------------------------------
mean_self = float(data.hspp["hspp_r_self"].mean())
mean_fam = float(data.hspp["hspp_r_fam"].mean())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Favors own writing",
    f"{mean_self:.2f}x",
    help="HSPP-Ratio (self): how much more often a judge over-credits its own writing "
    "versus unrelated models. 1.0 means no favoritism; above 1 means it favors itself.",
)
k2.metric(
    "Favors its family",
    f"{mean_fam:.2f}x",
    help="HSPP-Ratio (family): the same idea, but for sibling models from the same maker.",
)
k3.metric("Models compared", "6")
k4.metric("Example cost", "$283", help="What a pilot run of this size would cost.")
k5.metric("Prompts", "60")
st.write("")

METRIC_LABELS = {
    "recognition_accuracy": "Recognition accuracy",
    "attribution_f1": "Attribution macro-F1",
    "hspp_r_self": "HSPP-R (self)",
}
# recognition=blue, attribution=teal, HSPP-R=orange (match the preview)
METRIC_COLORS = alt.Scale(
    domain=list(METRIC_LABELS.values()),
    range=[ui.CATEGORICAL[0], ui.CATEGORICAL[2], ui.CATEGORICAL[1]],
)

# --- length-sweep money plot --------------------------------------------------
with st.container(border=True):
    ui.section(
        "Does favoritism kick in once the writing is recognizable?",
        "As the writing gets longer, three things tend to rise together: the model spotting "
        "its own work, a simple classifier guessing the author, and the model favoring "
        "itself. The shaded bands show the range of uncertainty.",
        "The RQ2 / H1-vs-H2 comparison. Recognition accuracy, TF-IDF attribution macro-F1, "
        "and HSPP-R (self) versus realized length (log scale), pooled across judges with "
        "95% bootstrap bands.",
    )
    pooled = data.length_curves.groupby(["bin", "metric"], as_index=False).agg(
        value=("value", "mean"), lo=("lo", "mean"), hi=("hi", "mean")
    )
    pooled["metric"] = pooled["metric"].map(METRIC_LABELS)
    base = alt.Chart(pooled).encode(
        x=alt.X("bin:Q", scale=alt.Scale(type="log"), title="realized length (tokens, log)")
    )
    band = base.mark_area(opacity=0.15).encode(
        y=alt.Y("lo:Q", title="value"),
        y2="hi:Q",
        color=alt.Color("metric:N", scale=METRIC_COLORS, title="Metric"),
    )
    line = base.mark_line(point=True, strokeWidth=2.5).encode(
        y="value:Q",
        color=alt.Color("metric:N", scale=METRIC_COLORS, title="Metric"),
        tooltip=["metric", "bin", "value"],
    )
    st.altair_chart(ui.style_chart((band + line).properties(height=360)), use_container_width=True)

# --- headline HSPP-R table ----------------------------------------------------
with st.container(border=True):
    ui.section(
        "How much each judge favors itself",
        "One row per model. A score of 1.0 means it grades everyone fairly; higher means it "
        "leans toward its own work (self) or its siblings (family).",
        "HSPP-Ratio, self and family, per judge. Leniency-normalized so a generally "
        "generous grader doesn't look biased (METRICS section 3.1).",
    )
    st.dataframe(
        data.hspp.rename(columns={"hspp_r_self": "favors self", "hspp_r_fam": "favors family"}),
        use_container_width=True,
        hide_index=True,
    )

# --- heatmaps -----------------------------------------------------------------
h1, h2 = st.columns(2)
with h1.container(border=True):
    ui.section(
        "Who over-credits whom",
        "Each cell is how often a judge (row) gives too much credit to a writer (column). "
        "Darker means more. The bright diagonal is judges over-crediting themselves.",
        "Overestimation rate O(J, G): among cases the reference says the writer failed, how "
        "often the judge still passes them. Hot diagonal and same-family blocks signal "
        "self- and family-preference.",
    )
    heat = (
        alt.Chart(data.overestimation)
        .mark_rect()
        .encode(
            x=alt.X("generator:N", title="Generator"),
            y=alt.Y("judge:N", title="Judge"),
            color=alt.Color("O:Q", scale=alt.Scale(range=ui.SEQUENTIAL), title="O(J,G)"),
            tooltip=["judge", "generator", "relation", "O"],
        )
        .properties(height=330)
    )
    st.altair_chart(ui.style_chart(heat, categorical=False), use_container_width=True)
with h2.container(border=True):
    ui.section(
        "Who gets bumped up or down",
        "Warm cells are writers a judge scores higher than the crowd does; cool cells are "
        "lower. Judges tend to bump up their own work.",
        "Centered score-delta: each judge's score for a system minus the reference, then "
        "centered on that judge's own average so only relative skew shows.",
    )
    delta = (
        alt.Chart(data.delta_matrix)
        .mark_rect()
        .encode(
            x=alt.X("generator:N", title="Generator"),
            y=alt.Y("judge:N", title="Judge"),
            color=alt.Color(
                "delta:Q",
                scale=alt.Scale(range=[ui.DIV_NEG, ui.DIV_MID, ui.DIV_POS], domainMid=0),
                title="Δ",
            ),
            tooltip=["judge", "generator", "delta"],
        )
        .properties(height=330)
    )
    st.altair_chart(ui.style_chart(delta, categorical=False), use_container_width=True)

# --- recognition by judge -----------------------------------------------------
with st.container(border=True):
    ui.section(
        "Can each model spot its own writing?",
        "How often each model correctly picks out its own text as it gets longer. The dashed "
        "line is a 50/50 guess. Lines are colored by model family.",
        "Pairwise self-recognition probe accuracy versus length, per judge, on the "
        "controlled-length texts. Separate calls from the grading, so recognition can't leak "
        "into the scores.",
    )
    fam = dict(ROSTER)
    recog = data.length_curves[data.length_curves.metric == "recognition_accuracy"].copy()
    recog["family"] = recog["judge"].map(fam)
    recog_chart = (
        alt.Chart(recog)
        .mark_line(point=True)
        .encode(
            x=alt.X("bin:Q", scale=alt.Scale(type="log"), title="length (tokens, log)"),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="accuracy"),
            color=alt.Color("family:N", title="Family"),
            detail="judge:N",
            tooltip=["judge", "family", "bin", "value"],
        )
        .properties(height=300)
    )
    chance = (
        alt.Chart(recog)
        .mark_rule(strokeDash=[4, 4], color=ui.MUTED)
        .encode(y="a:Q")
        .transform_calculate(a="0.5")
    )
    st.altair_chart(ui.style_chart(recog_chart + chance), use_container_width=True)

# --- paradigm + domain breakdowns --------------------------------------------
b1, b2 = st.columns(2)
with b1.container(border=True):
    ui.section(
        "By how the grading works",
        "Comparing two pieces head-to-head tends to show more favoritism than scoring each "
        "against a checklist.",
        "HSPP-R (self) by evaluation paradigm: pairwise comparison versus rubric-based (H5).",
    )
    st.altair_chart(
        ui.style_chart(
            alt.Chart(data.paradigm)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("paradigm:N", title="Paradigm"),
                y=alt.Y("hspp_r_self:Q", title="HSPP-R (self)"),
                color=alt.Color("paradigm:N", legend=None),
                tooltip=["paradigm", "hspp_r_self"],
            )
            .properties(height=280)
        ),
        use_container_width=True,
    )
with b2.container(border=True):
    ui.section(
        "By kind of task",
        "Favoritism shows up more on open-ended, taste-based tasks (like creative writing) "
        "than on tasks with a clear right answer.",
        "HSPP-R (self) by domain: the subjective arms versus the objective, "
        "programmatically-checkable arm.",
    )
    st.altair_chart(
        ui.style_chart(
            alt.Chart(data.domain)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("domain:N", title="Domain", sort=None),
                y=alt.Y("hspp_r_self:Q", title="HSPP-R (self)"),
                color=alt.Color("domain:N", legend=None),
                tooltip=["domain", "hspp_r_self"],
            )
            .properties(height=280)
        ),
        use_container_width=True,
    )

# --- diagnostics --------------------------------------------------------------
with st.container(border=True):
    ui.section(
        "Sanity checks",
        "Quick health checks on the experiment itself, so you can trust the numbers above.",
        "Position-bias rate (how often swapping the order flips the winner), length "
        "compliance, judge repeatability, and how much judges agree with each other "
        "(METRICS section 7).",
    )
    st.dataframe(data.diagnostics, use_container_width=True, hide_index=True)

# --- real run analysis (from data/reports/) -----------------------------------
st.divider()
ui.eyebrow("From your runs")
st.subheader("Your analyzed runs")
report_dir = ROOT / "data" / "reports"
reports = sorted(report_dir.glob("*.json")) if report_dir.exists() else []
if not reports:
    st.info(
        "Nothing here yet. Once you run an experiment and analyze it, its real results "
        "show up right here with the same charts."
    )
else:
    labels = {}
    for p in reports:
        try:
            r = json.loads(p.read_text())
            labels[f"{r.get('run_name', p.stem)} ({p.stem[:8]})"] = r
        except Exception:  # noqa: BLE001
            continue
    pick = st.selectbox("Run report", list(labels.keys()))
    rep = labels[pick]

    names = {
        "recognition_accuracy": "Recognizes itself at",
        "attribution_f1": "Author is guessable at",
        "hspp_r_self": "Starts favoring itself at",
    }
    helps = {
        "recognition_accuracy": "Onset length L*: the shortest length where self-recognition "
        "is reliably above chance.",
        "attribution_f1": "Onset length L*: the shortest length where the author classifier "
        "beats chance.",
        "hspp_r_self": "Onset length L*: the shortest length where self-preference (HSPP-R) "
        "is reliably above 1.0.",
    }
    cols = st.columns(3)
    for col, key in zip(cols, names, strict=True):
        onset = rep["onsets"].get(key)
        col.metric(names[key], f"{onset} tokens" if onset else "not yet", help=helps[key])

    with st.container(border=True):
        ui.section(
            "How your results change with length",
            "The same three lines as the example above, but from your run. Bands show the "
            "uncertainty; a wider band means fewer prompts.",
            "Recognition accuracy, attribution F1, and HSPP-R (self) versus length, with "
            "95% prompt-level bootstrap confidence intervals.",
        )
        rows = []
        for key, label in METRIC_LABELS.items():
            for pt in rep["curves"][key]:
                if pt["point"] is not None:
                    rows.append(
                        {
                            "bin": pt["bin"],
                            "metric": label,
                            "value": pt["point"],
                            "lo": pt["lo"],
                            "hi": pt["hi"],
                        }
                    )
        if rows:
            cdf = pd.DataFrame(rows)
            rbase = alt.Chart(cdf).encode(
                x=alt.X("bin:Q", scale=alt.Scale(type="log"), title="length (tokens, log)")
            )
            rband = rbase.mark_area(opacity=0.15).encode(
                y="lo:Q",
                y2="hi:Q",
                color=alt.Color("metric:N", scale=METRIC_COLORS, title="Metric"),
            )
            rline = rbase.mark_line(point=True, strokeWidth=2.5).encode(
                y=alt.Y("value:Q", title="value"),
                color=alt.Color("metric:N", scale=METRIC_COLORS, title="Metric"),
                tooltip=["metric", "bin", "value", "lo", "hi"],
            )
            st.altair_chart(
                ui.style_chart((rband + rline).properties(height=340)), use_container_width=True
            )
        else:
            st.caption(
                "Not enough data for a chart yet. A run with more prompts will fill this in."
            )

    if rep.get("hspp_table"):
        with st.container(border=True):
            ui.section(
                "How much each of your judges favored itself",
                "One row per model from your run. 1.0 is fair; higher means it leaned toward "
                "its own or its family's work.",
                "HSPP-Ratio (self and family) per judge, at the longest length bin.",
            )
            st.dataframe(rep["hspp_table"], use_container_width=True, hide_index=True)

    reg = rep.get("regression", {})
    with st.container(border=True):
        ui.section(
            "What drives the favoritism",
            "A statistical model estimating how much each factor (length, whose writing it "
            "is, the kind of task) pushes a judge toward over-crediting. Above 1 pushes "
            "toward favoritism; below 1 pushes against.",
            "Logistic regression of per-rubric overestimation on log length, authorship "
            "relation, and domain, with prompt-clustered standard errors (v1). Odds ratios "
            "with 95% intervals.",
        )
        if reg.get("status") == "ok":
            st.caption(f"Based on {reg['n_obs']:,} judgments across {reg['n_prompts']} prompts.")
            st.dataframe(
                [{"term": t, **v} for t, v in reg["terms"].items()],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(f"Not available: {reg.get('status')}")
