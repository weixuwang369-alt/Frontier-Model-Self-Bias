"""Results - your analyzed runs, with the illustrative example tucked into a dropdown.

The example (synthetic) charts live in a collapsed expander so the page opens straight to
"Your analyzed runs". The example runs fully without keys (``selfbias.synthetic`` plants a
literature-shaped signal so every chart renders, clearly labelled as illustrative). Once a
run has been analyzed (auto after a dashboard run, or ``selfbias analyze``), its saved
report is selectable below for real curves, onsets, and the mechanism regression.
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
st.caption(
    "Your own runs are below. The example results (made-up numbers, just to show what the "
    "charts look like) are tucked into the dropdown."
)

# Shared by both the example and your real runs, so defined once at the top.
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

# --- metric explainers (what it measures / the math / why it matters) ---------
# Reused as column-header info icons on the tables and as tile tooltips.
HELP_JUDGE = (
    "In plain terms: the model that produced the grades in this row. Its scores are what we "
    "examine for favoritism."
)
HELP_FAMILY = (
    "In plain terms: the grouping that marks sibling models from the same developer (for "
    "example, two OpenAI models share the family 'openai'). It lets us separate a model's "
    "preference for its own output from preference for its family's shared style."
)
HELP_HSPP_SELF = (
    "In plain terms: whether a model grades its own writing more leniently than it grades "
    "other models' writing. We compare how often it wrongly passes its own failing answers "
    "with how often it wrongly passes other models' failing answers.\n\n"
    "Example: the reference marks Claude's answer as failing a requirement, yet Claude "
    "passes it 30% of the time while passing GPT's failing answers only 15% of the time. "
    "The ratio is 30 / 15 = 2.0.\n\n"
    "Formula:\n\n"
    "`O(J,G) = #{r : reference = FAIL, judge J = PASS} / #{r : reference = FAIL}`\n\n"
    "`HSPP-R_self(J) = O(J,J) / mean[ O(J,G) over models G unrelated to J ]`\n\n"
    "The math behind it: O(J,G) is the overestimation rate - of the rubric points the "
    "reference marks failed for model G's text, the fraction judge J still passes. The self "
    "ratio divides J's rate on its own output by its average rate on unrelated models, which "
    "cancels out generic leniency.\n\n"
    "Typical range: 0 to about 3 (no hard cap). 0.9-1.1 = even-handed; 1.1-1.3 = mild "
    "self-preference; 1.3-1.7 = moderate; above 1.7 = strong; below 0.9 = unusually harder "
    "on itself."
)
HELP_HSPP_FAM = (
    "In plain terms: whether a model favors its siblings (models from the same developer), "
    "not only itself. Same construction as the self ratio, comparing how it grades its "
    "family's writing against unrelated models' writing.\n\n"
    "Example: two OpenAI models each pass the other's failing answers more often than they "
    "pass an Anthropic model's - a family-level leniency.\n\n"
    "Formula:\n\n"
    "`HSPP-R_family(J) = mean[ O(J,G) : G in family(J), G != J ] / mean[ O(J,G) : G "
    "unrelated ]`\n\n"
    "The math behind it: the numerator is J's average overestimation rate on same-family "
    "siblings; the denominator is its average rate on unrelated models.\n\n"
    "Typical range: 0 to about 3. 0.9-1.1 = even-handed; 1.1-1.3 = mild; 1.3-1.7 = moderate; "
    "above 1.7 = strong family preference. Blank when the roster has no same-family pair."
)
HELP_ONSET = {
    "recognition_accuracy": (
        "In plain terms: the shortest text length at which a model reliably identifies its "
        "own writing, better than a 50/50 guess.\n\n"
        "Example: '250 tokens' means at about 250 tokens and above the model recognizes its "
        "own text above chance; below that it cannot.\n\n"
        "Formula:\n\n"
        "`L* = min{ L : lower95(accuracy, L') > 0.5 for every bin L' >= L }`\n\n"
        "The math behind it: for each length bin we bootstrap a 95% confidence interval for "
        "recognition accuracy; L* is the smallest bin whose interval sits entirely above the "
        "0.5 chance line and stays there for all larger bins. Recognizing its own writing is "
        "the suspected trigger for self-preference (RQ2).\n\n"
        "Typical range: one of the length bins (here 50-500 tokens), or 'not yet' if no bin "
        "clears chance. Accuracy itself runs 0.5 (chance) to 1.0 (perfect); 0.6-0.7 = weak, "
        "0.7-0.85 = clear, above 0.85 = strong recognition."
    ),
    "attribution_f1": (
        "In plain terms: the shortest length at which a text carries a detectable style "
        "'fingerprint' - enough for a simple word-pattern classifier to identify its author "
        "better than guessing.\n\n"
        "Example: '100 tokens' means from about 100 tokens up a basic classifier names the "
        "author better than chance.\n\n"
        "Formula:\n\n"
        "`L* = min{ L : lower95(macroF1, L') > 1/M for every bin L' >= L },  M = #models`\n\n"
        "The math behind it: a TF-IDF classifier is trained and tested on separate prompts "
        "(so it cannot memorize wording); L* is the smallest bin whose 95% interval for "
        "macro-F1 clears the 1/M chance line. Detectable style is the 'familiarity' signal "
        "(RQ2).\n\n"
        "Typical range: a length bin or 'not yet'. Macro-F1 runs from chance (1/M, e.g. 0.33 "
        "for 3 models) to 1.0 (perfect attribution)."
    ),
    "hspp_r_self": (
        "In plain terms: the headline result - the text length at which self-favoritism "
        "becomes measurable.\n\n"
        "Example: 'Starts favoring itself at 500 tokens' means below 500 tokens graders are "
        "roughly fair, but at 500+ they reliably over-credit their own work.\n\n"
        "Formula:\n\n"
        "`L* = min{ L : lower95(HSPP-R_self, L') > 1.0 for every bin L' >= L }`\n\n"
        "The math behind it: the smallest bin whose 95% confidence interval for HSPP-R (self) "
        "sits entirely above 1.0 (the fair line). This is the core RQ2 threshold.\n\n"
        "Typical range: a length bin or 'not yet' if no bin's interval clears 1.0 (expected "
        "with few prompts, where the bands are wide)."
    ),
}
HELP_TERM = (
    "The factor whose independent effect on over-crediting is estimated, with the others "
    "held constant. Each row is labelled in plain language; the underlying model term is "
    "shown in parentheses:\n\n"
    "- Model self-preference (relation=self): the text was written by the grader itself, "
    "versus unrelated models (the baseline).\n"
    "- Same model family preference (relation=family): the text was written by a sibling "
    "model from the same developer.\n"
    "- Text length (log_len): the natural log of the text's token count.\n"
    "- Task type (domain=...): the task category, versus the baseline domain."
)
HELP_ODDS = (
    "The estimated effect of the factor on the odds that a grader over-credits an output, "
    "with the other factors held constant. A value of 1.0 indicates no effect; above 1.0 "
    "indicates the factor increases over-crediting; below 1.0 indicates it decreases it.\n\n"
    "Example: an odds ratio of 1.40 for relation=self indicates that, controlling for length "
    "and task, over-crediting is about 40% more likely when a model judges its own output.\n\n"
    "Formula:\n\n"
    "`logit P(overest=1) = b0 + b1*log(len) + b2*self + b3*family + sum_d bd*domain`\n\n"
    "`odds ratio = exp(b)`\n\n"
    "The math behind it: a logistic regression of the per-rubric over-crediting indicator on "
    "length, authorship relation, and domain, with standard errors clustered by prompt so "
    "repeated prompts do not inflate precision. The odds ratio is the exponentiated "
    "coefficient.\n\n"
    "Typical range: 0 to infinity, centered at 1.0. 0.8-1.25 = negligible; 1.25-1.5 = "
    "modest; 1.5-2.0 = substantial; above 2.0 = large; below 0.8 = the factor reduces "
    "over-crediting."
)
HELP_CI = (
    "The 95% confidence interval for the odds ratio: the range in which the true effect is "
    "expected to lie. If the interval excludes 1.0, the effect is statistically "
    "distinguishable from none; if it spans 1.0, the effect is not reliably different from "
    "zero.\n\n"
    "Formula:\n\n"
    "`95% CI = [ exp(b - 1.96*SE),  exp(b + 1.96*SE) ]`\n\n"
    "Typical range: both bounds are positive and bracket the odds ratio. A narrow interval "
    "reflects more data or less noise; a wide one (common with few prompts) reflects greater "
    "uncertainty."
)
HELP_PVALUE = (
    "The probability of observing an effect at least this large if the factor truly had no "
    "influence. Smaller values indicate stronger evidence that the effect is real.\n\n"
    "Formula:\n\n"
    "`p = P( |Z| >= |b / SE| ),  under the null hypothesis b = 0`\n\n"
    "Typical range: 0 to 1. By convention, p < 0.05 = significant, p < 0.01 = strong "
    "evidence, p >= 0.10 = no reliable effect. With small pilots, large p-values are "
    "expected even for genuine effects."
)
HELP_DIAG = (
    "In plain terms: checks on the experiment's own reliability, so the bias results above "
    "can be trusted.\n\n"
    "What each row means, with its healthy range:\n\n"
    "- position_bias_rate: how often swapping the order of the two texts flips the winner. "
    "0 to 1; 0 is ideal, above ~0.2 signals order sensitivity.\n"
    "- length_compliance: share of generations that hit the requested length band. 0 to 1; "
    "above ~0.9 is good.\n"
    "- repeatability (Krippendorff's alpha): a grader's agreement with itself on repeats. "
    "0 to 1; >=0.8 strong, ~0.67 acceptable.\n"
    "- inter_judge (Cohen's kappa): agreement between different graders. -1 to 1; >=0.6 "
    "substantial.\n\n"
    "Higher is better for all except position_bias_rate."
)

# layman labels for the regression 'term' column (the model term is named in the header
# info hover, e.g. relation=self -> Model self-preference).
_TERM_LABELS = {
    "relation=self": "Model self-preference",
    "relation=family": "Same model family preference",
    "log_len": "Text length (log)",
}


def _term_label(term: str) -> str:
    if term in _TERM_LABELS:
        return _TERM_LABELS[term]
    if term.startswith("domain="):
        return "Task type: " + term.split("=", 1)[1].replace("_", " ")
    return term


# --- example results (synthetic), collapsed by default --------------------------
with st.expander("Example results (sample data)", expanded=False):
    data = build_demo_dataset()

    # headline stat tiles
    mean_self = float(data.hspp["hspp_r_self"].mean())
    mean_fam = float(data.hspp["hspp_r_fam"].mean())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Favors own writing", f"{mean_self:.2f}x", help=HELP_HSPP_SELF)
    k2.metric("Favors its family", f"{mean_fam:.2f}x", help=HELP_HSPP_FAM)
    k3.metric("Models compared", "6")
    k4.metric("Example cost", "$283", help="What a pilot run of this size would cost.")
    k5.metric("Prompts", "60")
    st.write("")

    # length-sweep money plot
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
        st.altair_chart(
            ui.style_chart((band + line).properties(height=360)), use_container_width=True
        )

    # headline HSPP-R table
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
            column_config={
                "judge": st.column_config.TextColumn("judge", help=HELP_JUDGE),
                "family": st.column_config.TextColumn("family", help=HELP_FAMILY),
                "favors self": st.column_config.NumberColumn(
                    "favors self", help=HELP_HSPP_SELF, format="%.2f"
                ),
                "favors family": st.column_config.NumberColumn(
                    "favors family", help=HELP_HSPP_FAM, format="%.2f"
                ),
            },
        )

    # heatmaps
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

    # recognition by judge
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

    # paradigm + domain breakdowns
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

    # diagnostics
    with st.container(border=True):
        ui.section(
            "Sanity checks",
            "Quick health checks on the experiment itself, so you can trust the numbers above.",
            "Position-bias rate (how often swapping the order flips the winner), length "
            "compliance, judge repeatability, and how much judges agree with each other "
            "(METRICS section 7).",
        )
        st.dataframe(
            data.diagnostics,
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric": st.column_config.TextColumn("metric", help=HELP_DIAG),
                "value": st.column_config.NumberColumn("value", help=HELP_DIAG, format="%.2f"),
            },
        )

# --- real run analysis (from data/reports/) -----------------------------------
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
    cols = st.columns(3)
    for col, key in zip(cols, names, strict=True):
        onset = rep["onsets"].get(key)
        col.metric(names[key], f"{onset} tokens" if onset else "not yet", help=HELP_ONSET[key])
    st.caption(
        "Onset length L*: the shortest length bin whose 95% confidence interval clears its "
        "chance line (0.5 for recognition, 1/models for attribution, 1.0 for self-preference) "
        "and stays clear at every longer length."
    )

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
            st.dataframe(
                rep["hspp_table"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "judge": st.column_config.TextColumn("judge", help=HELP_JUDGE),
                    "hspp_r_self": st.column_config.NumberColumn(
                        "HSPP-R (self)", help=HELP_HSPP_SELF, format="%.3f"
                    ),
                    "hspp_r_fam": st.column_config.NumberColumn(
                        "HSPP-R (family)", help=HELP_HSPP_FAM, format="%.3f"
                    ),
                },
            )

    reg = rep.get("regression", {})
    with st.container(border=True):
        ui.section(
            "What drives the favoritism",
            "An estimate of each factor's independent effect on over-crediting - the text's "
            "length, who wrote it, and the task type - with the other factors held constant.",
            "Logistic regression of per-rubric overestimation on log length, authorship "
            "relation, and domain, with prompt-clustered standard errors (v1). Odds ratios "
            "with 95% intervals.",
        )
        if reg.get("status") == "ok":
            st.caption(f"Based on {reg['n_obs']:,} judgments across {reg['n_prompts']} prompts.")
            st.dataframe(
                [{"term": _term_label(t), **v} for t, v in reg["terms"].items()],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "term": st.column_config.TextColumn("Factor", help=HELP_TERM),
                    "odds_ratio": st.column_config.NumberColumn(
                        "odds ratio", help=HELP_ODDS, format="%.3f"
                    ),
                    "ci_lo": st.column_config.NumberColumn(
                        "95% CI low", help=HELP_CI, format="%.3f"
                    ),
                    "ci_hi": st.column_config.NumberColumn(
                        "95% CI high", help=HELP_CI, format="%.3f"
                    ),
                    "p_value": st.column_config.NumberColumn(
                        "p-value", help=HELP_PVALUE, format="%.4f"
                    ),
                },
            )
        else:
            st.caption(f"Not available: {reg.get('status')}")
