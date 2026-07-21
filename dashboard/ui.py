"""Shared UI theme for the dashboard.

Follows the viewer's system light/dark preference: the page chrome comes from Streamlit's
own theme (config.toml sets no colors, only the accent and fonts), and the custom card /
tile styling here adapts through ``prefers-color-scheme``. Chart styling uses neutral,
theme-agnostic axis colors plus the validated data palette.

``section(title, plain, technical)`` is the house pattern: a plain-language heading with a
hover info icon that holds the technical detail.
"""

from __future__ import annotations

import streamlit as st

ACCENT = "#4b4ea0"

# Validated data palettes (read well on both light and dark backgrounds).
CATEGORICAL = ["#3563E9", "#C4671F", "#1E9078", "#9A44C4", "#CB3B46"]
SEQUENTIAL = ["#fbeee0", "#f6d3ad", "#eaa86a", "#d67c33", "#b0521c", "#7d3410"]
DIV_NEG, DIV_MID, DIV_POS = "#3563E9", "#e9e9f0", "#C4671F"
MUTED = "#8a8a99"  # neutral that reads on both themes (chart axes, rules)

SERIF = "'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'SF Mono','JetBrains Mono','Roboto Mono',ui-monospace,Menlo,Consolas,monospace"

_CSS = f"""
<style>
  .block-container {{ padding-top:2.4rem; max-width:1180px; }}

  /* serif display headings; color is inherited so it adapts to light/dark */
  .stApp h1 {{ font-family:{SERIF}; font-weight:600; letter-spacing:-.01em;
    font-size:2rem; text-wrap:balance; }}
  .stApp h2 {{ font-family:{SERIF}; font-weight:600; letter-spacing:-.01em; font-size:1.35rem; }}
  .stApp h3 {{ font-family:{SERIF}; font-weight:600; font-size:1.12rem; }}

  .sb-eyebrow {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
    color:{MUTED}; font-weight:600; margin-bottom:.15rem; }}
  .sb-sub {{ font-size:.98rem; max-width:70ch; opacity:.85; }}

  /* stat tiles */
  [data-testid="stMetric"] {{ border:1px solid rgba(127,127,140,.22); border-radius:12px;
    padding:14px 16px; background:rgba(127,127,140,.04);
    box-shadow:0 1px 2px rgba(0,0,0,.05),0 6px 18px rgba(0,0,0,.05); }}
  [data-testid="stMetricLabel"] p {{ font-size:.72rem; letter-spacing:.03em;
    text-transform:uppercase; color:{MUTED}; font-weight:600; }}
  [data-testid="stMetricValue"] {{ font-family:{SERIF}; font-weight:600;
    font-variant-numeric:tabular-nums; }}

  /* compact tile that lists short values (e.g. providers) without a giant font */
  .sb-tile {{ border:1px solid rgba(127,127,140,.22); border-radius:12px;
    padding:14px 16px; background:rgba(127,127,140,.04); text-align:center;
    box-shadow:0 1px 2px rgba(0,0,0,.05),0 6px 18px rgba(0,0,0,.05); }}
  .sb-tile .sb-tile-label {{ font-size:.72rem; letter-spacing:.03em;
    text-transform:uppercase; color:{MUTED}; font-weight:600; }}
  /* fixed value height so the box matches the metric tiles; scrolls if it overflows */
  .sb-tile .sb-tile-list {{ font-family:{SERIF}; font-weight:600; font-size:1rem;
    line-height:1.35; margin-top:.4rem; height:2.5rem; overflow-y:auto;
    scrollbar-width:thin; }}

  /* bordered containers become cards (border comes from Streamlit's theme) */
  [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:14px;
    box-shadow:0 1px 2px rgba(0,0,0,.05),0 6px 18px rgba(0,0,0,.05); }}

  [data-testid="stDataFrame"] {{ font-variant-numeric:tabular-nums; }}
  code {{ font-family:{MONO}; }}

  @media (prefers-color-scheme: dark) {{
    [data-testid="stMetric"] {{ border-color:rgba(180,180,205,.20);
      background:rgba(180,180,205,.05);
      box-shadow:0 1px 2px rgba(0,0,0,.35),0 8px 24px rgba(0,0,0,.4); }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      box-shadow:0 1px 2px rgba(0,0,0,.35),0 8px 24px rgba(0,0,0,.4); }}
    .sb-tile {{ border-color:rgba(180,180,205,.20); background:rgba(180,180,205,.05);
      box-shadow:0 1px 2px rgba(0,0,0,.35),0 8px 24px rgba(0,0,0,.4); }}
  }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def stat_list(label: str, items: list[str]) -> str:
    """HTML for a compact stat tile whose value is a short vertical list.

    Matches the metric tiles but keeps long words (e.g. provider names) readable instead
    of blowing them up to the metric font size. Render with ``st.markdown(..., True)``.
    """

    rows = "<br>".join(items) if items else "-"
    return (
        f"<div class='sb-tile'><div class='sb-tile-label'>{label}</div>"
        f"<div class='sb-tile-list'>{rows}</div></div>"
    )


def eyebrow(text: str) -> None:
    st.markdown(f"<div class='sb-eyebrow'>{text}</div>", unsafe_allow_html=True)


def subtitle(text: str) -> None:
    st.markdown(f"<div class='sb-sub'>{text}</div>", unsafe_allow_html=True)


def section(title: str, plain: str | None = None, technical: str | None = None) -> None:
    """A plain-language heading with a hover info icon holding the technical detail."""

    st.subheader(title, help=technical)
    if plain:
        st.caption(plain)


def style_chart(chart, *, categorical: bool = True):
    """Consistent fonts, palette, and quiet axes that read on light and dark."""

    chart = (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=SANS,
            titleFont=SANS,
            labelColor=MUTED,
            titleColor=MUTED,
            labelFontSize=11,
            titleFontSize=12,
            gridColor="#808080",
            gridOpacity=0.14,
            domainColor=MUTED,
            domainOpacity=0.4,
            tickColor=MUTED,
            tickOpacity=0.4,
        )
        .configure_legend(labelFont=SANS, titleFont=SANS, labelColor=MUTED, titleColor=MUTED)
        .configure_title(font=SERIF, color=MUTED, fontSize=14, anchor="start")
    )
    if categorical:
        chart = chart.configure_range(category=CATEGORICAL)
    return chart
