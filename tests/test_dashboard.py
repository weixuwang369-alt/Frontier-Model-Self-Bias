"""Headless smoke tests for the Streamlit dashboard via AppTest.

Confirms every page renders without raising, keyless (Results on synthetic data,
Configure/Run on the committed example config). No browser, no API, no keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"

PAGES = [
    DASHBOARD / "app.py",
    DASHBOARD / "pages" / "1_Configure.py",
    DASHBOARD / "pages" / "2_Run.py",
    DASHBOARD / "pages" / "3_Results.py",
]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page.name} raised: {at.exception}"


def test_results_page_shows_demo_notice_and_charts():
    at = AppTest.from_file(str(DASHBOARD / "pages" / "3_Results.py"), default_timeout=60)
    at.run()
    assert not at.exception
    # Example-data notice present (plain-language wording); it now lives in a caption above
    # the collapsed example dropdown, so gather text across notice-style elements.
    blobs: list[str] = []
    for name in ("info", "caption", "markdown", "warning", "success"):
        try:
            blobs.extend(getattr(e, "value", "") for e in getattr(at, name))
        except Exception:  # noqa: BLE001 - element type not present in this version
            pass
    assert "example results" in " ".join(blobs).lower()
    # At least one dataframe (HSPP-R table / diagnostics) rendered.
    assert len(at.dataframe) >= 1
