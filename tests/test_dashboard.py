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
    # Example-data notice present (plain-language wording).
    infos = " ".join(i.value for i in at.info)
    assert "example results" in infos.lower()
    # At least one dataframe (HSPP-R table / diagnostics) rendered.
    assert len(at.dataframe) >= 1
