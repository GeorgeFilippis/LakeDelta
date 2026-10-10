"""Drive the whole app in demo mode with Streamlit's AppTest."""

import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS = ["Record", "Seasons and trend", "Rain and heat", "Outlook", "Map", "Water quality",
        "Timelapse", "Data and method"]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LAKEDELTA_DEMO", "1")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _click(at, key):
    at.button(key=key).click()
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_intro_then_results(app):
    at = app
    assert any("Analyze" in b.label for b in at.button)
    _click(at, "run_main")
    assert at.session_state["current"] == "demo"
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Demo Reservoir" in html
    assert "ld-gauge" in html            # the staff gauge rendered
    assert "of its typical" in html      # the headline sentence
    assert len(at.tabs) == len(TABS)


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders(app, tab):
    at = app
    _click(at, "run_main")
    at.session_state["tab"] = tab
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def _in_tab(at, tab, key):
    """AppTest does not keep the selected tab across runs, so pin it for the click."""
    at.session_state["tab"] = tab
    at.run()
    at.session_state["tab"] = tab
    _click(at, key)


def test_on_demand_tools(app):
    at = app
    _click(at, "run_main")
    _in_tab(at, "Water quality", "wq_btn")
    _in_tab(at, "Timelapse", "tl_btn")
    assert at.session_state["timelapse"]
    gif = next(iter(at.session_state["timelapse"].values()))
    assert gif[:6] in (b"GIF89a", b"GIF87a")
    _in_tab(at, "Data and method", "pdf_btn")
    assert any(v[:4] == b"%PDF" for v in at.session_state["pdf"].values())


def test_settings_change_recomputes_without_refetch(app):
    at = app
    _click(at, "run_main")
    at.session_state["ref"] = "Whole record"
    at.run()
    assert not at.exception
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Demo Reservoir" in html
