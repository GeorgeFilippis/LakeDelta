"""Drive the app as if Earth Engine were connected, with every EE call stubbed.

Demo mode skips several code paths (outline preview, place search, year
comparison, report maps); this test covers them.
"""

import io
import os

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from lakedelta import demo
from lakedelta import ee_data as E
from lakedelta import pipeline as P
from lakedelta.ee_auth import EEStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (120, 140, 150)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def live(monkeypatch):
    monkeypatch.delenv("LAKEDELTA_DEMO", raising=False)
    monkeypatch.chdir(ROOT)
    import lakedelta.app as app

    calls = {"fetch": 0, "preview": 0}
    b0 = demo.demo_bundle()

    def fake_fetch(site, progress=None):
        calls["fetch"] += 1
        if progress:
            progress.say("Tracing the outline")
            progress.chunks("JRC")(1, 2)
            progress.chunks("JRC")(2, 2)
        b = dict(b0)
        b["demo"] = False
        b["site"] = site
        return b

    def fake_preview(site):
        calls["preview"] += 1
        return {"footprint": b0["footprint"], "hydrolake": None,
                "info": {**b0["footprint_info"], "method": "stub", "warnings": []}}

    frames = demo.demo_timelapse_frames({2020: 12.0, 2021: 11.0, 2022: 10.0})
    gif = demo.frames_to_gif(frames)
    monkeypatch.setattr(app, "ee_status", lambda: EEStatus(True, "service-account", "ok", "proj"))
    monkeypatch.setattr(P, "fetch_bundle", fake_fetch)
    monkeypatch.setattr(P, "preview_outline", fake_preview)
    monkeypatch.setattr(P, "geocode", lambda q: [
        {"name": "Lake Kariba", "display": "Lake Kariba, Zambia/Zimbabwe", "lat": -16.9, "lon": 28.0,
         "radius_km": 150.0, "is_water": True, "kind": "reservoir"}])
    monkeypatch.setattr(E, "map_layers", lambda fp: {k: f"https://tiles.test/{k}/{{z}}/{{x}}/{{y}}"
                                                     for k in ("occurrence", "transition", "recent")})
    monkeypatch.setattr(E, "season_composite_tile", lambda fp, y, m: f"https://tiles.test/{y}/{{z}}/{{x}}/{{y}}")
    monkeypatch.setattr(E, "water_quality", lambda fp, scale, progress=None: demo.demo_water_quality())
    monkeypatch.setattr(E, "timelapse_gif", lambda fp, years, months: (gif, [2020, 2021, 2022]))
    monkeypatch.setattr(E, "report_images", lambda fp, c: {"lake": _png(), "catchment": _png()})
    app.BUNDLES._data.clear()
    app.ANALYSES._data.clear()
    app.WQ._data.clear()
    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at, calls


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]


def test_featured_lake_preview_then_analysis(live):
    at, calls = live
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Lake Mornos" in html                       # default featured lake
    at.button(key="preview_btn").click().run()
    _ok(at)
    assert calls["preview"] == 1
    at.button(key="run_main").click().run()
    _ok(at)
    assert calls["fetch"] == 1
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "In September 2026, Lake Mornos covered" in html
    # changing a settings value re-analyses without fetching again
    at.session_state["min_clear"] = 30
    at.run()
    _ok(at)
    assert calls["fetch"] == 1


def _tab(at, name, button=None):
    at.session_state["tab"] = name
    at.run()
    _ok(at)
    if button:
        at.session_state["tab"] = name
        at.button(key=button).click().run()
        _ok(at)


def test_live_tabs_and_tools(live):
    at, calls = live
    at.button(key="run_main").click().run()
    _ok(at)
    _tab(at, "Map")
    assert any(s.label == "Left" for s in at.selectbox)
    _tab(at, "Water quality", "wq_btn")
    _tab(at, "Timelapse", "tl_btn")
    _tab(at, "Data and method", "pdf_btn")
    pdf = next(iter(at.session_state["pdf"].values()))
    assert pdf[:4] == b"%PDF"
    assert calls["fetch"] == 1


def test_search_mode_selects_a_place(live):
    at, calls = live
    at.session_state["mode"] = "Search"
    at.run()
    _ok(at)
    at.text_input(key="query").input("Kariba").run()
    _ok(at)
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Lake Kariba" in html


def test_map_mode_uses_typed_coordinates(live):
    at, calls = live
    at.session_state["mode"] = "Map"
    at.run()
    at.number_input(key="lat").set_value(40.2).run()
    at.text_input(key="map_name").input("Test lake").run()
    _ok(at)
    html = " ".join(h.proto.body for h in at.get("html"))
    assert "Test lake" in html
    at.button(key="run_side").click().run()
    _ok(at)
    assert calls["fetch"] == 1
