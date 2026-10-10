"""LakeDelta Streamlit app: pick a lake, build its satellite record, read the results."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from . import analysis as A
from . import config as C
from . import demo
from . import ee_data as E
from . import figures as F
from . import gauge as G
from . import maps as M
from . import pipeline as P
from . import report as R
from . import timelapse as T
from . import ui
from . import __version__
from .ee_auth import initialize

MODES = ["Featured", "Search", "Map", "File"]
TABS = ["Record", "Seasons and trend", "Rain and heat", "Outlook", "Map", "Water quality",
        "Timelapse", "Data and method"]

BUNDLES = P.TTLCache(ttl_s=12 * 3600, max_entries=24)
ANALYSES = P.TTLCache(ttl_s=12 * 3600, max_entries=48)
WQ = P.TTLCache(ttl_s=12 * 3600, max_entries=24)

DEMO_SITE = {"name": demo.DEMO_NAME, "lat": demo.DEMO_LAT, "lon": demo.DEMO_LON, "radius_km": 10,
             "kind": "reservoir", "country": "nowhere (synthetic)",
             "note": "Every number below is invented to show how the analysis works.",
             "source": "demo"}


# ---------------------------------------------------------------------------
# Earth Engine and cached requests
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def ee_status():
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        secrets = None
    return initialize(secrets)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_geocode(query: str) -> list[dict]:
    return P.geocode(query)


@st.cache_data(ttl=12 * 3600, max_entries=64, show_spinner=False)
def cached_outline(key: str, site_json: str) -> dict:
    return P.preview_outline(json.loads(site_json))


@st.cache_data(ttl=3 * 3600, max_entries=32, show_spinner=False)
def cached_layers(fp_json: str) -> dict:
    return E.map_layers(json.loads(fp_json))


@st.cache_data(ttl=3 * 3600, max_entries=128, show_spinner=False)
def cached_composite(fp_json: str, year: int, months: tuple[int, ...]) -> str:
    return E.season_composite_tile(json.loads(fp_json), year, months)


@st.cache_data(ttl=12 * 3600, max_entries=32, show_spinner=False)
def cached_timelapse(fp_json: str, years: tuple[int, ...], months: tuple[int, ...]):
    return E.timelapse_gif(json.loads(fp_json), years, months)


@st.cache_data(ttl=12 * 3600, max_entries=32, show_spinner=False)
def cached_report_maps(fp_json: str, catch_json: str) -> dict:
    return E.report_images(json.loads(fp_json), json.loads(catch_json) if catch_json else None)


def _theme() -> str:
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001
        t = None
    return t if t in ("light", "dark") else "light"


def get_analysis(bundle_key: str, bundle: dict, settings: C.AnalysisSettings) -> A.Analysis:
    skey = (bundle_key, settings.reference_period, tuple(settings.season_months),
            settings.min_clear_fraction, settings.max_relative_uncertainty,
            settings.harmonize_sources)
    an = ANALYSES.get(skey)
    if an is None:
        an = A.analyze(bundle["jrc"], bundle["optical"], bundle["climate"], settings)
        ANALYSES.put(skey, an)
    return an


# ---------------------------------------------------------------------------
# Sidebar: choose a lake and settings
# ---------------------------------------------------------------------------

def _apply_pending():
    """Apply a map click from the previous run before widgets are created."""
    pend = st.session_state.pop("pending_point", None)
    if pend:
        st.session_state["mode"] = "Map"
        st.session_state["lat"] = round(pend["lat"], 4)
        st.session_state["lon"] = round(pend["lon"], 4)
        st.session_state.setdefault("radius", 15)
        st.session_state["map_name"] = st.session_state.get("map_name") or "Selected lake"


def sidebar(status, demo_mode: bool) -> tuple[dict | None, C.AnalysisSettings, bool]:
    st.session_state.setdefault("lat", C.CATALOG[0].lat)
    st.session_state.setdefault("lon", C.CATALOG[0].lon)
    st.session_state.setdefault("radius", 15)
    st.session_state.setdefault("map_name", "Selected lake")
    site = None
    with st.sidebar:
        st.markdown(ui.wordmark(), unsafe_allow_html=True)
        mode = st.segmented_control("Find a lake", MODES, default="Featured", key="mode",
                                    required=True, width="stretch")
        if mode == "Featured":
            key = st.selectbox(
                "Featured lakes", [lake.key for lake in C.CATALOG], key="featured",
                format_func=lambda k: f"{C.CATALOG_BY_KEY[k].name}, {C.CATALOG_BY_KEY[k].country}")
            lake = C.CATALOG_BY_KEY[key]
            site = {"name": lake.name, "lat": lake.lat, "lon": lake.lon, "radius_km": lake.radius_km,
                    "country": lake.country, "kind": lake.kind, "note": lake.note,
                    "source": "catalog"}
        elif mode == "Search":
            q = st.text_input("Lake or reservoir name", key="query", placeholder="Lake Kariba")
            if q and len(q.strip()) >= 3:
                try:
                    found = cached_geocode(q.strip())
                except Exception:  # noqa: BLE001
                    found = None
                    st.error("Place search is not responding. Try again, or choose Map and click "
                             "the lake.")
                if found:
                    i = st.radio("Matches", list(range(len(found))), key=f"pick::{q.strip()}",
                                 format_func=lambda j: found[j]["display"][:90])
                    r = found[i]
                    site = {"name": r["name"], "lat": r["lat"], "lon": r["lon"],
                            "radius_km": r["radius_km"], "source": "search",
                            "note": r["display"]}
                elif found is not None:
                    st.caption("No place by that name. Check the spelling, or choose Map and "
                               "click the lake.")
            else:
                st.caption("Type at least three letters. Results come from OpenStreetMap.")
        elif mode == "Map":
            st.caption("Click the lake on the map, or type its coordinates.")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", -90.0, 90.0, key="lat", format="%.4f", step=0.01)
            lon = c2.number_input("Longitude", -180.0, 180.0, key="lon", format="%.4f", step=0.01)
            name = st.text_input("Name", key="map_name")
            radius = st.slider("Search radius (km)", 2, 300, key="radius",
                               help="How far from the point to look for the lake's edges. "
                                    "Used when the lake is not in HydroLAKES.")
            site = {"name": name or "Selected lake", "lat": lat, "lon": lon,
                    "radius_km": float(radius), "source": "map"}
        else:
            up = st.file_uploader("Lake outline (GeoJSON, longitude and latitude)",
                                  type=["geojson", "json"], key="upload")
            name = st.text_input("Name", key="file_name", value="My lake")
            if up is not None:
                try:
                    parsed = P.parse_outline(up.getvalue())
                    site = {"name": name or "My lake", "lat": parsed["lat"], "lon": parsed["lon"],
                            "radius_km": 10, "geojson": parsed["geometry"], "source": "file"}
                except Exception as exc:  # noqa: BLE001
                    st.error(f"That file could not be read: {exc}")

        with st.expander("Settings"):
            ref = st.segmented_control("Normals from", ["1991–2020", "Whole record"],
                                       default="1991–2020", key="ref", required=True)
            season = st.pills("Season for yearly values and the timelapse", list(C.MONTH_ABBR),
                              selection_mode="multi", default=["Jun", "Jul", "Aug", "Sep"],
                              key="season")
            min_clear = st.slider("Share of the lake that must be seen clearly", 10, 60, 20,
                                  step=5, format="%d%%", key="min_clear",
                                  help="Months seen less clearly are left out.")
            harmonize = st.toggle("Match recent sensors to the JRC record", True, key="harmonize",
                                  help="Scales 2022 onward by the ratio found where both sources "
                                       "overlap (2019–2021).")
        run = st.button("Analyze lake", type="primary", width="stretch", key="run_side",
                        disabled=(site is None and not demo_mode))
        if demo_mode:
            st.caption(f"{status.message} The app is showing demo data; connect Earth Engine to "
                       "analyse the lake you pick here.")
        else:
            st.caption(f"Earth Engine connected. LakeDelta {__version__}.")

    months = tuple(sorted(C.MONTH_ABBR.index(m) + 1 for m in (season or ["Jun", "Jul", "Aug", "Sep"])))
    settings = C.AnalysisSettings(
        reference_period=(1991, 2020) if ref == "1991–2020" else (1800, 2200),
        season_months=months,
        min_clear_fraction=min_clear / 100.0,
        harmonize_sources=harmonize,
    )
    return site, settings, run


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------

def run_pipeline(site: dict, key: str, demo_mode: bool) -> dict | None:
    if demo_mode:
        with st.spinner("Building the demo record"):
            bundle = demo.demo_bundle()
            bundle["site"] = DEMO_SITE
        BUNDLES.put("demo", bundle)
        return bundle
    cached = BUNDLES.get(key)
    if cached is not None:
        return cached
    with st.status(f"Building the record for {site['name']}", expanded=True) as status:
        bar = st.empty()

        def say(msg: str):
            status.update(label=msg)
            st.write(msg)

        def chunks(label: str):
            def cb(done: int, total: int):
                bar.progress(done / total if total else 1.0, text=f"{label}: {done} of {total} parts")
            return cb

        try:
            bundle = P.fetch_bundle(site, P.Progress(say, chunks))
        except E.EEFetchError as exc:
            status.update(label="The analysis stopped", state="error", expanded=True)
            st.error(str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            status.update(label="The analysis stopped", state="error", expanded=True)
            st.error(f"Something went wrong while fetching data: {exc}")
            return None
        bar.empty()
        status.update(label="Record ready", state="complete", expanded=False)
    BUNDLES.put(key, bundle)
    return bundle


# ---------------------------------------------------------------------------
# Before analysis
# ---------------------------------------------------------------------------

def intro(site: dict, key: str, demo_mode: bool) -> bool:
    lede = R.lede_text(site, {})
    st.html(ui.name_block(site["name"], lede))
    left, right = st.columns([2.1, 1], gap="large")
    preview = st.session_state.get("preview", {}).get(key)
    if demo_mode and not preview:
        b = demo.demo_bundle()
        preview = {"footprint": b["footprint"], "info": b["footprint_info"]}
    with left:
        m = M.base_map(site["lat"], site["lon"], zoom=10, demo=demo_mode)
        if preview:
            M.add_outline(m, preview["footprint"])
            M.fit(m, preview["footprint"])
        elif site.get("geojson"):
            M.add_outline(m, site["geojson"], outline_name="Uploaded outline")
            M.fit(m, site["geojson"])
        M.add_point(m, site["lat"], site["lon"], site["name"])
        out = st_folium(m, height=470, use_container_width=True, returned_objects=["last_clicked"],
                        key=f"intro_map::{key}")
        click = (out or {}).get("last_clicked")
        if click and not demo_mode:
            token = (round(click["lat"], 5), round(click["lng"], 5))
            if st.session_state.get("handled_click") != token:
                st.session_state["handled_click"] = token
                st.session_state["pending_point"] = {"lat": click["lat"], "lon": click["lng"]}
                st.rerun()
        if preview:
            info = preview["info"]
            st.caption(f"Outline: {info['method']}. Water has covered up to "
                       f"{A.fmt_area(info['ever_water_km2'])} inside it since 1984.")
            for w in info.get("warnings", []):
                st.warning(w)
    with right:
        st.markdown(
            "LakeDelta measures how much of the lake was water in every month since March 1984, "
            "using Landsat and Sentinel-2 images in Google Earth Engine. It then compares the "
            "latest month with the same month in earlier years and with rain on the catchment.")
        st.markdown("The first run for a lake takes one to three minutes. Results are kept for "
                    "twelve hours, so returning visitors see them at once.")
        if not demo_mode:
            st.caption("Not on the lake? Click it on the map.")
        go = st.button(f"Analyze {site['name']}", type="primary", key="run_main")
        if not demo_mode and not preview:
            if st.button("Check the outline first", key="preview_btn",
                         help="Traces the lake outline (about 10 seconds) without the full analysis."):
                with st.spinner("Tracing the outline"):
                    try:
                        res = cached_outline(key, json.dumps(site))
                        st.session_state.setdefault("preview", {})[key] = res
                        st.rerun()
                    except E.EEFetchError as exc:
                        st.error(str(exc))
    return go


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def hero(an: A.Analysis, site: dict, bundle: dict):
    st.html(ui.name_block(site["name"], R.lede_text(site, bundle)))
    g = G.gauge_inputs(an)
    left, right = st.columns([1, 2.75], gap="large")
    with left:
        if g:
            st.markdown(G.gauge_svg(**g), unsafe_allow_html=True)
            month = A.MONTH_NAME[an.latest["date"].month - 1]
            st.caption(f"The tube shows the water surface. The strip beside it splits every {month} "
                       f"on record into the five classes; the open circle is a year earlier.")
    with right:
        lt = an.latest
        if lt:
            cond = lt["condition"]
            st.html(ui.status_block(A.month_label(lt["date"], long=True), A.fmt_area(lt["area"]),
                                    cond.label if cond else None, cond.color if cond else None,
                                    A.headline(site["name"], lt)))
        else:
            st.html(ui.status_block("No recent clear month", "n/a", None, None,
                                    A.headline(site["name"], None)))
        st.html(ui.readings(R.readings_rows(an)))
    notes = list(dict.fromkeys(an.notes + (bundle.get("footprint_info") or {}).get("warnings", [])))
    for n in notes:
        st.html(ui.note(n))


def tab_record(an: A.Analysis):
    win = st.segmented_control("Period shown", ["5 years", "10 years", "20 years", "Since 1984"],
                               default="Since 1984", key="win", required=True)
    last = an.record.index.max()
    start = None
    if win != "Since 1984":
        start = last - pd.DateOffset(years=int(win.split()[0]))
    items = [i for i in F.record_legend(_theme()) if start is not None or "Normal" not in i[2]]
    if an.outlook is None:
        items = [i for i in items if "Outlook" not in i[2]]
    st.html(ui.legend(items))
    st.plotly_chart(F.record_figure(an, start=start, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit",
                    key="fig_record")
    st.html(ui.note(
        "The line is each month's water surface after filling cloud gaps; the blue wash around it "
        "is the range of plausible fills. Months seen too poorly are left as gaps. Shorter periods "
        "add the normal range for each calendar month."))
    st.subheader("Month by month")
    st.html(ui.condition_legend())
    st.plotly_chart(F.calendar_figure(an, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit",
                    key="fig_calendar")


def tab_seasons(an: A.Analysis):
    season = A.season_label(an.settings.season_months)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("This year against the usual cycle")
        st.html(ui.legend(F.seasonal_legend(an, theme=_theme())))
        st.plotly_chart(F.seasonal_figure(an, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit",
                        key="fig_seasonal")
    with right:
        st.subheader("Year by year")
        options = {"Yearly mean": "mean", f"{season} mean": "season", "Yearly minimum": "min"}
        pick = st.segmented_control("Value", list(options), default="Yearly mean", key="basis",
                                    required=True, label_visibility="collapsed")
        col = {"mean": "mean", "season": "season_mean", "min": "min"}[options[pick]]
        tr_pick = A.trend_test(an.annual[col]) if col in an.annual else None
        pp = F.pal(_theme())
        legend_items = [("line", pp["water"], pick)]
        if tr_pick:
            legend_items.append(("line", pp["context_line"],
                                 f"Trend {A.fmt_signed(tr_pick['slope_per_decade'], 2)} km² per decade, "
                                 f"{A.fmt_p(tr_pick['p'])}"))
        st.html(ui.legend(legend_items))
        st.plotly_chart(F.annual_figure(an, options[pick], theme=_theme()), config=F.PLOTLY_CONFIG,
                        theme="streamlit", key="fig_annual")
    items = []
    tr = an.trend_annual
    if tr:
        verdict = "a significant" if tr["significant"] else "no significant"
        items.append(f"The yearly mean shows {verdict} trend from {tr['first']} to {tr['last']}: "
                     f"{A.fmt_signed(tr['slope_per_decade'], 2)} km² per decade "
                     f"({A.fmt_signed(tr['relative_per_decade'], 1)}% of the mean), 95% range "
                     f"{A.fmt_signed(tr['ci'][0], 2)} to {A.fmt_signed(tr['ci'][1], 2)}, {A.fmt_p(tr['p'])}.")
    ts = an.trend_season
    if ts:
        items.append(f"For {season} alone the change is {A.fmt_signed(ts['slope_per_decade'], 2)} km² "
                     f"per decade ({A.fmt_p(ts['p'])}).")
    sk = an.seasonal_trend
    if sk:
        items.append(f"A seasonal Kendall test on all months gives Kendall's tau {sk['tau']:+.2f} "
                     f"({A.fmt_p(sk['p'])}).")
    if items:
        st.html(ui.facts(items))


def tab_drivers(an: A.Analysis, bundle: dict):
    if an.climate is None or an.response is None:
        st.info("No climate record is available for this lake.")
        return
    best = int(an.response["r"].idxmax())
    cs = an.climate_summary or {}
    ci = bundle.get("catchment_info") or {}
    facts = []
    if ci.get("area_km2"):
        facts.append(f"Climate is averaged over {ci.get('method', 'the catchment')}: "
                     f"{A.fmt_area(ci['area_km2'])}.")
    facts.append(f"The lake tracks rain summed over about {best} months most closely "
                 f"(Spearman r = {an.response.loc[best, 'r']:.2f}).")
    if cs:
        facts.append(f"In the 12 months to {A.month_label(cs['date'], long=True)} the catchment had "
                     f"{cs['p12_pct_normal']:.0f}% of its normal precipitation"
                     + (f" and was {cs['temp_anom12']:+.1f} °C against normal." if cs.get('temp_anom12') is not None else "."))
        if cs.get("runoff12_pct_normal") is not None:
            facts.append(f"Modelled runoff from the catchment was {cs['runoff12_pct_normal']:.0f}% of normal "
                         "over the same 12 months.")
    st.html(ui.facts(facts))
    k = st.slider("Months of rain summed", 1, 36, best, key="spi_k")
    fig = F.drivers_figure(an, k=k, theme=_theme())
    if fig is not None:
        st.plotly_chart(fig, config=F.PLOTLY_CONFIG, theme="streamlit", key="fig_drivers")
    st.subheader("How long does the lake remember rain?")
    st.plotly_chart(F.response_figure(an.response, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit",
                    key="fig_response")
    st.html(ui.note(
        "Correlation is not attribution. Withdrawals, releases and irrigation are not observed here "
        "and can matter as much as rain, especially for reservoirs."))


def tab_outlook(an: A.Analysis):
    if an.outlook is None:
        st.info("The record is too short for an outlook. At least eight years of clear months are "
                "needed.")
        return
    st.html(f'<p class="ld-headline">{R.outlook_text(an)}</p>')
    pp = F.pal(_theme())
    st.html(ui.legend([("line", pp["water"], "Water surface"),
                       ("band", pp["water"], "Outlook, 50% and 80% ranges"),
                       ("line", pp["context_line"], "Normal median"),
                       ("band", pp["context"], "Normal range (middle half)")]))
    fig = F.outlook_figure(an, theme=_theme())
    st.plotly_chart(fig, config=F.PLOTLY_CONFIG, theme="streamlit", key="fig_outlook")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("How well has this worked before?")
        sk = F.skill_figure(an, theme=_theme())
        if sk is not None:
            st.plotly_chart(sk, config=F.PLOTLY_CONFIG, theme="streamlit", key="fig_skill")
        st.html(ui.note("Bars show how much smaller the hindcast errors were than always predicting "
                        "the normal value, testing each year with a model fitted on the other years."))
    with right:
        st.subheader("Outlook by month")
        o = an.outlook.table
        table = pd.DataFrame({
            "Month": [A.month_label(t) for t in o.index],
            "Median": o["median"].to_numpy(),
            "80% range": [f"{A.fmt_area(a, unit=False)} to {A.fmt_area(b, unit=False)}"
                          for a, b in zip(o["p10"], o["p90"])],
            "Normal": o["normal"].to_numpy(),
        })
        st.dataframe(table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4,
                     column_config={
                         "Median": st.column_config.NumberColumn("Median (km²)", format="%.2f"),
                         "80% range": st.column_config.TextColumn("80% range (km²)"),
                         "Normal": st.column_config.NumberColumn("Normal (km²)", format="%.2f"),
                     })


def tab_map(an: A.Analysis, bundle: dict, demo_mode: bool):
    site = bundle["site"]
    fp = bundle["footprint"]
    catch = bundle.get("catchment")
    show_catch = st.toggle("Show the whole catchment", False, key="show_catch")
    layers = None
    if not demo_mode:
        try:
            with st.spinner("Loading map layers from Earth Engine"):
                layers = cached_layers(json.dumps(fp))
        except E.EEFetchError as exc:
            st.warning(f"Satellite layers are unavailable right now ({exc}).")
    m = M.site_map(site, fp, catch, layers, demo=demo_mode, show_catchment_extent=show_catch)
    st_folium(m, height=560, use_container_width=True, returned_objects=[], key="map_main")
    st.html(ui.legend([("line", C.WATER, "Lake outline: water seen at least once, 1984–2021"),
                       ("line", C.SEDIMENT, "Catchment (dashed)")]))
    st.subheader("Compare two years")
    if demo_mode:
        st.info("The year comparison needs Earth Engine imagery, so it is not part of the demo.")
        return
    ann = an.annual["season_mean"].dropna()
    years = list(range(1984, int(an.record.index.max().year) + 1))
    default_a = int(ann.idxmax()) if len(ann) else years[0]
    default_b = int(ann.idxmin()) if len(ann) else years[-1]
    season = A.season_label(an.settings.season_months)
    c1, c2 = st.columns(2)
    ya = c1.selectbox("Left", years, index=years.index(default_a), key="cmp_a")
    yb = c2.selectbox("Right", years, index=years.index(default_b), key="cmp_b")
    try:
        with st.spinner("Building both composites"):
            ua = cached_composite(json.dumps(fp), int(ya), tuple(an.settings.season_months))
            ub = cached_composite(json.dumps(fp), int(yb), tuple(an.settings.season_months))
        st_folium(M.compare_map(ua, ub, fp), height=520, use_container_width=True,
                  returned_objects=[], key=f"map_cmp::{ya}::{yb}")
        st.caption(f"Drag the divider to compare {season} {ya} (left) with {season} {yb} (right). "
                   "The white line is the lake outline. Defaults are the largest and smallest "
                   f"{season} extents on record.")
    except E.EEFetchError as exc:
        st.warning(f"The composites could not be built ({exc}).")


def tab_quality(an: A.Analysis, bundle: dict, key: str, demo_mode: bool):
    st.html(ui.note(
        "Indicators come from the permanent-water core of the lake, away from the shore. NDTI rises "
        "with suspended sediment; NDCI rises with chlorophyll-a; surface temperature comes from "
        "Landsat thermal bands. They are relative indices without field calibration, so read "
        "changes against the lake's own seasonal range (grey band), not as absolute values."))
    wq = WQ.get(key)
    if wq is None:
        if not st.button("Compute water-quality indicators", key="wq_btn",
                         help="Reads every clear Sentinel-2 and Landsat 8/9 image since 2013. "
                              "Takes about a minute."):
            return
        if demo_mode:
            wq = demo.demo_water_quality()
        else:
            bar = st.progress(0.0, text="Reading images")
            try:
                wq = E.water_quality(bundle["footprint"], bundle["meta"]["hist_scale_m"],
                                     progress=lambda d, t: bar.progress(d / t if t else 1,
                                                                        text=f"Reading images: {d} of {t} parts"))
            except E.EEFetchError as exc:
                st.error(str(exc))
                return
            bar.empty()
        WQ.put(key, wq)
        st.rerun()
    df = F.water_quality_frame(wq)
    if df.dropna(how="all").empty:
        st.info("No clear images of open water were found.")
        return
    pp = F.pal(_theme())
    st.html(ui.legend([("box", pp["water"], "Monthly median"),
                       ("line", pp["water"], "12-month level, seasonal cycle removed"),
                       ("band", pp["context"], "Usual range for the month (middle half)")]))
    st.plotly_chart(F.water_quality_figure(df, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit",
                    key="fig_wq")


def tab_timelapse(an: A.Analysis, bundle: dict, key: str, demo_mode: bool):
    months = tuple(an.settings.season_months)
    season = A.season_label(months)
    ann = an.annual["season_mean"].dropna()
    areas = {int(y): float(v) for y, v in ann.items()}
    first = 1984 if not demo_mode else int(min(areas)) if areas else 1984
    last = int(an.record.index.max().year)
    st.html(ui.note(f"One frame per year: the median of clear {season} images (Landsat 5, 7, 8, 9 "
                    "and Sentinel-2). The label shows that season's mean water surface. Change the "
                    "season in Settings."))
    span = st.slider("Years", first, last, (max(first, last - 25), last), key="tl_years")
    tl_key = ("tl", key, span, months)
    gif = st.session_state.get("timelapse", {}).get(tl_key)
    if gif is None:
        if not st.button("Make the timelapse", key="tl_btn"):
            return
        years = list(range(span[0], span[1] + 1))
        with st.spinner("Rendering frames"):
            try:
                if demo_mode:
                    frames = demo.demo_timelapse_frames({y: areas.get(y, np.nan) for y in years
                                                        if y in areas})
                    kept = [y for y in years if y in areas]
                    gif = T.annotate(frames, kept, season, areas)
                else:
                    raw, kept = cached_timelapse(json.dumps(bundle["footprint"]), tuple(years), months)
                    gif = T.annotate(raw, kept, season, areas)
            except E.EEFetchError as exc:
                st.error(str(exc))
                return
        st.session_state.setdefault("timelapse", {})[tl_key] = gif
        st.rerun()
    st.image(gif, width=640)
    st.download_button("Download the GIF", gif, file_name=f"{_slug(bundle['site']['name'])}_timelapse.gif",
                       mime="image/gif", on_click="ignore", key="dl_gif")


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower() or "lake"


def monthly_csv(an: A.Analysis) -> bytes:
    rec = an.record.copy()
    out = pd.DataFrame({
        "month": rec.index.strftime("%Y-%m"),
        "area_km2": rec["area"].round(4), "area_low_km2": rec["lo"].round(4),
        "area_high_km2": rec["hi"].round(4), "observed_km2": rec["observed"].round(4),
        "gap_filled_km2": rec["filled"].round(4), "clear_fraction": rec["clear_fraction"].round(3),
        "used": rec["valid"], "spike": rec["outlier"], "source": rec["source"],
        "normal_median_km2": rec["normal_median"].round(4), "anomaly_km2": rec["anomaly"].round(4),
        "percent_of_normal": rec["pct_of_normal"].round(1), "percentile": rec["percentile"].round(1),
        "condition": rec["condition"],
    })
    if an.climate is not None:
        cl = an.climate.reindex(rec.index)
        out["precip_mm"] = cl["precip_mm"].round(1).to_numpy()
        out["temp_c"] = cl["temp_c"].round(2).to_numpy()
        out["runoff_mm"] = cl["runoff_mm"].round(1).to_numpy()
        out["spi12"] = cl["spi12"].round(2).to_numpy()
    return out.to_csv(index=False).encode("utf-8")


def yearly_csv(an: A.Analysis) -> bytes:
    a = an.annual.copy()
    a.index.name = "year"
    return a.round(4).to_csv().encode("utf-8")


def metadata(an: A.Analysis, bundle: dict) -> dict:
    s = an.settings
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lakedelta_version": __version__, "demo": bool(bundle.get("demo")),
        "site": {k: v for k, v in bundle["site"].items() if k != "geojson"},
        "outline": bundle.get("footprint_info"), "catchment": bundle.get("catchment_info"),
        "hydrolakes": bundle.get("hydrolake"),
        "settings": {"reference_period": list(an.normals.period), "season_months": list(s.season_months),
                     "min_clear_fraction": s.min_clear_fraction,
                     "max_relative_uncertainty": s.max_relative_uncertainty,
                     "harmonize_sources": s.harmonize_sources},
        "harmonization": {k: v for k, v in an.harmonization.items() if k != "pairs"},
        "datasets": [C.GSW_IMAGE, C.GSW_MONTHLY, C.S2_SR, C.CLOUD_SCORE_PLUS, *C.LANDSAT.values(),
                     C.ERA5_LAND_MONTHLY, C.HYDROBASINS.format(level="N"), C.HYDROLAKES],
        "scales_m": bundle.get("meta"),
    }


def tab_data(an: A.Analysis, bundle: dict, key: str, demo_mode: bool):
    slug = _slug(bundle["site"]["name"])
    st.subheader("Downloads")
    files = {
        f"{slug}_monthly.csv": monthly_csv(an),
        f"{slug}_yearly.csv": yearly_csv(an),
        f"{slug}_outline.geojson": json.dumps({"type": "Feature", "properties": {"name": bundle["site"]["name"]},
                                               "geometry": bundle["footprint"]}).encode(),
        f"{slug}_metadata.json": json.dumps(metadata(an, bundle), indent=2, default=str).encode(),
    }
    if bundle.get("catchment"):
        files[f"{slug}_catchment.geojson"] = json.dumps(
            {"type": "Feature", "properties": {"name": f"Catchment of {bundle['site']['name']}"},
             "geometry": bundle["catchment"]}).encode()
    pdf_key = ("pdf", key, an.settings.reference_period, tuple(an.settings.season_months))
    pdf = st.session_state.get("pdf", {}).get(pdf_key)
    if pdf:
        files[f"{slug}_report.pdf"] = pdf
    with st.container(horizontal=True, gap="small"):
        st.download_button("Monthly record (CSV)", files[f"{slug}_monthly.csv"],
                           file_name=f"{slug}_monthly.csv", mime="text/csv", on_click="ignore",
                           key="dl_month")
        st.download_button("Yearly values (CSV)", files[f"{slug}_yearly.csv"],
                           file_name=f"{slug}_yearly.csv", mime="text/csv", on_click="ignore",
                           key="dl_year")
        st.download_button("Outline (GeoJSON)", files[f"{slug}_outline.geojson"],
                           file_name=f"{slug}_outline.geojson", mime="application/geo+json",
                           on_click="ignore", key="dl_outline")
        if f"{slug}_catchment.geojson" in files:
            st.download_button("Catchment (GeoJSON)", files[f"{slug}_catchment.geojson"],
                               file_name=f"{slug}_catchment.geojson", mime="application/geo+json",
                               on_click="ignore", key="dl_catch")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
        st.download_button("Everything (ZIP)", buf.getvalue(), file_name=f"{slug}_lakedelta.zip",
                           mime="application/zip", on_click="ignore", key="dl_zip")
    if pdf:
        st.download_button("PDF report", pdf, file_name=f"{slug}_report.pdf", mime="application/pdf",
                           type="primary", on_click="ignore", key="dl_pdf")
    elif st.button("Prepare the PDF report", key="pdf_btn",
                   help="Builds a four-page report with maps, charts and the method."):
        with st.spinner("Writing the report"):
            maps = None
            if not demo_mode:
                try:
                    maps = cached_report_maps(json.dumps(bundle["footprint"]),
                                              json.dumps(bundle.get("catchment")) if bundle.get("catchment") else "")
                except E.EEFetchError:
                    maps = None
            pdf = R.build_report(an, bundle["site"], bundle, G.gauge_inputs(an), maps)
        st.session_state.setdefault("pdf", {})[pdf_key] = pdf
        st.rerun()

    st.subheader("How clearly each month was seen")
    st.plotly_chart(F.coverage_figure(an, theme=_theme()), config=F.PLOTLY_CONFIG, theme="streamlit", key="fig_cov")
    pp = F.pal(_theme())
    st.html(ui.legend([("box", pp["water"], "Used"), ("box", pp["context"], "Seen too poorly"),
                       ("box", pp["sediment"], "Set aside as a spike")]))
    h = an.harmonization
    ov = F.overlap_figure(h, theme=_theme())
    if ov is not None:
        st.subheader("Where the two sources overlap")
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.plotly_chart(ov, config=F.PLOTLY_CONFIG, theme="streamlit", key="fig_overlap")
        with right:
            txt = (f"{h['n']} months between 2019 and 2021 were seen clearly by both the JRC record "
                   f"and our Landsat 8/9 and Sentinel-2 classification. The JRC area was "
                   f"{h['ratio']:.3f} times ours")
            txt += ", so months from 2022 on were scaled by that factor." if h["applied"] else \
                "; no adjustment was applied."
            if h.get("r") is not None:
                txt += f" Correlation between the two was {h['r']:.3f}."
            if h.get("mape") is not None:
                txt += f" After scaling, the median difference was {h['mape']:.1f}%."
            st.html(f'<p class="ld-headline">{txt}</p>')
    st.subheader("Method")
    for p in R.method_paragraphs(an, bundle):
        st.markdown(p)
    st.subheader("Data sources")
    st.html(ui.facts(R.REFERENCES))


def results(bundle: dict, settings: C.AnalysisSettings, key: str, demo_mode: bool):
    try:
        an = get_analysis(key, bundle, settings)
    except ValueError as exc:
        st.error(f"The record could not be analysed: {exc}")
        return
    hero(an, bundle["site"], bundle)
    tabs = st.tabs(TABS, key="tab", on_change="rerun")
    renderers = [
        lambda: tab_record(an),
        lambda: tab_seasons(an),
        lambda: tab_drivers(an, bundle),
        lambda: tab_outlook(an),
        lambda: tab_map(an, bundle, demo_mode),
        lambda: tab_quality(an, bundle, key, demo_mode),
        lambda: tab_timelapse(an, bundle, key, demo_mode),
        lambda: tab_data(an, bundle, key, demo_mode),
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            if getattr(tab, "open", True):
                render()


def main():
    st.set_page_config(page_title="LakeDelta", page_icon=":material/water_drop:", layout="wide",
                       initial_sidebar_state="auto")
    st.html(ui.CSS)
    _apply_pending()
    status = ee_status()
    demo_mode = not status.ok
    site, settings, run_side = sidebar(status, demo_mode)
    if demo_mode:
        st.html(ui.demo_banner())
        site = DEMO_SITE
        key = "demo"
    else:
        if site is None:
            st.html(ui.name_block("Find a lake", "Pick a featured lake, search by name, click a "
                                                 "point on the map, or upload an outline."))
            return
        key = P.site_key(site)

    current = st.session_state.get("current")
    go_main = False
    if current != key:
        go_main = intro(site, key, demo_mode)
    if run_side or go_main:
        bundle = run_pipeline(site, key, demo_mode)
        if bundle is not None:
            st.session_state["current"] = key
            st.rerun()
        return
    if st.session_state.get("current") == key:
        bundle = BUNDLES.get(key)
        if bundle is None:
            bundle = run_pipeline(site, key, demo_mode)
            if bundle is None:
                return
        results(bundle, settings, key, demo_mode)
