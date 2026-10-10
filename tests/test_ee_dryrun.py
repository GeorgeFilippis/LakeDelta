"""Run every Earth Engine code path offline (see tests/ee_dryrun.py)."""


import pytest

from ee_dryrun import dry_run
from lakedelta import analysis as A
from lakedelta import ee_data as E

POLY = {"type": "Polygon", "coordinates": [[[22.1, 38.5], [22.2, 38.5], [22.2, 38.56],
                                            [22.1, 38.56], [22.1, 38.5]]]}


def fake_features(props_list):
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": None, "properties": p} for p in props_list]}


def groups(area_m2=2e7):
    return [{"code": 0 * 101 + 0, "sum": 5e6}, {"code": 1 * 101 + 98, "sum": area_m2},
            {"code": 2 * 101 + 40, "sum": 1e6}]


def responder(kind, payload):
    if kind == "mapid":
        return "https://tiles.dry/{z}/{x}/{y}"
    if kind in ("thumb", "video"):
        return "thumb-id"
    p = payload
    if "HydroLakes" in p:
        return {"type": "FeatureCollection", "features": [{
            "type": "Feature", "geometry": POLY,
            "properties": {"Lake_name": "Mornos", "Lake_type": 2, "Lake_area": 17.4,
                           "Wshd_area": 588.0, "Pour_long": 22.12, "Pour_lat": 38.52,
                           "dist_m": 0.0}}]}
    if "reduceToVectors" in p:
        return {"geometry": POLY, "n_patches": 1, "used_nearest": 0, "truncated": 0,
                "area_m2": 3.1e7}
    if "MonthlyHistory" in p:
        return fake_features([{"t": 446774400000 + i * 2.6e9, "groups": groups()} for i in range(3)])
    if "ERA5_LAND" in p:
        return fake_features([{"t": 441763200000, "total_precipitation_sum": 0.05,
                               "temperature_2m": 285.0, "runoff_sum": 0.01}])
    if '"ndci"' in p:
        return fake_features([{"t": 1500000000000, "ndti_median": -0.1, "ndti_count": 500,
                               "ndci_median": -0.05, "ndci_count": 500, "lst_median": 18.2,
                               "lst_count": 300}])
    if "List.sequence" in p and "S2_SR_HARMONIZED" in p:
        return fake_features([{"t": 1546300800000, "n_scenes": 9, "groups": groups(1.9e7)}])
    if '"MAIN_BAS"' in p and "reduceColumns" in p and '"NEXT_DOWN"' not in p:
        return [[1, 500.0, 99], [2, 800.0, 99]]
    if '"NEXT_DOWN"' in p:
        return [[1, 2], [2, 0], [3, 1], [4, 3], [5, 0], [6, 5]]
    if "AggregateFeatureCollection" in p or "aggregate_max" in p:
        return 900.0
    if "hybas" in p and "union" in p.lower():
        return {"geometry": POLY, "area_m2": 7.8e8}
    if '"ever"' in p:
        return {"ever": 2.5e7, "permanent": 9e6}
    if "Geometry.buffer" in p and '"area_m2"' in p:
        return {"geometry": POLY, "area_m2": 2.4e9}
    if "Collection.size" in p or "size" in p and p.count("ImageCollection") > 0 and "List" in p:
        return [3, 0, 5]
    if "Geometry.area" in p:
        return 3.1e7
    raise AssertionError("Unexpected request: " + p[:300])


@pytest.fixture
def log():
    return []


def test_find_hydrolake(log):
    with dry_run(responder, log):
        hl = E.find_hydrolake(22.14, 38.53)
    assert hl["properties"]["Lake_name"] == "Mornos"
    assert "Collection.limit" in log[0] or "limit" in log[0]


def test_delineate_with_and_without_seed(log):
    with dry_run(responder, log):
        a = E.delineate(22.14, 38.53, radius_m=10_000)
        b = E.delineate(22.14, 38.53, seed_geojson=POLY, seed_area_km2=17.4)
        c = E.delineate(59.6, 45.4, radius_m=260_000)
    assert a["vector_scale_m"] == 30
    assert b["margin_m"] >= 1500
    assert c["vector_scale_m"] > 30  # coarse grid for a big search area
    assert "reduceResolution" in log[-1]


def test_outline_for_site_uses_hydrolakes(log):
    site = {"name": "Mornos", "lat": 38.53, "lon": 22.14, "radius_km": 10}
    with dry_run(responder, log):
        out = E.outline_for_site(site)
    assert out["footprint"] == POLY
    assert out["hydrolake"]["properties"]["Lake_type"] == 2
    assert out["info"]["method"].startswith("JRC water ever seen, around")
    assert out["info"]["ever_water_km2"] == pytest.approx(25.0)


def test_outline_truncation_expands_then_warns(log):
    calls = []

    def truncated(kind, payload):
        if kind == "value" and "reduceToVectors" in payload:
            calls.append(payload)
            return {"geometry": POLY, "n_patches": 1, "used_nearest": 0,
                    "edge_crossing_m": 9000.0, "area_m2": 3.1e7}
        return responder(kind, payload)

    site = {"name": "Mornos", "lat": 38.53, "lon": 22.14, "radius_km": 10}
    with dry_run(truncated, log):
        out = E.outline_for_site(site)
    assert len(calls) == 3                      # two enlargements, then give up
    assert out["info"]["truncated"]
    assert any("edge of the search area" in w for w in out["info"]["warnings"])


def test_river_crossing_is_not_truncation(log):
    def river(kind, payload):
        if kind == "value" and "reduceToVectors" in payload:
            return {"geometry": POLY, "n_patches": 1, "used_nearest": 0,
                    "edge_crossing_m": 240.0, "area_m2": 3.1e7}
        return responder(kind, payload)

    with dry_run(river, log):
        out = E.delineate(22.14, 38.53, radius_m=10_000)
    assert not out["truncated"]


def test_trace_catchment_walks_upstream(log):
    with dry_run(responder, log):
        res = E.trace_catchment(POLY, pour_point=(22.12, 38.52), watershed_km2=588.0)
    assert res["info"]["level"] == 12
    # seeds {1, 2}; 3 drains to 1, 4 drains to 3 -> {1, 2, 3, 4}
    assert res["info"]["n_basins"] == 4
    assert res["info"]["area_km2"] == pytest.approx(780.0)


def test_upstream_ids_handles_int_float_mix():
    table = [[10, 20], [20.0, 0], [30, 10.0], [40, 99]]
    assert E.upstream_ids({20}, table) == {20, 10, 30}


def test_histogram_requests_parse(log):
    with dry_run(responder, log):
        jrc = E.jrc_histograms(POLY, 30, "1984-03-01", "1985-03-01")
        opt = E.optical_histograms(POLY, 30, "2019-01-01", "2019-03-01")
    assert len(jrc) == 3 * len(E.month_chunks("1984-03-01", "1985-03-01", 48))
    assert opt[0]["n_scenes"] == 9
    rec = A.months_from_records(jrc, A.SOURCE_JRC)
    assert rec["area"].iloc[0] == pytest.approx(20.0, abs=1.5)
    assert any("Image.remap" in p for p in log)
    assert any("linkCollection" in p for p in log)


def test_era5_and_water_quality(log):
    with dry_run(responder, log):
        clim = E.era5_monthly(POLY, 600.0, "1984-01-01", "1984-03-01")
        wq = E.water_quality(POLY, 30, "2017-01-01", "2017-03-01")
    assert clim[0]["temperature_2m"] == 285.0
    assert wq[0]["ndci"] == -0.05 and wq[0]["lswt"] == 18.2


def test_map_layers_and_composites(log):
    with dry_run(responder, log):
        layers = E.map_layers(POLY)
        url = E.season_composite_tile(POLY, 2010, [6, 7, 8, 9])
        url_wrap = E.season_composite_tile(POLY, 1995, [12, 1, 2])
    assert set(layers) == {"occurrence", "transition", "recent"}
    assert url.startswith("https://tiles.dry")
    assert url_wrap.startswith("https://tiles.dry")


def test_report_images_and_timelapse(monkeypatch, log):
    class R:
        content = b"PNGDATA"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(E.requests, "get", lambda url, timeout=0: R())
    with dry_run(responder, log):
        imgs = E.report_images(POLY, POLY)
        gif, years = E.timelapse_gif(POLY, [2010, 2011, 2012], [6, 7, 8, 9])
    assert set(imgs) == {"lake", "catchment"}
    assert gif == b"PNGDATA"
    assert years == [2010, 2012]  # the year with zero images is dropped


def test_scales_and_chunks():
    assert E.hist_scale(20) == 30
    assert E.hist_scale(60_000) > 300
    assert 1000 <= E.climate_scale(600) <= 11132
    ch = E.month_chunks("2019-01-01", "2020-01-01", 5)
    assert [n for _, n in ch] == [5, 5, 2]
    assert E.basin_level(500) == 12 and E.basin_level(50_000) == 8 and E.basin_level(9e5) == 6


def test_bbox_of_polygon():
    b = E.bbox_of(POLY, 0.0)
    assert b[0] == pytest.approx(22.1, abs=1e-2) and b[3] == pytest.approx(38.56, abs=1e-2)
