"""All Google Earth Engine requests made by LakeDelta.

Each public function returns plain Python data (dicts, lists, bytes) so the
results can be cached and analysed without Earth Engine. Heavy requests are
split into chunks of months that run in parallel and report progress.

Pipeline (see README for the method and citations):
  1. Outline: connected "ever water" area of JRC Global Surface Water around
     the chosen point, seeded by the HydroLAKES polygon when one is nearby.
  2. Catchment: HydroBASINS sub-basins upstream of the lake.
  3. Monthly histograms of (water / land / no data) x occurrence:
       - JRC monthly water history, Mar 1984 - Dec 2021
       - our classification of Landsat 8/9 + Sentinel-2, 2019 - now
  4. ERA5-Land monthly precipitation, temperature and runoff over the catchment.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import requests

from . import config as C

ProgressFn = Callable[[int, int], None]

OCC_PALETTE = ["e3eef6", "9fc4e0", "5a97c7", "2b6ca3", "123f66"]
TRANSITION_PALETTE = ["ffffff", "0000ff", "22b14c", "d1102d", "99d9ea", "b5e61d",
                      "e6a1aa", "ff7f27", "ffc90e", "7f7f7f", "c3c3c3"]
TRANSITION_LABELS = {
    1: "Permanent", 2: "New permanent", 3: "Lost permanent", 4: "Seasonal",
    5: "New seasonal", 6: "Lost seasonal", 7: "Seasonal to permanent",
    8: "Permanent to seasonal", 9: "Ephemeral permanent", 10: "Ephemeral seasonal",
}


class EEFetchError(RuntimeError):
    """A request to Earth Engine failed in a way the user should hear about."""


def _ee():
    import ee  # imported lazily so demo mode never needs it

    return ee


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

_TRANSIENT = ("Too many concurrent", "429", "503", "502", "500", "Deadline", "timed out",
              "temporarily", "Connection", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


def get_info(obj: Any, retries: int = 3, what: str = "request") -> Any:
    """getInfo() with retries on transient errors and a readable failure."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            return obj.getInfo()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            transient = any(t.lower() in msg.lower() for t in _TRANSIENT)
            if attempt < retries and transient:
                time.sleep(delay)
                delay *= 2
                continue
            raise EEFetchError(f"Earth Engine could not complete the {what}: {msg}") from exc
    raise EEFetchError(f"Earth Engine could not complete the {what}.")


def run_chunks(build: Callable[[Any], Any], chunks: Sequence[Any], what: str,
               progress: ProgressFn | None = None, workers: int = 4) -> list[Any]:
    """Evaluate build(chunk) for every chunk in parallel; keep the input order."""
    results: list[Any] = [None] * len(chunks)
    done = 0
    if progress:
        progress(0, len(chunks))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks) or 1))) as pool:
        futures = {pool.submit(lambda c=c: get_info(build(c), what=what)): i
                   for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if progress:
                progress(done, len(chunks))
    return results


def month_chunks(start: str, end: str, size: int) -> list[tuple[str, int]]:
    """Split [start, end) into (chunk_start, n_months) pieces of at most size months."""
    months = pd.date_range(start, end, freq="MS", inclusive="left")
    out = []
    for i in range(0, len(months), size):
        piece = months[i:i + size]
        out.append((piece[0].strftime("%Y-%m-%d"), len(piece)))
    return out


def current_month_start(now: pd.Timestamp | None = None) -> str:
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    return now.to_period("M").to_timestamp().strftime("%Y-%m-%d")


def hist_scale(area_km2: float) -> int:
    """Pixel size for area histograms: 30 m unless the outline is very large."""
    if not area_km2 or area_km2 <= 400:
        return 30
    return int(math.ceil(math.sqrt(area_km2 * 1e6 / 4.5e5) / 10.0) * 10)


def climate_scale(area_km2: float) -> int:
    if not area_km2:
        return 1000
    return int(min(11132, max(1000, math.sqrt(area_km2 * 1e6 / 40000.0))))


# ---------------------------------------------------------------------------
# Outline of the lake
# ---------------------------------------------------------------------------

HYDROLAKES_FIELDS = ["Hylak_id", "Lake_name", "Country", "Lake_type", "Lake_area", "Vol_total",
                     "Depth_avg", "Res_time", "Elevation", "Wshd_area", "Pour_long", "Pour_lat",
                     "Grand_id"]


def find_hydrolake(lon: float, lat: float, max_dist_m: float = 2000) -> dict | None:
    """Nearest HydroLAKES polygon within max_dist_m of the point, or None."""
    ee = _ee()
    pt = ee.Geometry.Point([lon, lat])
    fc = ee.FeatureCollection(C.HYDROLAKES).filterBounds(pt.buffer(max_dist_m, 10))

    def with_distance(f):
        return f.set("dist_m", f.geometry().distance(pt, 10))

    best = (fc.map(with_distance).sort("dist_m").limit(1)
            .map(lambda f: f.simplify(60)))
    res = get_info(best, what="HydroLAKES lookup")
    feats = (res or {}).get("features") or []
    if not feats:
        return None
    f = feats[0]
    props = {k: v for k, v in (f.get("properties") or {}).items()
             if k in HYDROLAKES_FIELDS or k == "dist_m"}
    return {"geometry": f.get("geometry"), "properties": props}


def occurrence(ee):
    """JRC occurrence (0-100) with a plain mask.

    The catalogue notes that the occurrence band carries partial masks equal to
    its value, which would down-weight rarely wet pixels in weighted reductions.
    Reductions on it therefore use unweighted reducers.
    """
    return ee.Image(C.GSW_IMAGE).select("occurrence").unmask(0).toInt16()


def _vector_scale(extent_m: float) -> int:
    return int(max(30, math.ceil(extent_m / 1581.0 / 10.0) * 10))


def delineate(lon: float, lat: float, *, seed_geojson: dict | None = None,
              seed_area_km2: float | None = None, radius_m: float = 10_000,
              margin_m: float | None = None, occ_min: int = 1) -> dict:
    """Connected area that has ever held water (JRC 1984-2021) around the site.

    With a seed polygon (HydroLAKES or an uploaded outline) the search is the
    seed grown by ``margin_m`` and every water patch touching the seed is kept.
    Without one, the search is a circle of ``radius_m`` and the patch under the
    point (or the nearest sizeable one) is kept. Large search areas are
    vectorised on a coarser grid using the maximum, so narrow arms survive.
    """
    ee = _ee()
    point = ee.Geometry.Point([lon, lat])
    occ = occurrence(ee)
    water = occ.gte(occ_min).toByte()

    if seed_geojson is not None:
        seed = ee.Geometry(seed_geojson)
        side = math.sqrt(max(seed_area_km2 or 1.0, 0.01)) * 1000.0
        margin_m = margin_m or min(30_000.0, max(1_500.0, 0.25 * side))
        region = seed.buffer(margin_m, 100)
        extent = side + 2 * margin_m
        scale = _vector_scale(extent)
        touch = seed
    else:
        region = point.buffer(radius_m, 50)
        extent = 2 * radius_m
        scale = _vector_scale(extent)
        touch = point.buffer(max(2 * scale, 150), 10)

    proj = ee.Projection("EPSG:4326").atScale(scale)
    if scale > 30:
        mask = water.reduceResolution(ee.Reducer.max(), True, 1024).reproject(proj)
    else:
        mask = water
    vectors = mask.toInt().selfMask().reduceToVectors(
        geometry=region, scale=scale, crs="EPSG:4326", geometryType="polygon",
        eightConnected=True, labelProperty="water", bestEffort=True, maxPixels=int(1e10),
        tileScale=4)
    vectors = vectors.map(lambda f: f.set("area_m2", f.geometry().area(scale)))
    hits = vectors.filterBounds(touch)
    nearest = (vectors.filter(ee.Filter.gte("area_m2", 50_000))
               .map(lambda f: f.set("dist_m", f.geometry().distance(point, scale)))
               .sort("dist_m").limit(1))
    chosen = ee.FeatureCollection(ee.Algorithms.If(hits.size().gt(0), hits, nearest))
    geom = (chosen.geometry(scale)
            .buffer(max(60, scale), scale / 2)
            .simplify(max(15, scale / 2)))
    # How far the outline runs along the edge of the search area. A river leaving
    # the lake crosses the edge over a few hundred metres; a lake cut off by the
    # edge runs along it for kilometres.
    inner = region.buffer(-2 * scale, scale)
    edge = region.difference(inner, scale)
    crossing_m = geom.intersection(edge, scale).area(scale).divide(2 * scale)
    result = ee.Dictionary({
        "geometry": geom,
        "n_patches": chosen.size(),
        "used_nearest": hits.size().eq(0),
        "edge_crossing_m": ee.Algorithms.If(chosen.size().gt(0), crossing_m, 0),
        "area_m2": geom.area(scale),
    })
    out = get_info(result, what="lake outline")
    out["truncated"] = bool((out.get("edge_crossing_m") or 0) > max(1500.0, 6 * scale))
    out["vector_scale_m"] = scale
    out["margin_m"] = margin_m
    out["radius_m"] = radius_m if seed_geojson is None else None
    return out


def footprint_stats(geojson: dict) -> dict:
    """Area of the outline, and how much of it was ever / permanently water."""
    ee = _ee()
    geom = ee.Geometry(geojson)
    area_km2 = get_info(geom.area(10), what="outline area") / 1e6
    scale = hist_scale(area_km2)
    occ = occurrence(ee)
    pa = ee.Image.pixelArea()
    img = (pa.multiply(occ.gte(1)).rename("ever")
           .addBands(pa.multiply(occ.gte(90)).rename("permanent")))
    d = get_info(img.reduceRegion(ee.Reducer.sum().unweighted(), geom, scale, None, None, True,
                                  int(1e10), 4),
                 what="outline statistics")
    return {"area_km2": area_km2, "ever_water_km2": (d.get("ever") or 0) / 1e6,
            "permanent_km2": (d.get("permanent") or 0) / 1e6, "scale_m": scale}


def outline_for_site(site: dict, progress: Callable[[str], None] | None = None) -> dict:
    """Find the lake outline for a site dict (name, lat, lon, radius_km, optional geojson)."""
    say = progress or (lambda _msg: None)
    warnings: list[str] = []
    hydrolake = None
    lat, lon = float(site["lat"]), float(site["lon"])

    if site.get("geojson"):
        say("Using the uploaded outline")
        geom = site["geojson"]
        stats = footprint_stats(geom)
        return {"footprint": geom, "hydrolake": None,
                "info": {**stats, "method": "uploaded outline", "truncated": False,
                         "warnings": warnings}}

    say("Looking up the lake in HydroLAKES")
    try:
        hydrolake = find_hydrolake(lon, lat)
    except EEFetchError as exc:
        warnings.append(f"HydroLAKES lookup failed, so the outline was traced from the point "
                        f"alone ({exc}).")

    say("Tracing the outline from 38 years of Landsat water maps")
    radius = float(site.get("radius_km") or 10) * 1000.0
    out = None
    if hydrolake and hydrolake.get("geometry"):
        props = hydrolake["properties"]
        margin = None
        for _ in range(3):
            out = delineate(lon, lat, seed_geojson=hydrolake["geometry"],
                            seed_area_km2=props.get("Lake_area"), margin_m=margin)
            if not out.get("truncated"):
                break
            margin = min((out.get("margin_m") or 1500) * 2.2, 120_000)
        method = "JRC water ever seen, around the HydroLAKES polygon"
    else:
        for _ in range(3):
            out = delineate(lon, lat, radius_m=radius)
            if not out.get("truncated"):
                break
            radius = min(radius * 1.8, 400_000)
        method = "JRC water ever seen, within the search radius"
    if not out or not out.get("n_patches"):
        raise EEFetchError("No water has been recorded near this point since 1984. "
                           "Move the point onto the lake or enlarge the search radius.")
    if out.get("truncated"):
        warnings.append("The outline still touches the edge of the search area, so part of the "
                        "lake may be missing. Increase the search radius and analyse again.")
    if out.get("used_nearest"):
        warnings.append("The point is not on water, so the nearest lake was used.")

    say("Measuring the outline")
    stats = footprint_stats(out["geometry"])
    info = {**stats, "method": method, "truncated": bool(out.get("truncated")),
            "vector_scale_m": out.get("vector_scale_m"), "warnings": warnings}
    return {"footprint": out["geometry"], "hydrolake": hydrolake, "info": info}


# ---------------------------------------------------------------------------
# Catchment
# ---------------------------------------------------------------------------

def upstream_ids(seeds: Iterable[Any], table: Iterable[Sequence[Any]]) -> set:
    """All basin ids draining (directly or not) into any seed basin."""
    children: dict[Any, list[Any]] = defaultdict(list)
    for hid, nxt in table:
        children[nxt].append(hid)
    found = set(seeds)
    stack = list(found)
    while stack:
        cur = stack.pop()
        for child in children.get(cur, ()):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def basin_level(upstream_km2: float | None) -> int:
    if not upstream_km2 or upstream_km2 < 20_000:
        return 12
    if upstream_km2 < 400_000:
        return 8
    return 6


def trace_catchment(footprint: dict, pour_point: tuple[float, float] | None = None,
                    watershed_km2: float | None = None) -> dict:
    ee = _ee()
    fp = ee.Geometry(footprint)
    if watershed_km2 is None:
        lvl12 = ee.FeatureCollection(C.HYDROBASINS.format(level=12)).filterBounds(fp)
        watershed_km2 = get_info(lvl12.aggregate_max("UP_AREA"), what="catchment size")
    level = basin_level(watershed_km2)
    basins = ee.FeatureCollection(C.HYDROBASINS.format(level=level))
    seeds_fc = basins.filterBounds(fp)
    if pour_point:
        seeds_fc = seeds_fc.merge(basins.filterBounds(ee.Geometry.Point(list(pour_point))))
    seed_rows = get_info(seeds_fc.reduceColumns(ee.Reducer.toList(3),
                                                ["HYBAS_ID", "UP_AREA", "MAIN_BAS"]).get("list"),
                         what="catchment seeds")
    if not seed_rows:
        raise EEFetchError("No HydroBASINS sub-basin overlaps this lake.")
    seeds = {r[0] for r in seed_rows}
    max_up = max(float(r[1] or 0) for r in seed_rows)
    mains = sorted({r[2] for r in seed_rows})
    radius_km = max(60.0, 3.0 * math.sqrt(max_up))
    cand = (basins.filter(ee.Filter.inList("MAIN_BAS", mains))
            .filter(ee.Filter.lte("UP_AREA", max_up))
            .filterBounds(fp.centroid(100).buffer(radius_km * 1000.0, 1000)))
    table = get_info(cand.reduceColumns(ee.Reducer.toList(2), ["HYBAS_ID", "NEXT_DOWN"]).get("list"),
                     what="river network")
    ids = sorted(upstream_ids(seeds, table or []))
    simp = max(100.0, math.sqrt(max_up) * 1000.0 / 200.0)
    union = basins.filter(ee.Filter.inList("HYBAS_ID", ids)).union(simp).geometry(simp)
    union = union.union(fp, simp).simplify(simp)
    res = get_info(ee.Dictionary({"geometry": union, "area_m2": union.area(simp)}),
                   what="catchment outline")
    return {"geometry": res["geometry"],
            "info": {"area_km2": res["area_m2"] / 1e6, "level": level, "n_basins": len(ids),
                     "method": f"HydroBASINS level {level}, upstream of the lake"}}


def local_climate_region(footprint: dict, buffer_km: float = 20.0) -> dict:
    ee = _ee()
    geom = ee.Geometry(footprint).buffer(buffer_km * 1000.0, 500).simplify(500)
    res = get_info(ee.Dictionary({"geometry": geom, "area_m2": geom.area(500)}),
                   what="climate area")
    return {"geometry": res["geometry"],
            "info": {"area_km2": res["area_m2"] / 1e6, "level": None, "n_basins": 0,
                     "method": f"{buffer_km:.0f} km around the lake (catchment unavailable)"}}


# ---------------------------------------------------------------------------
# Optical imagery (Landsat 5/7/8/9 and Sentinel-2) on a common band set
# ---------------------------------------------------------------------------

_LANDSAT_BANDS = {
    "L5": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "ST_B6", False),
    "L7": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "ST_B6", False),
    "L8": (["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], "ST_B10", True),
    "L9": (["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], "ST_B10", True),
}
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]


def landsat(sensor: str, region, start: str, end: str):
    ee = _ee()
    bands, thermal, has_cirrus = _LANDSAT_BANDS[sensor]
    bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
    if has_cirrus:
        bits |= 1 << 2

    def prep(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(bits).eq(0).And(img.select("QA_RADSAT").eq(0))
        sr = img.select(bands).multiply(0.0000275).add(-0.2).rename(COMMON)
        lst = img.select(thermal).multiply(0.00341802).add(149.0 - 273.15).rename("lst")
        out = sr.addBands(lst).updateMask(clear).toFloat() # <-- ADD .toFloat() HERE
        return ee.Image(out.set("system:time_start", img.get("system:time_start"))
                        .set("sensor", sensor))

    return (ee.ImageCollection(C.LANDSAT[sensor]).filterBounds(region)
            .filterDate(start, end).map(prep))


def sentinel2(region, start: str, end: str):
    ee = _ee()
    cs = ee.ImageCollection(C.CLOUD_SCORE_PLUS)

    def prep(img):
        clear = img.select("cs_cdf").gte(C.CS_PLUS_THRESHOLD).And(img.select("SCL").neq(11))
        sr = (img.select(["B2", "B3", "B4", "B8", "B11", "B12", "B5"]).divide(10000)
              .rename(COMMON + ["rededge1"]))
        return ee.Image(sr.updateMask(clear).toFloat()
                        .set("system:time_start", img.get("system:time_start"))
                        .set("sensor", "S2"))

    return (ee.ImageCollection(C.S2_SR).filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 85))
            .linkCollection(cs, ["cs_cdf"]).map(prep))


def add_water(occ):
    """Per-image water classification (Zou et al. 2018) with a guard outside the
    historic water extent: there, only strongly water-like pixels count."""

    def fn(img):
        mndwi = img.normalizedDifference(["green", "swir1"])
        ndvi = img.normalizedDifference(["nir", "red"])
        evi = img.expression("2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)",
                             {"N": img.select("nir"), "R": img.select("red"),
                              "B": img.select("blue")})
        water = mndwi.gt(ndvi).Or(mndwi.gt(evi)).And(evi.lt(0.1))
        water = water.And(occ.gte(1).Or(mndwi.gt(0.3)))
        return img.addBands(water.rename("water").toFloat())

    return fn


def recent_collection(region, start: str, end: str):
    """Landsat 8/9 + Sentinel-2, harmonized bands, cloud-masked."""
    return (landsat("L8", region, start, end)
            .merge(landsat("L9", region, start, end))
            .merge(sentinel2(region, start, end).select(COMMON)))


def rgb_collection_for_year(region, year: int):
    """Sensors that give the cleanest true-colour composite for a calendar year."""
    start, end = f"{year - 1}-01-01", f"{year + 2}-01-01"  # generous; caller filters dates
    if year <= 1998:
        col = landsat("L5", region, start, end)
    elif year <= 2002:
        col = landsat("L5", region, start, end).merge(landsat("L7", region, start, end))
    elif year <= 2011:
        col = landsat("L5", region, start, end)
    elif year == 2012:
        col = landsat("L7", region, start, end)
    elif year <= 2017:
        col = landsat("L8", region, start, end)
        if year == 2017:
            col = col.merge(sentinel2(region, start, end).select(COMMON))
    else:
        col = recent_collection(region, start, end)
    return col.select(["red", "green", "blue"])


# ---------------------------------------------------------------------------
# Monthly histograms
# ---------------------------------------------------------------------------

def _hist_reduce(cls_img, occ, geom, scale: int):
    ee = _ee()
    code = cls_img.multiply(101).add(occ).toInt16().rename("code")
    img = ee.Image.pixelArea().rename("area").addBands(code)
    return img.reduceRegion(ee.Reducer.sum().unweighted().group(1, "code"), geom, scale, None,
                            None, True, int(1e10), 4).get("groups")


def jrc_histograms(footprint: dict, scale: int, start: str = C.GSW_FIRST_MONTH,
                   end: str = C.GSW_END_EXCLUSIVE, progress: ProgressFn | None = None,
                   chunk_months: int = 48) -> list[dict]:
    ee = _ee()

    def build(chunk):
        c_start, n = chunk
        c_end = (pd.Timestamp(c_start) + pd.DateOffset(months=n)).strftime("%Y-%m-%d")
        geom = ee.Geometry(footprint)
        occ = occurrence(ee)
        col = ee.ImageCollection(C.GSW_MONTHLY).filterDate(c_start, c_end)

        def per_image(img):
            w = img.select("water").unmask(0)
            cls = w.remap([0, 1, 2], [2, 0, 1])
            return ee.Feature(None, {"t": img.get("system:time_start"),
                                     "groups": _hist_reduce(cls, occ, geom, scale)})

        return ee.FeatureCollection(col.map(per_image))

    chunks = month_chunks(start, end, chunk_months)
    results = run_chunks(build, chunks, "JRC monthly water history", progress)
    return [f["properties"] for r in results for f in (r or {}).get("features", [])]


def optical_histograms(footprint: dict, scale: int, start: str = C.OPTICAL_RECORD_START,
                       end: str | None = None, progress: ProgressFn | None = None,
                       chunk_months: int = 12) -> list[dict]:
    ee = _ee()
    end = end or current_month_start()

    def build(chunk):
        c_start, n = chunk
        c_end = (pd.Timestamp(c_start) + pd.DateOffset(months=n)).strftime("%Y-%m-%d")
        geom = ee.Geometry(footprint)
        occ = occurrence(ee)
        col = recent_collection(geom, c_start, c_end).map(add_water(occ)).select("water")
        dummy = ee.ImageCollection([ee.Image.constant(0).toFloat().rename("water")
                                    .updateMask(ee.Image.constant(0))])

        def per_month(i):
            d0 = ee.Date(c_start).advance(i, "month")
            d1 = d0.advance(1, "month")
            month = col.filterDate(d0, d1)
            frac = month.merge(dummy).mean()
            cls = frac.gte(0.5).unmask(2)
            return ee.Feature(None, {"t": d0.millis(), "n_scenes": month.size(),
                                     "groups": _hist_reduce(cls, occ, geom, scale)})

        return ee.FeatureCollection(ee.List.sequence(0, n - 1).map(per_month))

    chunks = month_chunks(start, end, chunk_months)
    if not chunks:
        return []
    results = run_chunks(build, chunks, "Landsat and Sentinel-2 water maps", progress)
    return [f["properties"] for r in results for f in (r or {}).get("features", [])]


# ---------------------------------------------------------------------------
# Climate (ERA5-Land)
# ---------------------------------------------------------------------------

ERA5_BANDS = ["total_precipitation_sum", "temperature_2m", "runoff_sum"]


def era5_monthly(region: dict, area_km2: float, start: str = "1984-01-01", end: str | None = None,
                 progress: ProgressFn | None = None, chunk_months: int = 120) -> list[dict]:
    ee = _ee()
    end = end or current_month_start()
    scale = climate_scale(area_km2)

    def build(chunk):
        c_start, n = chunk
        c_end = (pd.Timestamp(c_start) + pd.DateOffset(months=n)).strftime("%Y-%m-%d")
        geom = ee.Geometry(region)
        col = ee.ImageCollection(C.ERA5_LAND_MONTHLY).filterDate(c_start, c_end).select(ERA5_BANDS)

        def per_image(img):
            d = img.reduceRegion(ee.Reducer.mean(), geom, scale, None, None, True, int(1e10), 2)
            return ee.Feature(None, d).set("t", img.get("system:time_start"))

        return ee.FeatureCollection(col.map(per_image))

    results = run_chunks(build, month_chunks(start, end, chunk_months), "ERA5-Land climate",
                         progress)
    return [f["properties"] for r in results for f in (r or {}).get("features", [])]


# ---------------------------------------------------------------------------
# Water-quality indicators (on demand)
# ---------------------------------------------------------------------------

def water_quality(footprint: dict, scale: int, start: str = "2013-04-01", end: str | None = None,
                  progress: ProgressFn | None = None, chunk_months: int = 24) -> list[dict]:
    """Monthly medians over the permanent-water core:
    NDTI (turbidity) and NDCI (chlorophyll-a) from Sentinel-2,
    surface temperature from Landsat 8/9 thermal bands."""
    ee = _ee()
    end = end or current_month_start()

    def build(chunk):
        c_start, n = chunk
        c_end = (pd.Timestamp(c_start) + pd.DateOffset(months=n)).strftime("%Y-%m-%d")
        geom = ee.Geometry(footprint)
        occ = occurrence(ee)
        core = occ.gte(90).focalMin(60, "circle", "meters")
        classify = add_water(occ)

        def s2_idx(img):
            img = classify(img)
            keep = img.select("water").eq(1).And(core)
            ndti = img.normalizedDifference(["red", "green"]).rename("ndti")
            ndci = img.normalizedDifference(["rededge1", "red"]).rename("ndci")
            return ndti.addBands(ndci).updateMask(keep)

        def ls_idx(img):
            img = classify(img)
            keep = img.select("water").eq(1).And(core)
            return img.select("lst").updateMask(keep)

        s2 = sentinel2(geom, c_start, c_end).map(s2_idx)
        ls = landsat("L8", geom, c_start, c_end).merge(landsat("L9", geom, c_start, c_end)).map(ls_idx)
        zero = ee.Image.constant(0)
        dummy_s2 = ee.ImageCollection([ee.Image.constant([0, 0]).toFloat()
                                       .rename(["ndti", "ndci"]).updateMask(zero)])
        dummy_ls = ee.ImageCollection([zero.toFloat().rename("lst").updateMask(zero)])
        reducer = ee.Reducer.median().combine(ee.Reducer.count(), None, True)

        def per_month(i):
            d0 = ee.Date(c_start).advance(i, "month")
            d1 = d0.advance(1, "month")
            img = (s2.filterDate(d0, d1).merge(dummy_s2).median()
                   .addBands(ls.filterDate(d0, d1).merge(dummy_ls).median()))
            stats = img.reduceRegion(reducer, geom, scale, None, None, True, int(1e10), 4)
            return ee.Feature(None, stats).set("t", d0.millis())

        return ee.FeatureCollection(ee.List.sequence(0, n - 1).map(per_month))

    results = run_chunks(build, month_chunks(start, end, chunk_months), "water-quality indicators",
                         progress)
    rows = []
    for r in results:
        for f in (r or {}).get("features", []):
            p = f["properties"]
            rows.append({
                "t": p.get("t"),
                "ndti": p.get("ndti_median") if (p.get("ndti_count") or 0) >= 20 else None,
                "ndci": p.get("ndci_median") if (p.get("ndci_count") or 0) >= 20 else None,
                "lswt": p.get("lst_median") if (p.get("lst_count") or 0) >= 10 else None,
            })
    return rows


# ---------------------------------------------------------------------------
# Imagery for maps, the report and the timelapse
# ---------------------------------------------------------------------------

def bbox_of(geojson: dict, pad: float = 0.08) -> list[float]:
    xs, ys = [], []

    def walk(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        elif isinstance(c, (list, tuple)):
            for cc in c:
                walk(cc)

    if geojson.get("type") == "GeometryCollection":
        for g in geojson.get("geometries", []):
            walk(g.get("coordinates", []))
    else:
        walk(geojson.get("coordinates", []))
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) * pad + 1e-3, (y1 - y0) * pad + 1e-3
    return [x0 - dx, y0 - dy, x1 + dx, y1 + dy]


def _rect(bbox):
    ee = _ee()
    return ee.Geometry.Rectangle(bbox, None, False)


# def _outline(geojson: dict, color: str, width: int = 2):
#     ee = _ee()
#     fc = ee.FeatureCollection([ee.Feature(ee.Geometry(geojson))])
#     return ee.Image().byte().paint(fc, 1, width).visualize(palette=[color])
  
def _outline(geojson: dict, color: str, width: int = 2):
    ee = _ee()
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(geojson))])
    
    # We must explicitly cast this to a 3-band RGB visualizer
    # so it can be safely blended with the true-color imagery.
    painted = ee.Image().byte().paint(fc, 1, width)
    
    return painted.visualize(
        palette=[color],
        min=0,
        max=1,
        forceRgbOutput=True # <--- THIS IS THE MAGIC FIX
    )

def _true_color(col):
    return col.median().visualize(bands=["red", "green", "blue"], min=0.0, max=0.3, gamma=1.3)


def season_filter(season_yr: int, months: Sequence[int]):
    ee = _ee()
    from .analysis import season_date_ranges

    filters = [ee.Filter.date(a, b) for a, b in season_date_ranges(season_yr, months)]
    return ee.Filter.Or(*filters) if len(filters) > 1 else filters[0]


def tile_url(image) -> str:
    return image.getMapId()["tile_fetcher"].url_format


def map_layers(footprint: dict) -> dict[str, str]:
    ee = _ee()
    gsw = ee.Image(C.GSW_IMAGE)
    occ = gsw.select("occurrence")
    trans = gsw.select("transition")
    region = _rect(bbox_of(footprint, 0.3))
    end = current_month_start()
    start = (pd.Timestamp(end) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    recent = recent_collection(region, start, end).select(["red", "green", "blue"])
    return {
        "occurrence": tile_url(occ.updateMask(occ.gt(0)).visualize(min=0, max=100,
                                                                   palette=OCC_PALETTE)),
        "transition": tile_url(trans.updateMask(trans.gt(0)).visualize(
            min=0, max=10, palette=TRANSITION_PALETTE)),
        "recent": tile_url(_true_color(recent).clip(region)),
    }


def season_composite_tile(footprint: dict, season_yr: int, months: Sequence[int]) -> str:
    region = _rect(bbox_of(footprint, 0.3))
    col = rgb_collection_for_year(region, season_yr).filter(season_filter(season_yr, months))
    return tile_url(_true_color(col).clip(region))


def _download(url: str, what: str) -> bytes:
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        return r.content
    except requests.RequestException as exc:
        raise EEFetchError(f"Could not download the {what}: {exc}") from exc


def report_images(footprint: dict, catchment: dict | None, dims: int = 900) -> dict[str, bytes]:
    """PNG thumbnails for the PDF report: the lake, and the lake in its catchment."""
    ee = _ee()
    out: dict[str, bytes] = {}
    glo = ee.ImageCollection(C.COPERNICUS_DEM).select("DEM")
    dem = glo.mosaic().setDefaultProjection(glo.first().projection())
    shade = ee.Terrain.hillshade(dem).visualize(min=60, max=255, palette=["8b9296", "f3f5f5"])
    occ = ee.Image(C.GSW_IMAGE).select("occurrence")
    occ_vis = occ.updateMask(occ.gt(0)).visualize(min=0, max=100, palette=OCC_PALETTE)

    lake_box = bbox_of(footprint, 0.12)
    lake_img = shade.blend(occ_vis).blend(_outline(footprint, "14232E", 2))
    params = {"region": _rect(lake_box), "dimensions": dims, "format": "png", "crs": "EPSG:3857"}
    out["lake"] = _download(lake_img.getThumbURL(params), "lake map")

    if catchment:
        c_box = bbox_of(catchment, 0.06)
        fp_fill = (ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ee.Geometry(footprint))]), 1)
                   .visualize(palette=["2b6ca3"]))
        c_img = shade.blend(fp_fill).blend(_outline(catchment, "a86a2c", 3))
        params = {"region": _rect(c_box), "dimensions": dims, "format": "png", "crs": "EPSG:3857"}
        out["catchment"] = _download(c_img.getThumbURL(params), "catchment map")
    return out


def timelapse_gif(footprint: dict, years: Sequence[int], months: Sequence[int],
                  dims: int = 640, fps: float = 2.0) -> tuple[bytes, list[int]]:
    """Animated GIF of seasonal true-colour composites, one frame per season-year."""
    ee = _ee()
    region = _rect(bbox_of(footprint, 0.12))
    frames, sizes = [], []
    for y in years:
        col = rgb_collection_for_year(region, int(y)).filter(season_filter(int(y), months))
        sizes.append(col.size())
        frames.append(ee.Image(_true_color(col).blend(_outline(footprint, "ffffff", 1))
                               .set("year", int(y))))
    counts = get_info(ee.List(sizes), what="timelapse image counts")
    keep = [i for i, n in enumerate(counts) if n and n > 0]
    if not keep:
        raise EEFetchError("No clear imagery was found for the chosen season.")
    video = ee.ImageCollection([frames[i] for i in keep])
    params = {"dimensions": dims, "region": region, "framesPerSecond": fps, "crs": "EPSG:3857"}
    url = get_info_url(video, params)
    return _download(url, "timelapse"), [int(years[i]) for i in keep]


def get_info_url(collection, params: dict) -> str:
    try:
        return collection.getVideoThumbURL(params)
    except Exception as exc:  # noqa: BLE001
        raise EEFetchError(f"Earth Engine could not render the timelapse: {exc}") from exc
