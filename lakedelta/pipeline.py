"""Fetch everything one lake needs, from Earth Engine or from the demo generator.

These functions are UI-agnostic; app.py wraps them in Streamlit caches and
passes progress callbacks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import requests

from . import demo
from . import ee_data as E
from .config import USER_AGENT


@dataclass
class Progress:
    say: Callable[[str], None]
    chunks: Callable[[str], Callable[[int, int], None]]


def quiet_progress() -> Progress:
    return Progress(lambda _m: None, lambda _l: (lambda _d, _t: None))


def site_key(site: dict) -> str:
    """Stable identity of a site for caching and for matching results to the selection."""
    core = {
        "lat": round(float(site["lat"]), 4), "lon": round(float(site["lon"]), 4),
        "radius_km": round(float(site.get("radius_km") or 10), 1),
        "geojson": site.get("geojson"),
    }
    return json.dumps(core, sort_keys=True)


def fetch_bundle(site: dict, progress: Progress | None = None) -> dict:
    """Outline, catchment, monthly water histograms and climate for one site."""
    p = progress or quiet_progress()
    outline = E.outline_for_site(site, p.say)
    footprint = outline["footprint"]
    finfo = outline["info"]
    hl = outline.get("hydrolake")
    props = (hl or {}).get("properties") or {}

    p.say("Tracing the catchment upstream of the lake")
    pour = None
    if props.get("Pour_long") is not None and props.get("Pour_lat") is not None:
        pour = (float(props["Pour_long"]), float(props["Pour_lat"]))
    try:
        catch = E.trace_catchment(footprint, pour, props.get("Wshd_area"))
    except E.EEFetchError as exc:
        finfo.setdefault("warnings", []).append(
            f"The catchment could not be traced, so climate is averaged around the lake ({exc}).")
        catch = E.local_climate_region(footprint)

    scale = E.hist_scale(finfo.get("area_km2") or 0)
    p.say("Reading the monthly Landsat water record, 1984–2021")
    jrc = E.jrc_histograms(footprint, scale, progress=p.chunks("Landsat water record 1984–2021"))
    p.say("Classifying Landsat 8/9 and Sentinel-2 images since 2019")
    opt = E.optical_histograms(footprint, scale, progress=p.chunks("Landsat 8/9 and Sentinel-2"))
    p.say("Averaging ERA5-Land climate over the catchment")
    clim = E.era5_monthly(catch["geometry"], catch["info"]["area_km2"],
                          progress=p.chunks("ERA5-Land climate"))
    return {
        "demo": False,
        "site": site,
        "footprint": footprint,
        "footprint_info": finfo,
        "hydrolake": props or None,
        "catchment": catch["geometry"],
        "catchment_info": catch["info"],
        "jrc": jrc,
        "optical": opt,
        "climate": clim,
        "meta": {"hist_scale_m": scale, "climate_scale_m": E.climate_scale(catch["info"]["area_km2"]),
                 "jrc_months": len(jrc), "optical_months": len(opt)},
    }


def preview_outline(site: dict) -> dict:
    return E.outline_for_site(site)


def demo_bundle() -> dict:
    return demo.demo_bundle()


# ---------------------------------------------------------------------------
# Place search (OpenStreetMap Nominatim)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geocode(query: str, limit: int = 6) -> list[dict]:
    if not query or len(query.strip()) < 3:
        return []
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query.strip(), "format": "jsonv2", "limit": limit, "accept-language": "en"},
        headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    out = []
    for item in r.json():
        lat, lon = float(item["lat"]), float(item["lon"])
        bbox = [float(v) for v in item.get("boundingbox", [lat, lat, lon, lon])]  # s, n, w, e
        half_diag = _haversine_km(bbox[0], bbox[2], bbox[1], bbox[3]) / 2
        radius = max(3.0, min(300.0, half_diag * 1.15 + 3.0))
        kind = item.get("type", "").replace("_", " ")
        out.append({
            "name": item.get("name") or item.get("display_name", "").split(",")[0],
            "display": item.get("display_name", ""),
            "lat": lat, "lon": lon, "radius_km": round(radius, 1),
            "is_water": item.get("category") in ("natural", "water", "waterway")
            or kind in ("water", "reservoir", "lake", "lagoon", "basin"),
            "kind": kind,
        })
    out.sort(key=lambda d: (not d["is_water"],))
    return out


# ---------------------------------------------------------------------------
# Uploaded outlines
# ---------------------------------------------------------------------------

def parse_outline(raw: bytes) -> dict:
    """GeoJSON bytes -> single (multi)polygon geometry in lon/lat, simplified."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    obj = json.loads(raw.decode("utf-8"))
    geoms = []
    if obj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
    elif obj.get("type") == "Feature":
        geoms = [shape(obj["geometry"])]
    else:
        geoms = [shape(obj)]
    polys = [g for g in geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    if not polys:
        raise ValueError("The file has no polygon. Upload a GeoJSON polygon of the lake.")
    geom = unary_union(polys)
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= 180 and -90 <= miny <= 90 and -180 <= maxx <= 180 and -90 <= maxy <= 90):
        raise ValueError("Coordinates must be longitude and latitude (EPSG:4326).")
    geom = geom.simplify(0.0002, preserve_topology=True)
    pt = geom.representative_point()
    return {"geometry": json.loads(json.dumps(mapping(geom))), "lat": pt.y, "lon": pt.x}


class TTLCache:
    """Small shared cache for results that need progress reporting while computing."""

    def __init__(self, ttl_s: float, max_entries: int):
        import threading
        import time

        self._time = time.monotonic
        self._lock = threading.Lock()
        self.ttl = ttl_s
        self.max_entries = max_entries
        self._data: dict = {}

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            t, value = item
            if self._time() - t > self.ttl:
                del self._data[key]
                return None
            return value

    def put(self, key, value):
        with self._lock:
            self._data[key] = (self._time(), value)
            while len(self._data) > self.max_entries:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest]
