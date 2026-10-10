"""Folium maps: site picker, outline and catchment, Earth Engine layers, year swipe."""

from __future__ import annotations

import folium
from folium import plugins

from .config import SEDIMENT, WATER

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_GRAY = ESRI.format("Canvas/World_Light_Gray_Base")
ESRI_TOPO = ESRI.format("World_Topo_Map")
ESRI_IMAGERY = ESRI.format("World_Imagery")
ESRI_GRAY_ATTR = "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
ESRI_TOPO_ATTR = "Tiles &copy; Esri, HERE, Garmin, FAO, NOAA, USGS, &copy; OpenStreetMap contributors"
ESRI_ATTR = "Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community"
EE_ATTR = "Google Earth Engine"
JRC_ATTR = "JRC Global Surface Water 1984–2021 (Pekel et al. 2016), Google Earth Engine"


def _bounds(geojson: dict) -> list[list[float]] | None:
    xs, ys = [], []

    def walk(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        elif isinstance(c, (list, tuple)):
            for cc in c:
                walk(cc)

    if not geojson:
        return None
    if geojson.get("type") == "GeometryCollection":
        for g in geojson.get("geometries", []):
            walk(g.get("coordinates", []))
    else:
        walk(geojson.get("coordinates", []))
    if not xs:
        return None
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


def base_map(lat: float, lon: float, zoom: int = 10, demo: bool = False) -> folium.Map:
    if demo:
        m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
        return m
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(ESRI_GRAY, attr=ESRI_GRAY_ATTR, name="Light grey map", max_zoom=16).add_to(m)
    folium.TileLayer(ESRI_TOPO, attr=ESRI_TOPO_ATTR, name="Topographic map", show=False).add_to(m)
    folium.TileLayer(ESRI_IMAGERY, attr=ESRI_ATTR, name="Satellite (Esri)", show=False).add_to(m)
    return m


def add_outline(m: folium.Map, footprint: dict | None, catchment: dict | None = None,
                outline_name: str = "Lake outline (water ever seen)") -> None:
    if catchment:
        folium.GeoJson(
            catchment, name="Catchment",
            style_function=lambda _f: {"color": SEDIMENT, "weight": 2, "fill": False,
                                       "dashArray": "7 5"},
            tooltip="Catchment (HydroBASINS)").add_to(m)
    if footprint:
        folium.GeoJson(
            footprint, name=outline_name,
            style_function=lambda _f: {"color": WATER, "weight": 2, "fillColor": WATER,
                                       "fillOpacity": 0.08},
            tooltip=outline_name).add_to(m)


def add_point(m: folium.Map, lat: float, lon: float, label: str) -> None:
    folium.CircleMarker([lat, lon], radius=6, color="#ffffff", weight=2, fill=True,
                        fill_color=WATER, fill_opacity=1, tooltip=label).add_to(m)


def add_ee_layer(m: folium.Map, url: str, name: str, show: bool = False,
                 attr: str = EE_ATTR, opacity: float = 1.0) -> folium.TileLayer:
    layer = folium.TileLayer(tiles=url, attr=attr, name=name, overlay=True, control=True,
                             show=show, opacity=opacity, max_zoom=18)
    layer.add_to(m)
    return layer


def fit(m: folium.Map, *geojsons: dict | None) -> None:
    boxes = [b for b in (_bounds(g) for g in geojsons if g) if b]
    if not boxes:
        return
    s = min(b[0][0] for b in boxes)
    w = min(b[0][1] for b in boxes)
    n = max(b[1][0] for b in boxes)
    e = max(b[1][1] for b in boxes)
    m.fit_bounds([[s, w], [n, e]], padding=(12, 12))


def site_map(site: dict, footprint: dict | None = None, catchment: dict | None = None,
             layers: dict[str, str] | None = None, demo: bool = False,
             show_catchment_extent: bool = False) -> folium.Map:
    m = base_map(site["lat"], site["lon"], zoom=11, demo=demo)
    if layers:
        if layers.get("recent"):
            add_ee_layer(m, layers["recent"], "Latest clear images (last 3 months)")
        if layers.get("occurrence"):
            add_ee_layer(m, layers["occurrence"], "How often water was seen, 1984–2021",
                         show=True, attr=JRC_ATTR, opacity=0.85)
        if layers.get("transition"):
            add_ee_layer(m, layers["transition"], "Change in water, 1984–2021", attr=JRC_ATTR)
    add_outline(m, footprint, catchment)
    add_point(m, site["lat"], site["lon"], site.get("name", "Selected point"))
    if show_catchment_extent and catchment:
        fit(m, catchment, footprint)
    elif footprint:
        fit(m, footprint)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m


def compare_map(left_url: str, right_url: str, footprint: dict | None,
                demo: bool = False) -> folium.Map:
    b = _bounds(footprint) if footprint else None
    center = [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2] if b else [0, 0]
    m = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)
    left = folium.TileLayer(tiles=left_url, attr=EE_ATTR, name="left", overlay=True, control=False)
    right = folium.TileLayer(tiles=right_url, attr=EE_ATTR, name="right", overlay=True, control=False)
    left.add_to(m)
    right.add_to(m)
    plugins.SideBySideLayers(layer_left=left, layer_right=right).add_to(m)
    if footprint:
        folium.GeoJson(footprint, name="Lake outline",
                       style_function=lambda _f: {"color": "#ffffff", "weight": 1.5,
                                                  "fill": False, "opacity": 0.9}).add_to(m)
        fit(m, footprint)
    return m
