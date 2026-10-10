"""Synthetic data for demo mode and tests.

The generator produces exactly what the Earth Engine layer returns (monthly
occurrence histograms, ERA5-style climate records, GeoJSON outlines), so demo
mode runs the real analysis code end to end. Every value is invented; the app
labels demo results as synthetic wherever they appear.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DEMO_NAME = "Demo Reservoir"
DEMO_LAT, DEMO_LON = 10.0, 10.0  # arbitrary; the demo map has no basemap

N_OCC = 101


def _occurrence_areas() -> np.ndarray:
    """km² of the footprint in each occurrence bin 0..100."""
    a = np.zeros(N_OCC)
    a[0] = 6.0                                    # dry buffer inside the outline
    o = np.arange(1, 95)
    dyn = 0.6 + 0.9 * np.exp(-((o - 70) / 28.0) ** 2)  # drawdown zone
    a[1:95] = dyn / dyn.sum() * 16.0
    a[95:101] = 9.0 / 6.0                          # permanent core
    return a


def _water_fraction_by_bin(area_km2: float, a: np.ndarray) -> np.ndarray:
    """Fill bins from the highest occurrence down until the area is reached."""
    frac = np.zeros(N_OCC)
    remaining = area_km2
    for o in range(100, 0, -1):
        if remaining <= 0:
            break
        take = min(a[o], remaining)
        frac[o] = take / a[o] if a[o] > 0 else 0
        remaining -= take
    return frac


def _true_series(rng: np.random.Generator, dates: pd.DatetimeIndex):
    n = len(dates)
    m = dates.month.to_numpy()
    # Mediterranean precipitation: wet winters, dry summers (mm/month)
    clim_p = 70 + 55 * np.cos((m - 1) / 12 * 2 * np.pi)
    clim_p = np.clip(clim_p, 6, None)
    yearly = np.zeros(n)
    shock = 0.0
    for i in range(n):
        if i == 0 or dates[i].month == 10:
            shock = 0.55 * shock + rng.normal(0, 0.28)
        yearly[i] = shock
    yr = dates.year.to_numpy() + (m - 1) / 12
    droughts = (
        -0.45 * ((yr > 1989.5) & (yr < 1993.8))
        - 0.35 * ((yr > 2007.2) & (yr < 2008.9))
        - 0.55 * ((yr > 2021.8) & (yr < 2025.6))
        + 0.30 * ((yr > 2002.5) & (yr < 2004.2))
    )
    mult = np.exp(yearly + droughts + rng.normal(0, 0.35, n))
    precip = clim_p * mult
    temp = 16.5 + 9.5 * np.sin((m - 4.2) / 12 * 2 * np.pi) + 0.035 * (yr - 1984) + rng.normal(0, 0.8, n)
    runoff = np.clip(0.35 * precip * np.clip(mult, 0.2, None) - 8, 0, None)

    anom = pd.Series(precip - clim_p, index=dates).rolling(18, min_periods=1).sum().to_numpy()
    seasonal = 1.8 * np.cos((m - 4.5) / 12 * 2 * np.pi)
    area = 19.0 + seasonal + 0.0105 * anom - 0.06 * (yr - 1984)
    area = np.clip(area + rng.normal(0, 0.25, n), 9.5, 24.6)
    return area, precip, temp, runoff


def _histogram(rng, a, true_area, clear, bias=1.0):
    """Return EE-style grouped sums (m²) for one month."""
    observed_area = min(true_area * bias, a[1:].sum() * 0.999)
    wfrac = _water_fraction_by_bin(observed_area, a)
    # Clouds are patchy: some occurrence ranges are hidden more than others
    weights = np.clip(1 + 0.6 * np.sin(np.arange(N_OCC) / 13.0 + rng.uniform(0, 6)), 0.1, None)
    hidden = np.clip((1 - clear) * weights / weights.mean(), 0, 1)
    groups = []
    for o in range(N_OCC):
        if a[o] <= 0:
            continue
        nod = a[o] * hidden[o]
        clear_a = a[o] - nod
        wat = clear_a * wfrac[o]
        land = clear_a - wat
        # a little classification noise near the shoreline
        if 0 < wfrac[o] < 1 or (o > 0 and wfrac[o] != wfrac[o - 1]):
            flip = 0.08 * clear_a * rng.uniform()
            wat, land = max(wat - flip, 0), land + min(flip, wat)
        for cls, val in ((0, land), (1, wat), (2, nod)):
            if val > 0:
                groups.append({"code": cls * N_OCC + o, "sum": val * 1e6})
    return groups


def _lake_polygon(scale: float = 1.0, cx=DEMO_LON, cy=DEMO_LAT, seed: int = 3) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    pts = []
    k = 72
    harmonics = [(2, 0.22, rng.uniform(0, 6)), (3, 0.16, rng.uniform(0, 6)),
                 (5, 0.10, rng.uniform(0, 6)), (9, 0.05, rng.uniform(0, 6))]
    for i in range(k):
        th = 2 * math.pi * i / k
        r = 1 + sum(amp * math.sin(n * th + ph) for n, amp, ph in harmonics)
        rx = 0.045 * r * scale * 1.6
        ry = 0.045 * r * scale
        pts.append([cx + rx * math.cos(th), cy + ry * math.sin(th)])
    pts.append(pts[0])
    return pts


def demo_bundle(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    a = _occurrence_areas()
    dates = pd.date_range("1984-03-01", "2026-09-01", freq="MS")
    area, precip, temp, runoff = _true_series(rng, dates)

    jrc, opt = [], []
    for d, ar in zip(dates, area):
        t = int(d.timestamp() * 1000)
        winter = d.month in (11, 12, 1, 2, 3)
        early = d.year < 1999
        base_clear = rng.beta(2.2, 1.4) if winter else rng.beta(5, 1.3)
        if early:
            base_clear *= rng.uniform(0.3, 1.0)
        if rng.uniform() < 0.07:
            base_clear = rng.uniform(0, 0.08)
        if d < pd.Timestamp("2022-01-01"):
            jrc.append({"t": t, "groups": _histogram(rng, a, ar, base_clear)})
        if d >= pd.Timestamp("2019-01-01"):
            c2 = min(1.0, base_clear * rng.uniform(1.0, 1.4) + 0.1)
            opt.append({"t": t, "groups": _histogram(rng, a, ar, c2, bias=1.03),
                        "n_scenes": int(rng.integers(4, 16))})

    climate = []
    for d, p, tc, ro in zip(dates, precip, temp, runoff):
        if d > pd.Timestamp("2026-07-01"):
            continue  # ERA5-Land lags real time by about two months
        climate.append({"t": int(d.timestamp() * 1000),
                        "total_precipitation_sum": p / 1000.0,
                        "temperature_2m": tc + 273.15,
                        "runoff_sum": ro / 1000.0})

    footprint = {"type": "Polygon", "coordinates": [_lake_polygon(1.18)]}
    catch_pts = _lake_polygon(5.5, cx=DEMO_LON - 0.12, cy=DEMO_LAT + 0.10, seed=11)
    catchment = {"type": "Polygon", "coordinates": [catch_pts]}
    return {
        "demo": True,
        "site": {"name": DEMO_NAME, "lat": DEMO_LAT, "lon": DEMO_LON, "source": "demo",
                 "country": "nowhere (synthetic)", "kind": "reservoir"},
        "footprint": footprint,
        "footprint_info": {"area_km2": float(a.sum()), "ever_water_km2": float(a[1:].sum()),
                           "permanent_km2": float(a[90:].sum()), "method": "synthetic outline",
                           "scale_m": 30, "truncated": False, "warnings": []},
        "hydrolake": {"Lake_name": DEMO_NAME, "Lake_type": 2, "Lake_area": 21.4,
                      "Vol_total": 640.0, "Depth_avg": 29.9, "Res_time": 410.0,
                      "Wshd_area": 1180.0, "Elevation": 412},
        "catchment": catchment,
        "catchment_info": {"area_km2": 1180.0, "level": 12, "n_basins": 23,
                           "method": "synthetic catchment"},
        "jrc": jrc,
        "optical": opt,
        "climate": climate,
        "meta": {"hist_scale_m": 30, "climate_scale_m": 1000,
                 "jrc_months": len(jrc), "optical_months": len(opt)},
    }


def demo_water_quality(seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for d in pd.date_range("2013-04-01", "2026-09-01", freq="MS"):
        m = d.month
        rec = {"t": int(d.timestamp() * 1000)}
        lswt = 15 + 10 * math.sin((m - 4.5) / 12 * 2 * math.pi) + 0.04 * (d.year - 2013) + rng.normal(0, 0.7)
        rec["lswt"] = lswt if rng.uniform() > 0.15 else None
        if d >= pd.Timestamp("2017-04-01"):
            bloom = 0.06 * max(0, math.sin((m - 5) / 12 * 2 * math.pi)) + (0.03 if d.year >= 2022 else 0)
            rec["ndci"] = -0.08 + bloom + rng.normal(0, 0.015) if rng.uniform() > 0.1 else None
            rec["ndti"] = -0.12 + 0.05 * math.cos((m - 1) / 12 * 2 * math.pi) + rng.normal(0, 0.02) \
                if rng.uniform() > 0.1 else None
        out.append(rec)
    return out


def demo_timelapse_frames(annual_areas: dict[int, float], size: int = 560) -> list[Image.Image]:
    """Simple synthetic 'satellite' frames: the demo lake drawn at each year's size."""
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, (size // 4, size // 4))
    tex = Image.fromarray(np.uint8(np.clip(128 + 30 * base, 0, 255))).resize((size, size), Image.BICUBIC)
    tex = tex.filter(ImageFilter.GaussianBlur(2))
    land = Image.merge("RGB", (tex.point(lambda v: int(150 + v * 0.35)),
                               tex.point(lambda v: int(132 + v * 0.30)),
                               tex.point(lambda v: int(98 + v * 0.25))))
    amax = max(annual_areas.values()) if annual_areas else 1
    frames = []
    for year, ar in sorted(annual_areas.items()):
        img = land.copy()
        dr = ImageDraw.Draw(img)
        sc = 1.18 * math.sqrt(max(ar, 0.1) / amax)
        poly = _lake_polygon(sc, cx=0, cy=0)
        pts = [(size / 2 + x / 0.16 * size * 0.5, size / 2 - y / 0.16 * size * 0.5) for x, y in poly]
        # pale ring of exposed shoreline, then water
        ring = _lake_polygon(1.18, cx=0, cy=0)
        ring_pts = [(size / 2 + x / 0.16 * size * 0.5, size / 2 - y / 0.16 * size * 0.5) for x, y in ring]
        dr.polygon(ring_pts, fill=(196, 186, 160))
        dr.polygon(pts, fill=(28, 70, 104))
        frames.append(img.filter(ImageFilter.GaussianBlur(0.6)))
    return frames


def frames_to_gif(frames: list[Image.Image], fps: float = 2.0) -> bytes:
    if not frames:
        return b""
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    return buf.getvalue()
