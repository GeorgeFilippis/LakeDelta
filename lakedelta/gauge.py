"""The staff gauge: where this month's water surface sits in the lake's own history.

Drawn as inline SVG so it inherits the page's text colour (light or dark
theme) through ``currentColor``. The tube shows the current water surface;
the strip beside it splits the same calendar month's history into the five
condition classes; ticks are graduated in km² like a reservoir staff gauge.
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd

from . import analysis as A
from .config import CONDITIONS, FONT_DENSE, FONT_UI, WATER


def nice_step(span: float, target: int = 5) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_tick(v: float, step: float) -> str:
    for decimals in (0, 1, 2, 3):
        if abs(step * 10 ** decimals - round(step * 10 ** decimals)) < 1e-6:
            return f"{v:,.{decimals}f}"
    return f"{v:.3g}"


def gauge_svg(value: float, when_label: str, dist: dict, extremes: dict | None = None,
              year_ago: float | None = None, year_ago_label: str | None = None,
              width: int = 250, height: int = 340) -> str:
    """dist needs min, q10, q25, median, q75, q90, max for this calendar month."""
    extremes = extremes or {}
    candidates = [dist["min"], dist["max"], value] + ([year_ago] if year_ago is not None else [])
    vmin, vmax = min(candidates), max(candidates)
    span = vmax - vmin
    pad = span * 0.08 if span > 0 else max(abs(vmax) * 0.05, 0.1)
    lo, hi = max(0.0, vmin - pad), vmax + pad
    top, bottom = 26, height - 14

    def y(v: float) -> float:
        return bottom - (v - lo) / (hi - lo) * (bottom - top)

    tube_x, tube_w = 70, 32
    strip_x, strip_w = tube_x + tube_w + 5, 7
    label_x = strip_x + strip_w + 12
    out = []
    out.append(f'<text x="{tube_x - 10}" y="{top - 12}" text-anchor="end" font-size="11" '
               f'fill="currentColor" fill-opacity="0.6">km²</text>')

    # Graduations
    step = nice_step(hi - lo)
    minor = step / 5
    t = math.ceil(lo / minor) * minor
    while t <= hi + 1e-9:
        major = abs(t / step - round(t / step)) < 1e-6
        x1 = tube_x - (7 if major else 3)
        out.append(f'<line x1="{x1}" x2="{tube_x}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                   f'stroke="currentColor" stroke-opacity="{0.55 if major else 0.25}" stroke-width="1"/>')
        if major:
            out.append(f'<text x="{tube_x - 11}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" '
                       f'fill="currentColor" fill-opacity="0.75" '
                       f'style="font-variant-numeric: tabular-nums">{_fmt_tick(t, step)}</text>')
        t += minor

    # Tube with water
    yw = y(value)
    out.append(f'<rect x="{tube_x}" y="{top}" width="{tube_w}" height="{bottom - top}" rx="5" '
               f'fill="none" stroke="currentColor" stroke-opacity="0.28" stroke-width="1"/>')
    out.append(f'<rect x="{tube_x + 1.5}" y="{yw:.1f}" width="{tube_w - 3}" height="{max(bottom - yw - 1.5, 0):.1f}" '
               f'rx="3.5" fill="{WATER}" fill-opacity="0.26"/>')

    # Condition strip for this calendar month, with 2px gaps between classes
    bounds = [max(dist["min"], lo), dist["q10"], dist["q25"], dist["q75"], dist["q90"],
              min(dist["max"], hi)]
    for i, c in enumerate(CONDITIONS):
        y_top, y_bot = y(bounds[i + 1]), y(bounds[i])
        if y_bot - y_top <= 2.2:
            continue
        out.append(f'<rect x="{strip_x}" y="{y_top + 1:.1f}" width="{strip_w}" '
                   f'height="{y_bot - y_top - 2:.1f}" rx="1.5" fill="{c.color}">'
                   f'<title>{escape(c.label)}</title></rect>')

    # Last year's value as a small open mark on the strip
    if year_ago is not None:
        ya = y(year_ago)
        out.append(f'<circle cx="{strip_x + strip_w / 2:.1f}" cy="{ya:.1f}" r="4.5" fill="none" '
                   f'stroke="currentColor" stroke-width="1.5"/>')

    # The water line
    out.append(f'<line x1="{tube_x - 4}" x2="{label_x - 4}" y1="{yw:.1f}" y2="{yw:.1f}" '
               f'stroke="{WATER}" stroke-width="2" stroke-linecap="round"/>')

    # Labels on the right, placed by priority without overlapping
    labels: list[tuple[float, str, int]] = [
        (yw, f'<tspan font-size="16" font-weight="600" fill="currentColor">{escape(A.fmt_area(value))}</tspan>'
             f'<tspan x="{label_x}" dy="15" font-size="12" fill="currentColor" fill-opacity="0.7">'
             f'{escape(when_label)}</tspan>', 0),
    ]
    if year_ago is not None and year_ago_label:
        labels.append((y(year_ago), f'<tspan font-size="12" fill="currentColor" fill-opacity="0.7">'
                                    f'{escape(year_ago_label)}</tspan>', 1))
    if extremes.get("low") is not None:
        v, yr = extremes["low"]
        labels.append((y(v), f'<tspan font-size="12" fill="currentColor" fill-opacity="0.7">'
                             f'Lowest, {yr}</tspan>', 2))
    if extremes.get("high") is not None:
        v, yr = extremes["high"]
        labels.append((y(v), f'<tspan font-size="12" fill="currentColor" fill-opacity="0.7">'
                             f'Highest, {yr}</tspan>', 2))
    placed: list[tuple[float, float]] = []
    for yy, body, prio in sorted(labels, key=lambda t: t[2]):
        h = 30 if prio == 0 else 14
        y0 = yy - 4 if prio == 0 else yy - 7
        if any(not (y0 + h < p0 or y0 > p1) for p0, p1 in placed):
            continue
        placed.append((y0, y0 + h))
        base = yy + 5 if prio == 0 else yy + 4
        out.append(f'<text x="{label_x}" y="{base:.1f}">{body}</text>')

    desc = (f"Staff gauge: {A.fmt_area(value)} in {when_label}. Normal range "
            f"{A.fmt_area(dist['q25'])} to {A.fmt_area(dist['q75'])}.")
    return (f'<svg class="ld-gauge" viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px;display:block" role="img" aria-label="{escape(desc)}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="{escape(FONT_UI)}">'
            f'<title>{escape(desc)}</title>{"".join(out)}</svg>')


def gauge_inputs(an: A.Analysis) -> dict | None:
    """Collect what gauge_svg needs from an Analysis."""
    latest = an.latest
    if not latest:
        return None
    m = latest["date"].month
    row = an.normals.table.loc[m]
    rec = an.record
    same = rec[rec["valid"] & (rec.index.month == m)]
    others = same[same.index != latest["date"]]
    extremes = {}
    if len(others):
        lo_ts = others["area"].idxmin()
        hi_ts = others["area"].idxmax()
        extremes = {"low": (float(others.loc[lo_ts, "area"]), lo_ts.year),
                    "high": (float(others.loc[hi_ts, "area"]), hi_ts.year)}
    dist = {"min": float(same["area"].min()), "max": float(same["area"].max()),
            "q10": float(row["q10"]), "q25": float(row["q25"]), "median": float(row["median"]),
            "q75": float(row["q75"]), "q90": float(row["q90"])}
    if len(same) < 5 or not all(math.isfinite(v) for v in dist.values()) \
            or not math.isfinite(latest["area"]):
        return None  # too little history for this calendar month to draw a meaningful gauge
    ya = latest.get("year_ago")
    return {
        "value": latest["area"],
        "when_label": A.month_label(latest["date"]),
        "dist": dist,
        "extremes": extremes,
        "year_ago": ya,
        "year_ago_label": A.month_label(latest["date"] - pd.DateOffset(years=1))
        if ya is not None else None,
    }


FONT_STACKS = {"ui": FONT_UI, "dense": FONT_DENSE}
