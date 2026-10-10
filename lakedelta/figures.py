"""Plotly figures for the app.

Charts follow a few rules: one y-axis per chart (small multiples instead of
dual axes), thin marks, hairline recessive chrome, colour only where it
carries meaning (the lake in water blue, deficits in sediment brown, context
in grey), a legend whenever two or more series share a chart, and selective
direct labels. Backgrounds, grid and text colours come from the Streamlit
theme (``st.plotly_chart(..., theme="streamlit")``) so light and dark both work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis as A
from .config import (
    CONDITIONS,
    CONTEXT,
    CONTEXT_BAND,
    MONTH_ABBR,
    SEDIMENT,
    WATER,
    WATER_WASH,
    WATER_WASH_STRONG,
)

CONTEXT_LINE = "rgba(138, 151, 159, 0.85)"
CONTEXT_FAINT = "rgba(138, 151, 159, 0.12)"
HAIRLINE = "rgba(138, 151, 159, 0.35)"

# Data colours per theme. Dark steps are lighter so marks keep their contrast on
# the deep-water background; chrome still comes from the Streamlit theme.
PALETTES = {
    "light": {
        "water": WATER, "wash": WATER_WASH, "wash2": WATER_WASH_STRONG,
        "sediment": SEDIMENT, "context": CONTEXT, "context_line": CONTEXT_LINE,
        "band": CONTEXT_BAND, "faint": CONTEXT_FAINT, "hairline": HAIRLINE,
        "ring": "rgba(247, 249, 249, 0.95)",
    },
    "dark": {
        "water": "#5B9BD5", "wash": "rgba(91, 155, 213, 0.20)", "wash2": "rgba(91, 155, 213, 0.36)",
        "sediment": "#C98A4B", "context": "#93A1AA", "context_line": "rgba(170, 182, 190, 0.9)",
        "band": "rgba(138, 151, 159, 0.26)", "faint": "rgba(138, 151, 159, 0.14)",
        "hairline": "rgba(138, 151, 159, 0.42)", "ring": "rgba(15, 26, 33, 0.95)",
    },
}


def pal(theme: str | None = "light") -> dict[str, str]:
    return PALETTES.get(theme or "light", PALETTES["light"])

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _base(fig: go.Figure, height: int, *, legend: bool = True, top: int = 36, theme: str = "light") -> go.Figure:
    p = pal(theme)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=top, b=8),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12), itemclick=False,
                    itemdoubleclick=False),
        hoverlabel=dict(namelength=-1),
        xaxis=dict(showgrid=False, ticks="outside", ticklen=4, tickcolor=p["hairline"]),
        yaxis=dict(zeroline=False, ticks="", gridwidth=1),
    )
    return fig


def _gap_line(x, y, mask):
    """Series with None where mask is False so lines break at missing months."""
    y = np.where(mask, y, np.nan)
    return list(x), [None if not np.isfinite(v) else float(v) for v in y]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def _band_polygons(x, lo, hi, mask):
    """Closed polygons for each contiguous run where mask is True (gap-safe bands)."""
    px, py = [], []
    idx = np.flatnonzero(mask & np.isfinite(lo) & np.isfinite(hi))
    if len(idx) == 0:
        return px, py
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    xs = list(x)
    for run in runs:
        rx = [xs[i] for i in run]
        if len(run) == 1:  # widen a lone month a little so it stays visible
            t = rx[0]
            rx = [t - pd.Timedelta(days=10), t + pd.Timedelta(days=10)]
            rlo = [lo[run[0]]] * 2
            rhi = [hi[run[0]]] * 2
        else:
            rlo = [lo[i] for i in run]
            rhi = [hi[i] for i in run]
        px += rx + rx[::-1] + [None]
        py += list(rhi) + list(rlo)[::-1] + [None]
    return px, py


def record_legend(theme: str = "light") -> list[tuple[str, str, str]]:
    p = pal(theme)
    return [("line", p["water"], "Water surface"),
            ("band", p["water"], "Gap-filling range"),
            ("band", p["context"], "Normal range for the month (middle half)"),
            ("band", p["water"], "Outlook, 50% and 80% ranges")]


def record_figure(an: A.Analysis, start: pd.Timestamp | None = None,
                  show_outlook: bool = True, show_normal: bool | None = None,
                  height: int = 430, theme: str = "light") -> go.Figure:
    p = pal(theme)
    rec = an.record
    if start is not None:
        rec = rec[rec.index >= start]
    if show_normal is None:
        show_normal = start is not None  # the seasonal band is noise over four decades
    x = rec.index
    valid = rec["valid"].to_numpy()
    fig = go.Figure()

    outlook = an.outlook if show_outlook else None
    if show_normal:
        nx = list(x)
        q25 = list(rec["normal_q25"])
        q75 = list(rec["normal_q75"])
        if outlook is not None:
            t = an.normals.table
            for d in outlook.table.index:
                nx.append(d)
                q25.append(float(t.loc[d.month, "q25"]))
                q75.append(float(t.loc[d.month, "q75"]))
        fig.add_trace(go.Scatter(x=nx + nx[::-1], y=q75 + q25[::-1], mode="lines",
                                 line=dict(width=0), fill="toself", fillcolor=p["band"],
                                 hoverinfo="skip", name="Normal range"))
    # Uncertainty from gap filling, one closed shape per run of usable months
    bx, by = _band_polygons(x, rec["lo"].to_numpy(float), rec["hi"].to_numpy(float), valid)
    if bx:
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(width=0), fill="toself",
                                 fillcolor=p["wash"], hoverinfo="skip", name="Gap-filling range"))
    # The lake
    ax, ay = _gap_line(x, rec["area"].to_numpy(float), valid)
    pct = rec["percentile"].to_numpy(float)
    custom = np.stack([
        np.where(np.isfinite(pct), np.round(pct), np.nan),
        rec["normal_median"].to_numpy(float),
        rec["clear_fraction"].to_numpy(float) * 100,
    ], axis=1)
    fig.add_trace(go.Scatter(
        x=ax, y=ay, mode="lines", line=dict(color=p["water"], width=2), name="Water surface",
        connectgaps=False, customdata=custom,
        hovertemplate="<b>%{y:.2f} km²</b> water<br>Normal %{customdata[1]:.2f} km²"
                      "<br>Percentile %{customdata[0]:.0f}<br>%{customdata[2]:.0f}% seen clearly"
                      "<extra></extra>"))

    # Where the record switches source
    switch = rec[rec["source"] == A.SOURCE_OPT].index.min()
    if pd.notna(switch) and (start is None or switch > start):
        fig.add_vline(x=switch, line=dict(color=p["hairline"], width=1))
        fig.add_annotation(x=switch, y=1, yref="paper", text="Landsat 8/9 + Sentinel-2",
                           showarrow=False, xanchor="left", yanchor="top", xshift=4,
                           font=dict(size=11, color=p["context"]))

    # Outlook fan
    if outlook is not None:
        o = outlook.table
        t0 = outlook.issued_from
        a0 = float(an.record.loc[t0, "area"])
        ox = [t0] + list(o.index)
        for lo_c, hi_c, color in (("p10", "p90", p["wash"]), ("p25", "p75", p["wash2"])):
            lo_v = [a0] + list(o[lo_c])
            hi_v = [a0] + list(o[hi_c])
            fig.add_trace(go.Scatter(x=ox + ox[::-1], y=hi_v + lo_v[::-1], mode="lines",
                                     line=dict(width=0), fill="toself", fillcolor=color,
                                     hoverinfo="skip", name="Outlook range"))
        fig.add_trace(go.Scatter(
            x=ox, y=[a0] + list(o["median"]), mode="lines",
            line=dict(color=p["water"], width=1.5, dash="dot"), name="Outlook, median",
            customdata=np.stack([[a0] + list(o["p10"]), [a0] + list(o["p90"])], axis=1),
            hovertemplate="Outlook <b>%{y:.2f} km²</b><br>80%: %{customdata[0]:.2f}–"
                          "%{customdata[1]:.2f}<extra></extra>"))

    latest = an.latest
    if latest and (start is None or latest["date"] >= start):
        fig.add_trace(go.Scatter(
            x=[latest["date"]], y=[latest["area"]], mode="markers",
            marker=dict(size=9, color=p["water"], line=dict(width=2, color=p["ring"])),
            showlegend=False, hoverinfo="skip"))
    _base(fig, height, legend=False, top=14, theme=theme)
    fig.update_yaxes(title_text="Water surface (km²)", title_font=dict(size=12))
    fig.update_layout(hovermode="x unified")
    return fig


def calendar_figure(an: A.Analysis, height_per_year: int = 15, theme: str = "light") -> go.Figure:
    """Year x month grid coloured by condition class."""
    rec = an.record
    years = sorted(set(rec.index.year), reverse=True)
    idx = {c.key: i for i, c in enumerate(CONDITIONS)}
    z = np.full((len(years), 12), np.nan)
    text = np.full((len(years), 12), "", dtype=object)
    for ts, row in rec.iterrows():
        yi = years.index(ts.year)
        if row["valid"] and row["condition"] in idx:
            z[yi, ts.month - 1] = idx[row["condition"]]
            text[yi, ts.month - 1] = (
                f"<b>{A.month_label(ts)}</b><br>{A.fmt_area(row['area'])}, "
                f"{A.ordinal(int(round(row['percentile'])))} percentile"
                f"<br>{next(c.label for c in CONDITIONS if c.key == row['condition'])}")
        else:
            text[yi, ts.month - 1] = f"<b>{A.month_label(ts)}</b><br>No clear observation"
    n = len(CONDITIONS)
    scale = []
    for i, c in enumerate(CONDITIONS):
        scale.append([i / n, c.color])
        scale.append([(i + 1) / n, c.color])
    fig = go.Figure(go.Heatmap(
        z=z, x=list(MONTH_ABBR), y=[str(y) for y in years], zmin=-0.5, zmax=n - 0.5,
        colorscale=scale, showscale=False, xgap=2, ygap=2, text=text,
        hovertemplate="%{text}<extra></extra>"))
    h = max(220, 36 + height_per_year * len(years))
    _base(fig, h, legend=False, top=8, theme=theme)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(side="top", ticks="", showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed", type="category",
                     dtick=5 if len(years) > 25 else 1)
    return fig


def seasonal_figure(an: A.Analysis, height: int = 360, theme: str = "light") -> go.Figure:
    p = pal(theme)
    t = an.normals.table
    months = list(MONTH_ABBR)
    rec = an.record
    latest = an.latest
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=t["q90"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=months, y=t["q10"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=p["faint"], hoverinfo="skip",
                             name="Usual range (10th–90th percentile)"))
    fig.add_trace(go.Scatter(x=months, y=t["q75"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=months, y=t["q25"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=p["band"], hoverinfo="skip",
                             name="Normal range (middle half)"))
    fig.add_trace(go.Scatter(x=months, y=t["median"], mode="lines",
                             line=dict(color=p["context_line"], width=1), name="Median",
                             hovertemplate="Median %{y:.2f} km²<extra></extra>"))
    if latest:
        cur_year = latest["date"].year
        for year, color, width, name in ((cur_year - 1, p["context_line"], 1.5, str(cur_year - 1)),
                                         (cur_year, p["water"], 2.5, str(cur_year))):
            sel = rec[(rec.index.year == year)]
            y = np.where(sel["valid"], sel["area"], np.nan)
            full = np.full(12, np.nan)
            full[sel.index.month.to_numpy() - 1] = y
            fig.add_trace(go.Scatter(
                x=months, y=[None if not np.isfinite(v) else v for v in full],
                mode="lines+markers", line=dict(color=color, width=width),
                marker=dict(size=7 if year == cur_year else 5, color=color),
                name=name, connectgaps=False,
                hovertemplate=f"{name}: %{{y:.2f}} km²<extra></extra>"))
    _base(fig, height, legend=False, top=12, theme=theme)
    fig.update_yaxes(title_text="Water surface (km²)", title_font=dict(size=12))
    return fig


def seasonal_legend(an: A.Analysis, theme: str = "light") -> list[tuple[str, str, str]]:
    p = pal(theme)
    items = []
    if an.latest:
        y = an.latest["date"].year
        items += [("line", p["water"], str(y)), ("line", p["context"], str(y - 1))]
    return items + [("line", p["context"], "Median"), ("band", p["context"], "Middle half"),
                    ("band", "rgba(138, 151, 159, 0.45)", "10th to 90th percentile")]


def annual_figure(an: A.Analysis, basis: str = "mean", height: int = 320, theme: str = "light") -> go.Figure:
    p = pal(theme)
    ann = an.annual
    col = {"mean": "mean", "season": "season_mean", "min": "min"}[basis]
    s = ann[col].dropna() if col in ann else pd.Series(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers",
                             line=dict(color=p["water"], width=2), marker=dict(size=6, color=p["water"]),
                             name="Yearly value",
                             hovertemplate="%{x}: <b>%{y:.2f} km²</b><extra></extra>"))
    tr = A.trend_test(s)
    if tr:
        xs = np.array([s.index.min(), s.index.max()], dtype=float)
        ys = tr["intercept"] + tr["slope_per_decade"] / 10.0 * xs
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=p["context_line"], width=1.5),
                                 name="Trend (Sen's slope)", hoverinfo="skip"))
    _base(fig, height, legend=False, top=12, theme=theme)
    fig.update_layout(hovermode="x")
    fig.update_yaxes(title_text="km²", title_font=dict(size=12))
    fig.update_xaxes(dtick=5)
    return fig


def _signed_bar_colors(v: np.ndarray, p: dict | None = None) -> list[str]:
    p = p or pal()
    return [p["sediment"] if (np.isfinite(x) and x < 0) else p["water"] for x in v]


def drivers_figure(an: A.Analysis, k: int | None = None, start: pd.Timestamp | None = None,
                   height: int = 560, theme: str = "light") -> go.Figure | None:
    p = pal(theme)
    ci = an.climate
    if ci is None or ci.empty:
        return None
    if k is None:
        k = int(an.response["r"].idxmax()) if an.response is not None else 12
    rec = an.record
    if start is not None:
        rec = rec[rec.index >= start]
        ci = ci[ci.index >= start]
    spi = A.standardize_nonparametric(an.climate["precip_mm"].rolling(k, min_periods=k).sum())
    if start is not None:
        spi = spi[spi.index >= start]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=(
                            "Lake surface compared with normal (standard units)",
                            f"Rain on the catchment, summed over {k} months (SPI-{k})",
                            "Air temperature compared with normal, 12-month mean (°C)"))
    z = rec["z"].to_numpy(float)
    fig.add_trace(go.Bar(x=rec.index, y=z, marker=dict(color=_signed_bar_colors(z, p), line_width=0),
                         name="Lake", hovertemplate="%{y:.2f}<extra>Lake</extra>"), row=1, col=1)
    sv = spi.to_numpy(float)
    fig.add_trace(go.Bar(x=spi.index, y=sv, marker=dict(color=_signed_bar_colors(sv, p), line_width=0),
                         name=f"SPI-{k}", hovertemplate="%{y:.2f}<extra>Rain</extra>"),
                  row=2, col=1)
    ta = ci["temp_anom12"]
    fig.add_trace(go.Scatter(x=ta.index, y=ta.values, mode="lines",
                             line=dict(color=p["context_line"], width=1.5), name="Temperature",
                             hovertemplate="%{y:+.2f} °C<extra>Temperature</extra>"),
                  row=3, col=1)
    for r in (1, 2, 3):
        fig.add_hline(y=0, line=dict(color=p["hairline"], width=1), row=r, col=1)
    _base(fig, height, legend=False, top=28, theme=theme)
    fig.update_layout(bargap=0, hovermode="x unified")
    fig.update_annotations(font_size=13, xanchor="left", x=0)
    return fig


def response_figure(resp: pd.DataFrame, height: int = 300, theme: str = "light") -> go.Figure:
    p = pal(theme)
    best = int(resp["r"].idxmax())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resp.index, y=resp["r"], mode="lines", line=dict(color=p["water"], width=2),
                             name="Correlation",
                             hovertemplate="%{x} months: r = %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[best], y=[resp.loc[best, "r"]], mode="markers",
                             marker=dict(size=9, color=p["water"], line=dict(width=2, color=p["ring"])),
                             hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=best, y=resp.loc[best, "r"], yshift=10, showarrow=False, yanchor="bottom",
                       text=f"{best} months, r = {resp.loc[best, 'r']:.2f}", font=dict(size=12))
    _base(fig, height, legend=False, top=12, theme=theme)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(title_text="Months of rain summed", dtick=6, title_font=dict(size=12))
    fig.update_yaxes(title_text="Correlation with lake anomaly", range=[min(0, resp["r"].min() - 0.05), 1],
                     title_font=dict(size=12))
    return fig


# ---------------------------------------------------------------------------
# Outlook
# ---------------------------------------------------------------------------

def outlook_figure(an: A.Analysis, months_back: int = 36, height: int = 380, theme: str = "light") -> go.Figure | None:
    p = pal(theme)
    if an.outlook is None:
        return None
    start = an.outlook.issued_from - pd.DateOffset(months=months_back)
    fig = record_figure(an, start=start, show_outlook=True, show_normal=True, height=height, theme=theme)
    o = an.outlook.table
    fig.add_trace(go.Scatter(x=list(o.index), y=list(o["normal"]), mode="lines",
                             line=dict(color=p["context_line"], width=1), name="Normal median",
                             hovertemplate="Normal %{y:.2f} km²<extra></extra>"))
    return fig


def skill_figure(an: A.Analysis, height: int = 260, theme: str = "light") -> go.Figure | None:
    p = pal(theme)
    if an.outlook is None or an.outlook.skill.empty:
        return None
    s = an.outlook.skill
    v = (s["skill_vs_climatology"] * 100).to_numpy(float)
    fig = go.Figure(go.Bar(
        x=s.index, y=v, marker=dict(color=[p["water"] if x > 0 else p["sediment"] for x in v], line_width=0),
        width=0.6, hovertemplate="%{x} months ahead: %{y:.0f}% smaller error than climatology"
                                 "<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=p["hairline"], width=1))
    _base(fig, height, legend=False, top=12, theme=theme)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(title_text="Months ahead", dtick=1, title_font=dict(size=12))
    fig.update_yaxes(title_text="Error reduction vs. normal (%)", title_font=dict(size=12))
    return fig


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def coverage_figure(an: A.Analysis, height: int = 200, theme: str = "light") -> go.Figure:
    p = pal(theme)
    rec = an.record
    cf = (rec["clear_fraction"] * 100).to_numpy(float)
    colors = np.where(rec["valid"], p["water"], np.where(rec["outlier"], p["sediment"], p["context"]))
    fig = go.Figure(go.Bar(x=rec.index, y=cf, marker=dict(color=list(colors), line_width=0),
                           hovertemplate="%{x|%b %Y}: %{y:.0f}% seen clearly<extra></extra>"))
    fig.add_hline(y=an.settings.min_clear_fraction * 100, line=dict(color=p["hairline"], width=1))
    _base(fig, height, legend=False, top=8, theme=theme)
    fig.update_layout(bargap=0, hovermode="closest")
    fig.update_yaxes(title_text="Seen clearly (%)", range=[0, 100], title_font=dict(size=12))
    return fig


def overlap_figure(harm: dict, height: int = 320, theme: str = "light") -> go.Figure | None:
    p = pal(theme)
    pairs = harm.get("pairs")
    if pairs is None or len(pairs) == 0:
        return None
    lo = float(min(pairs.min())) * 0.97
    hi = float(max(pairs.max())) * 1.03
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=p["hairline"], width=1), name="Equal",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=pairs["area_opt"], y=pairs["area_jrc"], mode="markers",
        marker=dict(size=8, color=p["water"], line=dict(width=1.5, color=p["ring"])),
        name="Months seen by both", text=[A.month_label(d) for d in pairs.index],
        hovertemplate="%{text}<br>Ours %{x:.2f} km², JRC %{y:.2f} km²<extra></extra>"))
    _base(fig, height, legend=False, top=12, theme=theme)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Landsat 8/9 + Sentinel-2 (km²)", range=[lo, hi], showgrid=True,
                     title_font=dict(size=12))
    fig.update_yaxes(title_text="JRC Global Surface Water (km²)", range=[lo, hi],
                     title_font=dict(size=12), scaleanchor="x")
    return fig


# ---------------------------------------------------------------------------
# Water quality
# ---------------------------------------------------------------------------

WQ_SERIES = (
    ("ndti", "Turbidity index (NDTI)", "Sentinel-2", ""),
    ("ndci", "Chlorophyll index (NDCI)", "Sentinel-2", ""),
    ("lswt", "Surface water temperature (°C)", "Landsat 8/9", " °C"),
)


def water_quality_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records or []:
        if r.get("t") is None:
            continue
        rows.append({"date": pd.Timestamp(int(r["t"]), unit="ms").to_period("M").to_timestamp(),
                     **{k: (float(r[k]) if r.get(k) is not None else np.nan) for k, *_ in WQ_SERIES}})
    if not rows:
        return pd.DataFrame(columns=[k for k, *_ in WQ_SERIES])
    return pd.DataFrame(rows).groupby("date").last().sort_index()


def water_quality_figure(wq: pd.DataFrame, height: int = 620, theme: str = "light") -> go.Figure:
    p = pal(theme)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{label}, {src}" for _, label, src, _ in WQ_SERIES])
    for i, (key, label, _src, unit) in enumerate(WQ_SERIES, start=1):
        s = wq[key].dropna() if key in wq else pd.Series(dtype=float)
        if s.empty:
            continue
        clim = s.groupby(s.index.month).quantile([0.25, 0.75]).unstack()
        full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        q25 = clim[0.25].reindex(full.month).to_numpy()
        q75 = clim[0.75].reindex(full.month).to_numpy()
        fig.add_trace(go.Scatter(x=full, y=q75, mode="lines", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=full, y=q25, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=p["band"], hoverinfo="skip", showlegend=False),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers",
                                 marker=dict(size=5, color=p["water"]), name=label, showlegend=False,
                                 hovertemplate=f"%{{x|%b %Y}}: %{{y:.3f}}{unit}<extra></extra>"),
                      row=i, col=1)
        # 12-month average of anomalies from the monthly median, added back to the
        # overall median: shows slow change without the seasonal cycle or gap bias
        med_m = s.groupby(s.index.month).median()
        anom = (s - med_m.reindex(s.index.month).to_numpy()).reindex(full)
        level = float(s.median()) + anom.rolling(12, min_periods=6, center=True).mean()
        fig.add_trace(go.Scatter(x=full, y=level, mode="lines", line=dict(color=p["water"], width=2),
                                 hoverinfo="skip", showlegend=False), row=i, col=1)
    _base(fig, height, legend=False, top=28, theme=theme)
    fig.update_layout(hovermode="closest")
    fig.update_annotations(font_size=13, xanchor="left", x=0)
    return fig
