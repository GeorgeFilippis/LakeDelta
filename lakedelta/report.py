"""PDF report (fpdf2) with matplotlib figures in the same visual language as the app."""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import analysis as A  # noqa: E402
from .config import CONDITIONS, CONTEXT, MONTH_ABBR, SEDIMENT, WATER  # noqa: E402

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
INK = "#14232E"
INK2 = "#4A5A66"
MUTED = "#76848E"
RULE = "#D3DCE0"
BAND = "#DDE3E6"
WATER_LABEL = "#245A85"

_fonts_ready = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _valid_png(data: bytes | None) -> bool:
    if not data:
        return False
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _setup_matplotlib() -> None:
    global _fonts_ready
    if _fonts_ready:
        return
    for f in ("Barlow-Regular.ttf", "Barlow-Medium.ttf", "Barlow-SemiBold.ttf",
              "BarlowSemiCondensed-SemiBold.ttf"):
        path = os.path.join(FONT_DIR, f)
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": ["Barlow", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E6EBED",
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    _fonts_ready = True


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_record(an: A.Analysis, width_in: float = 7.1, height_in: float = 2.7) -> bytes:
    _setup_matplotlib()
    rec = an.record
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    x = rec.index
    v = rec["valid"].to_numpy()
    lo = np.where(v, rec["lo"], np.nan)
    hi = np.where(v, rec["hi"], np.nan)
    ax.fill_between(x, lo, hi, color=WATER, alpha=0.18, lw=0, label="Gap-filling range")
    ax.plot(x, np.where(v, rec["area"], np.nan), color=WATER, lw=1.3, label="Water surface")
    if an.outlook is not None:
        o = an.outlook.table
        t0 = an.outlook.issued_from
        a0 = float(rec.loc[t0, "area"])
        ox = [t0] + list(o.index)
        ax.fill_between(ox, [a0] + list(o["p10"]), [a0] + list(o["p90"]), color=WATER, alpha=0.15,
                        lw=0, label="Outlook, 80% range")
        ax.plot(ox, [a0] + list(o["median"]), color=WATER, lw=1.1, ls=(0, (2, 2)))
    if an.latest:
        ax.plot([an.latest["date"]], [an.latest["area"]], "o", ms=5, color=WATER, mec="white", mew=1.2)
    ax.set_ylabel("Water surface (km²)")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13), handlelength=1.4)
    ax.margins(x=0.01)
    return _png(fig)


def fig_calendar(an: A.Analysis, width_in: float = 7.1) -> bytes:
    _setup_matplotlib()
    rec = an.record
    years = sorted(set(rec.index.year), reverse=True)
    idx = {c.key: i for i, c in enumerate(CONDITIONS)}
    z = np.full((len(years), 12), np.nan)
    for ts, row in rec.iterrows():
        if row["valid"] and row["condition"] in idx:
            z[years.index(ts.year), ts.month - 1] = idx[row["condition"]]
    h = 0.6 + 0.135 * len(years)
    fig, ax = plt.subplots(figsize=(width_in, h))
    cmap = ListedColormap([c.color for c in CONDITIONS]).with_extremes(bad="white")
    ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-0.5, vmax=len(CONDITIONS) - 0.5,
              aspect="auto", interpolation="nearest")
    for i in range(len(years) + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2)
    for j in range(13):
        ax.axvline(j - 0.5, color="white", lw=1.2)
    ax.set_xticks(range(12), MONTH_ABBR)
    ax.xaxis.tick_top()
    step = 5 if len(years) > 25 else 1
    ticks = [i for i, y in enumerate(years) if y % step == 0]
    ax.set_yticks(ticks, [str(years[i]) for i in ticks])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c.color) for c in CONDITIONS]
    ax.legend(handles, [c.label for c in CONDITIONS], loc="upper left", ncol=5,
              bbox_to_anchor=(0, -0.01 - 0.25 / h), handlelength=1.0, handleheight=1.0)
    return _png(fig)


def fig_seasonal(an: A.Analysis, width_in: float = 3.45, height_in: float = 2.5) -> bytes:
    _setup_matplotlib()
    t = an.normals.table
    m = np.arange(12)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.fill_between(m, t["q10"], t["q90"], color=BAND, alpha=0.55, lw=0, label="10th–90th pct.")
    ax.fill_between(m, t["q25"], t["q75"], color=BAND, lw=0, label="Middle half")
    ax.plot(m, t["median"], color=CONTEXT, lw=1, label="Median")
    if an.latest:
        rec = an.record
        for yr, color, lw in ((an.latest["date"].year - 1, CONTEXT, 1.2),
                              (an.latest["date"].year, WATER, 1.8)):
            sel = rec[(rec.index.year == yr)]
            y = np.full(12, np.nan)
            y[sel.index.month - 1] = np.where(sel["valid"], sel["area"], np.nan)
            ax.plot(m, y, color=color, lw=lw, marker="o", ms=2.8, label=str(yr))
    ax.set_xticks(m, [a[0] for a in MONTH_ABBR])
    ax.set_title("Seasonal cycle", pad=4)
    ax.set_ylabel("km²")
    ax.legend(loc="upper center", fontsize=7.2, ncol=3, bbox_to_anchor=(0.5, -0.1),
              handlelength=1.2, columnspacing=0.8)
    return _png(fig)


def fig_annual(an: A.Analysis, width_in: float = 3.45, height_in: float = 2.5) -> bytes:
    _setup_matplotlib()
    s = an.annual["mean"].dropna()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot(s.index, s.values, color=WATER, lw=1.4, marker="o", ms=2.8)
    tr = an.trend_annual
    if tr:
        xs = np.array([s.index.min(), s.index.max()], dtype=float)
        ax.plot(xs, tr["intercept"] + tr["slope_per_decade"] / 10 * xs, color=CONTEXT, lw=1.1)
        ax.set_title(f"Yearly mean, {A.fmt_signed(tr['slope_per_decade'], 2)} km² per decade")
    else:
        ax.set_title("Yearly mean")
    ax.set_ylabel("km²")
    return _png(fig)


def fig_bullet(gauge: dict, width_in: float = 7.1, height_in: float = 0.85) -> bytes:
    """Horizontal gauge: this month's history in five classes, with the current value."""
    _setup_matplotlib()
    d = gauge["dist"]
    vals = [d["min"], d["max"], gauge["value"]] + ([gauge["year_ago"]] if gauge.get("year_ago") else [])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 0.1
    lo, hi = lo - pad, hi + pad
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    bounds = [d["min"], d["q10"], d["q25"], d["q75"], d["q90"], d["max"]]
    for i, c in enumerate(CONDITIONS):
        ax.barh(0, bounds[i + 1] - bounds[i], left=bounds[i], height=0.42, color=c.color,
                edgecolor="white", linewidth=1.5)
    ax.axvline(gauge["value"], color=WATER, lw=2.2, ymin=0.05, ymax=0.95)
    ax.text(gauge["value"], 0.42, f"{A.fmt_area(gauge['value'])}, {gauge['when_label']}",
            ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="semibold")
    if gauge.get("year_ago") is not None:
        ax.plot([gauge["year_ago"]], [0], "o", ms=6, mfc="white", mec=INK, mew=1.2)
        ax.text(gauge["year_ago"], -0.42, gauge["year_ago_label"], ha="center", va="top",
                fontsize=7.5, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.75, 0.85)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("km²", fontsize=8)
    return _png(fig)


def fig_drivers(an: A.Analysis, width_in: float = 7.1, height_in: float = 3.6) -> bytes | None:
    if an.climate is None or an.climate.empty:
        return None
    _setup_matplotlib()
    k = int(an.response["r"].idxmax()) if an.response is not None else 12
    spi = A.standardize_nonparametric(an.climate["precip_mm"].rolling(k, min_periods=k).sum())
    rec = an.record
    fig, axes = plt.subplots(3, 1, figsize=(width_in, height_in), sharex=True)
    for ax, s, title in ((axes[0], rec["z"], "Lake surface compared with normal (standard units)"),
                         (axes[1], spi, f"Rain on the catchment, {k}-month sum (SPI-{k})")):
        v = s.to_numpy(float)
        colors = [SEDIMENT if (np.isfinite(t) and t < 0) else WATER for t in v]
        ax.bar(s.index, np.nan_to_num(v), width=31, color=colors, lw=0)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.set_title(title, fontsize=9)
    ta = an.climate["temp_anom12"]
    axes[2].plot(ta.index, ta.values, color=INK2, lw=1)
    axes[2].axhline(0, color=RULE, lw=0.8)
    axes[2].set_title("Air temperature compared with normal, 12-month mean (°C)", fontsize=9)
    fig.tight_layout(h_pad=0.6)
    return _png(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


class Report(FPDF):
    def __init__(self, lake_name: str, demo: bool):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.lake_name = lake_name
        self.demo = demo
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, margin=18)
        self.add_font("Barlow", "", os.path.join(FONT_DIR, "Barlow-Regular.ttf"))
        self.add_font("Barlow", "B", os.path.join(FONT_DIR, "Barlow-SemiBold.ttf"))
        self.add_font("Barlow", "I", os.path.join(FONT_DIR, "Barlow-Italic.ttf"))
        self.add_font("Serif", "I", os.path.join(FONT_DIR, "SourceSerif4-MediumItalic.ttf"))
        dejavu = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        if dejavu and os.path.exists(dejavu):
            self.add_font("DejaVu", "", dejavu)
            self.set_fallback_fonts(["DejaVu"])
        self.set_title(f"LakeDelta report: {lake_name}")
        self.set_author("LakeDelta")
        self.set_creator("LakeDelta 2.0")

    def footer(self):
        self.set_y(-12)
        self.set_font("Barlow", "", 8)
        self.set_text_color(*_hex(MUTED))
        tag = "Demo data, not a real lake. " if self.demo else ""
        self.cell(0, 5, f"{tag}LakeDelta report for {self.lake_name}", align="L")
        self.set_x(self.l_margin)
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="R")

    # helpers ---------------------------------------------------------------
    def h2(self, text: str):
        self.ln(2)
        self.set_font("Barlow", "B", 13)
        self.set_text_color(*_hex(INK))
        self.multi_cell(0, 6.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str, size: float = 10, color: str = INK, lh: float = 5.0,
             markdown: bool = False):
        self.set_font("Barlow", "", size)
        self.set_text_color(*_hex(color))
        self.multi_cell(0, lh, text, new_x="LMARGIN", new_y="NEXT", align="L", markdown=markdown)
        self.ln(1.2)

    def image_bytes(self, data: bytes, w: float, x: float | None = None):
        self.image(io.BytesIO(data), x=x, w=w)

    def rule(self):
        self.set_draw_color(*_hex(RULE))
        self.set_line_width(0.25)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2)


def build_report(an: A.Analysis, site: dict, bundle: dict, gauge: dict | None,
                 maps: dict[str, bytes] | None = None) -> bytes:
    name = site.get("name", "Lake")
    demo = bool(bundle.get("demo"))
    pdf = Report(name, demo)
    pdf.alias_nb_pages()
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Barlow", "", 8.5)
    pdf.set_text_color(*_hex(MUTED))
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")
    pdf.cell(0, 4, f"LakeDelta report, generated {stamp}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Serif", "I", 26)
    pdf.set_text_color(*_hex(WATER_LABEL))
    pdf.multi_cell(0, 11, name, new_x="LMARGIN", new_y="NEXT")
    pdf.para(lede_text(site, bundle), size=10.5, color=INK2)

    latest = an.latest
    if latest:
        cond = latest["condition"]
        pdf.set_font("Barlow", "", 10)
        pdf.set_text_color(*_hex(INK2))
        pdf.cell(0, 5, A.month_label(latest["date"], long=True), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Barlow", "B", 26)
        pdf.set_text_color(*_hex(INK))
        pdf.cell(0, 11, A.fmt_area(latest["area"]), new_x="LMARGIN", new_y="NEXT")
        if cond:
            y = pdf.get_y() + 1.2
            pdf.set_fill_color(*_hex(cond.color))
            pdf.rect(pdf.l_margin, y, 3.6, 3.6, style="F")
            pdf.set_xy(pdf.l_margin + 5.5, y - 0.6)
            pdf.set_font("Barlow", "B", 10.5)
            pdf.cell(0, 5, cond.label, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)
    pdf.para(A.headline(name, latest), size=11, lh=5.4)

    rows = readings_rows(an)
    if rows:
        pdf.rule()
        col_w = width / 2
        pdf.set_font("Barlow", "", 9)
        for i in range(0, len(rows), 2):
            y0 = pdf.get_y()
            for j, (label, value, sub) in enumerate(rows[i:i + 2]):
                x = pdf.l_margin + j * col_w
                pdf.set_xy(x, y0)
                pdf.set_font("Barlow", "", 8.5)
                pdf.set_text_color(*_hex(MUTED))
                pdf.cell(col_w - 4, 4.2, label)
                pdf.set_xy(x, y0 + 4.4)
                pdf.set_font("Barlow", "B", 13)
                pdf.set_text_color(*_hex(INK))
                pdf.cell(col_w - 4, 6, value)
                pdf.set_xy(x, y0 + 10.6)
                pdf.set_font("Barlow", "", 8.5)
                pdf.set_text_color(*_hex(INK2))
                pdf.cell(col_w - 4, 4.2, sub)
            pdf.set_y(y0 + 16.5)
        pdf.rule()

    if gauge:
        pdf.set_font("Barlow", "B", 9.5)
        pdf.set_text_color(*_hex(INK))
        pdf.cell(0, 5, f"Where {A.month_label(latest['date'], long=True)} sits among every "
                       f"{A.MONTH_NAME[latest['date'].month - 1]} on record",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.image_bytes(fig_bullet(gauge), w=width)
        pdf.ln(2)
    pdf.h2("Monthly record since 1984")
    pdf.image_bytes(fig_record(an), w=width)

    pdf.add_page()
    pdf.h2("Month by month")
    pdf.para("Each cell compares one month with the same calendar month in the reference period. "
             "White cells had no clear enough observation.", size=9.5, color=INK2)
    pdf.image_bytes(fig_calendar(an), w=width)
    pdf.ln(2)
    y = pdf.get_y()
    if y > pdf.h - 80:
        pdf.add_page()
        y = pdf.get_y()
    half = (width - 6) / 2
    pdf.image(io.BytesIO(fig_seasonal(an)), x=pdf.l_margin, y=y, w=half)
    pdf.image(io.BytesIO(fig_annual(an)), x=pdf.l_margin + half + 6, y=y, w=half)
    pdf.set_y(y + half * 2.5 / 3.45 + 4)

    drivers = fig_drivers(an)
    if drivers or an.outlook is not None:
        pdf.add_page()
    if drivers:
        pdf.h2("Rain, temperature and the lake")
        pdf.para(drivers_text(an), size=10, color=INK)
        pdf.image_bytes(drivers, w=width)
        pdf.ln(2)
    if an.outlook is not None:
        pdf.h2("Outlook")
        pdf.para(outlook_text(an), size=10)
        _outlook_table(pdf, an, width)

    pdf.add_page()
    maps = {k: v for k, v in (maps or {}).items() if _valid_png(v)}
    if maps:
        pdf.h2("Maps")
        half = (width - 6) / 2
        y = pdf.get_y()
        if maps.get("lake"):
            pdf.image(io.BytesIO(maps["lake"]), x=pdf.l_margin, y=y, w=half)
        if maps.get("catchment"):
            pdf.image(io.BytesIO(maps["catchment"]), x=pdf.l_margin + half + 6, y=y, w=half)
        pdf.set_y(y + half + 3)
        pdf.para("Left: how often each pixel held water in 1984–2021 (light to dark blue), with the "
                 "lake outline used here. Right: the catchment (brown line) over terrain shading.",
                 size=8.5, color=INK2)
    pdf.h2("How these numbers were made")
    for p in method_paragraphs(an, bundle):
        pdf.para(p, size=9.5, color=INK, markdown=True)
    pdf.h2("Data sources")
    for ref in REFERENCES:
        pdf.para(ref, size=8.5, color=INK2, lh=4.3)
    return bytes(pdf.output())


def _outlook_table(pdf: Report, an: A.Analysis, width: float) -> None:
    o = an.outlook.table.head(12)
    cols = [("Month", 0.22), ("Median", 0.18), ("50% range", 0.25), ("80% range", 0.25),
            ("Normal", 0.10)]
    pdf.set_font("Barlow", "B", 9)
    pdf.set_text_color(*_hex(INK))
    for label, frac in cols:
        pdf.cell(width * frac, 6, label, border="B")
    pdf.ln()
    pdf.set_font("Barlow", "", 9)
    for ts, r in o.iterrows():
        vals = [A.month_label(ts), A.fmt_area(r["median"], unit=False),
                f"{A.fmt_area(r['p25'], unit=False)} to {A.fmt_area(r['p75'], unit=False)}",
                f"{A.fmt_area(r['p10'], unit=False)} to {A.fmt_area(r['p90'], unit=False)}",
                A.fmt_area(r["normal"], unit=False)]
        for v, (_, frac) in zip(vals, cols):
            pdf.cell(width * frac, 5.4, v)
        pdf.ln()
    pdf.set_font("Barlow", "", 8.5)
    pdf.set_text_color(*_hex(INK2))
    pdf.ln(1)
    pdf.multi_cell(0, 4.3, "All values in km².", new_x="LMARGIN", new_y="NEXT")


# ---------------------------------------------------------------------------
# Shared text (also used by the app)
# ---------------------------------------------------------------------------

LAKE_TYPES = {1: "Lake", 2: "Reservoir", 3: "Regulated lake"}


def lede_text(site: dict, bundle: dict) -> str:
    parts = []
    kind = site.get("kind")
    hl = (bundle.get("hydrolake") or {})
    hl = hl.get("properties", hl) if isinstance(hl, dict) else {}
    if not kind and hl.get("Lake_type") in LAKE_TYPES:
        kind = LAKE_TYPES[hl["Lake_type"]].lower()
    country = site.get("country") or hl.get("Country")
    if kind and country:
        parts.append(f"{kind[0].upper() + kind[1:]} in {country}.")
    elif kind:
        parts.append(f"{kind[0].upper() + kind[1:]}.")
    if site.get("note"):
        parts.append(site["note"])
    fi = bundle.get("footprint_info") or {}
    ci = bundle.get("catchment_info") or {}
    if fi.get("ever_water_km2"):
        s = f"Water has covered up to {A.fmt_area(fi['ever_water_km2'])} here since 1984"
        if ci.get("area_km2") and ci.get("level"):
            s += f", fed by a catchment of {A.fmt_area(ci['area_km2'])}"
        parts.append(s + ".")
    return " ".join(parts)


def readings_rows(an: A.Analysis) -> list[tuple[str, str, str]]:
    rows = []
    lt = an.latest
    if lt:
        mname = A.MONTH_NAME[lt["date"].month - 1]
        share = lt["pct_of_normal"]
        rows.append((f"Share of a typical {mname}",
                     f"{share:.0f}%" if share is not None and share == share else "n/a",
                     f"Normal is {A.fmt_area(lt['normal_median'])} "
                     f"({an.normals.period[0]}–{an.normals.period[1]})"))
        rows.append((f"Rank among {lt['n_same_month']} {mname}s",
                     f"{A.ordinal(lt['rank_low'])} smallest",
                     f"{A.ordinal(int(round(lt['percentile'])))} percentile"
                     if lt["percentile"] is not None else "No percentile"))
    tr = an.trend_annual
    if tr:
        sig = "significant" if tr["significant"] else "not significant"
        rows.append(("Yearly mean, change per decade",
                     f"{A.fmt_signed(tr['slope_per_decade'], 2)} km²",
                     f"{tr['first']} to {tr['last']}, {sig} ({A.fmt_p(tr['p'])})"))
    cs = an.climate_summary
    if cs:
        spi = cs.get("spi12")
        rows.append(("Catchment rain, last 12 months", f"{cs['p12_pct_normal']:.0f}%",
                     (f"of normal; SPI-12 {spi:+.1f}, {A.spi_class(spi)}" if spi is not None
                      else f"of normal, to {A.month_label(cs['date'])}")))
    return rows


def drivers_text(an: A.Analysis) -> str:
    parts = []
    if an.response is not None and len(an.response):
        k = int(an.response["r"].idxmax())
        r = float(an.response.loc[k, "r"])
        parts.append(f"The lake follows rain summed over about {k} months most closely "
                     f"(Spearman r = {r:.2f}).")
    cs = an.climate_summary
    if cs:
        parts.append(f"In the 12 months to {A.month_label(cs['date'], long=True)}, the catchment "
                     f"received {cs['p12_pct_normal']:.0f}% of its normal precipitation.")
        if cs.get("temp_anom12") is not None:
            parts.append(f"Air temperature averaged {cs['temp_anom12']:+.1f} °C against the "
                         "reference period.")
    return " ".join(parts) or "No climate data were available."


def outlook_text(an: A.Analysis) -> str:
    o = an.outlook
    if o is None:
        return ""
    t = o.table
    last = t.iloc[min(5, len(t) - 1)]
    sk = o.skill
    skill_txt = ""
    if not sk.empty and 6 in sk.index:
        s6 = sk.loc[6, "skill_vs_climatology"] * 100
        skill_txt = (f" In past years, forecasts made this way six months ahead had "
                     f"{abs(s6):.0f}% {'smaller' if s6 > 0 else 'larger'} errors than simply "
                     "assuming a normal month.")
    return (f"If the lake behaves as it has after similar months in the past, it would most likely "
            f"cover about {A.fmt_area(last['median'])} in {A.month_label(t.index[min(5, len(t) - 1)], long=True)} "
            f"(80% range {A.fmt_area(last['p10'], unit=False)} to {A.fmt_area(last['p90'])}). "
            "This is a statistical outlook based on persistence of the current anomaly, not a "
            "weather or inflow forecast." + skill_txt)


def method_paragraphs(an: A.Analysis, bundle: dict) -> list[str]:
    fi = bundle.get("footprint_info") or {}
    ci = bundle.get("catchment_info") or {}
    h = an.harmonization
    out = [
        "**Outline.** The lake outline is the connected area where the JRC Global Surface Water "
        "dataset recorded water at least once between 1984 and 2021, around the chosen point "
        f"({fi.get('method', 'see app')}), grown by about 60 m.",
        "**Monthly water.** For March 1984 to December 2021 the record uses the JRC monthly water "
        "history (Landsat 5, 7 and 8). From January 2022 it uses our own classification of "
        "Landsat 8/9 and Sentinel-2 surface reflectance: a pixel is water when MNDWI exceeds "
        "NDVI or EVI and EVI is below 0.1 (Zou et al. 2018); clouds are removed with QA bits "
        "(Landsat) and Cloud Score+ (Sentinel-2). A pixel is water for the month when most of "
        "its clear observations are water.",
        "**Gaps.** Clouds hide parts of the lake in many months. Within each month, the "
        "long-term water occurrence that best separates the clear water from the clear land "
        "becomes a threshold, and hidden pixels at or above it count as water (after Zhao and "
        "Gao 2018). The shaded range in the charts shows the spread of thresholds that fit "
        "the clear pixels almost as well. Months with less than "
        f"{an.settings.min_clear_fraction:.0%} of the lake seen, or a range wider than "
        f"{an.settings.max_relative_uncertainty:.0%} of the estimate, are left out, as are "
        "isolated spikes.",
    ]
    if h.get("n"):
        txt = (f"**Joining the sources.** In {h['n']} months seen clearly by both sources (2019–2021), "
               f"the JRC area was {h['ratio']:.3f} times ours")
        txt += (", so recent months were scaled by that factor." if h.get("applied")
                else "; no adjustment was applied.")
        if h.get("mape") is not None:
            txt += f" After scaling, the typical difference was {h['mape']:.1f}%."
        out.append(txt)
    out.append(
        f"**Normals and conditions.** Each month is ranked against the same calendar month in "
        f"{an.normals.period[0]}–{an.normals.period[1]}. Classes follow USGS WaterWatch: much below "
        "normal (below the 10th percentile), below normal (10th–24th), normal (25th–75th), above "
        "normal (76th–90th) and much above normal (above the 90th).")
    out.append(
        "**Trends and rain.** Trends use the Mann-Kendall test with the Hamed and Rao correction for autocorrelation, "
        "and Sen's slope. Rain uses ERA5-Land monthly precipitation averaged over the "
        f"{'HydroBASINS catchment' if ci.get('level') else 'area around the lake'}; the standardized "
        "precipitation index is computed non-parametrically (Farahmand and AghaKouchak 2015).")
    out.append(
        "**Limits.** Surface area is not volume: steep-sided reservoirs can lose much water while "
        "their area changes little. Floating vegetation, ice, turbid shallows and terrain shadow "
        "can be misclassified. Pixels are 30 m (10–20 m for Sentinel-2), so very small lakes are "
        "uncertain.")
    return out


REFERENCES = [
    "Pekel, J.-F., Cottam, A., Gorelick, N., Belward, A. S. (2016). High-resolution mapping of "
    "global surface water and its long-term changes. Nature 540, 418–422.",
    "Zhao, G., Gao, H. (2018). Automatic correction of contaminated images for assessment of "
    "reservoir surface area dynamics. Geophysical Research Letters 45, 6092–6099.",
    "Zou, Z. et al. (2018). Divergent trends of open-surface water body area in the contiguous "
    "United States from 1984 to 2016. PNAS 115, 3810–3815.",
    "Pasquarella, V. J. et al. (2023). Cloud Score+: A Cloud and Cloud Shadow Detection Method "
    "for Sentinel-2. CVPR Workshops.",
    "Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset "
    "for land applications. Earth System Science Data 13, 4349–4383.",
    "Lehner, B., Grill, G. (2013). Global river hydrography and network routing: baseline data "
    "and new approaches to study the world's large river systems. Hydrological Processes 27, "
    "2171–2186 (HydroBASINS).",
    "Messager, M. L. et al. (2016). Estimating the volume and age of water stored in global "
    "lakes using a geo-statistical approach. Nature Communications 7, 13603 (HydroLAKES).",
    "Farahmand, A., AghaKouchak, A. (2015). A generalized framework for deriving nonparametric "
    "standardized drought indicators. Advances in Water Resources 76, 140–145.",
    "Hamed, K. H., Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated "
    "data. Journal of Hydrology 204, 182–196. Sen, P. K. (1968). JASA 63, 1379–1389.",
    "Contains modified Copernicus Sentinel data and Copernicus Climate Change Service "
    "information; Landsat data courtesy of the U.S. Geological Survey.",
]
