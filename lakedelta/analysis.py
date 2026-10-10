"""Turn satellite water observations into a lake record and its statistics.

Everything in this module is plain numpy/pandas so it can be unit-tested
without Earth Engine. The Earth Engine layer delivers, for every month, the
area of the lake footprint split by (class x long-term water occurrence):

    class 0 = observed land, 1 = observed water, 2 = no clear observation
    occurrence = JRC Global Surface Water occurrence 0..100 %

From those histograms we estimate the full water area of partly cloudy months
(occurrence-threshold gap filling, after Zhao & Gao 2018), join the JRC record
(1984-2021) to our own Landsat 8/9 + Sentinel-2 classification (2019 onward),
and derive normals, anomalies, trends, a short-range outlook and the link to
catchment climate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

from .config import (
    MONTH_ABBR,
    MONTH_NAME,
    AnalysisSettings,
    Condition,
    condition_for,
)

N_OCC = 101
LAND, WATER, NODATA = 0, 1, 2
M2_PER_KM2 = 1e6

SOURCE_JRC = "JRC Global Surface Water (Landsat)"
SOURCE_OPT = "Landsat 8/9 and Sentinel-2"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_area(km2: float | None, unit: bool = True) -> str:
    """Area with three significant figures and thousands separators."""
    if km2 is None or not np.isfinite(km2):
        return "n/a"
    a = abs(km2)
    if a >= 99.95:
        s = f"{km2:,.0f}"
    elif a == 0:
        s = "0"
    else:
        s = format(km2, "#.3g").rstrip(".")
    return f"{s} km²" if unit else s


def fmt_signed(x: float, digits: int = 1, unit: str = "") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "+" if x > 0 else ("−" if x < 0 else "±")
    return f"{sign}{abs(x):,.{digits}f}{unit}"


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def month_label(ts: pd.Timestamp, long: bool = False) -> str:
    names = MONTH_NAME if long else MONTH_ABBR
    return f"{names[ts.month - 1]} {ts.year}"


# ---------------------------------------------------------------------------
# Season windows (may wrap the year end, e.g. Dec-Feb)
# ---------------------------------------------------------------------------

def season_start_month(months: Sequence[int]) -> int | None:
    """First month of a season that wraps the year end, else None."""
    ms = sorted(set(int(m) for m in months))
    if not ms or len(ms) == 12 or 12 not in ms or 1 not in ms:
        return None
    m = 12
    while (m - 1 if m > 1 else 12) in ms and m - 1 != 0:
        m -= 1
    return m


def season_year(ts: pd.Timestamp, months: Sequence[int]) -> int:
    """Season-years are named after the year in which the season ends."""
    start = season_start_month(months)
    if start is not None and ts.month >= start:
        return ts.year + 1
    return ts.year


def season_label(months: Sequence[int]) -> str:
    ms = sorted(set(int(m) for m in months))
    if not ms:
        return "no months"
    if len(ms) == 12:
        return "all months"
    start = season_start_month(ms)
    if start is not None:
        order = [m for m in range(start, 13)] + [m for m in range(1, start) if m in ms]
        order = [m for m in order if m in ms]
    else:
        order = ms
    contiguous = all(((order[i] % 12) + 1) == order[i + 1] for i in range(len(order) - 1))
    if contiguous and len(order) >= 2:
        return f"{MONTH_ABBR[order[0] - 1]}–{MONTH_ABBR[order[-1] - 1]}"
    return ", ".join(MONTH_ABBR[m - 1] for m in order)


def season_date_ranges(season_yr: int, months: Sequence[int]) -> list[tuple[str, str]]:
    """Calendar date ranges [start, end) covering one season-year."""
    start = season_start_month(months)
    out = []
    for m in sorted(set(int(x) for x in months)):
        y = season_yr - 1 if (start is not None and m >= start) else season_yr
        d0 = pd.Timestamp(year=y, month=m, day=1)
        d1 = d0 + pd.offsets.MonthBegin(1)
        out.append((d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")))
    return out


# ---------------------------------------------------------------------------
# Histogram gap filling
# ---------------------------------------------------------------------------

def groups_to_hist(groups: Iterable[dict] | None) -> np.ndarray:
    """Earth Engine grouped sums -> array (3, 101) of km² by class x occurrence."""
    h = np.zeros((3, N_OCC))
    if not groups:
        return h
    for g in groups:
        try:
            code = int(round(float(g["code"])))
            area = float(g.get("sum") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        cls, occ = divmod(code, N_OCC)
        if 0 <= cls <= 2 and 0 <= occ < N_OCC:
            h[cls, occ] += area / M2_PER_KM2
    return h


@dataclass
class MonthFill:
    area: float            # best estimate of water area, km²
    lo: float              # lower bound from the range of near-optimal thresholds
    hi: float              # upper bound
    observed_water: float  # water seen in clear pixels
    filled: float          # water assigned to unobserved pixels
    clear_fraction: float  # clear share of the potential-water zone
    threshold: int         # occurrence threshold used to fill gaps (1..101)
    consistency: float     # share of clear pixels the threshold explains
    potential: float       # area that could hold water (occurrence >= 1)


def fill_month(h: np.ndarray, tol: float = 0.005) -> MonthFill:
    """Estimate total water area for one month from its occurrence histogram.

    Clear pixels tell us which occurrence values were wet this month. We pick
    the occurrence threshold t that best separates clear water (occurrence >= t)
    from clear land (occurrence < t), then count every unobserved pixel with
    occurrence >= t as water. Thresholds whose error is within ``tol`` of the
    best (as a share of the clear area) give the uncertainty range.
    """
    land, water, nod = h[LAND], h[WATER], h[NODATA]
    pot_total = float(land[1:].sum() + water[1:].sum() + nod[1:].sum())
    clear_pot = float(land[1:].sum() + water[1:].sum())
    observed = float(water.sum())  # includes water seen outside the historic extent
    clear_fraction = clear_pot / pot_total if pot_total > 0 else 0.0

    w = water.copy()
    w[0] = 0.0
    l = land.copy()
    l[0] = 0.0
    cum_w = np.concatenate([[0.0], np.cumsum(w)])
    cum_l = np.concatenate([[0.0], np.cumsum(l)])
    ts = np.arange(1, N_OCC + 1)
    err = cum_w[ts] + (cum_l[-1] - cum_l[ts])
    e_min = float(err.min())
    near = ts[err <= e_min + tol * max(clear_pot, 1e-12)]
    t_star = int(np.round(np.median(near)))

    tail_n = np.concatenate([np.cumsum(nod[::-1])[::-1], [0.0]])
    filled = float(tail_n[t_star])
    area = observed + filled
    lo = observed + float(tail_n[int(near.max())])
    hi = observed + float(tail_n[int(near.min())])
    consistency = 1.0 - e_min / clear_pot if clear_pot > 0 else 0.0
    return MonthFill(area, lo, hi, observed, filled, clear_fraction, t_star,
                     consistency, pot_total + float(water[0]))


def _month_start(t_ms: float) -> pd.Timestamp:
    return pd.Timestamp(int(t_ms), unit="ms").to_period("M").to_timestamp()


def months_from_records(records: Sequence[dict], source: str) -> pd.DataFrame:
    """Records from the EE layer -> one row per month with the filled estimate."""
    rows = []
    for r in records or []:
        if r.get("t") is None:
            continue
        f = fill_month(groups_to_hist(r.get("groups")))
        rows.append({
            "date": _month_start(r["t"]),
            "area": f.area, "lo": f.lo, "hi": f.hi,
            "observed": f.observed_water, "filled": f.filled,
            "clear_fraction": f.clear_fraction, "threshold": f.threshold,
            "consistency": f.consistency, "potential": f.potential,
            "n_scenes": r.get("n_scenes"), "source": source,
        })
    cols = ["area", "lo", "hi", "observed", "filled", "clear_fraction", "threshold",
            "consistency", "potential", "n_scenes", "source"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows).groupby("date").last().sort_index()
    df.index.name = "date"
    return df[cols]


def _flag_usable(df: pd.DataFrame, settings: AnalysisSettings) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    denom = np.maximum(df["area"].to_numpy(float), np.maximum(0.01 * df["potential"].to_numpy(float), 1e-6))
    rel_unc = (df["hi"] - df["lo"]).to_numpy(float) / denom
    ok = (
        (df["clear_fraction"].to_numpy(float) >= settings.min_clear_fraction)
        & (rel_unc <= settings.max_relative_uncertainty)
        & (df["potential"].to_numpy(float) > 0)
    )
    return pd.Series(ok, index=df.index)


def hampel_flags(values: pd.Series, window: int = 2, k: float = 4.0,
                 min_rel: float = 0.08) -> pd.Series:
    """Flag isolated spikes: far from the median of their +-window neighbours."""
    x = values.to_numpy(float)
    out = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        lo, hi = max(0, i - window), min(len(x), i + window + 1)
        neigh = np.array([x[j] for j in range(lo, hi) if j != i and np.isfinite(x[j])])
        if len(neigh) < 2:
            continue
        med = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - med))) * 1.4826
        thr = max(k * mad, min_rel * abs(med), 1e-9)
        out[i] = abs(x[i] - med) > thr
    return pd.Series(out, index=values.index)


@dataclass
class Record:
    monthly: pd.DataFrame
    harmonization: dict
    potential_km2: float
    notes: list[str] = field(default_factory=list)


def build_record(jrc_records: Sequence[dict], opt_records: Sequence[dict],
                 settings: AnalysisSettings | None = None) -> Record:
    """Join both sources into one gap-filled monthly record with quality flags."""
    settings = settings or AnalysisSettings()
    notes: list[str] = []
    jrc = months_from_records(jrc_records, SOURCE_JRC)
    opt = months_from_records(opt_records, SOURCE_OPT)
    for df in (jrc, opt):
        df["usable"] = _flag_usable(df, settings) if not df.empty else pd.Series(dtype=bool)

    harm = {"applied": False, "ratio": 1.0, "n": 0, "mape": None, "r": None, "pairs": None}
    if not jrc.empty and not opt.empty:
        ov = jrc[["area", "usable"]].join(opt[["area", "usable"]], how="inner",
                                          lsuffix="_jrc", rsuffix="_opt")
        ov = ov[ov["usable_jrc"] & ov["usable_opt"] & (ov["area_opt"] > 0) & (ov["area_jrc"] > 0)]
        harm["n"] = int(len(ov))
        harm["pairs"] = ov[["area_jrc", "area_opt"]].copy()
        if len(ov) >= 6:
            ratio = float(np.median(ov["area_jrc"] / ov["area_opt"]))
            harm["ratio"] = ratio
            adj = ov["area_opt"] * ratio
            harm["mape"] = float(np.median(np.abs(adj - ov["area_jrc"]) / ov["area_jrc"]) * 100)
            if ov["area_jrc"].std() > 0 and ov["area_opt"].std() > 0 and len(ov) >= 3:
                harm["r"] = float(np.corrcoef(ov["area_jrc"], ov["area_opt"])[0, 1])
            if settings.harmonize_sources and 0.85 <= ratio <= 1.15:
                harm["applied"] = True
                for col in ("area", "lo", "hi"):
                    opt[col] = opt[col] * ratio
            elif settings.harmonize_sources:
                notes.append(
                    f"The two sources disagree by a factor of {ratio:.2f} where they overlap, "
                    "so recent months were left unadjusted. Check the lake outline.")
        else:
            notes.append("Too few clear months where both sources overlap to compare them; "
                         "recent months are shown unadjusted.")

    if not jrc.empty and not opt.empty:
        jrc_end = jrc.index.max()
        rec = pd.concat([jrc[jrc.index <= jrc_end], opt[opt.index > jrc_end]])
    elif not jrc.empty:
        rec = jrc
    else:
        rec = opt
    if rec.empty:
        raise ValueError("No monthly water observations were returned for this lake.")

    rec = rec.sort_index()
    full = pd.date_range(rec.index.min(), rec.index.max(), freq="MS", name="date")
    rec = rec.reindex(full)
    rec["usable"] = rec["usable"].astype("boolean").fillna(False).astype(bool)
    usable_area = rec["area"].where(rec["usable"])
    rec["outlier"] = hampel_flags(usable_area)
    rec["valid"] = rec["usable"] & ~rec["outlier"]
    rec["raw_area"] = rec["observed"] + rec["filled"]

    potential = float(np.nanmedian(rec["potential"])) if rec["potential"].notna().any() else float("nan")
    n_out = int(rec["outlier"].sum())
    if n_out:
        notes.append(f"{n_out} isolated spike{'s' if n_out != 1 else ''} were set aside as likely "
                     "cloud or shadow artefacts.")
    return Record(rec, harm, potential, notes)


# ---------------------------------------------------------------------------
# Normals, percentiles and anomalies
# ---------------------------------------------------------------------------

@dataclass
class Normals:
    table: pd.DataFrame                 # index 1..12
    samples: dict[int, np.ndarray]      # month -> reference values
    period: tuple[int, int]
    used_full_record: bool


def monthly_normals(rec: pd.DataFrame, ref: tuple[int, int] = (1991, 2020),
                    min_years: int = 10) -> Normals:
    v = rec.loc[rec["valid"], "area"].dropna()
    df = pd.DataFrame({"area": v.to_numpy(), "month": v.index.month, "year": v.index.year})
    in_ref = df[(df["year"] >= ref[0]) & (df["year"] <= ref[1])]
    counts = in_ref.groupby("month").size().reindex(range(1, 13), fill_value=0)
    use_full = bool((counts < min_years).any())
    base = df if use_full else in_ref
    if base.empty:
        raise ValueError("No usable months to build normals from.")
    g = base.groupby("month")["area"]
    table = pd.DataFrame({
        "n": g.size(), "median": g.median(), "mean": g.mean(), "std": g.std(),
        "q10": g.quantile(0.10), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
        "q90": g.quantile(0.90), "min": g.min(), "max": g.max(),
    }).reindex(range(1, 13))
    samples = {m: np.sort(base.loc[base["month"] == m, "area"].to_numpy()) for m in range(1, 13)}
    period = (int(base["year"].min()), int(base["year"].max()))
    return Normals(table, samples, period, use_full)


def percentile_of(x: float, sample: np.ndarray) -> float:
    if sample is None or len(sample) == 0 or not np.isfinite(x):
        return float("nan")
    return float(stats.percentileofscore(sample, x, kind="mean"))


def robust_scale(normals: Normals, potential: float) -> pd.Series:
    t = normals.table
    scale = (t["q75"] - t["q25"]) / 1.349
    floor = np.maximum(0.01 * t["median"].fillna(0), 0.001 * (potential if np.isfinite(potential) else 0))
    floor = np.maximum(floor, 1e-6)
    return np.maximum(scale.fillna(0), floor)


def add_anomalies(rec: pd.DataFrame, normals: Normals, potential: float) -> pd.DataFrame:
    rec = rec.copy()
    months = rec.index.month
    t = normals.table
    med = t["median"].reindex(months).to_numpy()
    scale = robust_scale(normals, potential).reindex(months).to_numpy()
    rec["normal_median"] = med
    rec["normal_q25"] = t["q25"].reindex(months).to_numpy()
    rec["normal_q75"] = t["q75"].reindex(months).to_numpy()
    rec["normal_q10"] = t["q10"].reindex(months).to_numpy()
    rec["normal_q90"] = t["q90"].reindex(months).to_numpy()
    rec["anomaly"] = rec["area"] - med
    rec["pct_of_normal"] = rec["area"] / med * 100.0
    rec["z"] = (rec["area"] - med) / scale
    pct = np.full(len(rec), np.nan)
    for i, (ts, row) in enumerate(rec.iterrows()):
        if row["valid"]:
            pct[i] = percentile_of(row["area"], normals.samples.get(ts.month))
    rec["percentile"] = pct
    rec["condition"] = [c.key if (c := condition_for(p)) else None for p in pct]
    for col in ("anomaly", "pct_of_normal", "z"):
        rec.loc[~rec["valid"], col] = np.nan
    return rec


# ---------------------------------------------------------------------------
# Annual series and trends
# ---------------------------------------------------------------------------

def annual_table(rec: pd.DataFrame, season_months: Sequence[int]) -> pd.DataFrame:
    v = rec.loc[rec["valid"], "area"].dropna()
    if v.empty:
        return pd.DataFrame(columns=["n", "mean", "min", "max", "season_n", "season_mean"])
    by_year = v.groupby(v.index.year)
    out = pd.DataFrame({"n": by_year.size(), "mean": by_year.mean(),
                        "min": by_year.min(), "max": by_year.max()})
    out.loc[out["n"] < 8, "mean"] = np.nan
    sm = set(int(m) for m in season_months)
    sv = v[v.index.month.isin(sm)]
    if not sv.empty:
        sy = pd.Index([season_year(ts, season_months) for ts in sv.index], name="year")
        g = sv.groupby(sy)
        season = pd.DataFrame({"season_n": g.size(), "season_mean": g.mean()})
        need = math.ceil(len(sm) / 2)
        season.loc[season["season_n"] < need, "season_mean"] = np.nan
        out = out.join(season, how="outer")
    else:
        out["season_n"] = 0
        out["season_mean"] = np.nan
    out.index.name = "year"
    return out.sort_index()


def trend_test(series: pd.Series, min_n: int = 8) -> dict | None:
    s = series.dropna()
    if len(s) < min_n:
        return None
    years = s.index.to_numpy(float)
    vals = s.to_numpy(float)
    if np.allclose(vals, vals[0]):
        return {"slope_per_decade": 0.0, "ci": (0.0, 0.0), "p": 1.0, "tau": 0.0, "n": len(s),
                "first": int(years[0]), "last": int(years[-1]), "significant": False,
                "relative_per_decade": 0.0}
    res = mk.hamed_rao_modification_test(vals)
    sen = stats.theilslopes(vals, years, alpha=0.95)
    mean = float(np.mean(vals))
    slope = float(sen.slope) * 10.0
    return {
        "slope_per_decade": slope,
        "ci": (float(sen.low_slope) * 10.0, float(sen.high_slope) * 10.0),
        "intercept": float(sen.intercept),
        "p": float(res.p), "tau": float(res.Tau), "n": int(len(s)),
        "first": int(years[0]), "last": int(years[-1]),
        "significant": bool(res.p < 0.05),
        "relative_per_decade": (slope / mean * 100.0) if mean > 0 else float("nan"),
    }


def seasonal_kendall(rec: pd.DataFrame, min_years: int = 8) -> dict | None:
    v = rec["area"].where(rec["valid"])
    if v.dropna().index.year.nunique() < min_years:
        return None
    start = pd.Timestamp(year=v.index.min().year, month=1, day=1)
    end = pd.Timestamp(year=v.index.max().year, month=12, day=1)
    full = v.reindex(pd.date_range(start, end, freq="MS"))
    res = mk.seasonal_test(full.to_numpy(float), period=12)
    return {"p": float(res.p), "tau": float(res.Tau),
            "slope_per_decade": float(res.slope) * 10.0,  # slope per season-year step
            "significant": bool(res.p < 0.05)}


# ---------------------------------------------------------------------------
# Latest condition and narrative facts
# ---------------------------------------------------------------------------

def latest_status(rec: pd.DataFrame, normals: Normals) -> dict | None:
    valid = rec[rec["valid"]]
    if valid.empty:
        return None
    ts = valid.index.max()
    row = valid.loc[ts]
    same_month = valid[valid.index.month == ts.month]
    earlier = same_month[same_month.index < ts]
    info: dict = {
        "date": ts, "area": float(row["area"]), "lo": float(row["lo"]), "hi": float(row["hi"]),
        "percentile": float(row["percentile"]) if np.isfinite(row["percentile"]) else None,
        "condition": condition_for(row["percentile"]),
        "pct_of_normal": float(row["pct_of_normal"]),
        "normal_median": float(row["normal_median"]),
        "normal_q25": float(row["normal_q25"]), "normal_q75": float(row["normal_q75"]),
        "source": row["source"], "clear_fraction": float(row["clear_fraction"]),
        "record_years": int(same_month.index.year.nunique()),
        "first_year": int(valid.index.min().year),
        "is_record_low": bool(len(earlier) and row["area"] < earlier["area"].min()),
        "is_record_high": bool(len(earlier) and row["area"] > earlier["area"].max()),
        "months_since_observation": None,
    }
    # Same month one year earlier
    prev = ts - pd.DateOffset(years=1)
    if prev in valid.index:
        info["year_ago"] = float(valid.loc[prev, "area"])
        info["change_vs_year_ago"] = info["area"] - info["year_ago"]
    else:
        info["year_ago"] = None
        info["change_vs_year_ago"] = None
    # "lowest/highest <month> since <year>"
    info["lowest_since"] = None
    info["highest_since"] = None
    if len(earlier):
        lower_or_equal = earlier[earlier["area"] <= row["area"]]
        higher_or_equal = earlier[earlier["area"] >= row["area"]]
        if info["percentile"] is not None and info["percentile"] < 50:
            info["lowest_since"] = int(lower_or_equal.index.max().year) if len(lower_or_equal) else None
        if info["percentile"] is not None and info["percentile"] > 50:
            info["highest_since"] = int(higher_or_equal.index.max().year) if len(higher_or_equal) else None
    # Length of the current spell below (or above) the normal range
    spell = 0
    spell_start = None
    side = None
    if info["percentile"] is not None:
        side = "below" if info["percentile"] < 25 else ("above" if info["percentile"] > 75 else None)
    if side:
        for d, p in valid["percentile"].iloc[::-1].items():
            if (side == "below" and p < 25) or (side == "above" and p > 75):
                spell += 1
                spell_start = d
            else:
                break
    info["spell_side"] = side
    info["spell_months"] = spell
    info["spell_start"] = spell_start
    # Rank among the same calendar month (1 = lowest)
    vals = np.sort(same_month["area"].to_numpy())
    info["rank_low"] = int(np.searchsorted(vals, row["area"], side="left") + 1)
    info["n_same_month"] = int(len(vals))
    return info


def headline(name: str, latest: dict | None) -> str:
    if not latest:
        return f"No recent clear-sky observation of {name} was good enough to report."
    ts = latest["date"]
    month = MONTH_NAME[ts.month - 1]
    pct = latest["pct_of_normal"]
    if pct is not None and np.isfinite(pct):
        parts = [f"In {month} {ts.year}, {name} covered {fmt_area(latest['area'])}, "
                 f"{pct:.0f}% of its typical {month} extent."]
    else:
        parts = [f"In {month} {ts.year}, {name} covered {fmt_area(latest['area'])}."]
    pctl = latest["percentile"]
    low_since = latest.get("lowest_since")
    high_since = latest.get("highest_since")
    if latest["is_record_low"]:
        parts.append(f"No earlier {month} since {latest['first_year']} was smaller.")
    elif latest["is_record_high"]:
        parts.append(f"No earlier {month} since {latest['first_year']} was larger.")
    elif low_since and pctl is not None and pctl < 25 and ts.year - low_since >= 3:
        parts.append(f"It was the smallest {month} extent since {low_since}.")
    elif high_since and pctl is not None and pctl > 75 and ts.year - high_since >= 3:
        parts.append(f"It was the largest {month} extent since {high_since}.")
    elif latest.get("change_vs_year_ago") is not None:
        ch = latest["change_vs_year_ago"]
        if abs(ch) >= 0.005 * max(latest["area"], 1e-9):
            more = "more" if ch > 0 else "less"
            parts.append(f"That is {fmt_area(abs(ch))} {more} than a year earlier.")
    start = latest.get("spell_start")
    if latest.get("spell_side") and latest.get("spell_months", 0) >= 3 and start is not None:
        parts.append(f"It has stayed {latest['spell_side']} its normal range in every clear month "
                     f"since {MONTH_NAME[start.month - 1]} {start.year}.")
    return " ".join(parts)


def fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


# ---------------------------------------------------------------------------
# Outlook: damped persistence of the standardized anomaly
# ---------------------------------------------------------------------------

@dataclass
class Outlook:
    table: pd.DataFrame     # index = target month; median, p10, p25, p75, p90, normal
    skill: pd.DataFrame     # index = horizon (months)
    issued_from: pd.Timestamp
    z0: float


def build_outlook(rec: pd.DataFrame, normals: Normals, potential: float,
                  horizon: int = 12, min_years: int = 8, min_pairs: int = 24) -> Outlook | None:
    """Forecast the next months from today's anomaly and how anomalies decayed before.

    For each lead time h we regress the standardized anomaly h months ahead on
    today's anomaly (pooled over all start months), and turn the residual
    spread into a range. Leave-one-year-out hindcasts score the model against
    climatology (always forecast the normal) and pure persistence.
    """
    v = rec["area"].where(rec["valid"])
    if v.dropna().index.year.nunique() < min_years:
        return None
    s = v.interpolate(limit=2, limit_area="inside")
    months = s.index.month
    med = normals.table["median"].reindex(months).to_numpy()
    scale_tbl = robust_scale(normals, potential)
    scale = scale_tbl.reindex(months).to_numpy()
    z = (s.to_numpy(float) - med) / scale
    t0 = v.last_valid_index()
    i0 = s.index.get_loc(t0)
    z0 = float(z[i0])
    years = s.index.year.to_numpy()
    cap = potential if np.isfinite(potential) and potential > 0 else np.inf

    rows, skill_rows = [], []
    for h in range(1, horizon + 1):
        x, y = z[:-h], z[h:]
        oy, ty = years[:-h], years[h:]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_pairs:
            continue
        x, y, oy, ty = x[ok], y[ok], oy[ok], ty[ok]
        beta, alpha = _ols(x, y)
        resid = y - (alpha + beta * x)
        q = np.quantile(resid, [0.10, 0.25, 0.50, 0.75, 0.90])
        zc = alpha + beta * z0
        target = t0 + pd.DateOffset(months=h)
        m = target.month
        mm, sc = float(normals.table.loc[m, "median"]), float(scale_tbl.loc[m])
        vals = np.clip(mm + sc * (zc + q), 0, cap)
        rows.append({"date": target, "p10": vals[0], "p25": vals[1], "median": vals[2],
                     "p75": vals[3], "p90": vals[4], "normal": mm, "lead": h})

        # Leave-one-year-out hindcast: errors in standardized units, then km²
        err_m, err_c, err_p = [], [], []
        target_scales = scale[h:][ok]
        for yr in np.unique(ty):
            test = ty == yr
            train = (ty != yr) & (oy != yr)
            if train.sum() < min_pairs // 2 or test.sum() == 0:
                continue
            b, a = _ols(x[train], y[train])
            pred = a + b * x[test]
            sc_t = target_scales[test]
            err_m.extend(np.abs((pred - y[test]) * sc_t))
            err_c.extend(np.abs((0 - y[test]) * sc_t))
            err_p.extend(np.abs((x[test] - y[test]) * sc_t))
        if err_m:
            mae_m, mae_c, mae_p = float(np.mean(err_m)), float(np.mean(err_c)), float(np.mean(err_p))
            skill_rows.append({"lead": h, "mae_model": mae_m, "mae_climatology": mae_c,
                               "mae_persistence": mae_p,
                               "skill_vs_climatology": 1 - mae_m / mae_c if mae_c > 0 else np.nan,
                               "n": len(err_m), "beta": beta})
    if not rows:
        return None
    table = pd.DataFrame(rows).set_index("date")
    skill = pd.DataFrame(skill_rows).set_index("lead") if skill_rows else pd.DataFrame()
    return Outlook(table, skill, t0, z0)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = float(np.mean(x)), float(np.mean(y))
    vx = float(np.mean((x - xm) ** 2))
    beta = float(np.mean((x - xm) * (y - ym)) / vx) if vx > 0 else 0.0
    return beta, ym - beta * xm


# ---------------------------------------------------------------------------
# Climate of the catchment
# ---------------------------------------------------------------------------

def climate_frame(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for r in records or []:
        if r.get("t") is None:
            continue
        p = r.get("total_precipitation_sum")
        t = r.get("temperature_2m")
        ro = r.get("runoff_sum")
        rows.append({
            "date": _month_start(r["t"]),
            "precip_mm": max(float(p) * 1000.0, 0.0) if p is not None else np.nan,
            "temp_c": float(t) - 273.15 if t is not None else np.nan,
            "runoff_mm": max(float(ro) * 1000.0, 0.0) if ro is not None else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["precip_mm", "temp_c", "runoff_mm"],
                            index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows).groupby("date").last().sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS", name="date")
    return df.reindex(full)


def standardize_nonparametric(series: pd.Series, min_n: int = 10) -> pd.Series:
    """Empirical standardized index by calendar month (Gringorten plotting positions).

    Farahmand & AghaKouchak (2015): the non-parametric counterpart of SPI.
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for m in range(1, 13):
        sel = series[(series.index.month == m)].dropna()
        n = len(sel)
        if n < min_n:
            continue
        ranks = sel.rank(method="average")
        p = (ranks - 0.44) / (n + 0.12)
        out.loc[sel.index] = stats.norm.ppf(p.to_numpy())
    return out


SPI_SCALES = (1, 3, 6, 12, 24, 36)


def climate_indices(clim: pd.DataFrame, ref: tuple[int, int] = (1991, 2020)) -> pd.DataFrame:
    df = clim.copy()
    p = df["precip_mm"]
    for k in SPI_SCALES:
        acc = p.rolling(k, min_periods=k).sum()
        df[f"p{k}"] = acc
        df[f"spi{k}"] = standardize_nonparametric(acc)
    p12 = df["p12"]
    norm12 = p12.groupby(p12.index.month).median()
    df["p12_pct_normal"] = p12 / norm12.reindex(p12.index.month).to_numpy() * 100
    t = df["temp_c"]
    in_ref = (t.index.year >= ref[0]) & (t.index.year <= ref[1])
    t_norm = t[in_ref].groupby(t[in_ref].index.month).mean()
    if t_norm.isna().all() or len(t_norm) < 12:
        t_norm = t.groupby(t.index.month).mean()
    df["temp_anom"] = t - t_norm.reindex(t.index.month).to_numpy()
    df["temp_anom12"] = df["temp_anom"].rolling(12, min_periods=10).mean()
    ro12 = df["runoff_mm"].rolling(12, min_periods=12).sum()
    df["runoff12"] = ro12
    ro_norm = ro12.groupby(ro12.index.month).median()
    df["runoff12_pct_normal"] = ro12 / ro_norm.reindex(ro12.index.month).to_numpy() * 100
    return df


def response_timescale(rec: pd.DataFrame, clim: pd.DataFrame, max_k: int = 36,
                       min_n: int = 36) -> pd.DataFrame | None:
    """Spearman correlation of the lake anomaly with precipitation summed over k months."""
    if clim is None or clim.empty or "z" not in rec:
        return None
    z = rec["z"]
    p = clim["precip_mm"]
    rows = []
    for k in range(1, max_k + 1):
        spi = standardize_nonparametric(p.rolling(k, min_periods=k).sum())
        joined = pd.concat([z, spi], axis=1, keys=["z", "spi"], sort=True).dropna()
        if len(joined) < min_n:
            continue
        r = stats.spearmanr(joined["z"], joined["spi"]).statistic
        rows.append({"k": k, "r": float(r), "n": len(joined)})
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("k")


def climate_summary(ci: pd.DataFrame) -> dict | None:
    if ci is None or ci.empty:
        return None
    last = ci["p12"].last_valid_index()
    if last is None:
        return None
    row = ci.loc[last]
    return {
        "date": last,
        "p12": float(row["p12"]),
        "p12_pct_normal": float(row["p12_pct_normal"]),
        "spi12": float(row["spi12"]) if np.isfinite(row["spi12"]) else None,
        "temp_anom12": float(row["temp_anom12"]) if np.isfinite(row["temp_anom12"]) else None,
        "runoff12_pct_normal": float(row["runoff12_pct_normal"])
        if np.isfinite(row.get("runoff12_pct_normal", np.nan)) else None,
    }


def spi_class(spi: float | None) -> str:
    if spi is None or not np.isfinite(spi):
        return "n/a"
    if spi <= -2:
        return "extremely dry"
    if spi <= -1.5:
        return "severely dry"
    if spi <= -1:
        return "moderately dry"
    if spi < 1:
        return "near normal"
    if spi < 1.5:
        return "moderately wet"
    if spi < 2:
        return "very wet"
    return "extremely wet"


# ---------------------------------------------------------------------------
# One-stop analysis
# ---------------------------------------------------------------------------

@dataclass
class Analysis:
    record: pd.DataFrame
    normals: Normals
    annual: pd.DataFrame
    trend_annual: dict | None
    trend_season: dict | None
    seasonal_trend: dict | None
    latest: dict | None
    outlook: Outlook | None
    climate: pd.DataFrame | None
    climate_summary: dict | None
    response: pd.DataFrame | None
    harmonization: dict
    potential_km2: float
    notes: list[str]
    settings: AnalysisSettings


def analyze(jrc_records: Sequence[dict], opt_records: Sequence[dict],
            climate_records: Sequence[dict] | None,
            settings: AnalysisSettings | None = None) -> Analysis:
    settings = settings or AnalysisSettings()
    record = build_record(jrc_records, opt_records, settings)
    normals = monthly_normals(record.monthly, settings.reference_period)
    rec = add_anomalies(record.monthly, normals, record.potential_km2)
    annual = annual_table(rec, settings.season_months)
    trend_annual = trend_test(annual["mean"]) if "mean" in annual else None
    trend_season = trend_test(annual["season_mean"]) if "season_mean" in annual else None
    seasonal = seasonal_kendall(rec)
    latest = latest_status(rec, normals)
    outlook = build_outlook(rec, normals, record.potential_km2, settings.outlook_horizon)
    clim = ci = resp = csum = None
    notes = list(record.notes)
    if normals.used_full_record:
        notes.append(f"Fewer than ten clear years per month fell inside {settings.reference_period[0]}–"
                     f"{settings.reference_period[1]}, so normals use the whole record "
                     f"({normals.period[0]}–{normals.period[1]}).")
    if climate_records:
        clim = climate_frame(climate_records)
        if not clim.empty:
            ci = climate_indices(clim, settings.reference_period)
            resp = response_timescale(rec, clim)
            csum = climate_summary(ci)
    return Analysis(rec, normals, annual, trend_annual, trend_season, seasonal, latest, outlook,
                    ci, csum, resp, record.harmonization, record.potential_km2, notes, settings)


def condition_counts(rec: pd.DataFrame, last_n_years: int | None = None) -> dict[str, int]:
    r = rec[rec["valid"]]
    if last_n_years:
        r = r[r.index >= r.index.max() - pd.DateOffset(years=last_n_years)]
    return r["condition"].value_counts().to_dict()


def condition_of_latest(analysis: Analysis) -> Condition | None:
    return analysis.latest["condition"] if analysis.latest else None
