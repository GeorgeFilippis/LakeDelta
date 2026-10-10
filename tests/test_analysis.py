import numpy as np
import pandas as pd
import pytest

from lakedelta import analysis as A
from lakedelta import demo
from lakedelta.config import AnalysisSettings, condition_for


def _hist(land=None, water=None, nodata=None):
    h = np.zeros((3, A.N_OCC))
    for cls, d in ((A.LAND, land), (A.WATER, water), (A.NODATA, nodata)):
        for occ, km2 in (d or {}).items():
            h[cls, occ] += km2
    return h


def _uniform_lake(true_threshold, cloud, a=None):
    """Footprint with 1 km² per occurrence bin; water where occ >= threshold."""
    a = np.ones(A.N_OCC) if a is None else a
    h = np.zeros((3, A.N_OCC))
    for o in range(A.N_OCC):
        clear = a[o] * (1 - cloud[o])
        wet = o >= true_threshold and o > 0
        h[A.WATER if wet else A.LAND, o] += clear
        h[A.NODATA, o] += a[o] * cloud[o]
    return h


def test_fill_month_fully_clear_is_exact():
    h = _uniform_lake(40, np.zeros(A.N_OCC))
    f = A.fill_month(h)
    assert f.clear_fraction == pytest.approx(1.0)
    assert f.area == pytest.approx(61.0)  # bins 40..100
    assert f.lo == pytest.approx(f.hi)
    assert f.filled == 0
    assert f.consistency == pytest.approx(1.0)


def test_fill_month_recovers_area_under_patchy_cloud():
    cloud = np.zeros(A.N_OCC)
    cloud[55:85] = 1.0            # a whole band of the lake is hidden
    cloud[10:20] = 0.5
    h = _uniform_lake(40, cloud)
    f = A.fill_month(h)
    assert f.area == pytest.approx(61.0, abs=1.0)
    assert f.lo <= f.area <= f.hi
    assert 30 <= f.threshold <= 41


def test_fill_month_without_clear_pixels_is_flagged_by_uncertainty():
    h = _uniform_lake(40, np.ones(A.N_OCC))
    f = A.fill_month(h)
    assert f.clear_fraction == 0
    assert f.hi - f.lo > 50  # nothing constrains the threshold


def test_groups_to_hist_converts_m2_to_km2_and_codes():
    groups = [{"code": 0 * 101 + 0, "sum": 2e6}, {"code": 1 * 101 + 99, "sum": 5e5},
              {"code": 2 * 101 + 50, "sum": 1e6}, {"code": 999, "sum": 1e6}]
    h = A.groups_to_hist(groups)
    assert h[A.LAND, 0] == pytest.approx(2.0)
    assert h[A.WATER, 99] == pytest.approx(0.5)
    assert h[A.NODATA, 50] == pytest.approx(1.0)
    assert h.sum() == pytest.approx(3.5)  # the out-of-range code is ignored


@pytest.mark.parametrize("months,label,start", [
    ([6, 7, 8, 9], "Jun–Sep", None),
    ([12, 1, 2], "Dec–Feb", 12),
    ([11, 12, 1, 2, 3], "Nov–Mar", 11),
    ([1, 3, 5], "Jan, Mar, May", None),
    (list(range(1, 13)), "all months", None),
])
def test_season_helpers(months, label, start):
    assert A.season_label(months) == label
    assert A.season_start_month(months) == start


def test_season_year_and_ranges_wrap_the_year_end():
    months = [12, 1, 2]
    assert A.season_year(pd.Timestamp("2020-12-01"), months) == 2021
    assert A.season_year(pd.Timestamp("2021-02-01"), months) == 2021
    ranges = A.season_date_ranges(2021, months)
    assert ranges[0] == ("2021-01-01", "2021-02-01")
    assert ("2020-12-01", "2021-01-01") in ranges


def test_hampel_flags_spikes_but_not_steps():
    idx = pd.date_range("2000-01-01", periods=9, freq="MS")
    spike = pd.Series([10, 10, 10, 10, 14, 10, 10, 10, 10], index=idx, dtype=float)
    step = pd.Series([10, 10, 10, 10, 14, 14, 14, 14, 14], index=idx, dtype=float)
    assert A.hampel_flags(spike).tolist() == [False] * 4 + [True] + [False] * 4
    assert not A.hampel_flags(step).any()


@pytest.fixture(scope="module")
def demo_analysis():
    b = demo.demo_bundle()
    return A.analyze(b["jrc"], b["optical"], b["climate"], AnalysisSettings())


def test_record_is_monthly_and_harmonized(demo_analysis):
    rec = demo_analysis.record
    assert rec.index.freqstr in ("MS", None)
    assert (rec.index[1:] - rec.index[:-1]).days.min() >= 28
    h = demo_analysis.harmonization
    assert h["applied"]
    assert h["ratio"] == pytest.approx(1 / 1.03, rel=0.02)
    assert rec["valid"].mean() > 0.7
    last = rec.index.max()
    assert last == pd.Timestamp("2026-09-01")
    assert rec.loc["2023-06-01", "source"] == A.SOURCE_OPT
    assert rec.loc["2010-06-01", "source"] == A.SOURCE_JRC


def test_normals_and_percentiles(demo_analysis):
    n = demo_analysis.normals
    assert n.period == (1991, 2020)
    t = n.table
    assert (t["q10"] <= t["q25"]).all() and (t["q25"] <= t["median"]).all()
    assert (t["median"] <= t["q75"]).all() and (t["q75"] <= t["q90"]).all()
    rec = demo_analysis.record
    pct = rec.loc[rec["valid"], "percentile"]
    assert pct.between(0, 100).all()


def test_normals_fall_back_to_full_record_when_reference_is_sparse():
    idx = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    rec = pd.DataFrame({"area": np.linspace(10, 12, len(idx)), "valid": True}, index=idx)
    n = A.monthly_normals(rec, (1991, 2020))
    assert n.used_full_record
    assert n.period == (2015, 2024)


def test_condition_classes():
    assert condition_for(5).key == "much_below"
    assert condition_for(10).key == "below"
    assert condition_for(50).key == "normal"
    assert condition_for(75).key == "normal"
    assert condition_for(80).key == "above"
    assert condition_for(95).key == "much_above"
    assert condition_for(100).key == "much_above"
    assert condition_for(float("nan")) is None


def test_trend_detects_decline(demo_analysis):
    tr = demo_analysis.trend_annual
    assert tr["significant"]
    assert tr["slope_per_decade"] < 0
    assert tr["ci"][0] <= tr["slope_per_decade"] <= tr["ci"][1]


def test_trend_none_for_short_series():
    s = pd.Series([1.0, 2.0, 3.0], index=[2000, 2001, 2002])
    assert A.trend_test(s) is None


def test_outlook_quantiles_are_ordered_and_skilful(demo_analysis):
    o = demo_analysis.outlook
    assert o is not None
    t = o.table
    assert len(t) == 12
    assert (t["p10"] <= t["p25"]).all() and (t["p25"] <= t["median"]).all()
    assert (t["median"] <= t["p75"]).all() and (t["p75"] <= t["p90"]).all()
    assert t.index.min() > o.issued_from
    # Persistence of the anomaly should beat climatology at short leads
    assert o.skill.loc[1, "skill_vs_climatology"] > 0


def test_nonparametric_index_is_standard_normal_like():
    idx = pd.date_range("1980-01-01", periods=12 * 40, freq="MS")
    rng = np.random.default_rng(0)
    s = pd.Series(rng.gamma(2.0, 30.0, len(idx)), index=idx)
    z = A.standardize_nonparametric(s)
    assert abs(z.mean()) < 0.05
    assert 0.9 < z.std() < 1.1
    jan = s[s.index.month == 1]
    assert z.loc[jan.idxmax()] == z[z.index.month == 1].max()


def test_climate_frame_units():
    t = int(pd.Timestamp("2020-05-01").timestamp() * 1000)
    df = A.climate_frame([{"t": t, "total_precipitation_sum": 0.0423,
                           "temperature_2m": 293.15, "runoff_sum": -0.0001}])
    row = df.iloc[0]
    assert row["precip_mm"] == pytest.approx(42.3)
    assert row["temp_c"] == pytest.approx(20.0)
    assert row["runoff_mm"] == 0.0  # GRIB packing noise clipped


def test_response_timescale_finds_generating_window(demo_analysis):
    r = demo_analysis.response
    assert r is not None
    best = int(r["r"].idxmax())
    assert 14 <= best <= 22  # the demo lake integrates ~18 months of rain
    assert r["r"].max() > 0.6


def test_headline_mentions_area_and_month(demo_analysis):
    text = A.headline("Demo Reservoir", demo_analysis.latest)
    assert "September 2026" in text
    assert "km²" in text
    assert "%" in text


def test_annual_table_requires_enough_months(demo_analysis):
    ann = demo_analysis.annual
    assert np.isnan(ann.loc[2026, "mean"])  # only part of 2026 observed
    assert np.isfinite(ann.loc[2010, "mean"])


def test_formatting():
    assert A.fmt_area(1234.5) == "1,234 km²" or A.fmt_area(1234.5) == "1,235 km²"
    assert A.fmt_area(14.23) == "14.2 km²"
    assert A.fmt_area(0.4567, unit=False) == "0.457"
    assert A.fmt_signed(-1.25, 1) == "−1.2" or A.fmt_signed(-1.25, 1) == "−1.3"
    assert A.ordinal(1) == "1st" and A.ordinal(12) == "12th" and A.ordinal(23) == "23rd"
    assert A.fmt_p(0.0) == "p < 0.001"


def test_short_record_still_produces_text_and_no_gauge():
    from lakedelta import gauge
    b = demo.demo_bundle()
    opt_only = A.analyze([], b["optical"], b["climate"], AnalysisSettings())
    assert opt_only.normals.used_full_record
    assert "km²" in A.headline("Lake", opt_only.latest)
    g = gauge.gauge_inputs(opt_only)
    assert g is None or all(np.isfinite(v) for v in g["dist"].values())
    if opt_only.outlook is not None:  # 2019-2026 spans eight calendar years
        assert not opt_only.outlook.skill.empty
