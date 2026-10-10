"""Check every Earth Engine request LakeDelta makes, against the real service.

The test suite runs the Earth Engine code offline; this script is the other
half: it confirms the datasets, band names and server-side logic work.

Usage (from the repository root):

    # with a service-account key file
    python scripts/ee_smoke_test.py --key path/to/service-account.json

    # or with your own login (earthengine authenticate) and a Cloud project
    python scripts/ee_smoke_test.py --project my-ee-project

    # add --full to run the whole pipeline for the lake and write outputs
    python scripts/ee_smoke_test.py --project my-ee-project --lake mornos --full

Each step prints PASS or FAIL with its duration. The exit code is the number
of failed steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lakedelta import analysis as A  # noqa: E402
from lakedelta import config as C  # noqa: E402
from lakedelta import ee_data as E  # noqa: E402

RESULTS: list[tuple[str, bool, float, str]] = []


def step(name):
    def deco(fn):
        def run(*args, **kwargs):
            t0 = time.time()
            try:
                out = fn(*args, **kwargs)
                RESULTS.append((name, True, time.time() - t0, ""))
                print(f"PASS  {name:<38} {time.time() - t0:6.1f} s")
                return out
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, time.time() - t0, str(exc)))
                print(f"FAIL  {name:<38} {time.time() - t0:6.1f} s  {exc}")
                if os.environ.get("SMOKE_TRACEBACK"):
                    traceback.print_exc()
                return None
        return run
    return deco


# def init(args):
#     import ee

#     if args.key:
#         from google.oauth2 import service_account

#         from lakedelta.ee_auth import EE_SCOPES

#         info = json.load(open(args.key))
#         creds = service_account.Credentials.from_service_account_info(info, scopes=EE_SCOPES)
#         ee.Initialize(credentials=creds, project=args.project or info.get("project_id"))
#     else:
#         ee.Initialize(project=args.project)
#     print(f"Earth Engine initialised (earthengine-api {ee.__version__}).")

def init(args):
    import ee

    if args.key:
        print("✅ SUCCESS: The script is using the service account key!") # <--- Add this line
        from google.oauth2 import service_account

        from lakedelta.ee_auth import EE_SCOPES

        info = json.load(open(args.key))
        creds = service_account.Credentials.from_service_account_info(info, scopes=EE_SCOPES)
        ee.Initialize(credentials=creds, project=args.project or info.get("project_id"))
    else:
        print("⚠️ NOTICE: No key provided, trying default login...") # <--- You can add this too
        ee.Initialize(project=args.project)
    print(f"Earth Engine initialised (earthengine-api {ee.__version__}).")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="service-account JSON key")
    ap.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--lake", default="mornos", choices=sorted(C.CATALOG_BY_KEY))
    ap.add_argument("--full", action="store_true", help="run the whole pipeline and write outputs")
    ap.add_argument("--out", default="smoke_output")
    args = ap.parse_args()
    if not args.key and not args.project:
        ap.error("give --key or --project")
    init(args)
    lake = C.CATALOG_BY_KEY[args.lake]
    site = {"name": lake.name, "lat": lake.lat, "lon": lake.lon, "radius_km": lake.radius_km}

    hl = step("HydroLAKES lookup")(E.find_hydrolake)(lake.lon, lake.lat)
    if hl:
        p = hl["properties"]
        print(f"      {p.get('Lake_name')!r}, type {p.get('Lake_type')}, {p.get('Lake_area')} km², "
              f"catchment {p.get('Wshd_area')} km²")
    outline = step("Lake outline")(E.outline_for_site)(site)
    if not outline:
        print("Cannot continue without an outline.")
        sys.exit(len([r for r in RESULTS if not r[1]]))
    info = outline["info"]
    print(f"      {info['method']}; outline {info['area_km2']:.1f} km², ever water "
          f"{info['ever_water_km2']:.1f} km², permanent {info['permanent_km2']:.1f} km², "
          f"truncated={info['truncated']}")
    fp = outline["footprint"]
    props = (outline.get("hydrolake") or {}).get("properties") or {}
    pour = (props["Pour_long"], props["Pour_lat"]) if props.get("Pour_long") is not None else None
    catch = step("Catchment")(E.trace_catchment)(fp, pour, props.get("Wshd_area"))
    if catch:
        print(f"      {catch['info']['method']}: {catch['info']['area_km2']:.0f} km², "
              f"{catch['info']['n_basins']} basins")
    scale = E.hist_scale(info["area_km2"])
    jrc = step("JRC monthly histograms (2019-2020)")(E.jrc_histograms)(fp, scale, "2019-01-01",
                                                                        "2021-01-01")
    if jrc:
        rec = A.months_from_records(jrc, A.SOURCE_JRC)
        print(f"      {len(rec)} months, area {rec['area'].min():.2f}–{rec['area'].max():.2f} km², "
              f"median clear share {rec['clear_fraction'].median():.0%}")
    opt = step("Landsat 8/9 + Sentinel-2 (6 months)")(E.optical_histograms)(fp, scale, "2020-01-01",
                                                                             "2020-07-01")
    if opt:
        rec = A.months_from_records(opt, A.SOURCE_OPT)
        print(f"      {len(rec)} months, area {rec['area'].min():.2f}–{rec['area'].max():.2f} km², "
              f"scenes per month {[r.get('n_scenes') for r in opt]}")
        if jrc:
            both = A.months_from_records(jrc, A.SOURCE_JRC).join(rec, rsuffix="_opt", how="inner")
            if len(both):
                print(f"      JRC / ours over the overlap: "
                      f"{(both['area'] / both['area_opt']).median():.3f}")
    region = catch["geometry"] if catch else fp
    area = catch["info"]["area_km2"] if catch else info["area_km2"]
    clim = step("ERA5-Land climate (2020)")(E.era5_monthly)(region, area, "2020-01-01", "2021-01-01")
    if clim:
        df = A.climate_frame(clim)
        print(f"      2020 precipitation {df['precip_mm'].sum():.0f} mm, mean temperature "
              f"{df['temp_c'].mean():.1f} °C")
    wq = step("Water-quality indicators (3 months)")(E.water_quality)(fp, scale, "2020-06-01",
                                                                     "2020-09-01")
    if wq:
        print(f"      {wq}")
    layers = step("Map tiles")(E.map_layers)(fp)
    if layers:
        print(f"      {sorted(layers)}")
    step("Season composite tile")(E.season_composite_tile)(fp, 2020, (6, 7, 8, 9))
    imgs = step("Report thumbnails")(E.report_images)(fp, catch["geometry"] if catch else None, 400)
    if imgs:
        print("      " + ", ".join(f"{k}: {len(v):,} bytes" for k, v in imgs.items()))
    gif = step("Timelapse GIF (3 years)")(E.timelapse_gif)(fp, [2018, 2019, 2020], (6, 7, 8, 9), 320)
    if gif:
        print(f"      {len(gif[0])} bytes, frames for {gif[1]}")

    if args.full:
        from lakedelta import gauge, pipeline, report

        os.makedirs(args.out, exist_ok=True)

        def full():
            t = time.time()
            bundle = pipeline.fetch_bundle({**site, "country": lake.country, "kind": lake.kind,
                                            "note": lake.note},
                                           pipeline.Progress(lambda m: print(f"      {m}"),
                                                             lambda label: (lambda d, n: None)))
            print(f"      fetched in {time.time() - t:.0f} s")
            an = A.analyze(bundle["jrc"], bundle["optical"], bundle["climate"])
            print("      " + A.headline(lake.name, an.latest))
            pdf = report.build_report(an, bundle["site"], bundle, gauge.gauge_inputs(an),
                                      E.report_images(bundle["footprint"], bundle["catchment"]))
            with open(os.path.join(args.out, f"{args.lake}_report.pdf"), "wb") as f:
                f.write(pdf)
            an.record.to_csv(os.path.join(args.out, f"{args.lake}_monthly.csv"))
            with open(os.path.join(args.out, f"{args.lake}_bundle.json"), "w") as f:
                json.dump({k: v for k, v in bundle.items() if k not in ("jrc", "optical")}, f,
                          default=str)
            return an

        step("Full pipeline and report")(full)()
        print(f"Outputs written to {os.path.abspath(args.out)}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} steps passed.")
    for name, _, _, msg in failed:
        print(f"  {name}: {msg}")
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
