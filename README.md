# LakeDelta

LakeDelta builds a monthly satellite record of a lake or reservoir's water surface from March 1984 to last month. It then answers one question: how full is the lake compared with its own history? Pick a lake, and the app traces its outline from 38 years of Landsat water maps. It fills cloud gaps month by month, ranks the latest month against the same month in earlier years, and relates the record to rain on the upstream catchment.

Everything runs in Google Earth Engine through a Streamlit app. Without Earth Engine credentials the app starts in demo mode with an invented reservoir, so the interface and the analysis can be tried anywhere.

## What changed from the first version

**Measurement**
- The lake is a traced outline, not a square around a point: the connected area where JRC Global Surface Water has ever seen water, seeded by the HydroLAKES polygon when one is nearby.
- Images are cloud-masked: QA bits for Landsat, Cloud Score+ for Sentinel-2. The deprecated `S2_SR` collection is replaced by `S2_SR_HARMONIZED`, which removes the 2022 offset that biased NDVI.
- The record starts in 1984. JRC monthly water history covers 1984–2021, and our own Landsat 8/9 + Sentinel-2 classification continues from there. The two are compared where they overlap (2019–2021) and joined.
- Partly cloudy months are gap-filled from long-term water occurrence (after Zhao and Gao 2018), and each month carries an uncertainty range. Months seen too poorly, and isolated spikes, are left out.

**Analysis**
- Climate is averaged over the HydroBASINS catchment upstream of the lake rather than a 5 km circle. It covers ERA5-Land precipitation, temperature and runoff, plus a non-parametric SPI and the rain-accumulation window the lake follows most closely.
- Condition is the percentile of the month against the same calendar month in 1991–2020, in USGS WaterWatch classes. This replaces a z-score of a few summers.
- Trends use Mann-Kendall with the Hamed–Rao autocorrelation correction and Sen's slope.
- The degree-2 polynomial forecast is replaced by a 12-month outlook. It assumes the current anomaly damps the way it has before, and comes with ranges and leave-one-year-out hindcast skill.
- Water-quality indicators (NDTI turbidity, NDCI chlorophyll, Landsat surface temperature) are taken over the permanent-water core, not NDVI over a mixed buffer.

**App**
- Season windows that skip months or cross the year end (Dec–Feb) now work.
- Results are cached by lake and settings in memory. The old app wrote files to a shared folder, so other visitors could be served outputs from someone else's lake or radius.

**Interface**
- New design:
  - a staff-gauge summary of where this month sits in the lake's history, a headline sentence and four key readings;
  - a month-by-month condition calendar and the seasonal cycle against normal;
  - a map with JRC layers and a two-year swipe comparison;
  - a labelled timelapse and a four-page PDF report;
  - CSV, GeoJSON and ZIP downloads;
  - light and dark themes, self-hosted fonts, and a layout that works on phones.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Demo mode: no Earth Engine needed
LAKEDELTA_DEMO=1 streamlit run streamlit_app.py
```

To use Earth Engine, give the app credentials in one of these ways:

- **Service account (as before).** Put the `[earth_engine]` table in `.streamlit/secrets.toml`; see `.streamlit/secrets.toml.example`. The app passes `project_id` to `ee.Initialize`.
- **Environment.** Set `EE_SERVICE_ACCOUNT_JSON` to the key's JSON text, optionally with `EE_PROJECT`.
- **Your own login.** Run `earthengine authenticate` once, then `EE_PROJECT=my-project streamlit run streamlit_app.py`.

The Cloud project must be registered for Earth Engine. HydroLAKES comes from the public community catalogue (`projects/sat-io/...`), and every other dataset is in the Earth Engine catalogue.

## Deploy on Streamlit Community Cloud

1. Replace the repository contents with this folder. Your secrets stay in the app settings and use the same `[earth_engine]` table.
2. The current deployment's main file is `streamlit_app.py.py`. This repository keeps that file as a two-line entry point, so the existing app keeps working with no settings change. For a new deployment, choose `streamlit_app.py`.
3. Use Python 3.11 or newer (3.12 or 3.13 recommended); development and tests used Python 3.13 and Streamlit 1.65. Community Cloud fixes the Python version when an app is deployed, so if the existing app runs on an older Python, delete and redeploy it with a newer one under Advanced settings. You can reuse the same subdomain and secrets.

A first analysis of a lake takes about one to three minutes, depending on its size. Results are cached for twelve hours.

## Check the Earth Engine side

The tests run every Earth Engine code path offline (see Tests), but only the real service can confirm asset IDs, band names and server-side behaviour. Run the smoke test once after deploying, or after any change to `lakedelta/ee_data.py`:

```bash
python scripts/ee_smoke_test.py --key path/to/service-account.json
# or: python scripts/ee_smoke_test.py --project my-ee-project
# add --full --lake mornos to run the whole pipeline and write a PDF and CSV
```

Each step prints PASS or FAIL with timings and a short sanity summary, such as the outline area and the JRC/ours ratio over the overlap. Set `SMOKE_TRACEBACK=1` to see full tracebacks.

## How it works

1. **Outline.** The app finds the HydroLAKES polygon nearest the chosen point, within 2 km. Around it, it keeps every patch of "water ever seen" (JRC occurrence of at least 1%) that touches the polygon, grows the result by about 60 m, and simplifies it. Without a HydroLAKES polygon it searches a circle and keeps the patch under the point. If the outline runs along the edge of the search area for more than about 1.5 km, the search area grows and the analysis runs again. A river leaving the lake does not count as running along the edge.
2. **Catchment.** HydroBASINS sub-basins that touch the lake (or hold the HydroLAKES pour point) are traced upstream through `NEXT_DOWN`. Level 12, 8 or 6 is chosen from the upstream area. If tracing fails, a 20 km ring around the lake is used and the app says so.
3. **Monthly water maps.** Months from March 1984 to December 2021 use the JRC monthly water history. From 2019 on, each Landsat 8/9 and Sentinel-2 image is classified: a pixel is water if MNDWI > NDVI or MNDWI > EVI, and EVI < 0.1 (Zou et al. 2018). Outside the historic water extent, the pixel also needs MNDWI > 0.3. A pixel is water for the month when most of its clear observations are water.
4. **Histograms.** For each month, Earth Engine returns the area of the outline split into observed water, observed land and no observation, crossed with long-term occurrence (0–100%). It returns only these numbers, about 300 per month, so all later steps run in Python and are unit-tested.
5. **Gap filling.** The occurrence threshold that best separates clear water from clear land is applied to the hidden pixels. Thresholds that fit almost as well give the uncertainty range. A month is used when at least 20% of the potential water area was seen clearly and the range is under 15% of the estimate. Isolated spikes are flagged with a Hampel filter.
6. **Joining the sources.** In months seen clearly by both, the median JRC/ours ratio scales the record from 2022 on, if it lies between 0.85 and 1.15. Otherwise the record stays unscaled and the app says so. The comparison is shown in the Data and method tab.
7. **Normals and condition.** Percentiles are computed against the same calendar month in 1991–2020, falling back to the whole record when a month has fewer than ten clear years. Classes: much below normal (below the 10th percentile), below normal (10th–24th), normal (25th–75th), above normal (76th–90th), much above normal (above the 90th).
8. **Trends and drivers.** The Mann-Kendall test with the Hamed–Rao correction and Sen's slope is applied to yearly and seasonal means, and a seasonal Kendall test to all months. Spearman correlation of the standardized lake anomaly with SPI accumulated over 1–36 months gives the lake's "memory" of rain.
9. **Outlook.** For each lead time up to 12 months, the anomaly that many months ahead is regressed on today's anomaly. Residual quantiles give the 50% and 80% ranges. Leave-one-year-out hindcasts report the skill against always forecasting the normal.

## Limits

- Surface area is not volume. A steep-sided reservoir can lose a lot of water while its area changes little.
- Floating vegetation, ice, very turbid shallows, terrain shadow and burnt or wet ground can be misclassified. Winter months on frozen lakes are usually left out.
- The JRC record ends in 2021. A reservoir built or raised since then has little occurrence history to fill gaps from, and the app flags the disagreement.
- Water-quality indices are relative and uncalibrated, so read them against the lake's own seasonal range.
- Withdrawals and releases are not observed. Correlation with rain is not attribution.

## Tests

```bash
pip install pytest
pytest            # 54 tests, about 30 seconds
```

- `tests/test_analysis.py` covers gap filling with known answers, season windows, spike detection, joining the sources, normals, trends, the outlook, SPI and formatting.
- `tests/test_ee_dryrun.py` runs every Earth Engine function offline. `tests/ee_dryrun.py` stands in for the server's algorithm registry and network calls, so mapped functions execute and expressions serialize. Wrong method names, arguments and Python errors fail here before deployment.
- `tests/test_app.py` drives the Streamlit app in demo mode with `AppTest`, opening every tab and building the water-quality chart, the timelapse and the PDF.
- `tests/test_app_live.py` drives the app as if Earth Engine were connected, with every Earth Engine call stubbed: outline preview, place search, typed coordinates, the year comparison and report maps.
- `tests/test_smoke_script.py` runs the smoke-test script itself through the offline harness.

## Project layout

```
streamlit_app.py         entry point (streamlit run streamlit_app.py)
streamlit_app.py.py      same, for the existing Community Cloud deployment
lakedelta/
  app.py                 Streamlit interface
  ee_data.py             every Earth Engine request
  ee_auth.py             credentials and demo-mode fallback
  pipeline.py            fetches one lake's data; place search; uploaded outlines
  analysis.py            gap filling, record, normals, trends, outlook, climate links
  figures.py             Plotly charts (light and dark palettes)
  gauge.py               the staff-gauge SVG
  maps.py                Folium maps
  report.py              PDF report (fpdf2 + matplotlib)
  timelapse.py           frame labels and sparkline for the timelapse
  demo.py                synthetic lake for demo mode and tests
  ui.py                  page CSS and small HTML components
  config.py              datasets, featured lakes, colours, defaults
  fonts/                 TTF copies of the fonts for the PDF (SIL OFL)
static/fonts/            web fonts served by Streamlit (SIL OFL)
scripts/ee_smoke_test.py check against the real Earth Engine service
tests/                   pytest suite and the offline Earth Engine harness
.streamlit/config.toml   theme (light and dark) and static serving
```

## Data and credits

- JRC Global Surface Water 1984–2021: Pekel et al. (2016), *Nature* 540, 418–422.
- Landsat Collection 2 Level-2: courtesy of the U.S. Geological Survey.
- Sentinel-2: contains modified Copernicus Sentinel data. Cloud Score+: Pasquarella et al. (2023), CC BY 4.0.
- ERA5-Land: Muñoz-Sabater et al. (2021), Copernicus Climate Change Service.
- HydroBASINS (Lehner and Grill 2013) and HydroLAKES (Messager et al. 2016): see the HydroSHEDS licence.
- Copernicus GLO-30 DEM, used for terrain shading in the report maps.
- Methods: Zhao and Gao (2018) for gap filling; Zou et al. (2018) for the water rule; Farahmand and AghaKouchak (2015) for the non-parametric SPI; Hamed and Rao (1998) and Sen (1968) for trends; USGS WaterWatch for condition classes.
- Place search: OpenStreetMap Nominatim. Basemaps: Esri.
- Fonts: Barlow (Jeremy Tribby) and Source Serif 4 (Adobe), SIL Open Font License.
