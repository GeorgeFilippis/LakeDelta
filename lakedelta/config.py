"""Constants shared across LakeDelta: datasets, featured lakes, colours, defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Earth Engine datasets (all public; see README for citations)
# --------------------------------------------------------------------------
GSW_IMAGE = "JRC/GSW1_4/GlobalSurfaceWater"
GSW_MONTHLY = "JRC/GSW1_4/MonthlyHistory"
GSW_FIRST_MONTH = "1984-03-01"
GSW_END_EXCLUSIVE = "2022-01-01"  # monthly history covers Mar 1984 - Dec 2021

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_PLUS_THRESHOLD = 0.60  # catalogue guidance: 0.50-0.65 works well

LANDSAT = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}

ERA5_LAND_MONTHLY = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
HYDROBASINS = "WWF/HydroSHEDS/v1/Basins/hybas_{level}"
HYDROLAKES = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
COPERNICUS_DEM = "COPERNICUS/DEM/GLO30"

# Sensor-era record built from our own classification starts here, three
# years before the JRC record ends, so the two can be compared and joined.
OPTICAL_RECORD_START = "2019-01-01"

# --------------------------------------------------------------------------
# Featured lakes. Points sit on (or next to) water; radius_km bounds the
# search for the lake outline when no HydroLAKES polygon is found nearby.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Lake:
    key: str
    name: str
    country: str
    kind: str  # "reservoir" | "lake" | "terminal lake"
    lat: float
    lon: float
    radius_km: float
    note: str = ""


CATALOG: tuple[Lake, ...] = (
    Lake("mornos", "Lake Mornos", "Greece", "reservoir", 38.530, 22.140, 10,
         "Main water supply for Athens."),
    Lake("yliki", "Lake Yliki", "Greece", "lake", 38.400, 23.267, 10,
         "Natural lake pumped into the Athens supply in dry years."),
    Lake("plastiras", "Lake Plastiras", "Greece", "reservoir", 39.297, 21.752, 12,
         "Hydropower and irrigation reservoir in Thessaly."),
    Lake("polyfytos", "Lake Polyfytos", "Greece", "reservoir", 40.233, 21.921, 25,
         "Largest reservoir on the Aliakmonas."),
    Lake("powell", "Lake Powell", "United States", "reservoir", 37.068, -111.240, 160,
         "Colorado River storage behind Glen Canyon Dam."),
    Lake("mead", "Lake Mead", "United States", "reservoir", 36.133, -114.453, 90,
         "Colorado River storage behind Hoover Dam."),
    Lake("great-salt", "Great Salt Lake", "United States", "terminal lake", 41.170, -112.580, 110,
         "Terminal lake whose level tracks the Wasatch snowpack."),
    Lake("urmia", "Lake Urmia", "Iran", "terminal lake", 37.700, 45.367, 110,
         "Hypersaline lake that has lost most of its surface."),
    Lake("aral", "Aral Sea", "Kazakhstan and Uzbekistan", "terminal lake", 45.396, 59.613, 260,
         "Fed by the Amu Darya and Syr Darya; largely dry since 2014."),
    Lake("poopo", "Lake Poopó", "Bolivia", "terminal lake", -18.550, -67.083, 70,
         "Shallow Altiplano lake that dried out in 2015."),
)

CATALOG_BY_KEY = {lake.key: lake for lake in CATALOG}

# --------------------------------------------------------------------------
# Condition classes (percentile of the value among the same calendar month in
# the reference period), following USGS WaterWatch conventions.
# Colours are mid-tones chosen to read on both the light and the dark theme.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    lo: float  # inclusive lower percentile bound
    hi: float  # exclusive upper percentile bound
    color: str
    text_on_fill: str  # ink to use for text laid on top of the fill


CONDITIONS: tuple[Condition, ...] = (
    Condition("much_below", "Much below normal", -1.0, 10.0, "#7C4A12", "#FFFFFF"),
    Condition("below", "Below normal", 10.0, 25.0, "#C99A5E", "#14232E"),
    Condition("normal", "Normal", 25.0, 75.0001, "#C9CFD2", "#14232E"),
    Condition("above", "Above normal", 75.0001, 90.0001, "#7FAFD3", "#14232E"),
    Condition("much_above", "Much above normal", 90.0001, 101.0, "#245E91", "#FFFFFF"),
)


def condition_for(percentile: float | None) -> Condition | None:
    if percentile is None or percentile != percentile:  # NaN check
        return None
    for c in CONDITIONS:
        if c.lo <= percentile < c.hi:
            return c
    return CONDITIONS[-1] if percentile >= 100 else CONDITIONS[0]


# --------------------------------------------------------------------------
# Colour tokens used by charts and custom HTML. Data colours are mid-tones that
# keep at least 3:1 contrast on both the light page (#F7F9F9) and the dark
# page (#0F1A21); chrome inherits the Streamlit theme via currentColor.
# --------------------------------------------------------------------------
WATER = "#2B6CA3"         # the lake: area line, primary marks
WATER_LABEL = "#245A85"   # hydrographic names (light theme)
WATER_LABEL_DARK = "#7FB2DC"
SEDIMENT = "#A86A2C"      # dry side: deficits, exposed bed
CONTEXT = "#8A979F"       # context marks (normal range, other years)
CONTEXT_BAND = "rgba(138, 151, 159, 0.22)"
WATER_WASH = "rgba(43, 108, 163, 0.14)"
WATER_WASH_STRONG = "rgba(43, 108, 163, 0.28)"
SEDIMENT_WASH = "rgba(168, 106, 44, 0.16)"

FONT_UI = "Barlow, 'Barlow Fallback', system-ui, sans-serif"
FONT_DENSE = "'Barlow Semi Condensed', Barlow, system-ui, sans-serif"
FONT_HYDRO = "'Source Serif 4', 'Source Serif Pro', Georgia, serif"

# --------------------------------------------------------------------------
# Analysis defaults
# --------------------------------------------------------------------------


@dataclass
class AnalysisSettings:
    reference_period: tuple[int, int] = (1991, 2020)
    season_months: tuple[int, ...] = (6, 7, 8, 9)
    min_clear_fraction: float = 0.20
    max_relative_uncertainty: float = 0.15
    harmonize_sources: bool = True
    outlook_horizon: int = 12
    extra: dict = field(default_factory=dict)


MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTH_NAME = ("January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December")

USER_AGENT = "LakeDelta/2.0 (+https://github.com/GeorgeFilippis/LakeDelta)"
