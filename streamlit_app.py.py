import streamlit as st
import ee
import geemap.foliumap as geemap
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
import json
from google.oauth2 import service_account
import base64 

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="LakeDelta Pro", layout="wide", page_icon="💧")

# ACADEMIC CSS
st.markdown("""
<style>
    .block-container { max-width: 95% !important; padding-top: 1rem; padding-left: 2rem; padding-right: 2rem; }
    
    .report-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-top: 5px solid #2c3e50;
        border-radius: 4px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .report-header {
        font-size: 1.2rem; font-weight: 700; color: #2c3e50; margin-bottom: 15px; 
        border-bottom: 1px solid #eee; padding-bottom: 8px; display: flex; align-items: center;
    }
    
    .metric-row { display: flex; flex-direction: row; justify-content: space-between; gap: 20px; margin-bottom: 15px; width: 100%; }
    .metric-box { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; flex: 1; text-align: center; min-width: 150px; }
    .metric-label { font-size: 0.7rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600; }
    .metric-value { font-size: 1.1rem; font-weight: 700; color: #212529; }
    
    .report-body { font-size: 0.95rem; line-height: 1.5; color: #343a40; text-align: justify; }
    
    .legend-box-split {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px; margin-bottom: 5px;
        text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    .legend-title { font-size: 0.75rem; font-weight: 700; color: #495057; margin-bottom: 5px; text-transform: uppercase; }
    .legend-labels { display: flex; justify-content: space-between; font-size: 0.65rem; color: #6c757d; margin-top: 3px; font-weight: 500; }
    
    .grad-freq { height: 8px; background: linear-gradient(90deg, #d7191c 0%, #fdae61 25%, #ffffbf 50%, #abd9e9 75%, #2c7bb6 100%); border-radius: 2px; }
    .grad-ghost { height: 8px; background: linear-gradient(90deg, #00ffff 0%, #0000ff 50%, #4b0082 100%); border-radius: 2px; }
    
    .hl-crit { color: #c0392b; font-weight: 700; background-color: #fadbd8; padding: 2px 6px; border-radius: 3px; }
    .hl-mod { color: #d35400; font-weight: 700; background-color: #fdebd0; padding: 2px 6px; border-radius: 3px; }
    .hl-stab { color: #27ae60; font-weight: 700; background-color: #d5f5e3; padding: 2px 6px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# --- EARTH ENGINE AUTHENTICATION ---

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    service_account_info = st.secrets["earth_engine"]
    scopes = ['https://www.googleapis.com/auth/earthengine']
    creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
    ee.Initialize(credentials=creds)
except Exception as e:
    st.error(f"🚨 Earth Engine Authentication Failed: {e}")
    st.info("Please ensure your .streamlit/secrets.toml file is correctly formatted.")
    st.stop()

# Output directory setup
OUTPUT_BASE = "Lake_Analysis_Output"
if not os.path.exists(OUTPUT_BASE):
    os.makedirs(OUTPUT_BASE)

CATALOG = {
    "Lake Mornos (Greece)": {"lat": 38.532, "lon": 22.124},
    "Lake Polyfytos (Greece)": {"lat": 40.233, "lon": 21.921},
    "Lake Yliki (Greece)": {"lat": 38.412, "lon": 23.275},
    "Lake Plastira (Greece)": {"lat": 39.236, "lon": 21.737},
    "Lake Powell (USA)": {"lat": 37.068, "lon": -111.236},
    "Lake Mead (USA)": {"lat": 36.133, "lon": -114.453},
    "Aral Sea (Kazakhstan)": {"lat": 45.396, "lon": 59.613}
}

# --- 2. ADVANCED AI & IMAGE PROCESSING ---

def add_roi_legend(image_path, title_text="Analysis Region"):
    """Adds a title header and a readable legend to the ROI context map."""
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, _ = img.shape
    
    # 1. Create Header (Title)
    header_h = 40
    header = np.zeros((header_h, w, 3), dtype=np.uint8) # Black background
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (tw, th), _ = cv2.getTextSize(title_text, font, 0.7, 2)
    cv2.putText(header, title_text, ((w-tw)//2, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 2. Create Footer (Legend)
    footer_h = 50
    footer = np.ones((footer_h, w, 3), dtype=np.uint8) * 255
    scale = 0.5
    thick = 1
    
    cv2.circle(footer, (40, 25), 6, (0, 0, 255), 2) # Red in BGR
    cv2.putText(footer, "ROI Limit", (55, 30), font, scale, (50,50,50), thick, cv2.LINE_AA)
    
    cv2.rectangle(footer, (160, 18), (175, 32), (255, 0, 0), -1) # Blue in BGR
    cv2.putText(footer, "Measured Water", (185, 30), font, scale, (50,50,50), thick, cv2.LINE_AA)
    
    cv2.rectangle(footer, (300, 18), (315, 32), (120, 120, 120), -1)
    cv2.putText(footer, "Context Area", (325, 30), font, scale, (50,50,50), thick, cv2.LINE_AA)
    
    final = np.vstack([header, img])
    cv2.imwrite(image_path, final)

def generate_roi_visual(roi_coords, buffer_m, out_dir, title_text="ROI Analysis"):
    """Generates a High-Context ROI Map with Natural Color background."""
    center = ee.Geometry.Point(roi_coords)
    roi_circle = center.buffer(buffer_m)
    viewport = center.buffer(buffer_m * 1.5).bounds()
    path_roi = os.path.join(out_dir, f"roi_inspect_{buffer_m}.jpg")
    
    col = (ee.ImageCollection("COPERNICUS/S2_SR")
           .filterBounds(viewport)
           .filterDate('2023-06-01', '2023-09-30')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
           .select(['B4', 'B3', 'B2', 'B11']))
    
    img = col.mosaic().clip(viewport)
    
    base = img.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000, gamma=1.2)
    water = img.normalizedDifference(['B3', 'B11']).gt(0).selfMask().clip(roi_circle).visualize(palette=['0066ff'], opacity=0.7)
    outline = ee.Image().paint(roi_circle, 2, 3).visualize(palette=['ff0000'])
    
    final_vis = base.blend(water).blend(outline)
    geemap.get_image_thumbnail(final_vis, path_roi, {'dimensions': 600, 'region': viewport, 'format': 'jpg'})
    add_roi_legend(path_roi, title_text)
    return path_roi

def add_academic_legend(image_path, mode="heatmap"):
    """Adds a MICRO-SIZED academic legend footer to the image (Burned in)."""
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, _ = img.shape
    footer_h = 60 
    footer = np.ones((footer_h, w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    label_scale = 0.28; title_scale = 0.35; thickness = 1
    
    if mode == "heatmap":
        bar_w = int(w * 0.5); bar_x = (w - bar_w) // 2; bar_y = 35
        cv2.rectangle(footer, (bar_x, bar_y), (bar_x + bar_w//5, bar_y+10), (28, 25, 215), -1) 
        cv2.rectangle(footer, (bar_x + bar_w//5, bar_y), (bar_x + 2*bar_w//5, bar_y+10), (97, 174, 253), -1)
        cv2.rectangle(footer, (bar_x + 2*bar_w//5, bar_y), (bar_x + 3*bar_w//5, bar_y+10), (191, 255, 255), -1)
        cv2.rectangle(footer, (bar_x + 3*bar_w//5, bar_y), (bar_x + 4*bar_w//5, bar_y+10), (233, 217, 171), -1)
        cv2.rectangle(footer, (bar_x + 4*bar_w//5, bar_y), (bar_x + bar_w, bar_y+10), (182, 123, 44), -1)
        
        cv2.putText(footer, "Ephemeral", (bar_x - 50, bar_y + 8), font, label_scale, (70,70,70), thickness, cv2.LINE_AA)
        cv2.putText(footer, "Permanent", (bar_x + bar_w + 5, bar_y + 8), font, label_scale, (70,70,70), thickness, cv2.LINE_AA)
        cv2.putText(footer, "FIG 1: Surface Water Frequency Analysis", ((w-200)//2, 20), font, title_scale, (0,0,0), thickness, cv2.LINE_AA)

    elif mode == "decline":
        start_x = w // 3
        cv2.rectangle(footer, (start_x, 35), (start_x + 15, 45), (255, 255, 0), -1)
        cv2.putText(footer, "Shallow Loss", (start_x + 20, 44), font, label_scale, (70,70,70), thickness, cv2.LINE_AA)
        cv2.rectangle(footer, (start_x + 150, 35), (start_x + 165, 45), (130, 0, 75), -1)
        cv2.putText(footer, "Deep Loss", (start_x + 170, 44), font, label_scale, (70,70,70), thickness, cv2.LINE_AA)
        cv2.putText(footer, "FIG 2: Spectral Decline Blueprint", ((w-180)//2, 20), font, title_scale, (0,0,0), thickness, cv2.LINE_AA)

    final_img = np.vstack([img, footer])
    cv2.imwrite(image_path, final_img)

@st.cache_data
def run_ai_forecast(df):
    X = df['Year'].values.reshape(-1, 1)
    y = df['Area_km'].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    last_year = int(df['Year'].iloc[-1])
    future_years = np.array([last_year, last_year + 1, last_year + 2]).reshape(-1, 1)
    preds = model.predict(poly.transform(future_years))
    
    preds[0] = df['Area_km'].iloc[-1]
    return future_years.flatten(), preds

def generate_styled_ai_insight(df, status, z_score, roi_img_path, year_range, months_txt):
    """Generates the Pro HTML report."""
    start_area = df.iloc[0]['Area_km']
    end_area = df.iloc[-1]['Area_km']
    change_pct = ((end_area - start_area) / start_area) * 100
    avg_rain = df['Rainfall_mm'].mean()
    last_rain = df.iloc[-1]['Rainfall_mm']
    
    if "CRITICAL" in status:
        border_color = "#c0392b"; h_class = "hl-crit"; icon = "🚨"
    elif "Decline" in status:
        border_color = "#d35400"; h_class = "hl-mod"; icon = "⚠️"
    else:
        border_color = "#27ae60"; h_class = "hl-stab"; icon = "✅"

    rain_status = f"<span class='{h_class}'>Below Average</span>" if last_rain < avg_rain else f"<span class='{h_class}'>Above Average</span>"
    outlook_txt = "further contraction" if change_pct < 0 else "stabilization"
    period_txt = f"{year_range[0]} - {year_range[1]}"

    html = ""
    html += f'<div class="report-container" style="border-top: 5px solid {border_color};">'
    html += f'<div class="report-header">{icon} &nbsp; Executive Hydrological Summary ({period_txt})</div>'
    html += f'<div class="metric-row">'
    html += f'<div class="metric-box"><div class="metric-label">Diagnosis</div><div class="metric-value" style="color:{border_color}">{status}</div></div>'
    html += f'<div class="metric-box"><div class="metric-label">Total Trend</div><div class="metric-value">{change_pct:+.1f}%</div></div>'
    html += f'<div class="metric-box"><div class="metric-label">Observed Months</div><div class="metric-value" style="font-size: 0.9rem;">{months_txt}</div></div>'
    html += f'</div>'
    html += f'<div class="report-body">'
    html += f'<b>Analysis:</b> Satellite earth observation data indicates a long-term <span class="{h_class}">{status.lower()}</span> trend in reservoir surface area. '
    html += f'Recent climatic inputs show precipitation levels are currently {rain_status} ({last_rain}mm) relative to the historical annual mean ({avg_rain:.0f}mm).<br><br>'
    html += f'<b>Outlook (Next 2 Years):</b> Based on polynomial regression modeling of historical trends, the system is projected to experience <b>{outlook_txt}</b> in the near term, barring significant meteorological shifts.'
    html += f'</div></div>'
    
    return html

@st.cache_data
def generate_spectral_forensics(roi_coords, buffer_m, year_max, year_min, out_dir):
    """Generates images with strict buffer application."""
    roi = ee.Geometry.Point(roi_coords).buffer(buffer_m)
    path_heat = os.path.join(out_dir, f"spectral_heatmap_{buffer_m}.jpg")
    path_diff = os.path.join(out_dir, f"spectral_decline_{buffer_m}.jpg")
    
    bands_needed = ['B4', 'B3', 'B2', 'B8', 'B11']

    # 1. HEATMAP
    col = (ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(roi).filterDate('2017-01-01', '2024-12-31')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).select(bands_needed))
    def classify_water(img): return img.normalizedDifference(['B3', 'B11']).gt(0).rename('water')
    freq = col.map(classify_water).mean().multiply(100)
    spectral_palette = ['9e0142', 'd53e4f', 'f46d43', 'fdae61', 'fee08b', 'ffffbf', 'e6f598', 'abdda4', '66c2a5', '3288bd', '5e4fa2']
    geemap.get_image_thumbnail(freq.visualize(min=0, max=100, palette=spectral_palette), path_heat, {'dimensions': 1000, 'region': roi, 'format': 'jpg'})
    
    # 2. DECLINE
    img_max = (ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(roi).filter(ee.Filter.calendarRange(year_max, year_max, 'year'))
                .select(bands_needed).median())
    img_min = (ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(roi).filter(ee.Filter.calendarRange(year_min, year_min, 'year'))
                .select(bands_needed).median())
    
    water_max = img_max.normalizedDifference(['B3', 'B11']).gt(0)
    water_min = img_min.normalizedDifference(['B3', 'B11']).gt(0)
    loss_mask = water_max.And(water_min.Not())
    
    base_gray = img_min.select('B8').visualize(min=500, max=4000, palette=['000000', 'ffffff'])
    ghost_vis = img_max.normalizedDifference(['B3', 'B11']).visualize(min=0, max=0.5, palette=['00ffff', '0000ff', '4b0082'])
    
    final_composite = base_gray.where(loss_mask, ghost_vis)
    geemap.get_image_thumbnail(final_composite, path_diff, {'dimensions': 1000, 'region': roi, 'format': 'jpg'})
    
    add_academic_legend(path_heat, mode="heatmap")
    add_academic_legend(path_diff, mode="decline")
    
    return path_heat, path_diff

# --- 3. STANDARD DATA FUNCTIONS ---
@st.cache_data
def get_water_balance(roi_coords, buffer_m, start, end):
    roi = ee.Geometry.Point(roi_coords).buffer(buffer_m)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(roi).filterDate(f'{start}-01-01', f'{end}-12-31')
    data = []
    for year in range(start, end + 1):
        try:
            img = era5.filter(ee.Filter.calendarRange(year, year, 'year')).sum()
            stats = img.reduceRegion(ee.Reducer.mean(), roi, 10000).getInfo()
            data.append({"Year": year, "Rainfall_mm": round(stats.get('total_precipitation_sum',0)*1000,1), "Evaporation_mm": abs(round(stats.get('total_evaporation_sum',0)*1000,1))})
        except: continue
    df = pd.DataFrame(data)
    if not df.empty: df['Net_Balance_mm'] = df['Rainfall_mm'] - df['Evaporation_mm']
    return df

@st.cache_data
def get_seasonal_profile(roi_coords, buffer_m):
    roi = ee.Geometry.Point(roi_coords).buffer(buffer_m)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(roi).filterDate('2017-01-01', '2024-12-31')
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = []
    for i, m in enumerate(months):
        try:
            img = era5.filter(ee.Filter.calendarRange(i+1, i+1, 'month')).mean()
            stats = img.reduceRegion(ee.Reducer.mean(), roi, 10000).getInfo()
            data.append({"Month": m, "Rain_Avg": stats.get('total_precipitation_sum',0)*1000, "Evap_Avg": abs(stats.get('total_evaporation_sum',0)*1000)})
        except: continue
    return pd.DataFrame(data)

def generate_pdf(out_dir, name, status, df, buffer_m):
    pdf = PDFReport(); pdf.add_page()
    pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, f"Target: {name}", 0, 1, 'L')
    pdf.set_font('Arial', '', 12); pdf.cell(0, 10, f"Analysis Period: 2017 - 2024", 0, 1, 'L')
    pdf.set_fill_color(255, 200, 200) if "CRITICAL" in status else pdf.set_fill_color(220, 255, 220)
    pdf.cell(0, 10, f"DIAGNOSIS: {status}", 1, 1, 'C', 1); pdf.ln(10)
    start = df.iloc[0]['Area_km']; end = df.iloc[-1]['Area_km']
    change = ((end - start) / start) * 100
    pdf.cell(0, 10, f"Total Change: {change:.1f}%  |  Current Area: {end} km2", 0, 1)
    
    images = ["Static_Hydrology.png", f"spectral_heatmap_{buffer_m}.jpg", f"spectral_decline_{buffer_m}.jpg"]
    for img in images:
        path = os.path.join(out_dir, img)
        if os.path.exists(path): pdf.image(path, x=10, w=180); pdf.ln(5)
    pdf_path = os.path.join(out_dir, f"{name}_Report.pdf")
    pdf.output(pdf_path, 'F'); return pdf_path

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15); self.cell(0, 10, 'Satellite Water Monitoring Report', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# --- 4. UI & MAIN LOGIC ---

st.title("🛰️ LakeDelta Pro")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

with st.sidebar:
    st.header("📍 Location Setup")
    mode = st.radio("Input Method", ["Catalog", "Search", "Manual Coordinates"])
    target = None
    if mode == "Catalog":
        name = st.selectbox("Choose Reservoir", list(CATALOG.keys()))
        if name: target = {"name": name, "lat": CATALOG[name]['lat'], "lon": CATALOG[name]['lon']}
    elif mode == "Search":
        query = st.text_input("Enter Location Name")
        if query:
            try:
                loc = Nominatim(user_agent="lake_app").geocode(query)
                if loc: st.success(f"Found: {loc.address}"); target = {"name": query, "lat": loc.latitude, "lon": loc.longitude}
            except: st.error("Search failed.")
    elif mode == "Manual Coordinates":
        name = st.text_input("Site Name", "My Lake")
        lat = st.number_input("Latitude", value=38.532)
        lon = st.number_input("Longitude", value=22.124)
        target = {"name": name, "lat": lat, "lon": lon}
        
    st.divider()
    
    # --- TIME & DATE CONTROL ---
    st.subheader("📅 Temporal Settings")
    year_range = st.slider("Analysis Years", 2017, 2025, (2017, 2025))
    
    month_options = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    selected_months = st.multiselect("Seasonal Window (Months to Analyze)", list(month_options.keys()), default=["Jun", "Jul", "Aug", "Sep"])
    
    st.divider()
    buffer_m = st.slider("Analysis Buffer (meters)", 1000, 20000, 5000, step=1000)
    run_clicked = st.button("🚀 Run Analysis", type="primary")

# --- LOGIC PART 1: PROCESSING ---
if run_clicked and target:
    if not selected_months:
        st.error("Please select at least one month to analyze.")
        st.stop()
        
    # Convert month names to integers
    target_months = [month_options[m] for m in selected_months]
    min_m = min(target_months)
    max_m = max(target_months)
    
    # Check if months are contiguous (optional optimization, but good for filtering)
    is_contiguous = (max_m - min_m + 1) == len(target_months)
    
    out_dir = os.path.join(OUTPUT_BASE, target['name'].replace(" ", "_"))
    temp_dir = os.path.join(out_dir, "frames")
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    current_coords = [target['lon'], target['lat']]
    
    # 1. FREEZE GEOMETRY
    roi_object = ee.Geometry.Point(current_coords).buffer(buffer_m).bounds()
    fixed_roi = roi_object.getInfo() 
    
    with st.spinner(f"Processing Data ({year_range[0]}-{year_range[1]})..."):
        # Base Collection Filtered by Month Range (Using min/max approximation for simple range)
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
              .filterBounds(roi_object)
              .filter(ee.Filter.calendarRange(min_m, max_m, 'month')) 
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .select(['B4', 'B3', 'B2', 'B8', 'B11']))
        
        records = []; local_paths = []
        bar = st.progress(0)
        
        # Loop through Selected Years
        years_to_process = range(year_range[0], year_range[1] + 1)
        total_years = len(years_to_process)
        
        for i, year in enumerate(years_to_process):
            try:
                img = s2.filter(ee.Filter.calendarRange(year, year, 'year')).median()
                stats = img.reduceRegion(ee.Reducer.mean(), roi_object, 100).getInfo()
                if not stats: continue

                # --- WATER AREA CALCULATION ---
                mndwi = img.normalizedDifference(['B3', 'B11'])
                water = mndwi.gt(0).rename('water')
                area = water.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), roi_object, 20, maxPixels=1e9).get('water').getInfo()
                area_km = round(area / 1e6, 2) if area else 0.0
                
                # --- NDVI SPLIT CALCULATIONS ---
                ndvi_raw = img.normalizedDifference(['B8', 'B4'])
                
                # 1. Total Buffer
                ndvi_total = ndvi_raw.reduceRegion(ee.Reducer.mean(), roi_object, 100).get('nd').getInfo()
                
                # 2. Water Only (Algae/Turbidity)
                ndvi_water = ndvi_raw.updateMask(water).reduceRegion(ee.Reducer.mean(), roi_object, 100).get('nd').getInfo()
                
                # 3. Land Only (Vegetation)
                ndvi_land = ndvi_raw.updateMask(water.Not()).reduceRegion(ee.Reducer.mean(), roi_object, 100).get('nd').getInfo()

                records.append({
                    "Year": year, 
                    "Area_km": area_km, 
                    "NDVI_Total": round(ndvi_total or 0, 3),
                    "NDVI_Water": round(ndvi_water or 0, 3),
                    "NDVI_Land": round(ndvi_land or 0, 3)
                })
                
                # 2. GENERATE AT DISPLAY SIZE (600px)
                vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.2}
                path = os.path.join(temp_dir, f"{year}.jpg")
                geemap.get_image_thumbnail(
                    img.visualize(**vis), 
                    path, 
                    {
                        'dimensions': 600,      
                        'region': fixed_roi,
                        'crs': 'EPSG:3857',
                        'format': 'jpg'
                    }
                )
                
                img_cv = cv2.imread(path); h, w, _ = img_cv.shape
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{year} | Area: {area_km} km2"
                banner_h = int(h * 0.10); overlay = img_cv.copy()
                cv2.rectangle(overlay, (0,0), (w, banner_h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, img_cv, 0.3, 0, img_cv)
                (t_w, t_h), _ = cv2.getTextSize(text, font, 0.8, 2)
                cv2.putText(img_cv, text, ((w - t_w) // 2, int((banner_h + t_h) / 2)), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(path, img_cv)
                local_paths.append(path)
            except Exception as e:
                print(f"Skipping {year}: {e}")
                continue
            bar.progress((i + 1) / total_years)

        if not records: 
            st.error("No valid satellite data found for this period.")
            st.stop()
        
        df = pd.DataFrame(records)
        clim_df = get_water_balance(current_coords, buffer_m, year_range[0], year_range[1])
        seas_df = get_seasonal_profile(current_coords, buffer_m)
        final_df = pd.merge(df, clim_df, on='Year', how='inner')
        future_years, future_preds = run_ai_forecast(final_df)
        z_score = (final_df.iloc[-1]['Area_km'] - final_df['Area_km'].mean()) / final_df['Area_km'].std()
        status = "CRITICAL DROUGHT" if z_score < -1.5 else "Moderate Decline" if z_score < -0.5 else "Stable"
        
        final_df.to_csv(os.path.join(out_dir, "Full_Analysis.csv"), index=False)
        geemap.make_gif(local_paths, os.path.join(out_dir, "Timelapse.gif"), fps=0.5)
        roi_img_path = generate_roi_visual(current_coords, buffer_m, out_dir, target['name'])
        
        st.session_state.analysis_results = {
            "final_df": final_df,
            "seas_df": seas_df,
            "future_years": future_years,
            "future_preds": future_preds,
            "z_score": z_score,
            "status": status,
            "out_dir": out_dir,
            "roi_img_path": roi_img_path,
            "target_name": target['name'],
            "buffer_m": buffer_m,
            "coords": current_coords,
            "year_range": year_range,
            "months_txt": ", ".join(selected_months)
        }
        st.session_state.analysis_complete = True
        st.rerun()

# --- LOGIC PART 2: DISPLAY ---
if st.session_state.analysis_complete:
    res = st.session_state.analysis_results
    final_df = res["final_df"]
    seas_df = res["seas_df"]
    future_years = res["future_years"]
    future_preds = res["future_preds"]
    status = res["status"]
    z_score = res["z_score"]
    out_dir = res["out_dir"]
    roi_img_path = res["roi_img_path"]
    target_name = res["target_name"]
    buffer_m = res["buffer_m"]
    saved_coords = res["coords"]
    year_range = res.get("year_range", (2017, 2025))
    months_txt = res.get("months_txt", "Summer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📸 Spectral Forensics", "🎬 Timelapse", "💾 Downloads"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(generate_styled_ai_insight(final_df, status, z_score, roi_img_path, year_range, months_txt), unsafe_allow_html=True)
        with c2:
            if roi_img_path and os.path.exists(roi_img_path):
                st.image(roi_img_path, caption="🔴 ROI Analysis Zone (5km)", use_container_width=True)
        
        # COMBO CHART
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=final_df['Year'], y=final_df['Rainfall_mm'], name="Rainfall (mm)", marker_color='rgba(0, 196, 154, 0.4)', width=0.3), secondary_y=True)
        fig.add_trace(go.Scatter(x=final_df['Year'], y=final_df['Area_km'], name="Water Area (km²)", mode='lines+markers', line=dict(color='#1E88E5', width=4), marker=dict(size=8, color='#1E88E5')), secondary_y=False)
        fig.add_trace(go.Scatter(x=future_years, y=future_preds, name="AI Forecast", mode='lines+markers', line=dict(color='#e67e22', width=3, dash='dash'), marker=dict(size=6, color='#e67e22', symbol='diamond')), secondary_y=False)
        fig.update_layout(title="Hydrological Correlation & AI Forecast", title_x=0.5, template="plotly_white", legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'), height=450, hovermode="x unified", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        st.plotly_chart(fig, use_container_width=True)
        
        # ECO-CHART (Split Logic + Color Coding + Thicker Lines)
        fig_eco = go.Figure()
        
        # Trace 1: Land Vegetation (GREEN, Thick Line)
        fig_eco.add_trace(go.Scatter(
            x=final_df['Year'], y=final_df['NDVI_Land'],
            name="Surrounding Vegetation",
            mode='lines+markers',
            line=dict(color='#27ae60', width=4), # Green + Thick
            marker=dict(symbol='circle', size=8)
        ))

        # Trace 2: Water Quality (BLUE, Thick Line)
        fig_eco.add_trace(go.Scatter(
            x=final_df['Year'], y=final_df['NDVI_Water'],
            name="Water Turbidity/Algae",
            mode='lines+markers',
            line=dict(color='#2980b9', width=4, dash='solid'), # Blue + Thick
            marker=dict(symbol='triangle-up', size=8)
        ))
        
        # Calculate dynamic range for zoom (Min/Max of both datasets)
        all_ndvi = pd.concat([final_df['NDVI_Land'], final_df['NDVI_Water']])
        y_min = all_ndvi.min() * 0.95
        y_max = all_ndvi.max() * 1.05

        fig_eco.update_layout(
            title="Eco-Health: Land (Green) vs Water (Blue) Signals",
            title_x=0.5,
            xaxis_title="Year",
            yaxis_title="NDVI Value",
            # SMART SCALING (Zoom in)
            yaxis=dict(range=[y_min, y_max]),
            template="plotly_white", 
            height=450,
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_eco, use_container_width=True)

        fig_s = go.Figure(data=[
            go.Bar(name='Precipitation', x=seas_df['Month'], y=seas_df['Rain_Avg'], marker_color='#66bb6a'),
            go.Bar(name='Evapotranspiration', x=seas_df['Month'], y=seas_df['Evap_Avg'], marker_color='#ef5350')
        ])
        fig_s.update_layout(title="Seasonal Climatology (Avg 2017-2024)", title_x=0.5, template="plotly_white", height=400, legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'))
        st.plotly_chart(fig_s, use_container_width=True)

    with tab2:
        st.markdown("<h3 style='text-align: center;'>📸 Advanced Spectral Forensics</h3>", unsafe_allow_html=True)
        year_max = int(final_df.loc[final_df['Area_km'].idxmax()]['Year'])
        year_min = int(final_df.loc[final_df['Area_km'].idxmin()]['Year'])
        path_heat = os.path.join(out_dir, f"spectral_heatmap_{buffer_m}.jpg")
        path_diff = os.path.join(out_dir, f"spectral_decline_{buffer_m}.jpg")
        
        if not (os.path.exists(path_heat) and os.path.exists(path_diff)):
            with st.spinner("Generating JRC Spectral Analysis..."):
                try:
                    path_heat, path_diff = generate_spectral_forensics(saved_coords, buffer_m, year_max, year_min, out_dir)
                except Exception as e:
                    st.error(f"Could not generate spectral images: {e}")

        if os.path.exists(path_heat) and os.path.exists(path_diff):
             c1, c2, c3, c4 = st.columns([1, 2, 2, 1]) 
             with c2:
                 st.markdown("""<div class="legend-box-split"><div class="legend-title">Fig 1: Water Frequency</div><div class="grad-freq"></div><div class="legend-labels"><span>Ephemeral</span><span>Permanent</span></div></div>""", unsafe_allow_html=True)
                 st.image(path_heat, use_container_width=True)
             with c3:
                 st.markdown("""<div class="legend-box-split"><div class="legend-title">Fig 2: Ghost Water Depth</div><div class="grad-ghost"></div><div class="legend-labels"><span>Shallow Loss</span><span>Deep Loss</span></div></div>""", unsafe_allow_html=True)
                 st.image(path_diff, use_container_width=True)
        else:
             st.info("Spectral images could not be loaded.")

    with tab3:
         st.markdown("<h3 style='text-align: center; color: #2c3e50;'>🛰️ Annual Satellite Timelapse</h3>", unsafe_allow_html=True)
         
         gif_path = os.path.join(out_dir, "Timelapse.gif")
         if os.path.exists(gif_path):
             with open(gif_path, "rb") as file_:
                 data_url = base64.b64encode(file_.read()).decode("utf-8")
             
             min_y = final_df['Year'].min()
             max_y = final_df['Year'].max()
             caption_text = f"<b>Figure 3:</b> Temporal Evolution from {min_y} to {max_y}"
             
             st.markdown(
                 f"""
                 <div style="text-align: center;">
                     <img src="data:image/gif;base64,{data_url}" width="600" style="border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                     <div style="margin-top: 10px; font-size: 0.9em; color: #555; font-family: 'Segoe UI', serif;">
                        {caption_text}
                     </div>
                 </div>
                 """,
                 unsafe_allow_html=True,
             )

    with tab4:
        st.header("Data Export")
        pdf_path = os.path.join(out_dir, f"{target_name}_Report.pdf")
        if not os.path.exists(pdf_path):
             pdf_path = generate_pdf(out_dir, target_name, status, final_df, buffer_m)
        with open(pdf_path, "rb") as f: 
            st.download_button("Download AI Report PDF", f, f"{target_name}_Report.pdf", key="dl_pdf")
        csv_path = os.path.join(out_dir, "Full_Analysis.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f: 
                st.download_button("Download CSV Data", f, "Data.csv", key="dl_csv")

    st.success("LakeDelta Analysis Ready.")
