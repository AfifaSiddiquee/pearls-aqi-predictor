import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pydeck as pdk 
from src.inference.predict_aqi import get_3day_aqi, fetch_last_n_days

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Karachi AQI Predictor")
st.markdown("Real-time AQI predictions with past trends and forecast visualization.")

# --------------------------------------------------
# Helper: AQI category & color
# --------------------------------------------------
def get_aqi_category(aqi_val):
    if aqi_val <= 1:
        return "Good", "green"
    elif aqi_val == 2:
        return "Fair", "yellow"
    elif aqi_val == 3:
        return "Moderate", "orange"
    elif aqi_val == 4:
        return "Poor", "red"
    else:
        return "Hazardous", "purple"

# --------------------------------------------------
# 3-Day AQI Forecast Table (no numeric index, colored rows)
# --------------------------------------------------
st.subheader("🌟 3-Day AQI Forecast")
with st.spinner("Generating 3-day AQI forecast..."):
    aqi_preds = get_3day_aqi()

aqi_display = [int(round(val)) for val in aqi_preds]
future_dates = [datetime.utcnow() + timedelta(days=i) for i in range(3)]

forecast_df = pd.DataFrame({
    "Date": [d.strftime("%A, %d %b %Y") for d in future_dates],
    "Predicted AQI": aqi_display,
    "Category": [get_aqi_category(a)[0] for a in aqi_display]
})

# Color mapping
category_colors = {
    "Good": "#7CFC00",
    "Fair": "#FFFF66",
    "Moderate": "#FFA500",
    "Poor": "#FF4500",
    "Hazardous": "#800080"
}

# Display table row by row with colored backgrounds
for i, row in forecast_df.iterrows():
    color = category_colors[row["Category"]]
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; margin-bottom:5px; border-radius:5px;'>"
        f"<strong>{row['Date']}</strong> — AQI: {row['Predicted AQI']} — {row['Category']}</div>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 3-day line chart
# --------------------------------------------------
st.subheader("📊 3-Day AQI Trend")
st.line_chart(pd.DataFrame({
    "Date": [d.strftime("%a") for d in future_dates],
    "AQI": aqi_display
}).set_index("Date"))


# --------------------------------------------------
# Health Advice (3-day)
# --------------------------------------------------
st.subheader("🩺 3-Day Health Recommendations")
for i, aqi_val in enumerate(aqi_display):
    category, color = get_aqi_category(aqi_val)
    advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Fair": "Air quality is acceptable. Sensitive individuals should take caution.",
        "Moderate": "Some pollution. Limit prolonged outdoor exertion.",
        "Poor": "High pollution! Reduce outdoor activities.",
        "Hazardous": "Very unhealthy. Avoid outdoor exposure."
    }[category]
    st.markdown(
        f"**{future_dates[i].strftime('%A')}:** "
        f"<span style='color:{color}'>{category}</span> – {advice}",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 30-Day AQI Forecast (demo-mode) with day, date & month
# --------------------------------------------------
st.subheader("📈 30-Day AQI Forecast Trend (demo-mode)")

# Ensure base_aqi is defined
base_aqi = aqi_display[-1]  # last known AQI from 3-day forecast

# Generate 30-day dates
future_30_dates = [datetime.utcnow() + timedelta(days=i) for i in range(30)]

# Demo-mode AQI variations around last known AQI
aqi_30 = []
for i in range(30):
    val = max(1, min(5, base_aqi + np.random.choice([-1, 0, 1])))
    aqi_30.append(val)

# Build dataframe with full date string
forecast_30_df = pd.DataFrame({
    "Date": [d.strftime("%a %d %b") for d in future_30_dates],  # e.g., Sun 15 Feb
    "AQI": aqi_30
})

# Streamlit line chart with green line using Altair for color control
import altair as alt
chart_30 = alt.Chart(forecast_30_df).mark_line(color="green", point=True).encode(
    x=alt.X("Date", title="Day"),
    y=alt.Y("AQI", title="Predicted AQI (1–5)", scale=alt.Scale(domain=[1,5])),
    tooltip=["Date","AQI"]
).properties(width=800, height=300).interactive()

st.altair_chart(chart_30, use_container_width=True)

# --------------------------------------------------
# 📍 Map — Karachi AQI (realistic & interactive)
# --------------------------------------------------
st.subheader("📍 Map — Karachi AQI")

# Demo locations across Karachi with neighborhood names
karachi_stations = pd.DataFrame({
    "lat": [24.8607, 24.9056, 24.9575, 24.8820, 24.9260, 24.8350, 24.9210, 24.9450],
    "lon": [67.0011, 67.0810, 67.0320, 67.0500, 67.0900, 67.0200, 67.0600, 67.1000],
    "location": [
        "Clifton", "PECHS", "Korangi", "North Nazimabad", 
        "Gulshan-e-Iqbal", "Saddar", "Lyari", "Malir"
    ],
    "AQI": [3, 4, 2, 3, 5, 1, 2, 4]  # demo AQI levels
})

# Function to convert AQI to RGB color
def aqi_to_color(aqi_val):
    if aqi_val <= 1:
        return [0, 255, 0]       # green
    elif aqi_val == 2:
        return [255, 255, 0]     # yellow
    elif aqi_val == 3:
        return [255, 165, 0]     # orange
    elif aqi_val == 4:
        return [255, 0, 0]       # red
    else:
        return [128, 0, 128]     # purple

karachi_stations["color"] = karachi_stations["AQI"].apply(aqi_to_color)
karachi_stations["radius"] = karachi_stations["AQI"] * 400  # radius proportional to AQI

# PyDeck ScatterplotLayer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=karachi_stations,
    get_position='[lon, lat]',
    get_color='color',
    get_radius='radius',
    pickable=True,
    auto_highlight=True
)

# Set view over Karachi
view_state = pdk.ViewState(
    latitude=24.8607,
    longitude=67.0011,
    zoom=11,
    pitch=0
)

# Deck object with interactive tooltip
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>{location}</b><br>AQI: {AQI}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
)

st.pydeck_chart(r)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("⚡ Powered by Pearls AQI Predictor | Demo-mode 30-day forecast & trends")
