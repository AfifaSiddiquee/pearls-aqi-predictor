import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pydeck as pdk
import shap
import matplotlib.pyplot as plt
from src.inference.predict_aqi import get_3day_aqi, fetch_last_n_days
aqi_preds, shap_vals, future_features = get_3day_aqi(return_explanations=True)

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
    aqi_preds, shap_values, future_features = get_3day_aqi(return_explanations=True)

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
# 3-Day AQI Trend (y-axis fixed 1–5)
# --------------------------------------------------
import altair as alt

st.subheader("📊 3-Day AQI Trend")

trend_df = pd.DataFrame({
    "Date": [d.strftime("%a %d") for d in future_dates],
    "AQI": aqi_display  # use actual predicted values
})

# Altair line chart with fixed y-axis
chart = (
    alt.Chart(trend_df)
    .mark_line(point=True, interpolate='monotone', color='green')
    .encode(
        x=alt.X("Date", title="Day"),
        y=alt.Y("AQI", title="Predicted AQI (1–5)", scale=alt.Scale(domain=[1, 5])),
        tooltip=["Date", "AQI"]
    )
    .properties(width=600, height=300)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)



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
# 🔍 SHAP Model Explanation
# --------------------------------------------------
st.subheader("🔍 Model Explanation — SHAP Analysis")

with st.spinner("Computing model explanations..."):
    try:
        # ---------------------------
        # Global Feature Importance
        # ---------------------------
        st.markdown("### 📊 Global Feature Importance")

        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_vals,  # from get_3day_aqi(return_explanations=True)
            future_features,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig1)

    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {e}")


# --------------------------------------------------
# 🧪 Live Pollutant Composition — Last 7 Days (Sorted Column Chart)
# --------------------------------------------------
st.subheader("🧪 Live Pollutant Composition — Last 7 Days")

last_7_days_df = fetch_last_n_days(7)

pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3"]
avg_pollutants = last_7_days_df[pollutants].mean().round(3)

total = avg_pollutants.sum()
percentages = (avg_pollutants / total * 100).round(2)

composition_df = pd.DataFrame({
    "Pollutant": avg_pollutants.index,
    "Percentage": percentages.values
})

# 🔥 SORT descending (highest to lowest)
composition_df = composition_df.sort_values(
    by="Percentage",
    ascending=False
)

import altair as alt

bar_chart = (
    alt.Chart(composition_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "Pollutant:N",
            sort="-y",   # ensures visual sorting matches percentage
            title="Pollutant"
        ),
        y=alt.Y("Percentage:Q", title="Contribution (%)"),
        color=alt.Color("Pollutant:N", legend=None),
        tooltip=[
            alt.Tooltip("Pollutant:N"),
            alt.Tooltip("Percentage:Q", format=".2f")
        ]
    )
    .properties(width=600, height=400)
)

text = bar_chart.mark_text(
    dy=-10
).encode(
    text=alt.Text("Percentage:Q", format=".2f")
)

st.altair_chart(bar_chart + text, use_container_width=True)

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

# More locations across Karachi
karachi_stations = pd.DataFrame({
    "lat": [
        24.8607, 24.9056, 24.9575, 24.8820, 24.9260, 24.8350, 24.9210, 24.9450,
        24.8500, 24.8700, 24.8900, 24.9100, 24.9300, 24.9500, 24.9700, 24.8800,
        24.8400, 24.8950, 24.9150, 24.9350
    ],
    "lon": [
        67.0011, 67.0810, 67.0320, 67.0500, 67.0900, 67.0200, 67.0600, 67.1000,
        67.0100, 67.0200, 67.0300, 67.0400, 67.0500, 67.0600, 67.0700, 67.0800,
        67.0900, 67.1000, 67.1100, 67.1200
    ],
    "location": [
        "Clifton", "PECHS", "Korangi", "North Nazimabad", "Gulshan-e-Iqbal", "Saddar", "Lyari", "Malir",
        "Shah Faisal", "Defence", "Gulistan-e-Jauhar", "Nazimabad", "SITE", "Korangi Creek", "Landhi", "Airport",
        "Baldia", "Orangi Town", "Hyderi", "Bahadurabad"
    ],
    "AQI": [
        3, 4, 2, 3, 5, 1, 2, 4,
        3, 2, 3, 4, 1, 2, 3, 4,
        2, 3, 4, 3
    ]  # demo AQI levels
})

# Convert AQI to RGB color
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
karachi_stations["radius"] = karachi_stations["AQI"] * 100  # smaller radius for many points

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

# Deck object with interactive tooltip (only AQI number)
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
