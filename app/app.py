import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.inference.predict_aqi import get_3day_aqi, fetch_last_n_days, load_model

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
# Load model & last 7 days for chart
# --------------------------------------------------
@st.cache_resource
def load_model_cached():
    return load_model()

@st.cache_data
def get_last_n_days_cached(n=7):
    return fetch_last_n_days(n)

model = load_model_cached()
last_7_days_df = get_last_n_days_cached(7)

# --------------------------------------------------
# 3-Day AQI Forecast (demo-mode)
# --------------------------------------------------
st.subheader("🌟 3-Day AQI Forecast")
aqi_preds = get_3day_aqi()
aqi_display = [int(round(val)) for val in aqi_preds]
future_dates = [datetime.utcnow() + timedelta(days=i) for i in range(3)]

# Display table
forecast_df = pd.DataFrame({
    "Date": [d.strftime("%A, %d %b %Y") for d in future_dates],
    "Predicted AQI": aqi_display
})
st.table(forecast_df)

# Line chart for 3-day forecast
st.line_chart(pd.DataFrame({
    "Date": [d.strftime("%a") for d in future_dates],
    "AQI": aqi_display
}).set_index("Date"))

# --------------------------------------------------
# Past 7-Day AQI Trend
# --------------------------------------------------
st.subheader("📆 Past 7-Day AQI Trend (daily)")
# Take last 7 days timestamps & AQI values
past_dates = last_7_days_df["timestamp"].dt.date.astype(str).tolist()
past_aqi = last_7_days_df["aqi"].tolist()

past_df = pd.DataFrame({
    "Date": past_dates,
    "AQI": past_aqi
})
st.line_chart(past_df.set_index("Date"))

# --------------------------------------------------
# 30-Day Forecast (demo-mode extension)
# --------------------------------------------------
st.subheader("📈 30-Day AQI Forecast Trend (demo-mode)")

# Extend 3-day forecast logic to 30 days
from src.inference.predict_aqi import fetch_last_n_days, forecast_pollutants_demo
last_n_days_for_forecast = fetch_last_n_days(7)
forecast_vals_list = forecast_pollutants_demo(last_n_days_for_forecast)

# Repeat pattern for 30 days
forecast_30 = []
for i in range(30):
    day_vals = forecast_vals_list[i % len(forecast_vals_list)]  # loop demo trend
    # Add day info
    forecast_30.append({
        "date": (datetime.utcnow() + timedelta(days=i)).strftime("%a %d"),
        "aqi": int(round(model.predict(pd.DataFrame([day_vals]))[0]))
    })

forecast_30_df = pd.DataFrame(forecast_30)
st.line_chart(forecast_30_df.set_index("date"))

# --------------------------------------------------
# AQI Categories & Health Advice (for 3-day)
# --------------------------------------------------
st.subheader("🩺 3-Day Health Recommendations")
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

for i, aqi_val in enumerate(aqi_display):
    category, color = get_aqi_category(aqi_val)
    advice = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Fair": "Air quality is acceptable. Sensitive individuals should take caution.",
        "Moderate": "Some pollution. Limit prolonged outdoor exertion.",
        "Poor": "High pollution! Reduce outdoor activities.",
        "Hazardous": "Very unhealthy. Avoid outdoor exposure."
    }[category]
    st.markdown(f"**{future_dates[i].strftime('%A')}:** <span style='color:{color}'>{category}</span> – {advice}", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚡ Powered by Pearls AQI Predictor | Demo-mode forecast & trends")
