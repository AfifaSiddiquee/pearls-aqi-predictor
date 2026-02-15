import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.inference.predict_aqi import get_3day_aqi, fetch_last_n_days, load_model

# --------------------------------------------------
# Streamlit caching for heavy operations
# --------------------------------------------------
@st.cache_resource
def load_model_cached():
    return load_model()

@st.cache_data
def get_last_n_days_cached(n=7):
    return fetch_last_n_days(n)

# --------------------------------------------------
# Main Streamlit App
# --------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="centered"
)

st.title("🌍 Karachi AQI 3-Day Forecast")
st.markdown("Predicting Air Quality Index (1–5) using real-time features.")

# --------------------------------------------------
# Load model and last 7 days features
# --------------------------------------------------
with st.spinner("Loading prediction system..."):
    model = load_model_cached()
    last_n_days = get_last_n_days_cached(7)

# --------------------------------------------------
# Generate 3-day AQI predictions
# --------------------------------------------------
with st.spinner("Generating 3-day AQI forecast..."):
    # Use your existing demo-mode logic
    aqi_preds = get_3day_aqi()

# --------------------------------------------------
# Display predictions
# --------------------------------------------------
future_dates = [
    datetime.utcnow(),
    datetime.utcnow() + pd.Timedelta(days=1),
    datetime.utcnow() + pd.Timedelta(days=2)
]

for i, date in enumerate(future_dates):
    st.subheader(date.strftime("%A, %d %b %Y"))
    st.metric("Predicted AQI Level (1-5)", aqi_preds[i])

# Optional: color-coded categories
def get_aqi_category(aqi_val):
    if aqi_val <= 1.5:
        return "Good", "green"
    elif aqi_val <= 2.5:
        return "Fair", "yellow"
    elif aqi_val <= 3.5:
        return "Moderate", "orange"
    elif aqi_val <= 4.5:
        return "Poor", "red"
    else:
        return "Hazardous", "purple"

st.markdown("---")
st.subheader("AQI Categories & Health Advice")
for i, aqi_val in enumerate(aqi_preds):
    category, color = get_aqi_category(aqi_val)
    st.markdown(
        f"**{future_dates[i].strftime('%A')}:** <span style='color:{color}'>{category}</span>",
        unsafe_allow_html=True
    )
