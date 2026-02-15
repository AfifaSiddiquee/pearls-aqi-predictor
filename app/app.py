import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.inference.predict_aqi import get_3day_aqi

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Karachi AQI 3-Day Forecast")
st.markdown("Predicting Air Quality Index (1–5) using real-time features.")

# --------------------------------------------------
# Fetch 3-day AQI predictions
# --------------------------------------------------
with st.spinner("Generating 3-day AQI forecast..."):
    aqi_preds = get_3day_aqi()

# Convert to whole numbers for display
aqi_display = [int(round(val)) for val in aqi_preds]

# Generate future dates
future_dates = [
    datetime.utcnow(),
    datetime.utcnow() + pd.Timedelta(days=1),
    datetime.utcnow() + pd.Timedelta(days=2)
]

# --------------------------------------------------
# AQI Category & Color Mapping
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
# Display 3-Day Forecast
# --------------------------------------------------
st.subheader("📊 AQI Forecast Table")
forecast_df = pd.DataFrame({
    "Date": [d.strftime("%A, %d %b %Y") for d in future_dates],
    "Predicted AQI": aqi_display,
    "Category": [get_aqi_category(a)[0] for a in aqi_display]
})
st.table(forecast_df)

# Display metric cards for each day
st.subheader("🌟 Daily AQI Levels")
cols = st.columns(3)
for i, col in enumerate(cols):
    category, color = get_aqi_category(aqi_display[i])
    col.metric(
        label=future_dates[i].strftime("%A"),
        value=str(aqi_display[i]),
        delta=category
    )

# --------------------------------------------------
# Interactive Line Chart
# --------------------------------------------------
st.subheader("📈 AQI Trend (3-Day Forecast)")
trend_df = pd.DataFrame({
    "Date": [d.strftime("%a") for d in future_dates],
    "AQI": aqi_display
})
st.line_chart(trend_df.set_index("Date"))

# --------------------------------------------------
# Health Advice Section
# --------------------------------------------------
st.subheader("🩺 Health Recommendations")
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

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("⚡ Powered by Pearls AQI Predictor | Demo-mode 3-day forecast")
