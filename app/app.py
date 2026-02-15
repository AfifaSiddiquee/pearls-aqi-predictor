import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.inference.predict_aqi import get_3day_aqi

st.set_page_config(page_title="Karachi AQI Predictor", layout="centered")
st.title("🌍 Karachi AQI 3-Day Forecast")

try:
    predictions = get_3day_aqi()
    labels = ["Today", "Tomorrow", "Day After Tomorrow"]
    
    for i, pred in enumerate(predictions):
        st.subheader(labels[i])
        st.metric("Predicted AQI Level (1-5)", pred)
        # Optional: add category & advice here
    
    # Line chart
    chart_df = pd.DataFrame({"Day": labels, "Predicted AQI": predictions})
    st.subheader("📈 Forecast Trend")
    st.line_chart(chart_df.set_index("Day"))

except Exception as e:
    st.error(f"Error loading prediction system: {e}")
