import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os

# Add repo root to sys.path so Python can find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.inference.predict_aqi import get_3day_aqi

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Karachi AQI Predictor")
st.markdown("Real-time AQI predictions with forecast visualization.")

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
# 3-Day AQI Forecast
# --------------------------------------------------
st.subheader("🌟 3-Day AQI Forecast")
aqi_preds = get_3day_aqi()
aqi_display = [int(round(val)) for val in aqi_preds]
future_dates = [datetime.utcnow() + timedelta(days=i) for i in range(3)]

cols = st.columns(3)
for i, col in enumerate(cols):
    category, color = get_aqi_category(aqi_display[i])
    col.metric(
        label=future_dates[i].strftime("%A"),
        value=str(aqi_display[i]),
        delta=category
    )

# 3-day line chart
st.line_chart(pd.DataFrame({
    "Date": [d.strftime("%a") for d in future_dates],
    "AQI": aqi_display
}).set_index("Date"))

# --------------------------------------------------
# 30-Day AQI Forecast (demo-mode)
# --------------------------------------------------
st.subheader("📈 30-Day AQI Forecast Trend (demo-mode)")

base_aqi = aqi_display[-1]  # last known AQI
aqi_30 = []
for i in range(30):
    # Slight variation to simulate trend
    val = max(1, min(5, base_aqi + np.random.choice([-1, 0, 1])))
    aqi_30.append(val)

forecast_30_df = pd.DataFrame({
    "Date": [(datetime.utcnow() + timedelta(days=i)).strftime("%a %d") for i in range(30)],
    "AQI": aqi_30
})
st.line_chart(forecast_30_df.set_index("Date"))

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
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("⚡ Powered by Pearls AQI Predictor | Demo-mode 30-day forecast & trends")
