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
import altair as alt

st.subheader("📊 3-Day AQI Trend")

# Simulate small variations for visualization only
display_aqi = [aqi_display[0] + i*0.2 for i in range(3)]  # e.g., 3, 3.2, 3.4

trend_df = pd.DataFrame({
    "Date": [d.strftime("%a %d") for d in future_dates],
    "AQI": display_aqi
})

# Create Altair line chart with points
chart = (
    alt.Chart(trend_df)
    .mark_line(point=True, interpolate='monotone')
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
# 30-Day AQI Forecast (demo-mode)
# --------------------------------------------------
st.subheader("📈 30-Day AQI Forecast Trend (demo-mode)")

# Simple demo-mode: rolling variations around last AQI
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
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("⚡ Powered by Pearls AQI Predictor | Demo-mode 30-day forecast & trends")
