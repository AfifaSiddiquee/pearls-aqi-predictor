import os
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import hopsworks
from datetime import datetime, timedelta

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Karachi AQI Predictor", layout="centered")

st.title("🌍 Karachi AQI 3-Day Forecast")
st.write("Real-time AQI prediction using ML model")

# --------------------------------------------------
# Load ML Model from DagsHub
# --------------------------------------------------
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(
        "https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/"
    )

    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

    model = mlflow.pyfunc.load_model("models:/AQI_Predictor_Best/latest")
    return model

# --------------------------------------------------
# Fetch Latest Features from Hopsworks
# --------------------------------------------------
@st.cache_resource
def get_latest_features():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()

    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()

    df = df.sort_values("timestamp", ascending=False)
    latest = df.iloc[0]

    return latest

# --------------------------------------------------
# AQI Category + Health Advice
# --------------------------------------------------
def aqi_category(aqi_value):
    if aqi_value <= 1:
        return "Good", "Air quality is satisfactory. Enjoy outdoor activities."
    elif aqi_value <= 2:
        return "Fair", "Air quality acceptable. Sensitive individuals should be cautious."
    elif aqi_value <= 3:
        return "Moderate", "Limit prolonged outdoor exertion."
    elif aqi_value <= 4:
        return "Poor", "Avoid outdoor activities if possible."
    else:
        return "Very Poor", "Stay indoors. Use masks if going outside."

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
try:
    model = load_model()
    latest = get_latest_features()

    # Create 3 future days
    future_dates = [
        datetime.utcnow(),
        datetime.utcnow() + timedelta(days=1),
        datetime.utcnow() + timedelta(days=2)
    ]

    rows = []

    for date in future_dates:
        row = {
            "pm25": latest["pm25"],
            "pm10": latest["pm10"],
            "co": latest["co"],
            "no2": latest["no2"],
            "so2": latest["so2"],
            "o3": latest["o3"],
            "hour": date.hour,
            "day": date.day,
            "month": date.month,
            "weekday": date.weekday()
        }
        rows.append(row)

    future_df = pd.DataFrame(rows)

    predictions = model.predict(future_df)
    predictions = np.round(predictions).astype(int)

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    st.subheader("📅 3-Day AQI Forecast")

    for i, pred in enumerate(predictions):
        label = ["Today", "Tomorrow", "Day After Tomorrow"][i]
        category, advice = aqi_category(pred)

        st.markdown(f"### {label}")
        st.metric("Predicted AQI Level (1-5)", pred)
        st.write(f"**Category:** {category}")
        st.write(f"💡 {advice}")
        st.divider()

    # Line Chart
    chart_df = pd.DataFrame({
        "Day": ["Today", "Tomorrow", "Day After Tomorrow"],
        "Predicted AQI": predictions
    })

    st.subheader("📈 Forecast Trend")
    st.line_chart(chart_df.set_index("Day"))

except Exception as e:
    st.error(f"Error loading prediction system: {e}")
