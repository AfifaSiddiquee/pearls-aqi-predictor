import os
import streamlit as st
import pandas as pd
from datetime import timedelta
import hopsworks
import mlflow
import mlflow.pyfunc

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")
st.title("🌫 Karachi AQI 3-Day Forecast Dashboard")

# -------------------------------------------------
# Connect to Feature Store
# -------------------------------------------------
@st.cache_resource
def connect_feature_store():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    return project.get_feature_store()

fs = connect_feature_store()
fg = fs.get_feature_group("karachi_aqi_features", version=1)
df = fg.read().sort_values("timestamp")

latest_row = df.iloc[-1:].copy()
last_timestamp = pd.to_datetime(latest_row["timestamp"].values[0])

# -------------------------------------------------
# Load Best Model from Dagshub MLflow
# -------------------------------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "AQI_Predictor_Best"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

# -------------------------------------------------
# AQI Category Mapping (1–5 Scale)
# -------------------------------------------------
def aqi_category(aqi_value):
    categories = {
        1: ("Good", "🟢",
            "Air quality is satisfactory, and air pollution poses little or no risk."),
        2: ("Fair", "🟡",
            "Acceptable air quality; sensitive individuals may experience slight effects."),
        3: ("Moderate", "🟠",
            "Air quality is acceptable, but sensitive groups may experience mild effects."),
        4: ("Poor", "🔴",
            "Sensitive groups may experience health effects; general public may also be affected."),
        5: ("Very Poor", "🟣",
            "Health warnings of emergency conditions; everyone may experience serious effects.")
    }

    return categories.get(int(round(aqi_value)),
                          ("Unknown", "⚪", "No description available."))

# -------------------------------------------------
# Generate 3-Day Forecast
# -------------------------------------------------
future_predictions = []

for i in range(1, 4):
    future_time = last_timestamp + timedelta(days=i)

    new_row = latest_row.copy()
    new_row["timestamp"] = future_time
    new_row["hour"] = future_time.hour
    new_row["day"] = future_time.day
    new_row["month"] = future_time.month
    new_row["weekday"] = future_time.weekday()

    X_future = new_row.drop(columns=["aqi", "timestamp", "city"])
    prediction = model.predict(X_future)[0]

    future_predictions.append({
        "Date": future_time.date(),
        "Predicted AQI": round(prediction)
    })

forecast_df = pd.DataFrame(future_predictions)

# -------------------------------------------------
# Display Forecast
# -------------------------------------------------
st.subheader("📅 3-Day AQI Forecast")

for _, row in forecast_df.iterrows():
    category, color, description = aqi_category(row["Predicted AQI"])

    st.metric(
        label=str(row["Date"]),
        value=f"{int(row['Predicted AQI'])} ({category})"
    )

    st.caption(f"{color} {description}")

    # 🚨 Hazard Alert
    if int(row["Predicted AQI"]) >= 4:
        st.warning("⚠️ Air quality is Poor or Very Poor. Limit outdoor exposure.")

    st.divider()

# -------------------------------------------------
# Historical Trend (Last 7 Days)
# -------------------------------------------------
st.subheader("📈 Last 7 Days AQI Trend")
last_7 = df.tail(7)
st.line_chart(last_7.set_index("timestamp")["aqi"])

# -------------------------------------------------
# Feature Importance (Tree Models Only)
# -------------------------------------------------
st.subheader("🔍 Feature Importance")

try:
    sklearn_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    importances = sklearn_model.feature_importances_
    feature_names = X_future.columns

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))

except Exception:
    st.info("Feature importance available only for tree-based models.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Built with Hopsworks Feature Store + MLflow (Dagshub) + Streamlit 🚀")
