import subprocess
import sys

# Install dependencies from requirements_app.txt at startup
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements_app.txt"])


import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import hopsworks
import dagshub

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AQI Predictor", layout="wide")
st.title("🌍 AQI Prediction Dashboard")

# -------------------------------
# Connect to Hopsworks
# -------------------------------
@st.cache_resource
def load_feature_data():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    return df

# -------------------------------
# Load Best Model from MLflow
# -------------------------------
@st.cache_resource
def load_model():
    dagshub.init(repo_owner="YOUR_USERNAME", repo_name="AQIPredictorProject", mlflow=True)
    model = mlflow.pyfunc.load_model("models:/AQI_Predictor_Best/Production")
    return model

# -------------------------------
# Prepare Future Dates
# -------------------------------
def prepare_future_data(df):
    latest_row = df.sort_values("date").iloc[-1:]

    future_rows = []
    for i in range(3):
        new_row = latest_row.copy()
        new_row["date"] = pd.to_datetime(new_row["date"]) + pd.Timedelta(days=i)
        future_rows.append(new_row)

    future_df = pd.concat(future_rows)
    return future_df

# -------------------------------
# MAIN
# -------------------------------
df = load_feature_data()
model = load_model()

future_df = prepare_future_data(df)

predictions = model.predict(future_df.drop(columns=["aqi", "date"], errors="ignore"))

future_df["Predicted AQI"] = predictions

# -------------------------------
# Display Dashboard
# -------------------------------
st.subheader("📊 Next 3 Days AQI Prediction")
st.dataframe(future_df[["date", "Predicted AQI"]])

st.line_chart(future_df.set_index("date")["Predicted AQI"])
