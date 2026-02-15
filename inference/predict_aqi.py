import os
import pandas as pd
import numpy as np
import mlflow
import hopsworks
from datetime import datetime, timedelta

# --------------------------------------------------
# Load model from MLflow
# --------------------------------------------------
def load_model():
    mlflow.set_tracking_uri(
        "https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/"
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    model = mlflow.pyfunc.load_model("models:/AQI_Predictor_Best/latest")
    return model

# --------------------------------------------------
# Fetch last N days features from Hopsworks
# --------------------------------------------------
def fetch_last_n_days(n=7):
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp", ascending=False)
    return df.head(n)

# --------------------------------------------------
# Forecast pollutants for next 3 days
# --------------------------------------------------
def forecast_pollutants(last_n_days_df):
    # Rolling mean over last N days
    pollutants = ["pm25","pm10","co","no2","so2","o3"]
    forecast_values = last_n_days_df[pollutants].mean().to_dict()
    
    # Optional: simple linear trend based on last 3 days
    if len(last_n_days_df) >= 3:
        for p in pollutants:
            trend = (last_n_days_df[p].iloc[0] - last_n_days_df[p].iloc[2]) / 2
            forecast_values[p] += trend  # simple linear trend
    
    return forecast_values

# --------------------------------------------------
# Generate 3-day features
# --------------------------------------------------
def generate_future_features():
    last_n_days = fetch_last_n_days(7)
    forecast_vals = forecast_pollutants(last_n_days)
    
    future_dates = [
        datetime.utcnow(),
        datetime.utcnow() + timedelta(days=1),
        datetime.utcnow() + timedelta(days=2)
    ]
    
    rows = []
    for date in future_dates:
        row = {
            **forecast_vals,
            "hour": date.hour,
            "day": date.day,
            "month": date.month,
            "weekday": date.weekday()
        }
        rows.append(row)
    
    future_df = pd.DataFrame(rows)
    return future_df

# --------------------------------------------------
# Main function to return 3-day AQI predictions
# --------------------------------------------------
def get_3day_aqi():
    model = load_model()
    future_df = generate_future_features()
    preds = model.predict(future_df)
    preds = np.round(preds).astype(int)
    return preds
