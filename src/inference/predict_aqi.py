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
# Forecast pollutants for next 3 days using rolling mean + trend
# --------------------------------------------------
def forecast_pollutants(last_n_days_df):
    pollutants = ["pm25","pm10","co","no2","so2","o3"]
    
    # Rolling mean over last N days
    mean_vals = last_n_days_df[pollutants].mean()
    
    # Compute linear trend based on last 3 days
    if len(last_n_days_df) >= 3:
        slopes = {}
        for p in pollutants:
            slopes[p] = (last_n_days_df[p].iloc[0] - last_n_days_df[p].iloc[2]) / 2
    else:
        slopes = {p: 0 for p in pollutants}

    # Generate 3-day forecast with day-by-day adjustment
    future_pollutants = []
    for i in range(3):
        day_vals = {}
        for p in pollutants:
            day_vals[p] = mean_vals[p] + slopes.get(p, 0) * i
            # Optional: small random fluctuation to avoid exact same values
            day_vals[p] += np.random.uniform(-0.5, 0.5)
        future_pollutants.append(day_vals)

    return future_pollutants

# --------------------------------------------------
# Generate future features including time features
# --------------------------------------------------
def generate_future_features():
    last_n_days = fetch_last_n_days(7)
    forecast_vals_list = forecast_pollutants(last_n_days)
    
    future_dates = [
        datetime.utcnow(),
        datetime.utcnow() + timedelta(days=1),
        datetime.utcnow() + timedelta(days=2)
    ]
    
    rows = []
    for i, date in enumerate(future_dates):
        row = forecast_vals_list[i].copy()
        row["hour"] = date.hour
        row["day"] = date.day
        row["month"] = date.month
        row["weekday"] = date.weekday()
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
