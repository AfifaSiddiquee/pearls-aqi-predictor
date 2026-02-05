import os
import pandas as pd
import hopsworks
from src.ingestion.fetch_aqi import fetch_aqi

def run_feature_pipeline():
    record = fetch_aqi()
    df = pd.DataFrame([record])

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday

    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    df[pollutant_cols] = df[pollutant_cols].fillna(-1)

    # 🔹 Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city"],
        event_time="timestamp",
        description="AQI features from OpenWeather"
    )

    fg.insert(df)

    print("Feature pipeline completed & data inserted into Hopsworks")

if __name__ == "__main__":
    run_feature_pipeline()
