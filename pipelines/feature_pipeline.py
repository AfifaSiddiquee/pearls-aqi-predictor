import os
import pandas as pd
from src.ingestion.fetch_aqi import fetch_aqi

def run_feature_pipeline():
    record = fetch_aqi()
    df = pd.DataFrame([record])

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday

    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    df[pollutant_cols] = df[pollutant_cols].fillna(-1)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/aqi_features.csv", index=False)

if __name__ == "__main__":
    run_feature_pipeline()
