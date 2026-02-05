import os
import pandas as pd
import hopsworks
from src.ingestion.fetch_aqi import fetch_aqi

def run_feature_pipeline():
    try:
        # Step 1: Fetch AQI
        print("Fetching AQI...")
        record = fetch_aqi()
        print("Record fetched:", record)

        df = pd.DataFrame([record])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='raise')

        # Step 2: Feature engineering
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["weekday"] = df["timestamp"].dt.weekday

        pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
        df[pollutant_cols] = df[pollutant_cols].fillna(-1)

        # Step 3: Connect to Hopsworks
        print("Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        print("Logged in to Hopsworks:", project)

        fs = project.get_feature_store()

        # Step 4: Create or get feature group
        print("Creating feature group...")
        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            description="AQI features from OpenWeather"
        )
        print("Feature group created:", fg.name)

        # Step 5: Insert features safely
        print("Inserting features into Hopsworks...")
        fg.insert(df, write_options={"wait_for_job": True, "upsert": True})
        print("Feature pipeline completed & data inserted into Hopsworks")

    except Exception as e:
        print("🚨 Feature pipeline failed:", e)
        raise

if __name__ == "__main__":
    run_feature_pipeline()
