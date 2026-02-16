import os
from datetime import datetime, timedelta
import pandas as pd
import hopsworks
from src.ingestion.fetch_aqi import fetch_aqi
from src.ingestion.fetch_historical_aqi import fetch_historical_aqi


def run_feature_pipeline():
    # 1️⃣ Connect to Hopsworks project
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()

    # 2️⃣ Define feature group
    fg_name = "karachi_aqi_features"
    fg = fs.get_feature_group(name=fg_name, version=1)

    if fg is None:
        print("Feature group does not exist. Creating new feature group...")
        fg = fs.create_feature_group(
            name=fg_name,
            version=1,
            description="AQI and pollutant data for Karachi",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False  
        )
    else:
        print("Feature group exists. Appending new data...")

    # 3️⃣ Check if feature group is empty
    try:
        existing = fg.select_all().read()
        is_empty = existing.empty
    except Exception:
        is_empty = True

    # 4️⃣ Historical backfill (run manually once recommended)
    if is_empty:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=120)  # last 4 months
        print(f"Fetching historical AQI data from {start_date.date()} to {end_date.date()}...")
        df_hist = fetch_historical_aqi(start_date, end_date)

        if not df_hist.empty:
            # Add time-based features
            df_hist["hour"] = df_hist["timestamp"].dt.hour
            df_hist["day"] = df_hist["timestamp"].dt.day
            df_hist["month"] = df_hist["timestamp"].dt.month
            df_hist["weekday"] = df_hist["timestamp"].dt.weekday

            fg.insert(df_hist, write_options={"wait_for_job": True})
            print("Historical data inserted.")

    # 5️⃣ Fetch latest AQI data
    print("Fetching latest AQI data...")
    latest_record = fetch_aqi()
    df_latest = pd.DataFrame([latest_record])

    # Add time-based features for latest record
    df_latest["hour"] = df_latest["timestamp"].dt.hour
    df_latest["day"] = df_latest["timestamp"].dt.day
    df_latest["month"] = df_latest["timestamp"].dt.month
    df_latest["weekday"] = df_latest["timestamp"].dt.weekday

    fg.insert(df_latest)
    print("Latest data appended to feature group.")


if __name__ == "__main__":
    run_feature_pipeline()
