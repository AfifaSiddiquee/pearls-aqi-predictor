import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import hopsworks
from src.ingestion.fetch_aqi import fetch_aqi
from src.ingestion.fetch_historical_aqi import fetch_historical_aqi

def run_feature_pipeline():
    # Connect to Hopsworks project
    project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
    fs = project.get_feature_store()  # Should now work with hopsworks==4.2.*

    # Define feature group
    fg_name = "karachi_aqi_features"
    try:
        fg = fs.get_feature_group(fg_name)
        print("Feature group exists. Appending new data...")
    except Exception:
        fg = fs.create_feature_group(
            name=fg_name,
            version=1,
            description="AQI and pollutant data for Karachi",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=True
        )
        print("Feature group created.")

    # 1️⃣ Fetch historical data (6 months) if feature group empty
    try:
    existing = fg.select_all().read()
    is_empty = existing.empty
    except:
    is_empty = True

    if is_empty:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=180)
        print("Fetching 6 months historical AQI data...")
        df_hist = fetch_historical_aqi(start_date, end_date)
        if not df_hist.empty:
            fg.insert(df_hist, write_options={"wait_for_job": True})
            print("Historical data inserted.")

    # 2️⃣ Fetch latest daily data and append
    print("Fetching latest AQI data...")
    latest_record = fetch_aqi()
    df_latest = pd.DataFrame([latest_record])
    fg.insert(df_latest, write_options={"wait_for_job": True})
    print("Latest data appended to feature group.")

if __name__ == "__main__":
    run_feature_pipeline()
