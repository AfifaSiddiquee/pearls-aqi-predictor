import os
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import hopsworks

from src.ingestion.fetch_aqi import fetch_aqi
from src.ingestion.fetch_historical_aqi import fetch_historical_aqi

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Safe Insert Function (Retry + No Materialization)
# --------------------------------------------------
def safe_insert(fg, df, label="data"):
    for attempt in range(3):
        try:
            fg.insert(
                df,
                write_options={
                    "start_offline_materialization": False
                }
            )
            print(f"✅ {label} inserted successfully.")
            return
        except Exception as e:
            print(f"⚠ Attempt {attempt+1}/3 failed: {e}")
            time.sleep(10)

    raise Exception(f"❌ Failed to insert {label} after 3 attempts.")


# --------------------------------------------------
# Main Feature Pipeline
# --------------------------------------------------
def run_feature_pipeline():

    # 1️⃣ Connect to Hopsworks
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    host="eu-west.cloud.hopsworks.ai")
    fs = project.get_feature_store()

    # 2️⃣ Get or Create Feature Group
    fg_name = "karachi_aqi_features"

    try:
        fg = fs.get_feature_group(name=fg_name, version=1)
        print("Feature group exists. Appending new data...")
    except:
        print("Feature group does not exist. Creating new feature group...")
        fg = fs.create_feature_group(
            name=fg_name,
            version=1,
            description="AQI and pollutant data for Karachi",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=False
        )

    # 3️⃣ Check if Feature Group is empty
    try:
        existing = fg.select_all().read()
        is_empty = existing.empty
    except Exception:
        is_empty = True

    # 4️⃣ Historical Backfill (only if empty)
    if is_empty:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=120)

        print(f"Fetching historical AQI data from {start_date.date()} to {end_date.date()}...")
        df_hist = fetch_historical_aqi(start_date, end_date)

        if not df_hist.empty:

            # Time features
            df_hist["hour"] = df_hist["timestamp"].dt.hour
            df_hist["day"] = df_hist["timestamp"].dt.day
            df_hist["month"] = df_hist["timestamp"].dt.month
            df_hist["weekday"] = df_hist["timestamp"].dt.weekday

            safe_insert(fg, df_hist, label="Historical data")

    # 5️⃣ Fetch Latest AQI
    print("Fetching latest AQI data...")
    latest_record = fetch_aqi()
    df_latest = pd.DataFrame([latest_record])

    # Time features
    df_latest["hour"] = df_latest["timestamp"].dt.hour
    df_latest["day"] = df_latest["timestamp"].dt.day
    df_latest["month"] = df_latest["timestamp"].dt.month
    df_latest["weekday"] = df_latest["timestamp"].dt.weekday

    safe_insert(fg, df_latest, label="Latest data")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()
