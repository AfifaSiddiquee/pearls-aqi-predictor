import pandas as pd
import hopsworks
from datetime import datetime

from src.ingestion.fetch_aqi import fetch_aqi


def run_feature_pipeline():
    try:
        print("Fetching AQI...")
        record = fetch_aqi()
        print("Record fetched:", record)

        # Convert single record to DataFrame
        df = pd.DataFrame([record])

        print("Connecting to Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Feature Group definition
        feature_group = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["city", "timestamp"],
            description="Daily AQI and pollutant features for Karachi",
            online_enabled=False
        )

        print("Inserting data into Feature Store...")
        feature_group.insert(df, write_options={"wait_for_job": True})

        print("✅ Feature pipeline completed successfully")

    except Exception as e:
        print("🚨 Feature pipeline failed:", e)
        raise


if __name__ == "__main__":
    run_feature_pipeline()
