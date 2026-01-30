import pandas as pd
import os

def process_historical_features(input_csv="data/raw/historical_aqi.csv",
                                output_csv="data/processed/historical_features.csv"):
    """
    Process raw historical AQI data into features for model training.
    """

    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    
    # 1. Extract time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday

    # 2. Handle missing pollutant values
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    df[pollutant_cols] = df[pollutant_cols].fillna(-1)

    # 3. Select final features
    feature_columns = pollutant_cols + ["hour", "day", "month", "weekday"]
    X = df[feature_columns]
    y = df["aqi"]

    # 4. Save processed features
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed historical features saved to {output_csv}")

    return X, y

if __name__ == "__main__":
    process_historical_features()
