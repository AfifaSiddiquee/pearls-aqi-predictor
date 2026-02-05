import os
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

# Karachi coordinates
LAT = 24.8607
LON = 67.0011

def fetch_historical_aqi(start_date, end_date):
    """
    Fetch historical AQI and pollutant data from OpenWeather API
    between start_date and end_date (datetime objects)
    Returns a DataFrame
    """
    records = []    
    current_date = start_date

    while current_date <= end_date:
        start_unix = int(current_date.timestamp())
        end_unix = int((current_date + timedelta(days=1)).timestamp() - 1)

        params = {
            "lat": LAT,
            "lon": LON,
            "start": start_unix,
            "end": end_unix,
            "appid": API_KEY
        }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data_list = response.json().get("list", [])
            for data in data_list:
                record = {
                    "city": "Karachi",
                    "timestamp": datetime.utcfromtimestamp(data["dt"]),
                                        "aqi": data["main"]["aqi"],
                    "pm25": data["components"].get("pm2_5"),
                    "pm10": data["components"].get("pm10"),
                    "co": data["components"].get("co"),
                    "no2": data["components"].get("no2"),
                    "so2": data["components"].get("so2"),
                    "o3": data["components"].get("o3")
                }
                records.append(record)
        except Exception as e:
            print(f"Error fetching data for {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    print("Fetching historical AQI data...")
    df = fetch_historical_aqi(start_date, end_date)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/historical_aqi.csv", index=False)

    print(f"Saved {len(df)} records")
