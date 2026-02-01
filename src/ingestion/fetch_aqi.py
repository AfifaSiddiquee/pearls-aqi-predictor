import os
import requests
from datetime import datetime

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

LAT = 24.8607
LON = 67.0011

def fetch_aqi():
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()["list"][0]

    return {
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
