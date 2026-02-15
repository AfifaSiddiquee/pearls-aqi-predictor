import os
import pandas as pd
import numpy as np
import mlflow
import hopsworks
import shap
from datetime import datetime, timedelta

# --------------------------------------------------
# Load model from MLflow (DagsHub)
# --------------------------------------------------
def load_model():
    mlflow.set_tracking_uri(
        "https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/"
    )

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    model = mlflow.pyfunc.load_model("models:/AQI_Predictor_Best/latest")
    return model


# --------------------------------------------------
# Fetch last N days features from Hopsworks
# --------------------------------------------------
def fetch_last_n_days(n=7):
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_aqi_features", version=1)

    df = fg.read()
    df = df.sort_values("timestamp", ascending=False)

    return df.head(n)


# --------------------------------------------------
# Forecast pollutants for next 3 days (demo logic)
# --------------------------------------------------
def forecast_pollutants_demo(last_n_days_df):

    pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3"]

    mean_vals = last_n_days_df[pollutants].mean()

    if len(last_n_days_df) >= 3:
        slopes = {
            p: (last_n_days_df[p].iloc[0] - last_n_days_df[p].iloc[2]) / 2
            for p in pollutants
        }
    else:
        slopes = {p: 0 for p in pollutants}

    future_pollutants = []

    for i in range(3):
        day_vals = {}
        for p in pollutants:
            day_vals[p] = (
                mean_vals[p]
                + slopes.get(p, 0) * i * 1.5
                + np.random.uniform(-1, 1)
            )
        future_pollutants.append(day_vals)

    return future_pollutants


# --------------------------------------------------
# Generate future features
# --------------------------------------------------
def generate_future_features():

    last_n_days = fetch_last_n_days(7)
    forecast_vals_list = forecast_pollutants_demo(last_n_days)

    future_dates = [
        datetime.utcnow(),
        datetime.utcnow() + timedelta(days=1),
        datetime.utcnow() + timedelta(days=2),
    ]

    rows = []

    for i, date in enumerate(future_dates):
        row = forecast_vals_list[i].copy()
        row["hour"] = date.hour
        row["day"] = date.day
        row["month"] = date.month
        row["weekday"] = date.weekday()
        rows.append(row)

    future_df = pd.DataFrame(rows)

    return future_df


# --------------------------------------------------
# Main function — AQI Prediction + Optional SHAP
# --------------------------------------------------
def get_3day_aqi(return_explanations=False):

    model = load_model()
    future_df = generate_future_features()

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------
    preds = model.predict(future_df)
    preds = np.round(preds, 1)

    # ---------------------------------------------
    # SHAP Explanation (optional)
    # ---------------------------------------------
    if return_explanations:
    try:
        # Attempt Tree SHAP
        if hasattr(model, "_model_impl"):
            inner_model = model._model_impl.python_model.model
        elif hasattr(model, "python_model"):
            inner_model = model.python_model.model
        else:
            inner_model = model

        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(future_df)

    except Exception:
        # Fallback: permutation explainer (works on pyfunc)
        explainer = shap.Explainer(model.predict, future_df, algorithm="permutation")
        shap_values = explainer(future_df)

    return preds, shap_values, future_df

