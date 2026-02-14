import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def run_training_pipeline():
    # 1️⃣ Load local CSV
    df = pd.read_csv("data/processed/historical_features.csv")

    # 2️⃣ Split features & target
    X = df.drop(columns=["aqi", "timestamp", "city"])
    y = df["aqi"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 3️⃣ Train models
    all_metrics = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    all_metrics["RandomForest"] = (rf, evaluate(y_test, preds))

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    all_metrics["Ridge"] = (ridge, evaluate(y_test, preds))

    # Neural Net
    nn = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    nn.compile(optimizer="adam", loss="mse")
    nn.fit(X_train, y_train, epochs=20, verbose=0)
    preds = nn.predict(X_test).flatten()
    all_metrics["NeuralNet"] = (nn, evaluate(y_test, preds))

    # 4️⃣ Select best model
    best_model_name = min(all_metrics, key=lambda k: all_metrics[k][1]["rmse"])
    best_model_obj = all_metrics[best_model_name][0]

    # 5️⃣ Save locally for Streamlit
    os.makedirs("app", exist_ok=True)
    if best_model_name in ["RandomForest", "Ridge", "XGBRegressor"]:
        joblib.dump(best_model_obj, "app/best_model.pkl")
        print(f"Saved {best_model_name} as app/best_model.pkl ✅")
    else:
        best_model_obj.save("app/best_model.h5")
        print("Saved NeuralNet as app/best_model.h5 ✅")

    print(f"Best model: {best_model_name} with metrics: {all_metrics[best_model_name][1]}")

if __name__ == "__main__":
    run_training_pipeline()
