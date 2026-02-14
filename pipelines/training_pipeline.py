import os
import hopsworks
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    "https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/"
)

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_training_pipeline():

    # 1️⃣ Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    fg = fs.get_feature_group("karachi_aqi_features", version=1)
    df = fg.read()

    # 2️⃣ Split features & target
    X = df.drop(columns=["aqi", "timestamp", "city"])
    y = df["aqi"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 3️⃣ Train models
    all_metrics = {}

    # Random Forest
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "model")
        all_metrics["RandomForest"] = (rf, metrics)

    # Ridge
    with mlflow.start_run(run_name="Ridge"):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(ridge, "model")
        all_metrics["Ridge"] = (ridge, metrics)

    # Neural Net
    with mlflow.start_run(run_name="NeuralNet"):
        nn = Sequential([
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        nn.compile(optimizer="adam", loss="mse")
        nn.fit(X_train, y_train, epochs=20, verbose=0)
        preds = nn.predict(X_test).flatten()
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.tensorflow.log_model(nn, "model")
        all_metrics["NeuralNet"] = (nn, metrics)

    # 4️⃣ Select best model (lowest RMSE)
    best_model_name = min(all_metrics, key=lambda k: all_metrics[k][1]["rmse"])
    best_model_obj = all_metrics[best_model_name][0]

    # 5️⃣ Save best model as pickle (only for scikit-learn)
    os.makedirs("models", exist_ok=True)
    if best_model_name in ["RandomForest", "Ridge"]:
        joblib.dump(best_model_obj, "models/best_model.pkl")
        print(f"Best model ({best_model_name}) saved as models/best_model.pkl")
    else:
        # NeuralNet (TF) save as .h5
        best_model_obj.save("models/best_model.h5")
        print("Best model (NeuralNet) saved as models/best_model.h5")

    # 6️⃣ Register best model in MLflow (existing code)
    client = MlflowClient()
    MODEL_NAME = "AQI_Predictor_Best"
    # Only needed if you want to keep MLflow registry
    try:
        client.create_registered_model(MODEL_NAME)
    except:
        pass

    print(f"Best model registered in MLflow: {best_model_name}")


if __name__ == "__main__":
    run_training_pipeline()
