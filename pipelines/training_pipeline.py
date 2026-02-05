import os
import hopsworks
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/AfifaSiddiquee/AQIPredictorProject.mlflow/")

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}



def run_training_pipeline():

    # 1️⃣ Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()

    # 2️⃣ Split features & target
    X = df.drop(columns=["aqi", "timestamp"])
    y = df["aqi"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 3️⃣ Train models
    all_metrics = {}

    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "model")
        all_metrics["RandomForest"] = (mlflow.active_run().info.run_id, metrics)

    with mlflow.start_run(run_name="Ridge"):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)
        metrics = evaluate(y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(ridge, "model")
        all_metrics["Ridge"] = (mlflow.active_run().info.run_id, metrics)

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
        all_metrics["NeuralNet"] = (mlflow.active_run().info.run_id, metrics)

    # 4️⃣ Select best model
    best_model = min(all_metrics, key=lambda k: all_metrics[k][1]["rmse"])
    best_run_id = all_metrics[best_model][0]

    # 5️⃣ Register best model
    client = MlflowClient()

    MODEL_NAME = "AQI_Predictor_Best"
    model_uri = f"runs:/{best_run_id}/model"

    try:
        client.create_registered_model(MODEL_NAME)
    except:
        pass

    client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=best_run_id
    )

    print(f"Best model registered: {best_model}")


if __name__ == "__main__":
    run_training_pipeline()
