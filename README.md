# Pearls AQI Predictor

A real-time **Air Quality Index (AQI) forecasting system** for Karachi, Pakistan, predicting AQI for the next 3 days using a **serverless, automated ML stack**. The project integrates automated data collection, feature engineering, machine learning model training, and an interactive dashboard for visualizing AQI trends and pollutant contributions.

---

## 🌟 Key Features

- **Automated Feature Pipeline**
  - Fetches live and historical AQI and pollutant data via **OpenWeather API**.
  - Computes time-based features (`hour`, `day`, `month`, `weekday`) and pollutant trends.
  - Stores processed features in **Hopsworks Feature Store** for scalable and consistent data access.

- **Historical Data Backfill & EDA**
  - Uses past 6+ months of AQI data for model training.
  - Performs Exploratory Data Analysis (EDA) to identify patterns, trends, and correlations.
  - Supports **SHAP** explanations to understand feature contributions.

- **Machine Learning Pipeline**
  - Trains multiple models: **Random Forest, Ridge Regression, Neural Network**.
  - Evaluates performance using **RMSE, MAE, R²** metrics.
  - Registers best-performing model (**Random Forest**) in **Dagshub MLflow**.
  - **Random Forest metrics**:
    - MAE: 0.0096
    - R²: 0.9913
    - RMSE: 0.0861
  - Random Forest consistently performs best due to its ensemble approach, handling nonlinear pollutant interactions effectively.

- **CI/CD Automation**
  - Feature pipeline runs hourly; training pipeline runs daily.
  - Implemented using **GitHub Actions** for fully automated workflows.

- **Interactive Dashboard**
  - Built with **Streamlit**, **Altair**, and **PyDeck**.
  - Displays:
    - Real-time AQI (1–5 scale) with color-coded categories.
    - 3-day AQI forecasts and 30-day demo trends.
    - SHAP-based top feature contributions.
    - Live pollutant composition for the last 7 days.
    - Map visualization of AQI across Karachi stations.
  - Provides health recommendations and AQI alerts.

---

## 🛠 Technology Stack

| **Component**              | **Technology / Tool**                        |
|-----------------------------|---------------------------------------------|
| Data Collection             | OpenWeather API                             |
| Feature Store               | Hopsworks                                   |
| Model Registry              | Dagshub MLflow                              |
| Machine Learning Models     | Random Forest, Ridge, Neural Network        |
| ML Libraries                | scikit-learn, xgboost, TensorFlow           |
| Model Explainability        | SHAP                                        |
| Dashboard / Frontend        | Streamlit, Altair, PyDeck                   |
| CI/CD                       | GitHub Actions                              |
| Programming Language        | Python 3.11                                 |
| Dependency Management       | requirements.txt, requirements_pipeline.txt |

---

## ⚙️ Project Structure
pearls-aqi-predictor
├── .github/workflows # CI/CD workflows for feature & training pipelines
├── data
│ ├── raw # Raw AQI & pollutant CSV files
│ └── processed # Processed feature datasets
├── notebooks # EDA & exploration notebooks
├── pipelines # Feature & training pipelines
├── src
│ ├── config # Configuration files
│ ├── features # Feature engineering scripts
│ ├── ingestion # API data fetching scripts
│ └── inference # Model prediction scripts
├── app # Streamlit dashboard
├── requirements.txt # Dependencies for app
├── requirements_pipeline.txt # Dependencies for pipelines
└── README.md

