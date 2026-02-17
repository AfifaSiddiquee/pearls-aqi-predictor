# Karachi — Air Quality Monitoring & Forecast Dashboard

A real-time **Air Quality Index (AQI) forecasting system** for Karachi, Pakistan, predicting AQI for the next 3 days using a **100% serverless stack**. This project combines automated data collection, feature engineering, machine learning model training, and interactive visualization in a dashboard.

 **Interactive App**: [https://pearls-aqi-predictor-9ptrenpz4c2easxuwyt4le.streamlit.app/]
---

## 🌟 Key Features

1. **Automated Feature Pipeline**
   - Fetches live and historical weather & pollutant data using **OpenWeather API**.
   - Computes time-based features (`hour`, `day`, `month`, `weekday`) and derived features like pollutant trends.
   - Stores processed features in **Hopsworks Feature Store** for consistent and reusable data.

2. **Historical Data Backfill & EDA**
   - Generates training datasets using past 6+ months of AQI data.
   - Performs Exploratory Data Analysis (EDA) to identify trends and correlations.
   - Supports model explainability using **SHAP** to determine feature contributions.

3. **Machine Learning Pipeline**
   - Trains multiple models: **Random Forest, Ridge Regression, Neural Network**.
   - Evaluates performance using **RMSE, MAE, R²** metrics.
   - Registers the best-performing model (**Random Forest**) in **Dagshub MLflow** for versioning and deployment.
   - Random Forest achieved:
     - MAE: 0.0096
     - R²: 0.9913
     - RMSE: 0.0861
   - Random Forest consistently performed best due to its ability to handle feature interactions and nonlinear pollutant patterns in AQI data.

4. **CI/CD Automation**
   - Feature pipeline runs hourly.
   - Training pipeline runs daily to retrain and update models automatically.
   - Implemented using **GitHub Actions**.

5. **Interactive Dashboard**
   - Built with **Streamlit**, **Altair**, and **PyDeck**.
   - Displays:
     - Real-time AQI (1–5 scale) with color-coded categories.
     - 3-day AQI forecasts.
     - 30-day demo trend.
     - SHAP-based top feature contributions.
     - Live pollutant composition for the last 7 days.
     - Map visualization across Karachi locations.
   - Provides health recommendations and AQI alerts.

---

## 🛠 Technology Stack

| **Component**              | **Technology / Tool**                        |
|----------------------------|---------------------------------------------|
| Data Collection            | OpenWeather API                             |
| Feature Store              | Hopsworks                                   |
| Model Registry             | Dagshub MLflow                              |
| Machine Learning Models    | Random Forest, Ridge, Neural Network        |
| ML Libraries               | scikit-learn, xgboost, TensorFlow           |
| Model Explainability       | SHAP                                        |
| Dashboard / Frontend       | Streamlit, Altair, PyDeck                   |
| CI/CD                      | GitHub Actions                              |
| Programming Language       | Python 3.11                                 |
| Dependency Management      | requirements.txt, requirements_pipeline.txt |

---

## ⚙️ Project Structure

```
pearls-aqi-predictor
├── .github/workflows          # CI/CD workflows (feature & training pipelines)
├── data
│   ├── raw                    # Raw AQI & pollutant CSV files
│   └── processed              # Processed feature datasets
├── notebooks                  # EDA & exploration notebooks
├── pipelines                  # Feature & training pipelines
├── src
│   ├── config                 # Configurations
│   ├── features               # Feature engineering scripts
│   ├── ingestion              # API data fetching scripts
│   └── inference              # Model prediction scripts
├── app                        # Streamlit dashboard
├── requirements.txt           # Dependencies for app
├── requirements_pipeline.txt  # Dependencies for pipelines
└── README.md
```

---

## ⚡ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AfifaSiddiquee/pearls-aqi-predictor.git
   cd pearls-aqi-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or for pipeline-specific dependencies:
   ```bash
   pip install -r requirements_pipeline.txt
   ```

3. **Set environment variables**
   ```bash
   export OPENWEATHER_API_KEY=<YOUR_OPENWEATHER_KEY>
   export HOPSWORKS_API_KEY=<YOUR_HOPSWORKS_KEY>
   export MLFLOW_TRACKING_USERNAME=<DAGSHUB_USERNAME>
   export MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>
   ```

4. **Run feature pipeline** (historical + live)
   ```bash
   python pipelines/feature_pipeline.py
   ```

5. **Run training pipeline**
   ```bash
   python pipelines/training_pipeline.py
   ```

6. **Launch dashboard**
   ```bash
   streamlit run app/app.py
   ```

---

## 📊 AQI Scale & Categories

| AQI Value | Category    | Color    |
|-----------|-------------|----------|
| 1         | Good        | Green    |
| 2         | Fair        | Yellow   |
| 3         | Moderate    | Orange   |
| 4         | Poor        | Red      |
| 5         | Hazardous   | Purple   |

---

## 🔍 Notes

- Uses OpenWeather API for live AQI & pollutant data (1–5 scale).
- Hopsworks ensures scalable feature storage.
- Dagshub MLflow provides versioned model registry for reproducibility.
- Random Forest outperforms Ridge & Neural Network due to its ensemble structure, handling pollutant feature interactions and non-linearities effectively.
- Dashboard includes real-time AQI, 3-day forecast, SHAP analysis, and interactive maps.

---

## 👩‍💻 Author

**Afifa Siddiquee**  
⚡ AI & Data Science Intern | Pearls AQI Predictor






