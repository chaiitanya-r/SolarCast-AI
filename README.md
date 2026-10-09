# ☀️ SolarCast AI — Solar Irradiance Forecasting & Energy Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![PyTest](https://img.shields.io/badge/PyTest-Passing%20(4%2F4)-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Machine Learning, Deep Learning, and time-series analytics platform for **Solar Irradiance Forecasting ($W/m^2$)**, photovoltaic power output estimation, and automated meteorological dispatch. Features an interactive operations dashboard, cached in-memory model serving, and a leak-free temporal data engineering pipeline.

---

## 📌 Problem Statement

Solar energy is one of the world's fastest-growing renewable power sources, but its power generation is inherently volatile due to cloud cover, diurnal sun geometry, and sudden weather fluctuations.

- **Grid Instability:** Solar farms feeding directly into the electrical grid can cause power surges or brownouts if sunlight drops unexpectedly.
- **Battery Dispatch Inefficiency:** Facility operators struggle to determine when to store solar energy versus when to discharge battery reserves.
- **Financial Penalties:** Energy market bidding requires accurate forward commitments; deviations from promised generation incur heavy grid penalties.

**SolarCast AI** solves this by providing continuous, high-accuracy solar irradiance predictions (Global Horizontal Irradiance in $W/m^2$) and weather regime classification using deep neural networks and gradient boosting, paired with an intuitive operational dashboard for plant engineers.

---

## ⚙️ How It Works

```text
  ┌───────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
  │  IoT SCADA / Weather  │ ────► │ Data Validation Layer  │ ────► │  Feature Engineering   │
  │     Sensor Stream     │       │ (Bounds & Null Checks) │       │ (Cyclical & Lag Trans) │
  └───────────────────────┘       └────────────────────────┘       └────────────────────────┘
                                                                               │
                                                                               ▼
  ┌───────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
  │  Streamlit Dashboard  │ ◄──── │   Fast Model Serving   │ ◄──── │  PyTorch DNN & XGBoost │
  │    & Executive NLP    │       │ (In-Memory Weight Caching│     │   (Inference Engine)   │
  └───────────────────────┘       └────────────────────────┘       └────────────────────────┘
```

The system operates across four core architectural stages:

1. **Leak-Free Data Pipeline (`src/preprocessing`):**
   - **Chronological Split (80/20):** Held-out test set strictly reserves the final 20% of time to prevent future data leakage common in weather time-series.
   - **Training-Only Scalers:** `StandardScaler` and `MinMaxScaler` fit solely on training data.
   - **Historical-Only Lags:** Features use only past hours (`lag_1h`, `rolling_mean_3h`, trigonometric `hour_sin`/`month_cos`, clearness index) relative to each timestamp.
2. **Deep Learning & Classical ML Inference (`src/deep_learning`, `src/classification`):**
   - Predicts continuous Global Horizontal Irradiance ($W/m^2$) using a PyTorch Deep Neural Network (DNN) achieving **97.45% $R^2$** and **42.06 RMSE**.
   - Categorizes solar regimes into Low ($<200$), Medium ($200–600$), and High ($>600\ W/m^2$) with XGBoost and Random Forest achieving **93.97% accuracy**.
3. **High-Performance In-Memory Serving Engine (`app.py`):**
   - Pre-loads trained PyTorch and scikit-learn models once into memory on startup via resource caching, delivering **sub-5ms inference latency**.
   - Enforces physical nighttime clamping (hours $19:00–05:00$ automatically clamp to $0\ W/m^2$).
4. **Natural Language Dispatch Reports (`src/nlp`):**
   - Translates raw sensor metrics into automated, plain-English operational guidance for solar operators.

---

## ✨ Web Application Features

The interactive web dashboard (`app.py`) provides:

- **⚡ Real-Time Forecaster (Nowcast Simulator):** Move sliders for Hour, Month, Temperature, and Historical Lags to see immediate changes in predicted sunlight ($W/m^2$), regime badge, and rooftop power output ($kW$).
- **🚀 1-Click Weather Scenarios:** Instant preset buttons (*Sunny Summer Noon*, *Mild Autumn Afternoon*, *Overcast Rainy Morning*, *Midnight*) for quick testing.
- **☀️ Diurnal Solar Arc Trajectory:** 24-hour visual curve showing how sunlight changes throughout the day based on selected atmospheric inputs.
- **📈 24-Hour Horizon & Batch CSV Processing:** Load sample 24-hour telemetry or upload custom SCADA CSVs to visualize actual vs. predicted curves, inspect Mean Absolute Error, and download processed forecasts as CSV.
- **🔬 Model Benchmarks Leaderboard:** Empirical accuracy, error rates, and inference speeds across all 12 trained models.
- **🏗️ Software Architecture & API Contract:** Full microservice architecture diagrams and sample JSON REST API schemas.

---

## 🚀 How to Run on Your PC

Follow these step-by-step instructions to run the application on any computer (Windows, macOS, or Linux):

### 1. Clone the Repository
```bash
git clone https://github.com/chaiitanya-r/Solar_Irradiance_Prediction.git
cd Solar_Irradiance_Prediction
```

### 2. Set Up a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Launch the Web Dashboard

#### Recommended (Direct Command):
```bash
# Windows
.\.venv\Scripts\python.exe -m streamlit run app.py

# macOS / Linux
python -m streamlit run app.py
```

#### Windows 1-Click Shortcuts:
- **In PowerShell:** `.\run_dashboard.ps1`
- **In Command Prompt:** `run_dashboard.bat`

Once started, open your browser and go to:
👉 **[http://localhost:8501](http://localhost:8501)**

To stop the dashboard, press **`Ctrl + C`** in your terminal.

---

### 5. Run Automated Unit Tests

Validate telemetry preprocessing, cyclical trigonometric bounds, and physical nighttime safety:

```bash
pytest tests/test_pipeline.py -v
```

Expected output:
```text
tests/test_pipeline.py::test_compute_features_shape_and_types PASSED     [ 25%]
tests/test_pipeline.py::test_cyclical_encoding_bounds PASSED             [ 50%]
tests/test_pipeline.py::test_nighttime_clamping_safety PASSED            [ 75%]
tests/test_pipeline.py::test_noon_peak_positive_generation PASSED        [100%]
======================== 4 passed in 4.74s ========================
```

---

### 6. Run via Docker (Optional)

Run the entire application in an isolated container without installing Python or dependencies locally:

```bash
# Build the Docker image
docker build -t solarcast-ai .

# Run the container
docker run -p 8501:8501 solarcast-ai
```
Visit **[http://localhost:8501](http://localhost:8501)**.

---

### 7. Run the Full Model Training Pipeline (Optional)

To retrain all ML/DL models from scratch across all 13 pipeline stages:

```bash
python main.py
```

---

## 📊 Model Benchmark Results

Evaluated on 45,000+ hourly sensor observations (chronologically held-out test set):

### Continuous Solar Regression ($W/m^2$)
| Model Engine | Architecture | RMSE | MAE | $R^2$ Score | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch DNN 🥇** | **Input(8) → 256 → 128 → 64 → 32 → 1** | **42.06** | **27.87** | **97.45%** | **1.2 ms** |
| PyTorch ANN 🥈 | Input(8) → 128 → 64 → 1 | 43.63 | 27.70 | 97.25% | 0.8 ms |
| Bidirectional LSTM 🥉 | 2-Layer BiLSTM (128 units) + Dense | 56.38 | 34.61 | 95.41% | 4.5 ms |
| Standard LSTM | 2-Layer LSTM (128 units) + Dense | 75.00 | 46.94 | 91.88% | 3.8 ms |

### Solar Regime Classification (Low / Medium / High)
| Classifier | Accuracy | Macro F1 | ROC-AUC | Training Speed |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost 🥇** | **93.97%** | **0.9397** | **0.9911** | **Fast (1.1s)** |
| Gradient Boosting | 93.92% | 0.9393 | 0.9904 | Moderate (14.1s) |
| Support Vector Machine (SVM) | 93.87% | 0.9388 | 0.9900 | Slow (34.5s) |
| Random Forest | 93.44% | 0.9345 | 0.9894 | Fast (5.9s) |
| Logistic Regression | 93.00% | 0.9300 | 0.9870 | Instant (0.3s) |

---

## 📁 Repository Structure

```text
solar-irradiance-prediction/
├── app.py                     # Streamlit Interactive Web Application
├── Dockerfile                 # Container deployment specification
├── .dockerignore              # Exclusions for Docker image builds
├── run_dashboard.bat          # 1-Click Windows Command Prompt launcher
├── run_dashboard.ps1          # 1-Click Windows PowerShell launcher
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Solar gold aesthetic & server configuration
├── tests/
│   └── test_pipeline.py       # PyTest automated unit test suite
├── data/
│   ├── raw/                   # Raw sensor datasets
│   └── processed/             # Cleaned & scaled telemetry arrays
├── results/
│   ├── models/                # Trained PyTorch (.pth) & scikit-learn (.joblib) weights
│   ├── plots/                 # Visualizations (SHAP, loss curves, confusion matrices)
│   └── reports/               # Metrics JSON, CSV results & summaries
├── src/
│   ├── classification/        # ML Classifiers (XGBoost, RF, SVM, etc.)
│   ├── clustering/            # Unsupervised algorithms (K-Means, GMM, Hierarchical)
│   ├── deep_learning/         # PyTorch architectures (DNN, ANN, LSTM, BiLSTM)
│   ├── evaluation/            # Metrics calculators & plotting utilities
│   ├── explainability/        # SHAP feature importance & attribution
│   ├── nlp/                   # Automated rule-based report generator
│   ├── preprocessing/         # Cleaning, cyclical feature engineering, scaling
│   └── utils/                 # Path helpers & GPU/device managers
└── main.py                    # 13-stage batch training pipeline orchestrator
```
