# AI-Based Solar Irradiance Forecasting and Climate Pattern Analytics Framework

Research-grade ML pipeline for solar irradiance prediction using tabular sensor data.

## Evaluation design (avoiding leakage)

- **Targets:** `irradiance_class` and regression targets use the **current** hour’s irradiance.
- **Features:** Irradiance history uses **only past hours** (lags, shifted rolling mean, clearness from lag + expanding past max). `DGR` uses an expanding same-day mean that excludes the current hour (no peeking at later afternoon when predicting morning).
- **Scaling:** `StandardScaler` / `MinMaxScaler` are **fit on the training split only**; the test set is transformed with those parameters.
- **Split:** By default `USE_CHRONOLOGICAL_SPLIT = True` in `scaling.py` — the **last 20% of time** is held out for test (stricter than random split for autocorrelated weather). Set `USE_CHRONOLOGICAL_SPLIT = False` for stratified random split (e.g. for class-balance experiments).

After changing preprocessing, re-run from feature engineering through downstream stages.

## Setup

```bash
pip install -r requirements.txt
```

### GPU training (recommended)

Install PyTorch with CUDA (pick the CUDA version that matches your driver from [pytorch.org](https://pytorch.org)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Deep learning modules (ANN, DNN, LSTM) use GPU via PyTorch CUDA.

At pipeline start you should see:

```text
[GPU] PyTorch training on: cuda:0 — NVIDIA GeForce ...
```

Strict CUDA is enabled by default (`SOLAR_REQUIRE_CUDA=1`). If CUDA is unavailable the pipeline fails fast.

Relax this behavior:
- Windows CMD: `set SOLAR_REQUIRE_CUDA=0`
- PowerShell: `$env:SOLAR_REQUIRE_CUDA=0`

Place `SolarIridescenceDataset.csv` in `data/raw/`, or generate synthetic demo data:

```bash
python scripts/generate_sample_data.py
```

## Run Full Pipeline

```bash
python main.py
```

## Run Individual Modules

```bash
python -m src.preprocessing.data_cleaning
python -m src.classification.classification_models
python -m src.deep_learning.lstm_model
```

## Removed modules

- CNN/image dataset pipeline removed.
- Spectral clustering removed.

## Outputs

| Directory | Contents |
|-----------|----------|
| `data/processed/` | Cleaned, engineered, clustered CSVs |
| `results/plots/` | All visualizations |
| `results/models/` | Saved sklearn / PyTorch models |
| `results/reports/` | Metrics CSVs, JSON, NLP samples |
