# Solar Predictive Analytics — Project Summary

**Dataset:** SolarIridescenceDataset.csv — 78,138 raw rows → 45,584 after cleaning  
**Features:** hour_sin, hour_cos, month_sin, month_cos, Temperature, lag_1h_irradiance, rolling_mean_3h, clearness_index  
**Target:** Irradiance (W/m²) — regression; irradiance_class (Low/Medium/High) — classification  
**Train/Test Split:** 80/20 chronological (no leakage)

---

## Project Structure

```
Solar_Predictive_Analytics_Project/
├── data/           raw → processed → final
├── src/
│   ├── preprocessing/    data_cleaning, feature_engineering, scaling
│   ├── classification/   8 classifiers
│   ├── clustering/       kmeans, kmedoids, hierarchical, gaussian_mixture
│   ├── deep_learning/    ann, dnn, lstm (+ bilstm)
│   ├── explainability/   shap_analysis
│   ├── evaluation/       metrics, plots
│   └── nlp/              solar_report_generator
├── results/        plots/, models/, reports/
└── main.py
```

---

## Classification Results

| Model | Accuracy | F1 | ROC-AUC | Train Time |
|---|---|---|---|---|
| LogisticRegression | 0.9300 | 0.9300 | 0.9870 | 0.3s |
| NaiveBayes | 0.7683 | 0.7659 | 0.9283 | 0.0s |
| KNN | 0.9300 | 0.9301 | 0.9795 | 0.1s |
| SVM | 0.9387 | 0.9388 | 0.9900 | 34.5s |
| DecisionTree | 0.9286 | 0.9287 | 0.9795 | 0.1s |
| RandomForest | 0.9344 | 0.9345 | 0.9894 | 5.9s |
| GradientBoosting | 0.9392 | 0.9393 | 0.9904 | 14.1s |
| XGBoost | 0.9397 | 0.9397 | 0.9911 | 1.1s |

---

## Clustering Results

| Algorithm | k | Silhouette Score | Notes |
|---|---|---|---|
| KMeans | 2 | — | Elbow method |
| KMedoids | 3 | — | PAM algorithm |
| Hierarchical | 3 | — | Ward linkage, ARI vs KMeans: — |
| Gaussian Mixture | 3 | — | BIC-selected components |

---

## Deep Learning Results

| Model | Epochs Run | RMSE | MAE | R² | MAPE | Train Time |
|---|---|---|---|---|---|---|
| ANN | 41/500 | 43.6304 | 27.6975 | 0.9725 | 18.1270% | 63.4s |
| DNN | 137/600 | 42.0586 | 27.8653 | 0.9745 | 23.1833% | 395.2s |
| LSTM (Attention) | 50/50 | 75.0021 | 46.9439 | 0.9188 | 28.0758% | 39.9s |
| BiLSTM | 50/50 | 56.3806 | 34.6096 | 0.9541 | 19.9771% | 70.4s |

### Architecture Details

**ANN:** Input(8) → Linear(128) → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(1)  
Optimizer: Adam lr=1e-3 | Scheduler: CosineAnnealingLR | Early stop patience: 40

**DNN:** Input(8) → Linear(256) → BN → ReLU → Linear(128) → BN → ReLU → Linear(64) → BN → ReLU → Linear(32) → Linear(1)  
Optimizer: Adam lr=5e-4 | Scheduler: ReduceLROnPlateau(patience=10, factor=0.5) | Early stop patience: 50

**LSTM:** Sequence length=24 | Hidden=128 | Layers=2 | Attention mechanism (learned temporal weights)  
Optimizer: Adam lr=5e-4 | Scheduler: CosineAnnealingLR | Epochs: 50

**BiLSTM:** Same as LSTM but bidirectional (processes sequence forward + backward)

---

## Explainability (SHAP)

Model explained: RandomForest classifier  
Top features by mean |SHAP|:
1. hour_sin: 0.0662
2. hour_cos: 0.0597
3. month_sin: 0.0522

---

## Hardware

- CPU: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch CUDA)
- PyTorch device: cuda:0
- sklearn/clustering: CPU

---

## Output Files

- `results/plots/`   — 68 plot files
- `results/models/`  — 8 saved model files  
- `results/reports/` — classification_results.csv, deep_learning_comparison.csv, sample_nlp_reports.txt, project_summary.md
