"""
Solar Predictive Analytics — full pipeline orchestrator.

Runs all preprocessing, ML, deep learning, explainability, and NLP stages.
Each stage is isolated with try/except so failures do not halt the pipeline.
"""

from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path
from typing import Callable

from src.utils.helpers import get_project_root


def run_stage(index: int, total: int, name: str, func: Callable) -> bool:
    print("──────────────────────────────────────")
    print(f" [{index}/{total}] {name}")
    print("──────────────────────────────────────")
    t0 = time.perf_counter()
    try:
        func()
        elapsed = time.perf_counter() - t0
        print(f"✓ Done  ({elapsed:.1f}s)\n")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"✗ Failed: {exc}  ({elapsed:.1f}s)")
        traceback.print_exc()
        print()
        return False


def run_cleaning() -> None:
    from src.preprocessing.data_cleaning import run_cleaning as fn

    fn()


def run_feature_engineering() -> None:
    from src.preprocessing.feature_engineering import run_feature_engineering as fn

    fn()


def run_scaling() -> None:
    from src.preprocessing.scaling import run_scaling as fn

    fn()


def run_classification() -> None:
    from src.classification.classification_models import run_classification as fn

    fn()


def run_kmeans() -> None:
    from src.clustering.kmeans import run_kmeans as fn

    fn()


def run_kmedoids() -> None:
    from src.clustering.kmedoids import run_kmedoids as fn

    fn()


def run_hierarchical() -> None:
    from src.clustering.hierarchical import run_hierarchical as fn

    fn()


def run_gmm() -> None:
    from src.clustering.gaussian_mixture import run_gmm as fn

    fn()


def run_ann() -> None:
    from src.deep_learning.ann_model import run_ann as fn

    fn()


def run_dnn() -> None:
    from src.deep_learning.dnn_model import run_dnn as fn

    fn()


def run_lstm() -> None:
    from src.deep_learning.lstm_model import run_lstm as fn

    fn()


def run_shap() -> None:
    from src.explainability.shap_analysis import run_shap as fn

    fn()


def run_model_comparison() -> None:
    from src.deep_learning.model_comparison import run_model_comparison as fn

    fn()


def _fmt(val: float | int | None, digits: int = 4) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def generate_summary() -> None:
    """Build project_summary.md from saved metrics and report files."""
    root = get_project_root()
    reports = root / "results" / "reports"
    plots_dir = root / "results" / "plots"
    models_dir = root / "results" / "models"

    raw_rows = cleaned_rows = "—"
    try:
        import pandas as pd

        raw_path = root / "data" / "raw" / "SolarIridescenceDataset.csv"
        cleaned_path = root / "data" / "processed" / "cleaned_solar_data.csv"
        if raw_path.exists():
            raw_rows = f"{len(pd.read_csv(raw_path)):,}"
        if cleaned_path.exists():
            cleaned_rows = f"{len(pd.read_csv(cleaned_path)):,}"
    except Exception:
        pass

    cls_lines = []
    cls_path = reports / "classification_results.csv"
    if cls_path.exists():
        import pandas as pd

        cls_df = pd.read_csv(cls_path)
        for _, row in cls_df.iterrows():
            cls_lines.append(
                f"| {row['model']} | {_fmt(row.get('accuracy'))} | {_fmt(row.get('f1'))} | "
                f"{_fmt(row.get('roc_auc'))} | {_fmt(row.get('train_time_s'), 1)}s |"
            )
    else:
        cls_lines.append("| — | — | — | — | — |")

    ann = _load_json(reports / "ann_metrics.json")
    dnn = _load_json(reports / "dnn_metrics.json")
    lstm = _load_json(reports / "lstm_metrics.json")
    bilstm = _load_json(reports / "bilstm_metrics.json")

    dl_rows = []
    for label, m, max_ep in [
        ("ANN", ann, 500),
        ("DNN", dnn, 600),
        ("LSTM (Attention)", lstm, 50),
        ("BiLSTM", bilstm, 50),
    ]:
        if m:
            dl_rows.append(
                f"| {label} | {m.get('epochs_run', '—')}/{max_ep} | {_fmt(m.get('rmse'))} | "
                f"{_fmt(m.get('mae'))} | {_fmt(m.get('r2'))} | {_fmt(m.get('mape'))}% | "
                f"{_fmt(m.get('train_time_s'), 1)}s |"
            )
    if not dl_rows:
        dl_rows.append("| — | — | — | — | — | — | — |")

    shap_feats = []
    shap_path = reports / "shap_top_features.json"
    if shap_path.exists():
        top = json.loads(shap_path.read_text(encoding="utf-8"))
        for i, item in enumerate(top[:3], start=1):
            shap_feats.append(f"{i}. {item['feature']}: {item['mean_abs_shap']:.4f}")
    if not shap_feats:
        shap_feats = ["1. —", "2. —", "3. —"]

    ari_val = "—"
    try:
        import pandas as pd
        from sklearn.metrics import adjusted_rand_score

        clustered = pd.read_csv(root / "data" / "processed" / "clustered_data.csv")
        eng = pd.read_csv(root / "data" / "processed" / "engineered_solar_data.csv")
        if "kmeans_cluster" in clustered.columns and "hierarchical_cluster" in eng.columns:
            merged = eng[["hierarchical_cluster"]].join(
                clustered[["kmeans_cluster"]], how="inner"
            ).dropna()
            if len(merged) > 0:
                ari_val = f"{adjusted_rand_score(merged['kmeans_cluster'], merged['hierarchical_cluster']):.4f}"
    except Exception:
        pass

    kmeans_k = "—"
    kmeans_sil = "—"
    try:
        import pandas as pd

        clustered = pd.read_csv(root / "data" / "processed" / "clustered_data.csv")
        if "kmeans_cluster" in clustered.columns:
            kmeans_k = str(clustered["kmeans_cluster"].nunique())
    except Exception:
        pass

    cpu_name = platform.processor() or platform.machine() or "Unknown"
    gpu_name = "Not available"
    torch_device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            torch_device = "cuda:0"
    except Exception:
        pass

    plot_count = len(list(plots_dir.glob("*"))) if plots_dir.exists() else 0
    model_count = len(list(models_dir.glob("*"))) if models_dir.exists() else 0

    md = f"""# Solar Predictive Analytics — Project Summary

**Dataset:** SolarIridescenceDataset.csv — {raw_rows} raw rows → {cleaned_rows} after cleaning  
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
{chr(10).join(cls_lines)}

---

## Clustering Results

| Algorithm | k | Silhouette Score | Notes |
|---|---|---|---|
| KMeans | {kmeans_k} | {kmeans_sil} | Elbow method |
| KMedoids | 3 | — | PAM algorithm |
| Hierarchical | 3 | — | Ward linkage, ARI vs KMeans: {ari_val} |
| Gaussian Mixture | 3 | — | BIC-selected components |

---

## Deep Learning Results

| Model | Epochs Run | RMSE | MAE | R² | MAPE | Train Time |
|---|---|---|---|---|---|---|
{chr(10).join(dl_rows)}

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
{chr(10).join(shap_feats)}

---

## Hardware

- CPU: {cpu_name}
- GPU: {gpu_name} (PyTorch CUDA)
- PyTorch device: {torch_device}
- sklearn/clustering: CPU

---

## Output Files

- `results/plots/`   — {plot_count} plot files
- `results/models/`  — {model_count} saved model files  
- `results/reports/` — classification_results.csv, deep_learning_comparison.csv, sample_nlp_reports.txt, project_summary.md
"""

    out_path = reports / "project_summary.md"
    reports.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print("  Saved: project_summary.md")


def main() -> None:
    from src.utils.device import log_device_once, require_cuda
    from src.utils.helpers import ensure_dirs

    ensure_dirs()
    log_device_once()
    require_cuda()

    stages = [
        ("Data Cleaning", run_cleaning),
        ("Feature Engineering", run_feature_engineering),
        ("Scaling + Split", run_scaling),
        ("Classification Models", run_classification),
        ("KMeans Clustering", run_kmeans),
        ("KMedoids Clustering", run_kmedoids),
        ("Hierarchical Clustering", run_hierarchical),
        ("Gaussian Mixture", run_gmm),
        ("ANN Model", run_ann),
        ("DNN Model", run_dnn),
        ("LSTM Model", run_lstm),
        ("SHAP Explainability", run_shap),
        ("Model Comparison Table", run_model_comparison),
    ]

    total = len(stages)
    ok = 0
    for i, (name, fn) in enumerate(stages, start=1):
        if run_stage(i, total, name, fn):
            ok += 1

    if ok == total:
        try:
            generate_summary()
        except Exception as exc:
            print(f"  [warn] Could not generate project summary: {exc}")

    print("══════════════════════════════════════")
    print(f"  {ok}/{total} stages completed successfully")
    print("  Results → results/")
    print("══════════════════════════════════════")


if __name__ == "__main__":
    main()
