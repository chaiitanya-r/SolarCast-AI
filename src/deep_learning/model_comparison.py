"""
Aggregate deep-learning model metrics into a comparison table and bar chart.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dirs, get_project_root, timer

plt.style.use("seaborn-v0_8")

METRIC_FILES = [
    ("ANN", "ann_metrics.json"),
    ("DNN", "dnn_metrics.json"),
    ("LSTM", "lstm_metrics.json"),
]


@timer
def run_model_comparison() -> pd.DataFrame:
    ensure_dirs()
    root = get_project_root()
    reports_dir = root / "results" / "reports"
    plots_dir = root / "results" / "plots"

    rows = []
    for name, fname in METRIC_FILES:
        path = reports_dir / fname
        if not path.exists():
            print(f"Skipping {name}: {fname} not found")
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "Model": data.get("model", name),
                "RMSE": data.get("rmse", np.nan),
                "MAE": data.get("mae", np.nan),
                "R²": data.get("r2", np.nan),
                "MAPE": data.get("mape", np.nan),
                "Training Time (s)": data.get("train_time_s", np.nan),
            }
        )

    bilstm_path = reports_dir / "bilstm_metrics.json"
    if bilstm_path.exists():
        with open(bilstm_path, encoding="utf-8") as f:
            bilstm_m = json.load(f)
        rows.append(
            {
                "Model": "BiLSTM",
                "RMSE": bilstm_m.get("rmse", float("nan")),
                "MAE": bilstm_m.get("mae", float("nan")),
                "R²": bilstm_m.get("r2", float("nan")),
                "MAPE": bilstm_m.get("mape", float("nan")),
                "Training Time (s)": bilstm_m.get("train_time_s", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    out_path = reports_dir / "deep_learning_comparison.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Deep Learning Model Comparison ===")
    print(df.to_string(index=False, float_format="%.4f"))

    if len(df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        x = np.arange(len(df))
        axes[0].bar(x, df["RMSE"], color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df["Model"], rotation=30, ha="right")
        axes[0].set_title("RMSE Comparison")
        axes[0].set_ylabel("RMSE")

        axes[1].bar(x, df["R²"], color="seagreen")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df["Model"], rotation=30, ha="right")
        axes[1].set_title("R² Comparison")
        axes[1].set_ylabel("R²")

        plt.suptitle("Deep Learning Models — Performance Comparison")
        plt.tight_layout()
        fig.savefig(plots_dir / "model_comparison.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved comparison to {out_path}")
    return df


def main() -> pd.DataFrame:
    return run_model_comparison()


if __name__ == "__main__":
    main()
