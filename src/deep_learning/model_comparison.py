"""
Aggregate deep-learning model metrics into a comparison table and bar chart.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dirs, get_project_root, save_plot, timer

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
            print(f"  [warn] Skipping {name}: {fname} not found")
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
        try:
            plt.style.use("seaborn-v0_8")
            rmse_df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(rmse_df["Model"], rmse_df["RMSE"], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(rmse_df))))
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_title("RMSE Comparison (Lower is Better)")
            ax.set_xlabel("RMSE")
            ax.set_ylabel("Model")
            plt.tight_layout()
            save_plot(fig, "model_comparison_rmse.png")
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] Could not generate RMSE chart: {exc}")

        try:
            plt.style.use("seaborn-v0_8")
            r2_df = df.sort_values("R²", ascending=False).reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(r2_df["Model"], r2_df["R²"], color=plt.cm.tab10(np.linspace(0, 1, len(r2_df))))
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_title("R² Comparison (Higher is Better)")
            ax.set_xlabel("R²")
            ax.set_ylabel("Model")
            plt.tight_layout()
            save_plot(fig, "model_comparison_r2.png")
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] Could not generate R² chart: {exc}")

        try:
            plt.style.use("seaborn-v0_8")
            radar_df = df.copy()
            radar_metrics = ["RMSE", "MAE", "R²", "MAPE", "Training Time (s)"]
            norm_df = radar_df.copy()
            for col in ["RMSE", "MAE", "MAPE", "Training Time (s)"]:
                cmin, cmax = norm_df[col].min(), norm_df[col].max()
                denom = (cmax - cmin) if (cmax - cmin) != 0 else 1
                norm_df[col] = 1 - ((norm_df[col] - cmin) / denom)
            cmin, cmax = norm_df["R²"].min(), norm_df["R²"].max()
            denom = (cmax - cmin) if (cmax - cmin) != 0 else 1
            norm_df["R²"] = (norm_df["R²"] - cmin) / denom

            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, polar=True)
            palette = plt.cm.tab10(np.linspace(0, 1, len(norm_df)))
            for i, (_, row) in enumerate(norm_df.iterrows()):
                values = row[radar_metrics].values.astype(float).tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2, label=row["Model"], color=palette[i])
                ax.fill(angles, values, alpha=0.08, color=palette[i])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(["RMSE (inv)", "MAE (inv)", "R²", "MAPE (inv)", "Training Time (inv)"])
            ax.set_title("Model Comparison Radar (Normalized 0-1)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            plt.tight_layout()
            save_plot(fig, "model_comparison_radar.png")
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] Could not generate radar chart: {exc}")

        try:
            plt.style.use("seaborn-v0_8")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(df["Training Time (s)"], df["RMSE"], color="tab:blue", s=80, alpha=0.8)
            for _, row in df.iterrows():
                ax.annotate(str(row["Model"]), (row["Training Time (s)"], row["RMSE"]), xytext=(5, 5), textcoords="offset points")
            ax.set_title("RMSE vs Training Time Tradeoff")
            ax.set_xlabel("Training Time (s)")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_plot(fig, "model_comparison_tradeoff.png")
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] Could not generate tradeoff scatter: {exc}")

    print("  Saved: deep_learning_comparison.csv")
    return df

def main() -> pd.DataFrame:
    return run_model_comparison()

if __name__ == "__main__":
    main()
