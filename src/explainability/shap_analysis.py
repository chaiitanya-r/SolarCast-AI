"""
SHAP explainability for the Random Forest irradiance classifier.

Generates summary, importance bar, and waterfall plots.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.preprocessing.scaling import load_split_data
from src.utils.helpers import ensure_dirs, get_project_root, load_model, save_plot, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")


@timer
def run_shap() -> None:
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    model = load_model("random_forest_classifier.joblib")
    data = load_split_data()
    X_test = data["X_test"]
    feature_names = data["feature_names"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])

    if isinstance(shap_values, list):
        sv = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        sv = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    top_features = [(feature_names[i], float(mean_abs[i])) for i in order[:5]]
    reports_dir = root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    shap_json_path = reports_dir / "shap_top_features.json"
    shap_json_path.write_text(
        json.dumps([{"feature": f, "mean_abs_shap": v} for f, v in top_features], indent=2),
        encoding="utf-8",
    )
    print("  Saved: shap_top_features.json")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Waterfall plots: one per sample, for the predicted class ---
    # shap_values from TreeExplainer for multiclass is list of arrays [class0_array, class1_array, class2_array]
    # each array is shape (n_test_samples, n_features)
    n_classes = len(shap_values) if isinstance(shap_values, list) else shap_values.shape[-1]
    feature_names = list(X_test.columns) if hasattr(X_test, "columns") else [f"f{i}" for i in range(X_test.shape[1])]

    X_test_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    X_sample = X_test[:3] if hasattr(X_test, "iloc") else X_test[:3]
    X_sample_arr = X_sample.values if hasattr(X_sample, "values") else X_sample

    clf = model
    for i in range(3):
        pred_class = int(clf.predict(X_sample_arr[i : i + 1])[0])
        pred_class = min(pred_class, n_classes - 1)

        if isinstance(shap_values, list):
            sv_for_sample = shap_values[pred_class][i]
            base_val = (
                explainer.expected_value[pred_class]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value
            )
        else:
            sv_for_sample = shap_values[i, :, pred_class]
            base_val = (
                explainer.expected_value[pred_class]
                if hasattr(explainer.expected_value, "__len__")
                else explainer.expected_value
            )

        explanation = shap.Explanation(
            values=sv_for_sample,
            base_values=float(base_val),
            data=X_sample_arr[i],
            feature_names=feature_names,
        )

        fig_w, ax_w = plt.subplots(figsize=(10, 5))
        plt.sca(ax_w)
        shap.plots.waterfall(explanation, show=False, max_display=8)
        plt.title(f"SHAP Waterfall — Sample {i+1} (Predicted class {pred_class})")
        plt.tight_layout()
        save_plot(fig_w, f"shap_waterfall_{i+1}.png")
        plt.close()
        print(f"  Saved: shap_waterfall_{i+1}.png")

    sv_class2 = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]
    top_dep_features = [feature_names[i] for i in np.argsort(np.abs(sv_class2).mean(axis=0))[::-1][:3]]
    X_test_df = pd.DataFrame(X_test_arr[:200], columns=feature_names)

    for feat in top_dep_features:
        try:
            plt.style.use("seaborn-v0_8")
            fig_dep, ax_dep = plt.subplots(figsize=(8, 5))
            feat_idx = feature_names.index(feat)
            ax_dep.scatter(
                X_test_df[feat],
                sv_class2[:200, feat_idx],
                alpha=0.4,
                s=10,
                c=sv_class2[:200, feat_idx],
                cmap="coolwarm",
            )
            ax_dep.axhline(0, color="red", linewidth=0.8, linestyle="--")
            ax_dep.set_xlabel(feat)
            ax_dep.set_ylabel(f"SHAP value for '{feat}' (class: High)")
            ax_dep.set_title(f"SHAP Dependence Plot — {feat}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_plot(fig_dep, f"shap_dependence_{feat}.png")
            plt.close(fig_dep)
            print(f"  Saved: shap_dependence_{feat}.png")
        except Exception as e:
            print(f"  [warn] Dependence plot failed for {feat}: {e}")

    try:
        plt.style.use("seaborn-v0_8")
        sample_idx = 0
        pred_class = int(clf.predict(X_test_arr[sample_idx : sample_idx + 1])[0])
        sv_single = (
            shap_values[pred_class][sample_idx]
            if isinstance(shap_values, list)
            else shap_values[sample_idx, :, pred_class]
        )
        base_val = (
            explainer.expected_value[pred_class]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )

        fig_fp = plt.figure(figsize=(14, 3))
        shap.force_plot(
            float(base_val),
            sv_single,
            X_test_arr[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        plt.title("SHAP Force Plot — Sample 1")
        plt.tight_layout()
        save_plot(fig_fp, "shap_force_plot_sample1.png")
        plt.close(fig_fp)
        print("  Saved: shap_force_plot_sample1.png")
    except Exception as e:
        print(f"  [warn] Force plot failed: {e}")


def main() -> None:
    run_shap()


if __name__ == "__main__":
    main()
