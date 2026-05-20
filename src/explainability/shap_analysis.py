"""
SHAP explainability for the Random Forest irradiance classifier.

Generates summary, importance bar, and waterfall plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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
    top_features = [(feature_names[i], mean_abs[i]) for i in order[:5]]
    print("\n=== Top 5 Features by Mean |SHAP| ===")
    for feat, val in top_features:
        print(f"  {feat}: {val:.4f}")

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
        print(f"  Saved waterfall plot for sample {i+1}")

    print(f"SHAP plots saved to {plots_dir}")


def main() -> None:
    run_shap()


if __name__ == "__main__":
    main()
