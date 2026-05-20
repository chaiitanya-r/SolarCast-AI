"""
Train and compare multiple classifiers on irradiance_class labels.

Outputs: per-model confusion matrices, overlaid ROC plot,
         classification_results.csv, and saved Random Forest for SHAP.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]

from src.evaluation.metrics import classification_report_dict
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
from src.preprocessing.scaling import load_split_data
from src.utils.helpers import ensure_dirs, get_project_root, save_model, save_plot, timer

np.random.seed(42)

RUN_HYPERPARAMETER_TUNING = False
CLASS_LABELS = ["Low", "Medium", "High"]


def _get_models() -> dict:
    xgb_model = {}
    if XGBClassifier is not None:
        xgb_model = {
            "XGBoost": XGBClassifier(
                n_estimators=100,
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
                device="cuda",
            )
        }
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        **xgb_model,
    }


@timer
def run_classification() -> pd.DataFrame:
    """Train all classifiers, evaluate, and save results."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"
    reports_dir = root / "results" / "reports"

    data = load_split_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_class_train"], data["y_class_test"]
    models = _get_models()
    if RUN_HYPERPARAMETER_TUNING:
        print("Running GridSearchCV for RandomForest...")
        param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        models["RandomForest"] = gs.best_estimator_
        print(f"Best RF params: {gs.best_params_}")

    results = []
    fitted_models = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = classification_report_dict(y_test, y_pred, y_proba)
        metrics["model"] = name
        metrics["train_time_s"] = train_time
        results.append(metrics)
        fitted_models[name] = model

        plot_confusion_matrix(
            y_test,
            y_pred,
            CLASS_LABELS,
            f"Confusion Matrix — {name}",
            str(plots_dir / f"cm_{name}.png"),
        )
        print(
            f"  {name}  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
            f"auc={metrics.get('roc_auc', 0):.4f}  ({train_time:.1f}s)"
        )

    plot_roc_curves(
        fitted_models,
        X_test,
        y_test,
        str(plots_dir / "roc_all_classifiers.png"),
    )

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"]
    ]
    results_path = reports_dir / "classification_results.csv"
    results_df.to_csv(results_path, index=False)

    save_model(fitted_models["RandomForest"], "random_forest_classifier.joblib")

    try:
        plt.style.use("seaborn-v0_8")
        metrics_cols = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(results_df))
        width = 0.2
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, ax = plt.subplots(figsize=(16, 7))
        for i, metric in enumerate(metrics_cols):
            bars = ax.bar(
                x + (i - 1.5) * width,
                results_df[metric].values,
                width=width,
                label=metric.capitalize(),
                color=colors[i],
            )
            ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title("Classification Metrics Comparison Across Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["model"], rotation=30, ha="right")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, "classification_metrics_comparison.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate grouped metrics chart: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        roc_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(roc_df["model"], roc_df["roc_auc"], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(roc_df))))
        ax.invert_yaxis()
        for container in [bars]:
            ax.bar_label(container, fmt="%.4f", padding=3)
        ax.set_title("ROC-AUC Ranking Across Classifiers")
        ax.set_xlabel("ROC-AUC")
        ax.set_ylabel("Model")
        plt.tight_layout()
        save_plot(fig, "classification_roc_auc_ranking.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate ROC-AUC ranking chart: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        time_df = results_df.sort_values("train_time_s", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(time_df["model"], time_df["train_time_s"], color="tab:purple")
        ax.set_xscale("log")
        for container in [bars]:
            ax.bar_label(container, fmt="%.4f", padding=3)
        ax.set_title("Classifier Training Time Comparison (Log Scale)")
        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel("Model")
        plt.tight_layout()
        save_plot(fig, "classification_training_time.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate training time chart: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes_flat = axes.ravel()
        for idx, (name, model) in enumerate(fitted_models.items()):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks(np.arange(len(CLASS_LABELS)))
            ax.set_yticks(np.arange(len(CLASS_LABELS)))
            ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right")
            ax.set_yticklabels(CLASS_LABELS)
        for j in range(len(fitted_models), len(axes_flat)):
            axes_flat[j].axis("off")
        fig.colorbar(im, ax=axes_flat, fraction=0.02, pad=0.01)
        fig.suptitle("All Model Confusion Matrices")
        plt.tight_layout()
        save_plot(fig, "all_confusion_matrices.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate confusion matrix grid: {exc}")

    print("\n=== Classification Comparison ===")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("  Saved: classification_results.csv")
    return results_df


def main() -> pd.DataFrame:
    return run_classification()


if __name__ == "__main__":
    main()
