"""
Train and compare multiple classifiers on irradiance_class labels.

Outputs: per-model confusion matrices, overlaid ROC plot,
         classification_results.csv, and saved Random Forest for SHAP.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
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

from src.evaluation.metrics import classification_report_dict, format_metrics_table
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
from src.preprocessing.scaling import load_split_data
from src.utils.helpers import ensure_dirs, get_project_root, save_model, timer

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
    print(
        "GPU note: sklearn classifiers run on CPU. "
        "Only XGBoost is configured to use CUDA when available."
    )

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
        print(f"\nTraining {name}...")
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
        print(format_metrics_table(metrics, title=name))

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

    print(
        "\nNote: irradiance-derived features use past observations only; "
        "scalers are fit on the training split only (see scaling.py)."
    )
    print("\n=== Classification Comparison ===")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved results to {results_path}")
    return results_df


def main() -> pd.DataFrame:
    return run_classification()


if __name__ == "__main__":
    main()
