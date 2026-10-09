"""
Pure evaluation metric utilities (no side effects).

Inputs: ground-truth and prediction arrays.
Outputs: dictionaries of scalar metrics and formatted tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Return accuracy, weighted precision/recall/F1, and OvR ROC-AUC."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, MSE, R², nRMSE, and MAPE."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / y_range) if y_range > 1e-8 else float("nan")
    mask = np.abs(y_true) > 1e-8
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "nrmse": nrmse,
        "mape": mape,
    }

def format_metrics_table(metrics_dict: dict[str, Any], title: str = "Metrics") -> str:
    """Pretty-print a metrics dictionary as an aligned text table."""
    lines = [title, "-" * 40]
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            lines.append(f"  {key:20s}: {value:12.4f}")
        else:
            lines.append(f"  {key:20s}: {value}")
    lines.append("-" * 40)
    return "\n".join(lines)
