"""
Reusable matplotlib plotting utilities for evaluation and reporting.

All functions save figures via helpers.save_plot or an explicit save_path.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

plt.style.use("seaborn-v0_8")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models_dict: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
    n_classes: int = 3,
) -> None:
    """Overlay one-vs-rest ROC curves for multiple fitted classifiers."""
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            continue
        y_score = model.predict_proba(X_test)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=f"{name} (class {i}, AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classifiers (OvR)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    title: str,
    save_path: str,
) -> None:
    """Plot training vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str,
    max_points: int = 500,
) -> None:
    """Line plot of actual vs predicted values."""
    n = min(len(y_true), max_points)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_true[:n], label="Actual", alpha=0.8)
    ax.plot(y_pred[:n], label="Predicted", alpha=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Irradiance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Scatter plot of residuals vs predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    save_path: str,
) -> None:
    """Horizontal bar chart of feature importances."""
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
