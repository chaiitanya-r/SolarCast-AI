"""
PyTorch Artificial Neural Network for irradiance regression.

Architecture: Input(8) → 128 → ReLU → Dropout(0.2) → 64 → ReLU → 1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.deep_learning._training_utils import (
    evaluate_and_save,
    set_seeds,
    train_regressor,
)
from src.preprocessing.scaling import inverse_transform_target, load_split_data
from src.utils.device import require_cuda
from src.utils.helpers import save_plot, timer

set_seeds(42)
ANN_EPOCHS = 500
ANN_PATIENCE = 40
ANN_LR = 1e-3


class ANN(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@timer
def run_ann() -> dict:
    require_cuda()
    data = load_split_data()
    X, y = data["X_train"], data["y_train"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = ANN(input_dim=X.shape[1])
    train_losses, val_losses, elapsed = train_regressor(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=ANN_EPOCHS,
        lr=ANN_LR,
        patience=ANN_PATIENCE,
        model_label="ANN",
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ANN_EPOCHS, eta_min=1e-6
        ),
    )
    metrics = evaluate_and_save(
        model,
        "ANN",
        train_time=elapsed,
        loss_plot="ann_loss.png",
        pred_plot="ann_predictions.png",
        model_file="ann_model.pth",
        train_losses=train_losses,
        val_losses=val_losses,
    )
    try:
        plt.style.use("seaborn-v0_8")
        model.eval()
        X_test = data["X_test"]
        y_test = data["y_test"]
        with torch.no_grad():
            y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy().ravel()
        y_true = inverse_transform_target(y_test)
        y_pred = inverse_transform_target(y_pred_scaled)
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, s=12, color="tab:blue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("ANN Residual Plot")
        ax.set_xlabel("Predicted Irradiance")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "ann_residuals.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="tab:purple")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("ANN Residual Error Distribution")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        save_plot(fig, "ann_error_distribution.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, s=12, color="tab:green")
        line_min, line_max = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
        ax.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1.5)
        ax.set_title("ANN Actual vs Predicted Scatter")
        ax.set_xlabel("Actual Irradiance")
        ax.set_ylabel("Predicted Irradiance")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "ann_actual_vs_predicted_scatter.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate ANN residual diagnostics: {exc}")
    return metrics


def main() -> dict:
    return run_ann()


if __name__ == "__main__":
    main()
