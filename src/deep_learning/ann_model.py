"""
PyTorch Artificial Neural Network for irradiance regression.

Architecture: Input(8) → 128 → ReLU → Dropout(0.2) → 64 → ReLU → 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.deep_learning._training_utils import (
    evaluate_and_save,
    set_seeds,
    train_regressor,
)
from src.preprocessing.scaling import load_split_data
from src.utils.device import require_cuda
from src.utils.helpers import timer

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
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ANN_EPOCHS, eta_min=1e-6
        ),
    )
    return evaluate_and_save(
        model,
        "ANN",
        train_time=elapsed,
        loss_plot="ann_loss.png",
        pred_plot="ann_predictions.png",
        model_file="ann_model.pth",
        train_losses=train_losses,
        val_losses=val_losses,
    )


def main() -> dict:
    return run_ann()


if __name__ == "__main__":
    main()
