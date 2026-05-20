"""
PyTorch Deep Neural Network for irradiance regression.

Architecture: Input(8) → 256 → 128 → 64 → 32 → 1 with BatchNorm and Dropout(0.3)
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
DNN_EPOCHS = 600
DNN_PATIENCE = 50


class DNN(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, 256, 128, 64, 32]
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


@timer
def run_dnn() -> dict:
    require_cuda()
    data = load_split_data()
    X, y = data["X_train"], data["y_train"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = DNN(input_dim=X.shape[1])
    train_losses, val_losses, elapsed = train_regressor(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=DNN_EPOCHS,
        lr=0.0005,
        patience=DNN_PATIENCE,
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=True
        ),
    )
    return evaluate_and_save(
        model,
        "DNN",
        train_time=elapsed,
        loss_plot="dnn_loss.png",
        pred_plot="dnn_predictions.png",
        model_file="dnn_model.pth",
        train_losses=train_losses,
        val_losses=val_losses,
    )


def main() -> dict:
    return run_dnn()


if __name__ == "__main__":
    main()
