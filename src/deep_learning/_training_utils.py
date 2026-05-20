"""Shared PyTorch training utilities for regression models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_loss_curve, plot_predictions
from src.preprocessing.scaling import inverse_transform_target, load_split_data
from src.utils.device import (
    dataloader_kwargs,
    get_torch_device,
    is_cuda,
    log_device_once,
    set_training_seeds,
    to_device,
)
from src.utils.helpers import get_project_root


def set_seeds(seed: int = 42) -> None:
    set_training_seeds(seed)


def train_regressor(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    patience: int = 20,
    batch_size: int = 64,
    scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
) -> tuple[list[float], list[float], float]:
    """Train a PyTorch regressor with early stopping; return losses and elapsed time."""
    log_device_once()
    device = get_torch_device()
    model = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = to_device(torch.tensor(X_val, dtype=torch.float32))
    y_v = to_device(torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        **dataloader_kwargs(batch_size=batch_size, shuffle=True),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = to_device(xb)
            yb = to_device(yb)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    if is_cuda():
        torch.cuda.empty_cache()
    return train_losses, val_losses, elapsed


def evaluate_and_save(
    model: nn.Module,
    model_name: str,
    *,
    train_time: float,
    loss_plot: str,
    pred_plot: str,
    model_file: str,
    train_losses: list[float],
    val_losses: list[float],
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict:
    """Evaluate on test set, inverse-transform targets, save metrics and plots."""
    log_device_once()
    root = get_project_root()
    data = load_split_data()
    if X_test is None:
        X_test = data["X_test"]
    if y_test is None:
        y_test = data["y_test"]

    device = get_torch_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_scaled = (
            model(to_device(torch.tensor(X_test, dtype=torch.float32)))
            .cpu()
            .numpy()
            .ravel()
        )

    y_true = inverse_transform_target(y_test)
    y_pred = inverse_transform_target(preds_scaled)
    metrics = regression_metrics(y_true, y_pred)
    metrics["model"] = model_name
    metrics["train_time_s"] = train_time
    metrics["device"] = str(device)

    plots_dir = root / "results" / "plots"
    models_dir = root / "results" / "models"
    reports_dir = root / "results" / "reports"

    plot_loss_curve(
        train_losses,
        val_losses,
        f"{model_name} Training Loss",
        str(plots_dir / loss_plot),
    )
    plot_predictions(y_true, y_pred, f"{model_name} Predictions", str(plots_dir / pred_plot))
    torch.save(model.state_dict(), models_dir / model_file)

    report_path = reports_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n=== {model_name} Test Metrics ({device}) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return metrics
