"""
PyTorch LSTM models for 24-hour-ahead irradiance forecasting.

Trains unidirectional and bidirectional LSTMs and compares metrics.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.deep_learning._training_utils import set_seeds
from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_loss_curve
from src.preprocessing.scaling import inverse_transform_target, load_split_data
from src.utils.device import (
    dataloader_kwargs,
    get_torch_device,
    log_device_once,
    require_cuda,
    to_device,
)
from src.utils.helpers import ensure_dirs, get_project_root, timer
from src.utils.helpers import save_plot

set_seeds(42)
plt.style.use("seaborn-v0_8")

LSTM_EPOCHS = 50
LSTM_PATIENCE = 10
LSTM_LR = 5e-4
SEQ_LEN = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 256

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of length seq_len; target is next-step irradiance."""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden: int = HIDDEN_SIZE,
        layers: int = NUM_LAYERS,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden * mult, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class LSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (attn_weights * out).sum(dim=1)
        out = self.relu(self.fc1(self.dropout(context)))
        result = self.fc2(out)
        if return_attention:
            return result, attn_weights.squeeze(-1)
        return result

def _evaluate_lstm(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    device = get_torch_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(to_device(torch.tensor(X_test, dtype=torch.float32))).cpu().numpy().ravel()
    y_true = inverse_transform_target(y_test)
    y_pred = inverse_transform_target(preds)
    return regression_metrics(y_true, y_pred)

@timer
def run_lstm() -> dict:
    require_cuda()
    log_device_once()
    ensure_dirs()
    root = get_project_root()
    data = load_split_data()
    X, y = data["X_train"], data["y_train"]
    X_test_raw, y_test_raw = data["X_test"], data["y_test"]

    X_seq, y_seq = build_sequences(X, y)
    X_test_seq, y_test_seq = build_sequences(
        np.vstack([X[-SEQ_LEN:], X_test_raw]), np.concatenate([y[-SEQ_LEN:], y_test_raw])
    )

    input_dim = X.shape[1]
    results: dict[str, dict] = {}

    for bidirectional, label in [(False, "LSTM"), (True, "BiLSTM")]:
        if bidirectional:
            model = LSTMRegressor(
                input_dim=input_dim,
                hidden=HIDDEN_SIZE,
                layers=NUM_LAYERS,
                bidirectional=True,
            )
        else:
            model = LSTMWithAttention(
                input_size=input_dim,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
            )
        X_tr, X_val, y_tr, y_val = train_test_split(X_seq, y_seq, test_size=0.15, random_state=42)

        if len(X_tr) > 8000:
            sub_idx = np.random.choice(len(X_tr), 8000, replace=False)
            X_tr = X_tr[sub_idx]
            y_tr = y_tr[sub_idx]

        device = get_torch_device()
        model = model.to(device)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1)),
            **dataloader_kwargs(batch_size=BATCH_SIZE, shuffle=True),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=LSTM_EPOCHS, eta_min=1e-6
        )
        criterion = nn.MSELoss()
        X_v = to_device(torch.tensor(X_val, dtype=torch.float32))
        y_v = to_device(torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_losses, val_losses = [], []
        best_state, best_val, wait = None, float("inf"), 0
        import time

        t0 = time.perf_counter()
        for epoch in range(LSTM_EPOCHS):
            model.train()
            ep_loss = 0.0
            for xb, yb in loader:
                xb, yb = to_device(xb), to_device(yb)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * len(xb)
            train_losses.append(ep_loss / len(X_tr))

            model.eval()
            with torch.no_grad():
                vloss = criterion(model(X_v), y_v).item()
            val_losses.append(vloss)
            scheduler.step()
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= LSTM_PATIENCE:
                    break
        elapsed = time.perf_counter() - t0
        epochs_run = len(val_losses)
        if best_state:
            model.load_state_dict(best_state)

        status = "early stop" if epochs_run < LSTM_EPOCHS else "done"
        print(f"  {label}  → epoch {epochs_run}/{LSTM_EPOCHS}  val_loss={best_val:.5f}  [{status}]")

        metrics = _evaluate_lstm(model, X_test_seq, y_test_seq)
        metrics["model"] = label
        metrics["train_time_s"] = elapsed
        metrics["epochs_run"] = epochs_run
        metrics["device"] = str(get_torch_device())
        results[label] = metrics

        report_name = "lstm_metrics.json" if not bidirectional else "bilstm_metrics.json"
        (root / "results" / "reports" / report_name).write_text(json.dumps(metrics, indent=2))

        if not bidirectional:
            plot_loss_curve(
                train_losses,
                val_losses,
                "LSTM Training Loss",
                str(root / "results" / "plots" / "lstm_loss.png"),
            )
            torch.save(model.state_dict(), root / "results" / "models" / "lstm_model.pth")
        else:
            torch.save(model.state_dict(), root / "results" / "models" / "bilstm_model.pth")

    ROOT = get_project_root()
    reports_dir = ROOT / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lstm_metrics = results.get("LSTM", {})
    bilstm_metrics = results.get("BiLSTM", {})
    with open(reports_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**lstm_metrics, "model": "LSTM"}, f, indent=2)
    with open(reports_dir / "bilstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**bilstm_metrics, "model": "BiLSTM"}, f, indent=2)

    model_uni = LSTMWithAttention(
        input_size=input_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    model_uni.load_state_dict(torch.load(root / "results" / "models" / "lstm_model.pth", weights_only=True))
    model_uni = model_uni.to(get_torch_device()).eval()
    with torch.no_grad():
        preds = model_uni(to_device(torch.tensor(X_test_seq[:200], dtype=torch.float32))).cpu().numpy().ravel()
    y_true = inverse_transform_target(y_test_seq[:200])
    y_pred = inverse_transform_target(preds)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label="Actual", alpha=0.8)
    ax.plot(y_pred, label="Predicted", alpha=0.8)
    ax.set_title("LSTM 24h-Ahead Forecast (200 timesteps)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Irradiance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "lstm_forecast.png")
    plt.close(fig)

    try:
        plt.style.use("seaborn-v0_8")
        model_bi = LSTMRegressor(
            input_dim=input_dim,
            hidden=HIDDEN_SIZE,
            layers=NUM_LAYERS,
            bidirectional=True,
        )
        bilstm_path = root / "results" / "models" / "bilstm_model.pth"
        if bilstm_path.exists():
            model_bi.load_state_dict(torch.load(bilstm_path, weights_only=True))
        model_bi = model_bi.to(get_torch_device()).eval()
        with torch.no_grad():
            preds_bi = model_bi(to_device(torch.tensor(X_test_seq[:200], dtype=torch.float32))).cpu().numpy().ravel()
        y_pred_bi = inverse_transform_target(preds_bi)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(y_true, label="Actual", linewidth=2)
        ax.plot(y_pred, label="LSTM Prediction", alpha=0.85)
        ax.plot(y_pred_bi, label="BiLSTM Prediction", alpha=0.85)
        ax.set_title("LSTM vs BiLSTM Forecast Comparison (First 200 Timesteps)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Irradiance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "lstm_bilstm_forecast_comparison.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate LSTM/BiLSTM forecast comparison: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        residuals_lstm = y_true - y_pred
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_pred, residuals_lstm, alpha=0.5, s=12, color="tab:blue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("LSTM Residual Plot")
        ax.set_xlabel("Predicted Irradiance")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "lstm_residuals.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate LSTM residual plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        model_bi_res = LSTMRegressor(
            input_dim=input_dim,
            hidden=HIDDEN_SIZE,
            layers=NUM_LAYERS,
            bidirectional=True,
        )
        bilstm_path = root / "results" / "models" / "bilstm_model.pth"
        model_bi_res.load_state_dict(torch.load(bilstm_path, weights_only=True))
        model_bi_res = model_bi_res.to(get_torch_device()).eval()
        with torch.no_grad():
            y_pred_bi_scaled = model_bi_res(
                to_device(torch.tensor(X_test_seq[:200], dtype=torch.float32))
            ).cpu().numpy().ravel()
        y_pred_bi_res = inverse_transform_target(y_pred_bi_scaled)
        residuals_bi = y_true - y_pred_bi_res
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y_pred_bi_res, residuals_bi, alpha=0.5, s=12, color="tab:orange")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("BiLSTM Residual Plot")
        ax.set_xlabel("Predicted Irradiance")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "bilstm_residuals.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate BiLSTM residual plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        n_samples = min(5, len(X_test_seq))
        sample_idx = np.random.choice(len(X_test_seq), size=n_samples, replace=False)
        X_attn = to_device(torch.tensor(X_test_seq[sample_idx], dtype=torch.float32))
        with torch.no_grad():
            _, attn_weights = model_uni(X_attn, return_attention=True)
        attn_np = attn_weights.cpu().numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(attn_np, cmap="YlOrRd", annot=False, ax=ax)
        ax.set_title("LSTM Attention Weights Heatmap (Sampled Test Sequences)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Sequence Sample")
        plt.tight_layout()
        save_plot(fig, "lstm_attention_heatmap.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate LSTM attention heatmap: {exc}")

    return results

def main() -> dict:
    return run_lstm()

if __name__ == "__main__":
    main()
