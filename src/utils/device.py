"""
GPU / device selection for PyTorch training.

Set SOLAR_USE_GPU=0 to force CPU.
Set SOLAR_GPU_INDEX to select the CUDA device index.
Set SOLAR_REQUIRE_CUDA=1 to fail fast if CUDA is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

# Prefer GPU when available; override with env SOLAR_USE_GPU=0
USE_GPU = os.environ.get("SOLAR_USE_GPU", "1").strip().lower() not in ("0", "false", "no")
# Index into visible GPUs (after CUDA_VISIBLE_DEVICES is applied)
GPU_INDEX = int(os.environ.get("SOLAR_GPU_INDEX", "0"))

_torch_device: torch.device | None = None
_device_logged = False


def get_torch_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    global _torch_device
    if _torch_device is not None:
        return _torch_device

    if USE_GPU:
        if torch.cuda.is_available():
            _torch_device = torch.device(f"cuda:{GPU_INDEX}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _torch_device = torch.device("mps")
        else:
            _torch_device = torch.device("cpu")
    else:
        _torch_device = torch.device("cpu")

    return _torch_device


def is_cuda() -> bool:
    return get_torch_device().type == "cuda"


def dataloader_kwargs(batch_size: int = 64, shuffle: bool = True) -> dict[str, Any]:
    """DataLoader settings tuned for GPU transfer."""
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 0,  # safe on Windows; increase on Linux if needed
    }
    if is_cuda():
        kwargs["pin_memory"] = True
    return kwargs


def set_training_seeds(seed: int = 42) -> None:
    """Set random seeds for numpy and torch (including CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def log_device_once() -> None:
    """Print which device training will use (once per process)."""
    global _device_logged
    if _device_logged:
        return
    _device_logged = True

    dev = get_torch_device()
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev.index or 0)
        mem_gb = torch.cuda.get_device_properties(dev.index or 0).total_memory / 1e9
        print(f"[GPU] PyTorch training on: {dev} — {name} ({mem_gb:.1f} GB)")
    elif dev.type == "mps":
        print("[GPU] PyTorch training on: Apple Metal (MPS)")
    else:
        msg = "[CPU] PyTorch training on CPU"
        if USE_GPU:
            msg += " — no GPU detected. Install a CUDA-enabled PyTorch build."
        print(msg)


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(get_torch_device(), non_blocking=is_cuda())


def require_cuda() -> None:
    """Fail fast when strict CUDA mode is requested and CUDA is unavailable."""
    strict = os.environ.get("SOLAR_REQUIRE_CUDA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if strict and not is_cuda():
        raise RuntimeError(
            "CUDA is required (SOLAR_REQUIRE_CUDA=1) but not available. "
            "Install CUDA-enabled PyTorch or set SOLAR_REQUIRE_CUDA=0."
        )
