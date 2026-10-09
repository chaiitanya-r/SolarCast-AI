"""
Shared utility helpers for the Solar Predictive Analytics pipeline.

Provides project-root resolution, directory setup, plot/model I/O,
processed-data loading, and a timing decorator.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

import joblib
import matplotlib.pyplot as plt
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])

def get_project_root() -> Path:
    """Return the project root directory (contains main.py and data/)."""
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "main.py").exists() and (parent / "data").exists():
            return parent
    return current.parents[2]

def ensure_dirs() -> None:
    """Create data and results subdirectories if they do not exist."""
    root = get_project_root()
    dirs = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "final",
        root / "results" / "plots",
        root / "results" / "models",
        root / "results" / "reports",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def save_plot(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to results/plots/ and close it."""
    ensure_dirs()
    path = get_project_root() / "results" / "plots" / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def save_model(model: Any, filename: str) -> Path:
    """Persist a sklearn-compatible model with joblib to results/models/."""
    ensure_dirs()
    path = get_project_root() / "results" / "models" / filename
    joblib.dump(model, path)
    return path

def load_model(filename: str) -> Any:
    """Load a joblib model from results/models/."""
    path = get_project_root() / "results" / "models" / filename
    return joblib.load(path)

def load_processed(name: str) -> pd.DataFrame:
    """Load a CSV from data/processed/ (name with or without .csv)."""
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    path = get_project_root() / "data" / "processed" / name
    return pd.read_csv(path)

def timer(func: F) -> F:
    """Decorator that prints elapsed wall-clock time for a function call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper
