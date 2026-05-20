"""
Solar Predictive Analytics — full pipeline orchestrator.

Runs all preprocessing, ML, deep learning, explainability, and NLP stages.
Each stage is isolated with try/except so failures do not halt the pipeline.
"""

from __future__ import annotations

import traceback
from typing import Callable

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_banner(text: str) -> None:
    width = len(text) + 4
    print(f"\n{CYAN}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{RESET}\n")


def run_stage(name: str, func: Callable) -> bool:
    print(f"{CYAN}▶ {name}...{RESET}")
    try:
        func()
        print(f"{GREEN}✓ {name} completed{RESET}\n")
        return True
    except Exception as exc:
        print(f"{RED}✗ {name} failed: {exc}{RESET}")
        traceback.print_exc()
        print()
        return False


def run_cleaning() -> None:
    from src.preprocessing.data_cleaning import run_cleaning as fn

    fn()


def run_feature_engineering() -> None:
    from src.preprocessing.feature_engineering import run_feature_engineering as fn

    fn()


def run_scaling() -> None:
    from src.preprocessing.scaling import run_scaling as fn

    fn()


def run_classification() -> None:
    from src.classification.classification_models import run_classification as fn

    fn()


def run_kmeans() -> None:
    from src.clustering.kmeans import run_kmeans as fn

    fn()


def run_kmedoids() -> None:
    from src.clustering.kmedoids import run_kmedoids as fn

    fn()


def run_hierarchical() -> None:
    from src.clustering.hierarchical import run_hierarchical as fn

    fn()


def run_gmm() -> None:
    from src.clustering.gaussian_mixture import run_gmm as fn

    fn()


def run_ann() -> None:
    from src.deep_learning.ann_model import run_ann as fn

    fn()


def run_dnn() -> None:
    from src.deep_learning.dnn_model import run_dnn as fn

    fn()


def run_lstm() -> None:
    from src.deep_learning.lstm_model import run_lstm as fn

    fn()


def run_shap() -> None:
    from src.explainability.shap_analysis import run_shap as fn

    fn()


def run_model_comparison() -> None:
    from src.deep_learning.model_comparison import run_model_comparison as fn

    fn()


def main() -> None:
    from src.utils.device import log_device_once, require_cuda
    from src.utils.helpers import ensure_dirs

    ensure_dirs()
    log_device_once()
    require_cuda()
    print_banner("SOLAR PREDICTIVE ANALYTICS — FULL PIPELINE")

    stages = [
        ("Data Cleaning", run_cleaning),
        ("Feature Engineering", run_feature_engineering),
        ("Scaling + Split", run_scaling),
        ("Classification Models", run_classification),
        ("KMeans Clustering", run_kmeans),
        ("KMedoids Clustering", run_kmedoids),
        ("Hierarchical Clustering", run_hierarchical),
        ("Gaussian Mixture", run_gmm),
        ("ANN Model", run_ann),
        ("DNN Model", run_dnn),
        ("LSTM Model", run_lstm),
        ("SHAP Explainability", run_shap),
        ("Model Comparison Table", run_model_comparison),
    ]

    ok = sum(run_stage(name, fn) for name, fn in stages)
    print_banner(f"PIPELINE COMPLETE — {ok}/{len(stages)} stages succeeded")
    print(f"Results saved to results/\n")


if __name__ == "__main__":
    main()
