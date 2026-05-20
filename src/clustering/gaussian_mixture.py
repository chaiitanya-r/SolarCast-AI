"""
Gaussian Mixture Model clustering with BIC/AIC model selection.

Fits GMM with n_components=3 and plots BIC/AIC for n=2..8.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")

FEATURES = ["Temperature", "Irradiance"]
N_COMPONENTS = 3
MAX_ROWS = 12000


@timer
def run_gmm() -> pd.DataFrame:
    """Fit GMM, plot BIC/AIC curves, scatter clusters, print covariances."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    df = load_processed("engineered_solar_data")
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"GMM speed mode: sampled {MAX_ROWS} rows")
    X = df[FEATURES].values

    n_range = range(2, 9)
    bics, aics = [], []
    for n in n_range:
        gmm_tmp = GaussianMixture(n_components=n, random_state=42)
        gmm_tmp.fit(X)
        bics.append(gmm_tmp.bic(X))
        aics.append(gmm_tmp.aic(X))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(n_range), bics, "b-o", label="BIC")
    ax.plot(list(n_range), aics, "r-o", label="AIC")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Score")
    ax.set_title("GMM Model Selection — BIC / AIC")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "gmm_bic_aic.png", dpi=150)
    plt.close(fig)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
    df["gmm_cluster"] = gmm.fit_predict(X)

    print(f"\n=== GMM Covariance Matrices (k={N_COMPONENTS}) ===")
    for i, cov in enumerate(gmm.covariances_):
        print(f"Component {i}:\n{cov}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["Temperature"],
        df["Irradiance"],
        c=df["gmm_cluster"],
        cmap="coolwarm",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Irradiance")
    ax.set_title(f"GMM Clusters (n={N_COMPONENTS})")
    plt.tight_layout()
    fig.savefig(plots_dir / "gmm_clusters.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {plots_dir}")
    return df


def main() -> pd.DataFrame:
    return run_gmm()


if __name__ == "__main__":
    main()
