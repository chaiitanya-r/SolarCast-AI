"""
KMeans clustering with elbow and silhouette analysis.

Input:  engineered solar data (Temperature, Irradiance, Hour)
Output: clustered_data.csv, elbow/silhouette/scatter plots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")

CLUSTER_FEATURES = ["Temperature", "Irradiance"]
EXTRA_3D = ["Hour", "Temperature", "Irradiance"]
MAX_ROWS = 15000


def _find_best_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    root = get_project_root()
    plots_dir = root / "results" / "plots"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(k_range), inertias, "bo-")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans Elbow Method")
    plt.tight_layout()
    fig.savefig(plots_dir / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(k_range), silhouettes, "go-")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans Silhouette Scores")
    plt.tight_layout()
    fig.savefig(plots_dir / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    if silhouettes[int(np.argmax(silhouettes))] - silhouettes[2] < 0.02:
        best_k = 3
    return best_k


@timer
def run_kmeans(k: int | None = None) -> pd.DataFrame:
    """Fit KMeans, save cluster assignments and diagnostic plots."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    df = load_processed("engineered_solar_data")
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"KMeans speed mode: sampled {MAX_ROWS} rows")
    X = df[CLUSTER_FEATURES].values

    best_k = k or _find_best_k(X)
    print(f"Selected k={best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["Temperature"],
        df["Irradiance"],
        c=df["kmeans_cluster"],
        cmap="viridis",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Irradiance")
    ax.set_title(f"KMeans Clusters (k={best_k})")
    plt.tight_layout()
    fig.savefig(plots_dir / "kmeans_2d_scatter.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.scatter(
        df["Hour"],
        df["Temperature"],
        df["Irradiance"],
        c=df["kmeans_cluster"],
        cmap="viridis",
        alpha=0.3,
        s=5,
    )
    ax3d.set_xlabel("Hour")
    ax3d.set_ylabel("Temperature")
    ax3d.set_zlabel("Irradiance")
    ax3d.set_title(f"KMeans 3D Clusters (k={best_k})")
    plt.tight_layout()
    fig.savefig(plots_dir / "kmeans_3d_scatter.png", dpi=150)
    plt.close(fig)

    out_path = root / "data" / "processed" / "clustered_data.csv"
    df.to_csv(out_path, index=False)
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Saved clustered data to {out_path}")
    return df


def main() -> pd.DataFrame:
    return run_kmeans()


if __name__ == "__main__":
    main()
