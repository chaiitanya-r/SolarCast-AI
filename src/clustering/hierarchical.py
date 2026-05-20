"""
Hierarchical clustering with Ward linkage and dendrogram visualization.

Compares cut-tree labels (k=3) with KMeans via Adjusted Rand Index.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")

FEATURES = ["Temperature", "Irradiance"]
K = 3
SUBSAMPLE = 5000


@timer
def run_hierarchical() -> pd.DataFrame:
    """Build dendrogram on subsample, cut at k=3, compare with KMeans."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    df = load_processed("engineered_solar_data")
    X = df[FEATURES].values

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(SUBSAMPLE, len(X)), replace=False)
    X_sub = X[idx]

    Z = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward, truncated)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(plots_dir / "dendrogram.png", dpi=150)
    plt.close(fig)

    hier_labels_sub = fcluster(Z, t=K, criterion="maxclust") - 1

    if "kmeans_cluster" in df.columns:
        kmeans_sub = df.iloc[idx]["kmeans_cluster"].values
        ari = adjusted_rand_score(kmeans_sub, hier_labels_sub)
        print(f"\nAdjusted Rand Index (KMeans vs Hierarchical, k={K}): {ari:.4f}")
    else:
        print("KMeans labels not found — run kmeans.py first for ARI comparison.")

    df.loc[df.index[idx], "hierarchical_cluster"] = hier_labels_sub
    print(f"Saved dendrogram to {plots_dir / 'dendrogram.png'}")
    return df


def main() -> pd.DataFrame:
    return run_hierarchical()


if __name__ == "__main__":
    main()
