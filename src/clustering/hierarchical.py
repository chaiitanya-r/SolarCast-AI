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

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, save_plot, timer

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

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, leaf_rotation=90)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward, truncated)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    save_plot(fig, "dendrogram.png")
    plt.close(fig)

    hier_labels = fcluster(Z, t=K, criterion="maxclust") - 1
    ari = None

    try:
        clustered_df = load_processed("clustered_data.csv")
        if "kmeans_cluster" in clustered_df.columns:
            kmeans_labels_for_ari = clustered_df.iloc[idx]["kmeans_cluster"].values
            ari = adjusted_rand_score(hier_labels, kmeans_labels_for_ari[: len(hier_labels)])
            print(f"  ARI (Hierarchical vs KMeans): {ari:.4f}")
        else:
            print("  [warn] kmeans_cluster column not found in clustered_data.csv")
    except Exception as e:
        print(f"  [warn] Could not compute ARI: {e}")

    df.loc[df.index[idx], "hierarchical_cluster"] = hier_labels

    try:
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(12, 6))
        ari_plot = float(ari) if ari is not None else 0.0
        bar = ax.bar(["Hierarchical vs KMeans"], [ari_plot], color="tab:blue")
        ax.bar_label(bar, fmt="%.4f", padding=3)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Perfect agreement (1.0)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Adjusted Rand Index Comparison")
        ax.set_xlabel("Clustering Comparison")
        ax.set_ylabel("ARI Score")
        ax.legend()
        if ari is not None:
            ax.text(
                0,
                float(ari) + 0.02 if float(ari) <= 0.95 else float(ari) - 0.06,
                f"ARI = {ari:.4f}",
                ha="center",
            )
        plt.tight_layout()
        save_plot(fig, "hierarchical_ari.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate ARI comparison chart: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        plot_df = df.dropna(subset=["hierarchical_cluster"])
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(
            plot_df["Temperature"],
            plot_df["Irradiance"],
            c=plot_df["hierarchical_cluster"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Hierarchical Cluster")
        ax.set_title("Hierarchical Cluster Assignments")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Irradiance")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "hierarchical_clusters.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate hierarchical cluster scatter: {exc}")

    print("  Saved: dendrogram.png")
    return df

def main() -> pd.DataFrame:
    return run_hierarchical()

if __name__ == "__main__":
    main()
