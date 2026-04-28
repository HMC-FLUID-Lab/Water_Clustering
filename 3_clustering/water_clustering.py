"""
water_clustering.py
===================
Clustering analysis for water molecular dynamics simulations.

Replicates and extends Diya's Colab notebook for identifying Locally Favored
Tetrahedral Structures (LFTS) and Disordered Normal-Liquid Structures (DNLS)
in water, following the Shi & Tanaka two-state model (JACS 2020).

Supported clustering methods:
  - DBSCAN       (density-based, identifies noise + clusters)
  - K-Means      (centroid-based, forces exactly N clusters)
  - GMM          (Gaussian Mixture Model, best physical match to Tanaka framework)
  - dbscan_gmm   (two-stage: DBSCAN denoising → GMM)
  - HDBSCAN      (hierarchical DBSCAN; density-adaptive, no epsilon needed)
  - hdbscan_gmm  (two-stage: HDBSCAN denoising → GMM)

Optional pre-processing:
  - UMAP  (--umap flag): reduces scaled features to N-D embedding before any
           clustering method, often revealing cleaner Gaussian cluster structure.

Usage:
  python water_clustering.py \
      --mat_file  OrderParam_Run21_swm4ndp_T-20.0.mat \
      --zeta_file OrderParamZeta_Run21_swm4ndp_T-20.0.mat \
      --n_runs    20 \
      --method    dbscan \
      --eps       0.2 \
      --out_dir   ./results
"""

# Fix OpenBLAS thread overflow issue - MUST be set before importing numpy/scipy
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# Suppress threadpoolctl version detection errors (harmless)
import sys
import warnings
warnings.filterwarnings("ignore", message=".*'NoneType' object has no attribute 'split'.*")

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

from plot_style import feature_axis_label
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from time import time

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Cluster colour palette ────────────────────────────────────────────────────
# Index matches sorted cluster label: -1 = noise, 0 = first cluster, 1 = second
_CLUSTER_COLORS = {
    -1: "#aaaaaa",   # noise     → light grey
     0: "#298c8c",   # cluster 0 → med teal
     1: "#800074",   # cluster 1 → med magenta/purple
     2: "#f4a261",   # cluster 2 → warm amber  (fallback)
     3: "#2166ac",   # cluster 3 → blue        (fallback)
}

# K-means: blue + pink
_KMEANS_COLORS = {-1: "#888888", 0: "#2563eb", 1: "#ec4899", 2: "#7c3aed", 3: "#0d9488"}

# HDBSCAN→GMM: orange + steel blue (swap anytime; keep noise grey)
_HDBSCAN_GMM_COLORS = {
    -1: "#b0b0b0",
    0: "#e6550d",
    1: "#3182bd",
    2: "#fdae61",
    3: "#4daf4a",
}

# DBSCAN→GMM: teal + terracotta (LFTS / DNLS); distinct from default teal/magenta
_DBSCAN_GMM_COLORS = {
    -1: "#9ca3af",   # noise — cool grey
    0: "#0f766e",    # LFTS — deep lagoon teal
    1: "#c2410c",    # DNLS — burnt sienna
    2: "#7c3aed",    # fallback cluster
    3: "#ca8a04",    # fallback cluster
}


def _palette(unique_labels):
    """Return a colour list aligned with sorted unique_labels."""
    return [_CLUSTER_COLORS.get(lbl, "#333333") for lbl in unique_labels]


def _palette_for_method(method: str, unique_labels):
    """Method-specific palette: kmeans uses blue/pink, others use default."""
    m = method.lower()
    if "kmeans" in m:
        return [_KMEANS_COLORS.get(lbl, "#333333") for lbl in unique_labels]
    if "hdbscan_gmm" in m:
        return [_HDBSCAN_GMM_COLORS.get(lbl, "#333333") for lbl in unique_labels]
    if m == "dbscan_gmm":
        return [_DBSCAN_GMM_COLORS.get(lbl, "#333333") for lbl in unique_labels]
    return _palette(unique_labels)


def _cluster_display_name(lbl, method: str) -> str:
    """Legend label: kmeans/dbscan_gmm 0→LFTS, 1→DNLS; others keep Cluster/Noise."""
    if lbl == -1:
        return "Noise"
    if "kmeans" in method.lower() or "dbscan_gmm" in method.lower():
        if lbl == 0:
            return "LFTS"
        if lbl == 1:
            return "DNLS"
    return f"Cluster {lbl}"
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Note: You may see "AttributeError: 'NoneType' object has no attribute 'split'"
# from threadpoolctl. These are harmless and can be ignored. They don't affect results.


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_order_params(mat_file: str, zeta_file: str, n_runs: int) -> pd.DataFrame:
    """
    Load structural order parameters from two MATLAB .mat files.

    The .mat files store data as cell arrays: one cell per MD run.
    All runs are concatenated into a single flat array.

    Parameters
    ----------
    mat_file  : path to .mat file with q_all, Q6_all, LSI_all, Sk_all
    zeta_file : path to .mat file with zeta_all
    n_runs    : number of independent MD runs to concatenate

    Returns
    -------
    df : pd.DataFrame with columns [q_all, Q6_all, LSI_all, Sk_all, zeta_all]
         One row per water molecule across all runs.
    """
    water  = loadmat(mat_file)
    water1 = loadmat(zeta_file)

    def flatten(mat_dict, key):
        out = []
        for i in range(n_runs):
            out.extend(mat_dict[key][i])
        return np.array(out, dtype=float)

    df = pd.DataFrame({
        "q_all"    : flatten(water,  "q_all"),
        "Q6_all"   : flatten(water,  "Q6_all"),
        "LSI_all"  : flatten(water,  "LSI_all"),
        "Sk_all"   : flatten(water,  "Sk_all"),
        "zeta_all" : flatten(water1, "zeta_all"),
    })

    # Clean data: replace inf values with NaN, then drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    
    if n_after < n_before:
        print(f"Warning: Removed {n_before - n_after} rows with inf/NaN values")
    
    print(f"Loaded {len(df):,} water molecules from {n_runs} runs.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MinMax scaling so each feature lives in [0, 1].

    This is critical for DBSCAN: the epsilon radius is measured in the
    scaled feature space, so all descriptors contribute equally to the
    Euclidean distance regardless of their physical units.

    Returns a new DataFrame; the original is unchanged.
    """
    scaler    = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    return df_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UMAP DIMENSIONALITY REDUCTION  (optional pre-processing)
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(df_scaled: pd.DataFrame,
             n_components: int = 2,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             metric: str = "euclidean",
             random_state: int = 42) -> pd.DataFrame:
    """
    Reduce the scaled feature matrix with UMAP before clustering.

    UMAP preserves both local neighbourhood structure and global topology,
    making downstream GMM / DBSCAN more effective when features live on a
    curved manifold in high-dimensional space.

    Typical usage in this pipeline
    ───────────────────────────────
    • 5-D → 2-D  then GMM  : reveals cluster shape in a human-readable plane
    • 5-D → 2-D  then DBSCAN: density is better defined in 2-D than 5-D
    • 5-D → 2-D  then DBSCAN→GMM: full denoising + probabilistic assignment

    Parameters
    ----------
    n_components  : target dimensionality (2 = 2D embedding, typical)
    n_neighbors   : UMAP local neighbourhood size; larger = more global structure
    min_dist      : minimum distance between embedded points; smaller = tighter clusters
    metric        : distance metric in input space ('euclidean', 'cosine', etc.)
    random_state  : for reproducibility

    Returns
    -------
    df_umap : DataFrame with columns ['umap_0', 'umap_1', ...] same row order
    """
    if not HAS_UMAP:
        raise ImportError(
            "umap-learn is not installed. "
            "Install with: conda install -c conda-forge umap-learn"
        )

    print(f"UMAP  {df_scaled.shape[1]}-D → {n_components}-D  "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist})")
    t0 = time()

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=False,
    )
    embedding = reducer.fit_transform(df_scaled.values)
    dt = time() - t0

    cols = [f"umap_{i}" for i in range(n_components)]
    df_umap = pd.DataFrame(embedding, columns=cols, index=df_scaled.index)
    print(f"  Wall time : {dt:.2f} s")
    return df_umap


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(df_scaled: pd.DataFrame, eps: float = 0.2,
               min_samples: int = 5) -> np.ndarray:
    """
    DBSCAN clustering on the scaled feature matrix.

    Label convention (DBSCAN-specific):
      -1  →  noise / low-density transition molecules
       0  →  cluster 0  (typically the dense DNLS-like majority)
       1  →  cluster 1  (typically the sparse LFTS-like minority)

    This three-way labelling is why the scatter plot from Diya's notebook
    shows what looks like "3 groups" even though DBSCAN reports 2 clusters:
    the -1 noise points occupy the low-density bridge between the two
    physical states.

    Parameters
    ----------
    eps         : neighbourhood radius in scaled feature space (0–1 per axis)
    min_samples : minimum points required to form a core point

    Returns
    -------
    labels : int array of length n_samples
    """
    t0     = time()
    db     = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(df_scaled)
    dt     = time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"DBSCAN  eps={eps:.3f}  min_samples={min_samples}")
    print(f"  Clusters (excl. noise): {n_clusters}")
    print(f"  Noise points          : {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    print(f"  Wall time             : {dt:.2f} s")

    if n_clusters > 1:
        mask   = labels != -1
        score  = silhouette_score(df_scaled[mask], labels[mask])
        print(f"  Silhouette score      : {score:.4f}  (non-noise only)")

    return labels


def run_kmeans(df_scaled: pd.DataFrame, n_clusters: int = 2,
               random_state: int = 42) -> np.ndarray:
    """K-Means forcing exactly n_clusters partitions."""
    t0     = time()
    km     = KMeans(n_clusters=n_clusters, random_state=random_state,
                    init="k-means++", n_init=10)
    labels = km.fit_predict(df_scaled)
    dt     = time() - t0

    score = silhouette_score(df_scaled, labels)
    print(f"K-Means  n_clusters={n_clusters}")
    print(f"  Silhouette score : {score:.4f}")
    print(f"  Wall time        : {dt:.2f} s")
    return labels


def run_gmm(df_scaled: pd.DataFrame, n_components: int = 2,
            random_state: int = 42) -> np.ndarray:
    """
    Gaussian Mixture Model clustering.

    GMM is the theoretically preferred method for this physics problem
    because Shi & Tanaka's two-state model predicts each structural state
    follows a Gaussian distribution in order-parameter space.  GMM also
    returns soft probabilities (responsibility), which maps directly to
    the LFTS fraction s used in the thermodynamic two-state model.
    """
    t0  = time()
    gm  = GaussianMixture(n_components=n_components,
                          random_state=random_state,
                          covariance_type="full",
                          n_init=5)
    gm.fit(df_scaled)
    labels = gm.predict(df_scaled)
    probs  = gm.predict_proba(df_scaled)   # soft assignments
    dt     = time() - t0

    score = silhouette_score(df_scaled, labels)
    print(f"GMM  n_components={n_components}")
    print(f"  Silhouette score : {score:.4f}")
    print(f"  BIC              : {gm.bic(df_scaled):.1f}")
    print(f"  AIC              : {gm.aic(df_scaled):.1f}")
    print(f"  Wall time        : {dt:.2f} s")

    # Identify which component is LFTS (more ordered = higher value)
    # Prefer zeta_all; fall back to q_all; if neither present, skip labelling.
    for order_col in ("zeta_all", "q_all"):
        if order_col in df_scaled.columns:
            order_means = [
                df_scaled[order_col][labels == c].mean()
                for c in range(n_components)
            ]
            lfts_comp = int(np.argmax(order_means))
            print(f"  Probable LFTS component: {lfts_comp} "
                  f"(mean {order_col}_scaled={order_means[lfts_comp]:.3f}, "
                  f"ranked by '{order_col}')")
            s_fraction = np.mean(labels == lfts_comp)
            print(f"  Estimated LFTS fraction s ≈ {s_fraction:.3f}")
            break
    else:
        print("  Note: neither 'zeta_all' nor 'q_all' in features; "
              "skipping LFTS component identification.")

    return labels, probs


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(df_scaled: pd.DataFrame, labels: np.ndarray,
                 method: str, out_dir: str, df_raw: pd.DataFrame | None = None):
    """Scatter plot: q vs ζ (raw when df_raw given) or first two scaled features."""
    # Prefer q_all vs zeta_all; use raw data when df_raw provided
    use_raw = df_raw is not None and "q_all" in df_raw.columns and "zeta_all" in df_raw.columns
    features = list(df_raw.columns) if use_raw else list(df_scaled.columns)

    if len(features) < 2:
        print(f"  Skipping scatter plot (need at least 2 features)")
        return

    if use_raw:
        x_feat, y_feat = "q_all", "zeta_all"
        x_label, y_label = r"$q$", r"$\zeta$"
        title_suffix = r"$q$ vs $\zeta$  (key two-state diagnostic)"
        df_plot = df_raw
    else:
        if "q_all" in features and "zeta_all" in features:
            x_feat, y_feat = "q_all", "zeta_all"
        else:
            x_feat, y_feat = features[0], features[-1]
        x_label = f"{x_feat} (scaled)"
        y_label = f"{y_feat} (scaled)"
        title_suffix = f"{x_feat} vs {y_feat}"
        df_plot = df_scaled

    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels))
    palette = _palette_for_method(method, unique)
    for lbl, col in zip(unique, palette):
        mask = labels == lbl
        name = f"{_cluster_display_name(lbl, method)} ({np.sum(mask):,})"
        ax.scatter(df_plot.loc[mask, x_feat],
                   df_plot.loc[mask, y_feat],
                   s=4, alpha=0.3, color=col, label=name)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"{method.upper()} — {title_suffix}", fontsize=12)
    ax.legend(markerscale=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{method}_scatter.png"), dpi=150)
    plt.close(fig)


def plot_zeta_distribution(df_raw: pd.DataFrame, labels: np.ndarray,
                           method: str, out_dir: str):
    """
    Distribution of ζ (zeta) split by cluster (if available).

    This is the primary physical validation plot: a successful clustering
    should show bimodal separation in ζ, with one Gaussian peak per
    cluster matching the LFTS (high ζ) and DNLS (low ζ) populations.
    """
    if "zeta_all" not in df_raw.columns:
        print(f"  Skipping zeta distribution plot (zeta_all not in features)")
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    unique  = sorted(set(labels))
    palette = _palette_for_method(method, unique)
    for lbl, col in zip(unique, palette):
        mask = labels == lbl
        label = f"{_cluster_display_name(lbl, method)} (n={np.sum(mask):,})"
        sns.histplot(df_raw.loc[mask, "zeta_all"], ax=ax, kde=True,
                     color=col, alpha=0.5, label=label, stat="density")
    ax.set_xlabel("ζ  [Å]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{method.upper()} — ζ distribution per cluster\n"
                 "(bimodal separation validates LFTS / DNLS assignment)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{method}_zeta_distribution.png"), dpi=150)
    plt.close(fig)


def plot_pairplot(df_scaled: pd.DataFrame, labels: np.ndarray,
                  method: str, out_dir: str):
    """Pairplot of all five order parameters coloured by cluster."""
    df_plot = df_scaled.copy()
    df_plot = df_plot.rename(
        columns={c: feature_axis_label(c) for c in df_plot.columns}
    )
    unique = sorted(set(labels))
    lbl_to_disp = {lbl: _cluster_display_name(lbl, method) for lbl in unique}
    display_names = [lbl_to_disp[lbl] for lbl in unique]
    df_plot["cluster"] = [lbl_to_disp[l] for l in labels]
    colors = _palette_for_method(method, unique)
    palette = {disp: col for disp, col in zip(display_names, colors)}
    g = sns.pairplot(df_plot, hue="cluster", palette=palette,
                     plot_kws={"s": 4, "alpha": 0.3}, diag_kind="kde")
    g.figure.suptitle(f"{method.upper()} — Pairplot of scaled order parameters",
                      y=1.01, fontsize=11)
    g.figure.savefig(os.path.join(out_dir, f"{method}_pairplot.png"),
                     dpi=120, bbox_inches="tight")
    plt.close("all")


def plot_all_distributions(df_raw: pd.DataFrame, labels: np.ndarray,
                           method: str, out_dir: str):
    """Per-feature histograms coloured by cluster label (only selected features)."""
    features = list(df_raw.columns)
    n_features = len(features)
    
    if n_features == 0:
        print(f"  Skipping distribution plots (no features)")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]
    
    unique  = sorted(set(labels))
    palette = _palette_for_method(method, unique)

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for lbl, col in zip(unique, palette):
            mask  = labels == lbl
            name  = _cluster_display_name(lbl, method)
            sns.histplot(df_raw.loc[mask, feat], ax=ax, color=col,
                         alpha=0.5, kde=True, label=name, stat="density")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
    
    # Turn off extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")
    
    if n_features > 0:
        axes[0].legend(fontsize=8)
    
    fig.suptitle(f"{method.upper()} — Distribution of raw order parameters",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{method}_all_distributions.png"), dpi=150)
    plt.close(fig)


def plot_umap_embedding(df_umap: pd.DataFrame, labels: np.ndarray,
                        df_raw: pd.DataFrame, method: str, out_dir: str):
    """
    2-D UMAP scatter coloured by cluster, with zeta_all as point colour
    in a second panel when available.
    """
    n_comp = df_umap.shape[1]
    x_col, y_col = "umap_0", "umap_1"

    has_zeta = "zeta_all" in df_raw.columns
    n_panels = 2 if has_zeta else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # Panel 1 — coloured by cluster label
    ax = axes[0]
    unique  = sorted(set(labels))
    palette = _palette_for_method(method, unique)
    for lbl, col in zip(unique, palette):
        mask = labels == lbl
        name = f"{_cluster_display_name(lbl, method)} ({mask.sum():,})"
        ax.scatter(df_umap.loc[mask, x_col], df_umap.loc[mask, y_col],
                   s=3, alpha=0.3, color=col, label=name)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2" if n_comp >= 2 else "", fontsize=11)
    ax.set_title(f"{method.upper()} on UMAP embedding — cluster labels")
    ax.legend(markerscale=4, fontsize=9)

    # Panel 2 — coloured by zeta value (physical validation)
    if has_zeta:
        ax2 = axes[1]
        sc = ax2.scatter(df_umap[x_col], df_umap[y_col],
                         c=df_raw["zeta_all"].values,
                         s=3, alpha=0.4, cmap="coolwarm")
        plt.colorbar(sc, ax=ax2, label="ζ (Å)")
        ax2.set_xlabel("UMAP 1", fontsize=11)
        ax2.set_ylabel("UMAP 2" if n_comp >= 2 else "", fontsize=11)
        ax2.set_title("UMAP embedding — coloured by ζ")

    fig.tight_layout()
    fname = os.path.join(out_dir, f"{method}_umap_embedding.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved UMAP embedding plot: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Water structure clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Input files ───────────────────────────────────────────────────────────
    p.add_argument("-m", "--mat_file",  required=True,
                   help="Path to OrderParam .mat file  "
                        "(e.g. /data/OrderParam_Run21_tip4p2005_T240.mat)")
    p.add_argument("-z", "--zeta_file", required=True,
                   help="Path to OrderParamZeta .mat file  "
                        "(e.g. /data/OrderParamZeta_Run21_tip4p2005_T240.mat)")

    # ── Simulation parameters ─────────────────────────────────────────────────
    p.add_argument("-n", "--n_runs", type=int, default=20,
                   help="Number of MD runs to concatenate")

    # ── Clustering parameters ─────────────────────────────────────────────────
    p.add_argument("--method",
                   choices=["dbscan", "kmeans", "gmm", "dbscan_gmm",
                            "hdbscan", "hdbscan_gmm", "all"],
                   default="all", help="Clustering method")
    p.add_argument("--eps",         type=float, default=0.2,
                   help="DBSCAN epsilon radius in scaled space")
    p.add_argument("--min_samples", type=int,   default=5,
                   help="DBSCAN min_samples (also used as HDBSCAN min_samples "
                        "when --hdbscan-min-samples is not set)")
    p.add_argument("-k", "--n_clusters", type=int, default=2,
                   help="Number of clusters for KMeans / GMM")

    # ── HDBSCAN parameters ────────────────────────────────────────────────────
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=50,
                   help="HDBSCAN min_cluster_size — smallest group treated as a "
                        "real cluster (main tuning knob; ~0.5-2%% of data). "
                        "Default: 50")
    p.add_argument("--hdbscan-min-samples", type=int, default=None,
                   help="HDBSCAN min_samples — controls noise aggressiveness; "
                        "higher = more points labelled noise. "
                        "Defaults to --hdbscan-min-cluster-size when not set.")
    p.add_argument("--hdbscan-epsilon", type=float, default=0.0,
                   help="HDBSCAN cluster_selection_epsilon — optional distance "
                        "threshold to merge nearby clusters (like DBSCAN eps). "
                        "Default: 0.0 (disabled)")
    p.add_argument("--hdbscan-selection", choices=["eom", "leaf"], default="eom",
                   help="HDBSCAN cluster selection method: "
                        "'eom' (excess of mass, default) finds larger clusters; "
                        "'leaf' finds smaller, more fine-grained clusters.")

    # ── GMM denoising ─────────────────────────────────────────────────────────
    p.add_argument("--confidence", type=float, default=None,
                   help="GMM confidence threshold for denoising (0-1). "
                        "Points with max posterior probability below this value "
                        "are treated as noise/transition-state molecules and labeled -1. "
                        "Recommended: 0.7-0.9. Only applies to GMM. "
                        "If not set, all points are assigned to a cluster.")
    
    # ── Cluster filtering ─────────────────────────────────────────────────────
    p.add_argument("--min-cluster-size", type=int, default=None,
                   help="Minimum cluster size for DBSCAN+GMM. "
                        "Clusters smaller than this are reclassified as noise. "
                        "Only applies to dbscan_gmm method. "
                        "Example: --min-cluster-size 100")

    # ── Feature selection ─────────────────────────────────────────────────────
    p.add_argument("--features", nargs='+', default=None,
                   help="Features to use for clustering (default: all). "
                        "Available: q_all Q6_all LSI_all Sk_all zeta_all. "
                        "Example: --features q_all zeta_all")

    # ── UMAP pre-processing ───────────────────────────────────────────────────
    p.add_argument("--umap", action="store_true", default=False,
                   help="Apply UMAP dimensionality reduction before clustering. "
                        "Reduces scaled features to --umap-n-components dimensions, "
                        "then feeds the embedding into the chosen clustering method. "
                        "Requires: conda install -c conda-forge umap-learn")
    p.add_argument("--umap-n-components", type=int, default=2,
                   help="UMAP target dimensionality (default: 2)")
    p.add_argument("--umap-n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors — controls local vs global structure "
                        "(default: 15; try 5-50)")
    p.add_argument("--umap-min-dist", type=float, default=0.1,
                   help="UMAP min_dist — tightness of clusters in embedding "
                        "(default: 0.1; smaller = tighter)")
    p.add_argument("--umap-metric", type=str, default="euclidean",
                   help="Distance metric for UMAP (default: euclidean)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("-o", "--out_dir", default="./clustering_results",
                   help="Directory for output plots and CSV")

    return p.parse_args()

def run_dbscan_gmm(df_scaled: pd.DataFrame, eps: float = 0.2,
                   min_samples: int = 5, n_components: int = 2,
                   random_state: int = 42, min_cluster_size: int = None) -> np.ndarray:
    """
    Two-stage pipeline:
      Stage 1 — DBSCAN removes low-density noise points (label -1).
      Stage 2 — GMM clusters the clean subset into n_components Gaussians.
      Stage 3 — (Optional) Reclassify clusters smaller than min_cluster_size as noise.
    Points removed by DBSCAN retain label -1 in the final output.
    
    Parameters
    ----------
    min_cluster_size : int, optional
        Minimum number of points required for a GMM cluster to be valid.
        Clusters smaller than this are reclassified as noise (-1).
    """
    print("Stage 1: DBSCAN noise removal …")
    db_labels = run_dbscan(df_scaled, eps=eps, min_samples=min_samples)
    clean_mask = db_labels != -1
    n_noise = (~clean_mask).sum()
    n_clean = clean_mask.sum()
    print(f"  Removed {n_noise:,} noise points  ({100*n_noise/len(db_labels):.1f}%)")
    print(f"  Clean points for GMM: {n_clean:,}  ({100*n_clean/len(db_labels):.1f}%)")

    print("Stage 2: GMM clustering on clean subset …")
    df_clean = df_scaled[clean_mask]
    gmm_labels_clean, probs_clean = run_gmm(df_clean, n_components=n_components,
                                            random_state=random_state)

    # Merge back — noise points stay -1
    final_labels = np.full(len(df_scaled), -1, dtype=int)
    final_labels[clean_mask] = gmm_labels_clean

    # Stage 3: Filter small clusters
    if min_cluster_size is not None and min_cluster_size > 0:
        print(f"Stage 3: Filtering clusters smaller than {min_cluster_size} points …")
        unique_clusters = [c for c in np.unique(final_labels) if c >= 0]
        
        for cluster_id in unique_clusters:
            cluster_size = np.sum(final_labels == cluster_id)
            if cluster_size < min_cluster_size:
                print(f"  Cluster {cluster_id}: {cluster_size:,} points → reclassified as noise")
                final_labels[final_labels == cluster_id] = -1
            else:
                print(f"  Cluster {cluster_id}: {cluster_size:,} points → kept")
        
        # Report final statistics
        n_noise_final = np.sum(final_labels == -1)
        print(f"  Total noise after filtering: {n_noise_final:,} ({100*n_noise_final/len(final_labels):.1f}%)")

    # Silhouette on clean points only
    clean_final_mask = final_labels != -1
    if np.sum(clean_final_mask) > 0:
        clean_labels = final_labels[clean_final_mask]
        if len(np.unique(clean_labels)) > 1:
            sil = silhouette_score(df_scaled[clean_final_mask], clean_labels)
            print(f"  Silhouette (final clean subset): {sil:.4f}")

    return final_labels


def run_hdbscan(df_scaled: pd.DataFrame,
                min_cluster_size: int = 50,
                min_samples: int = None,
                cluster_selection_epsilon: float = 0.0,
                cluster_selection_method: str = "eom") -> np.ndarray:
    """
    HDBSCAN clustering on the scaled feature matrix.

    HDBSCAN (Hierarchical DBSCAN) improves on DBSCAN by:
    - Not requiring an epsilon parameter (uses min_cluster_size instead)
    - Adapting to varying local density across the feature space
    - Returning a stable hierarchy via the condensed cluster tree

    Label convention:
      -1  → noise / transition-state molecules
       0  → cluster 0
       1  → cluster 1  (etc.)

    Parameters
    ----------
    min_cluster_size          : smallest group considered a real cluster
                                (main tuning knob; ~0.5–2% of data is typical)
    min_samples               : controls noise sensitivity — higher = more noise;
                                defaults to min_cluster_size when None
    cluster_selection_epsilon : merge clusters closer than this distance
                                (like DBSCAN eps, but optional)
    cluster_selection_method  : 'eom' (excess of mass, default) or 'leaf'
    """
    t0 = time()
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        n_jobs=-1,
    )
    labels = hdb.fit_predict(df_scaled)
    dt = time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f"HDBSCAN  min_cluster_size={min_cluster_size}  "
          f"min_samples={min_samples}  "
          f"selection={cluster_selection_method}")
    print(f"  Clusters (excl. noise): {n_clusters}")
    print(f"  Noise points          : {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    print(f"  Wall time             : {dt:.2f} s")

    if n_clusters > 1:
        mask  = labels != -1
        score = silhouette_score(df_scaled[mask], labels[mask])
        print(f"  Silhouette score      : {score:.4f}  (non-noise only)")

    return labels


def run_hdbscan_gmm(df_scaled: pd.DataFrame,
                    min_cluster_size: int = 50,
                    min_samples: int = None,
                    cluster_selection_epsilon: float = 0.0,
                    cluster_selection_method: str = "eom",
                    n_components: int = 2,
                    random_state: int = 42) -> np.ndarray:
    """
    Two-stage pipeline:
      Stage 1 — HDBSCAN removes low-density noise points (label -1).
      Stage 2 — GMM clusters the clean subset into n_components Gaussians.

    Advantage over dbscan_gmm: no epsilon parameter to tune — HDBSCAN
    automatically adapts to varying density in the order-parameter space.
    Points removed by HDBSCAN retain label -1 in the final output.
    """
    print("Stage 1: HDBSCAN noise removal …")
    hdb_labels = run_hdbscan(df_scaled,
                              min_cluster_size=min_cluster_size,
                              min_samples=min_samples,
                              cluster_selection_epsilon=cluster_selection_epsilon,
                              cluster_selection_method=cluster_selection_method)

    clean_mask = hdb_labels != -1
    n_noise    = (~clean_mask).sum()
    n_clean    = clean_mask.sum()
    print(f"  Removed {n_noise:,} noise points  ({100*n_noise/len(hdb_labels):.1f}%)")
    print(f"  Clean points for GMM: {n_clean:,}  ({100*n_clean/len(hdb_labels):.1f}%)")

    if n_clean < n_components * 10:
        print("  WARNING: too few clean points for GMM — returning HDBSCAN labels.")
        return hdb_labels

    print("Stage 2: GMM clustering on clean subset …")
    df_clean = df_scaled[clean_mask]
    gmm_labels_clean, _ = run_gmm(df_clean, n_components=n_components,
                                   random_state=random_state)

    # Merge back — noise points stay -1
    final_labels = np.full(len(df_scaled), -1, dtype=int)
    final_labels[clean_mask] = gmm_labels_clean

    # Final silhouette on clean points
    clean_final_mask = final_labels != -1
    if np.sum(clean_final_mask) > 0 and len(np.unique(final_labels[clean_final_mask])) > 1:
        sil = silhouette_score(df_scaled[clean_final_mask],
                               final_labels[clean_final_mask])
        print(f"  Silhouette (final clean subset): {sil:.4f}")

    return final_labels


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load & scale ─────────────────────────────────────────────────────────
    df_raw    = load_order_params(args.mat_file, args.zeta_file, args.n_runs)
    
    # ── Feature selection ────────────────────────────────────────────────────
    all_features = ["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"]
    
    if args.features is not None:
        # Validate feature names
        invalid = [f for f in args.features if f not in all_features]
        if invalid:
            print(f"ERROR: Invalid feature(s): {invalid}")
            print(f"Valid features: {all_features}")
            return
        
        selected_features = args.features
        print(f"\nUsing selected features: {selected_features}")
        print(f"(Excluded: {[f for f in all_features if f not in selected_features]})")
        
        # Select only specified features for clustering
        df_raw_clustering = df_raw[selected_features].copy()
    else:
        selected_features = all_features
        print(f"\nUsing all features: {selected_features}")
        df_raw_clustering = df_raw.copy()
    
    df_scaled = scale_features(df_raw_clustering)

    print(f"\nFeature summary (raw):\n{df_raw_clustering.describe().T[['mean','std','min','max']]}\n")

    # ── Optional UMAP pre-processing ─────────────────────────────────────────
    df_umap = None
    df_cluster_input = df_scaled   # what gets passed to clustering functions

    if args.umap:
        if not HAS_UMAP:
            print("ERROR: --umap requested but umap-learn is not installed.")
            print("  Install with: conda install -c conda-forge umap-learn")
            return
        print("─" * 50)
        df_umap = run_umap(
            df_scaled,
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
        )
        df_cluster_input = df_umap   # clustering runs on the UMAP embedding
        print(f"  Clustering will use UMAP embedding "
              f"({df_umap.shape[1]}-D) instead of raw scaled features.\n")

    results = {}  # method → labels

    # ── Run requested methods ────────────────────────────────────────────────
    if args.method in ("dbscan", "all"):
        print("─" * 50)
        labels_db = run_dbscan(df_cluster_input, eps=args.eps,
                               min_samples=args.min_samples)
        results["dbscan"] = labels_db

    if args.method in ("kmeans", "all"):
        print("─" * 50)
        labels_km = run_kmeans(df_cluster_input, n_clusters=args.n_clusters)
        results["kmeans"] = labels_km

    if args.method in ("dbscan_gmm",):
        print("─" * 50)
        labels_dg = run_dbscan_gmm(df_cluster_input, eps=args.eps,
                                   min_samples=args.min_samples,
                                   n_components=args.n_clusters,
                                   min_cluster_size=args.min_cluster_size)
        results["dbscan_gmm"] = labels_dg

    if args.method in ("hdbscan", "all"):
        print("─" * 50)
        labels_hdb = run_hdbscan(
            df_cluster_input,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_epsilon=args.hdbscan_epsilon,
            cluster_selection_method=args.hdbscan_selection,
        )
        results["hdbscan"] = labels_hdb

    if args.method in ("hdbscan_gmm",):
        print("─" * 50)
        labels_hgm = run_hdbscan_gmm(
            df_cluster_input,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_epsilon=args.hdbscan_epsilon,
            cluster_selection_method=args.hdbscan_selection,
            n_components=args.n_clusters,
        )
        results["hdbscan_gmm"] = labels_hgm

    if args.method in ("gmm", "all"):
        print("─" * 50)
        labels_gm, probs_gm = run_gmm(df_cluster_input, n_components=args.n_clusters)

        # ── Optional confidence-based denoising ───────────────────────────
        if args.confidence is not None:
            max_prob    = probs_gm.max(axis=1)
            noise_mask  = max_prob < args.confidence
            labels_gm_denoised = labels_gm.copy()
            labels_gm_denoised[noise_mask] = -1
            n_noise     = noise_mask.sum()
            n_kept      = (~noise_mask).sum()
            clean_labels = labels_gm_denoised[~noise_mask]
            if len(np.unique(clean_labels)) > 1:
                sil_clean = silhouette_score(
                    df_cluster_input[~noise_mask], clean_labels)
            else:
                sil_clean = float("nan")
            print(f"  Confidence threshold : {args.confidence}")
            print(f"  Points kept          : {n_kept:,}  ({100*n_kept/len(labels_gm):.1f}%)")
            print(f"  Points removed (noise): {n_noise:,}  ({100*n_noise/len(labels_gm):.1f}%)")
            print(f"  Silhouette (clean)   : {sil_clean:.4f}")
            results["gmm_denoised"] = labels_gm_denoised

        results["gmm"] = labels_gm

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    for method, labels in results.items():
        plot_scatter(df_scaled, labels, method, args.out_dir, df_raw=df_raw)
        plot_zeta_distribution(df_raw, labels, method, args.out_dir)
        plot_all_distributions(df_raw, labels, method, args.out_dir)
        plot_pairplot(df_scaled, labels, method, args.out_dir)
        if df_umap is not None:
            plot_umap_embedding(df_umap, labels, df_raw, method, args.out_dir)
        print(f"  Saved plots for {method}")

    # ── Save labels ───────────────────────────────────────────────────────────
    out = df_raw.copy()
    for method, labels in results.items():
        out[f"label_{method}"] = labels
    csv_path = os.path.join(args.out_dir, "cluster_labels.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nLabels saved to {csv_path}")
    print(f"All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()