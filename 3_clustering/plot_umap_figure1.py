#!/usr/bin/env python3
"""
plot_umap_figure1.py
====================
Generates Figure 1 — UMAP Embedding by Cluster Label.

  • 2D UMAP projection at the representative supercooling temperature
  • Colored by HDBSCAN+GMM cluster:  LFTS = blue,  DNLS = red,  noise = gray
  • Two panels:  TIP4P/2005 (left)  |  TIP5P (right)
  • Population counts shown in each panel legend

Usage (minimal — uses defaults for both models at T=-20):
  python plot_umap_figure1.py

Usage (explicit):
  python plot_umap_figure1.py \\
    --tip4p2005-mat   /path/to/OrderParam_tip4p2005_T-20_Run01.mat \\
    --tip4p2005-zeta  /path/to/OrderParamZeta_tip4p2005_T-20_Run01.mat \\
    --tip4p2005-runs  20 \\
    --tip4p2005-temp  -20 \\
    --tip5p-mat       /path/to/OrderParam_tip5p_T-20_Run01.mat \\
    --tip5p-zeta      /path/to/OrderParamZeta_tip5p_T-20_Run01.mat \\
    --tip5p-runs      20 \\
    --tip5p-temp      -20 \\
    --output-dir      ./umap_figure1
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Import reusable functions from the existing pipeline ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from water_clustering import (
    load_order_params,
    scale_features,
    run_umap,
    run_dbscan_gmm,
    run_hdbscan_gmm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────
COLOR_LFTS  = "#2166ac"   # blue
COLOR_DNLS  = "#d62728"   # red
COLOR_NOISE = "#aaaaaa"   # gray


def _assign_physical_labels(labels: np.ndarray, df_raw) -> np.ndarray:
    """
    Relabel so that cluster 1 = LFTS (higher mean zeta) and cluster 0 = DNLS.
    Noise (-1) is unchanged.
    """
    if "zeta_all" not in df_raw.columns:
        return labels                      # can't determine — return as-is

    zeta = df_raw["zeta_all"].values
    unique_clean = [l for l in np.unique(labels) if l != -1]

    if len(unique_clean) < 2:
        return labels

    mean_zeta = {l: zeta[labels == l].mean() for l in unique_clean}
    # The cluster with HIGHER mean zeta is LFTS → should be label 1
    lfts_id = max(mean_zeta, key=mean_zeta.get)
    dnls_id = min(mean_zeta, key=mean_zeta.get)

    remapped = labels.copy()
    remapped[labels == lfts_id] = 1
    remapped[labels == dnls_id] = 0
    return remapped


def _color_for_label(lbl: int) -> str:
    if lbl == -1:
        return COLOR_NOISE
    if lbl == 1:
        return COLOR_LFTS
    return COLOR_DNLS


def _name_for_label(lbl: int) -> str:
    if lbl == -1:
        return "Noise"
    if lbl == 1:
        return "LFTS / S-state"
    return "DNLS / ρ-state"


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(mat_file, zeta_file, n_runs,
                  n_neighbors, min_dist, method,
                  # hdbscan_gmm params
                  hdbscan_min_cluster_size=200,
                  # dbscan_gmm params
                  dbscan_eps=0.2, dbscan_min_samples=5):
    """
    Load → scale → UMAP → clustering (hdbscan_gmm or dbscan_gmm)
    → physically-consistent labels.
    Returns (df_umap, labels, df_raw).
    """
    print(f"\n  Loading: {os.path.basename(mat_file)}")
    df_raw    = load_order_params(mat_file, zeta_file, n_runs)
    df_scaled = scale_features(df_raw)

    print("  Running UMAP …")
    df_umap = run_umap(df_scaled, n_components=2,
                       n_neighbors=n_neighbors, min_dist=min_dist)

    if method == "hdbscan_gmm":
        print("  Running HDBSCAN+GMM …")
        labels = run_hdbscan_gmm(df_umap,
                                  min_cluster_size=hdbscan_min_cluster_size)
    else:  # dbscan_gmm
        print(f"  Running DBSCAN+GMM  (eps={dbscan_eps}, min_samples={dbscan_min_samples}) …")
        labels = run_dbscan_gmm(df_umap,
                                eps=dbscan_eps,
                                min_samples=dbscan_min_samples)

    labels = _assign_physical_labels(labels, df_raw)

    n_lfts  = (labels == 1).sum()
    n_dnls  = (labels == 0).sum()
    n_noise = (labels == -1).sum()
    print(f"  LFTS={n_lfts:,}  DNLS={n_dnls:,}  Noise={n_noise:,}")

    return df_umap, labels, df_raw


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _draw_panel(ax, df_umap, labels, title):
    """Draw one UMAP panel with cluster coloring and population legend."""
    unique = sorted(set(labels))

    # Draw noise first (bottom layer), then clusters on top
    draw_order = ([l for l in unique if l == -1] +
                  [l for l in unique if l != -1])

    legend_handles = []
    for lbl in draw_order:
        mask  = labels == lbl
        color = _color_for_label(lbl)
        alpha = 0.25 if lbl == -1 else 0.45
        size  = 2    if lbl == -1 else 3

        ax.scatter(df_umap.loc[mask, "umap_0"],
                   df_umap.loc[mask, "umap_1"],
                   s=size, alpha=alpha, color=color, rasterized=True,
                   linewidths=0)

        patch = mpatches.Patch(color=color,
                               label=_name_for_label(lbl))
        legend_handles.append(patch)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(handles=legend_handles, fontsize=9,
              frameon=True, framealpha=0.9, edgecolor="black",
              markerscale=4, loc="best")


def plot_figure1(tip4p_umap, tip4p_labels,
                 tip5p_umap, tip5p_labels,
                 tip4p_temp, tip5p_temp,
                 output_dir, method="hdbscan_gmm"):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"wspace": 0.30}
    )

    _draw_panel(ax_left,  tip4p_umap, tip4p_labels,
                f"TIP4P/2005  (T = {tip4p_temp:+.0f}°C)")
    _draw_panel(ax_right, tip5p_umap, tip5p_labels,
                f"TIP5P  (T = {tip5p_temp:+.0f}°C)")

    method_label = method.upper().replace("_", "+")
    fig.suptitle(
        f"UMAP Embedding — {method_label} Cluster Assignment\n"
        "LFTS (S-state / tetrahedral)  vs  DNLS (ρ-state / disordered)",
        fontsize=12, y=1.01
    )

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"figure1_umap_clusters_{method}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAM_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "order_params"))

def parse_args():
    p = argparse.ArgumentParser(
        description="Figure 1 — UMAP embedding by cluster label (TIP4P/2005 vs TIP5P)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # TIP4P/2005
    g4 = p.add_argument_group("TIP4P/2005")
    g4.add_argument("--tip4p2005-mat",
                    default=f"{_PARAM_DIR}/OrderParam_tip4p2005_T-20_Run01.mat")
    g4.add_argument("--tip4p2005-zeta",
                    default=f"{_PARAM_DIR}/OrderParamZeta_tip4p2005_T-20_Run01.mat")
    g4.add_argument("--tip4p2005-runs", type=int, default=20)
    g4.add_argument("--tip4p2005-temp", type=float, default=-20.0,
                    help="Temperature label for plot title (°C)")

    # TIP5P
    g5 = p.add_argument_group("TIP5P")
    g5.add_argument("--tip5p-mat",
                    default=f"{_PARAM_DIR}/OrderParam_tip5p_T-20_Run01.mat")
    g5.add_argument("--tip5p-zeta",
                    default=f"{_PARAM_DIR}/OrderParamZeta_tip5p_T-20_Run01.mat")
    g5.add_argument("--tip5p-runs", type=int, default=20)
    g5.add_argument("--tip5p-temp", type=float, default=-20.0,
                    help="Temperature label for plot title (°C)")

    # Method
    p.add_argument("--method",
                   choices=["hdbscan_gmm", "dbscan_gmm"],
                   default="hdbscan_gmm",
                   help="Clustering method to use after UMAP")

    # UMAP tuning
    p.add_argument("--umap-n-neighbors",  type=int,   default=30)
    p.add_argument("--umap-min-dist",     type=float, default=0.05)

    # HDBSCAN+GMM tuning
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=200,
                   help="HDBSCAN min_cluster_size (higher = less noise)")

    # DBSCAN+GMM tuning
    p.add_argument("--dbscan-eps",         type=float, default=0.2,
                   help="DBSCAN epsilon radius in UMAP space")
    p.add_argument("--dbscan-min-samples", type=int,   default=5,
                   help="DBSCAN min_samples")

    p.add_argument("--output-dir", default="./umap_figure1")
    return p.parse_args()


def main():
    args = parse_args()

    method_label = args.method.upper().replace("_", "+")
    print("=" * 65)
    print(f" FIGURE 1 — UMAP Embedding by Cluster Label")
    print(f" Method: {method_label}  |  Colors: LFTS=blue  DNLS=red")
    print("=" * 65)

    shared_kw = dict(
        n_neighbors             = args.umap_n_neighbors,
        min_dist                = args.umap_min_dist,
        method                  = args.method,
        hdbscan_min_cluster_size= args.hdbscan_min_cluster_size,
        dbscan_eps              = args.dbscan_eps,
        dbscan_min_samples      = args.dbscan_min_samples,
    )

    # ── TIP4P/2005 ────────────────────────────────────────────────────────────
    print("\n── TIP4P/2005 ──────────────────────────────────────────────")
    tip4p_umap, tip4p_labels, _ = run_condition(
        args.tip4p2005_mat, args.tip4p2005_zeta, args.tip4p2005_runs,
        **shared_kw,
    )

    # ── TIP5P ─────────────────────────────────────────────────────────────────
    print("\n── TIP5P ───────────────────────────────────────────────────")
    tip5p_umap, tip5p_labels, _ = run_condition(
        args.tip5p_mat, args.tip5p_zeta, args.tip5p_runs,
        **shared_kw,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n── Plotting ────────────────────────────────────────────────")
    plot_figure1(
        tip4p_umap, tip4p_labels,
        tip5p_umap, tip5p_labels,
        tip4p_temp = args.tip4p2005_temp,
        tip5p_temp = args.tip5p_temp,
        output_dir = args.output_dir,
        method     = args.method,
    )

    print("\n" + "=" * 65)
    print(f" DONE — output in: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
