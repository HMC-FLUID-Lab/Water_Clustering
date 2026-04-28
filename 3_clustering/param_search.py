#!/usr/bin/env python3
"""
param_search.py
================
DBSCAN denoising parameter search.

Since DBSCAN is used purely for noise removal, the silhouette score
is computed between the noise group (label=-1) and the clean-data group
(all non-noise points merged as one).  Higher silhouette → cleaner separation.

Output
------
  dbscan_param_heatmap.png   — heatmap: x=eps, y=min_samples,
                               colour=noise %, text=silhouette score
  param_search_summary.csv   — raw results table

Usage
-----
    python clustering/param_search.py \\
        -m  /abs/path/OrderParam.mat \\
        -z  /abs/path/OrderParamZeta.mat \\
        -n  1 \\
        -o  ./clustering/param_search_results

Speed notes
-----------
  --fit-sample-size 20000   sub-samples the data for the grid search (fast).
                             Set to 0 to use all data (can take hours for >50k).
  --eps-steps 10            number of eps values to test in [eps-min, eps-max].
"""

# ── thread limits (must come before numpy) ────────────────────────────────────
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"]      = "8"
os.environ["OMP_NUM_THREADS"]      = "8"
os.environ["NUMEXPR_NUM_THREADS"]  = "8"

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import (
    set_default_plot,
    style_dbscan_param_heatmap_axes,
    dbscan_param_heatmap_figure_caption,
)

set_default_plot()
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_order_params(mat_file: str, zeta_file: str, n_runs: int) -> pd.DataFrame:
    """
    Load structural order parameters.  Handles two .mat formats:
      - Cell array (dtype=object): old format, iterate n_runs cells.
      - 2-D array  (dtype=float, shape=(n_frames, n_molecules)): new format,
        flatten the entire array regardless of n_runs.
    """
    water  = loadmat(mat_file)
    water1 = loadmat(zeta_file)

    def flatten(d, key):
        arr = d[key]
        if arr.dtype == object:
            out = []
            for i in range(n_runs):
                out.extend(np.asarray(arr[i]).ravel())
            return np.array(out, dtype=float)
        else:
            return arr.ravel().astype(float)

    df = pd.DataFrame({
        "q_all"    : flatten(water,  "q_all"),
        "Q6_all"   : flatten(water,  "Q6_all"),
        "LSI_all"  : flatten(water,  "LSI_all"),
        "Sk_all"   : flatten(water,  "Sk_all"),
        "zeta_all" : flatten(water1, "zeta_all"),
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    fmt = "cell array" if water["q_all"].dtype == object else "2-D array"
    print(f"Loaded {len(df):,} water molecules ({fmt} format).")
    return df


def scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(df.values)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SILHOUETTE — DENOISING CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def silhouette_denoising(X: np.ndarray, labels: np.ndarray,
                         sample_size: int = 10_000) -> float:
    """
    Silhouette treating noise(-1) as group-0 and all signal as group-1.
    Returns NaN if either group has fewer than 5 points.
    """
    n_noise  = int(np.sum(labels == -1))
    n_signal = int(np.sum(labels != -1))
    if n_noise < 5 or n_signal < 5:
        return np.nan
    denoising_labels = np.where(labels == -1, 0, 1)
    try:
        ss = min(sample_size, len(X))
        return float(silhouette_score(X, denoising_labels,
                                      sample_size=ss, random_state=42))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DBSCAN GRID SEARCH  →  HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan_grid(X: np.ndarray, eps_range, min_samples_range,
                    sample_size: int = 10_000):
    ne, nm = len(eps_range), len(min_samples_range)
    sil_mat    = np.full((nm, ne), np.nan)
    noise_mat  = np.full((nm, ne), np.nan)
    nclust_mat = np.zeros((nm, ne), dtype=int)

    print(f"\nDBSCAN grid: {ne} eps × {nm} min_samples = {ne*nm} combos  "
          f"(ball_tree, n_jobs=-1)")

    with tqdm(total=ne * nm, desc="DBSCAN") as pbar:
        for i, ms in enumerate(min_samples_range):
            for j, eps in enumerate(eps_range):
                labels = DBSCAN(
                    eps=eps, min_samples=ms,
                    algorithm="ball_tree", n_jobs=-1
                ).fit_predict(X)

                n_noise   = int(np.sum(labels == -1))
                n_clust   = len(set(labels)) - (1 if -1 in labels else 0)
                sil       = silhouette_denoising(X, labels, sample_size)
                noise_pct = n_noise / len(labels) * 100

                sil_mat[i, j]    = sil
                noise_mat[i, j]  = n_noise / len(labels)
                nclust_mat[i, j] = n_clust

                tqdm.write(f"  DBSCAN eps={eps:.3f} ms={ms:3d} | "
                           f"noise={noise_pct:5.1f}%  sil={sil:.4f}" if not np.isnan(sil)
                           else f"  DBSCAN eps={eps:.3f} ms={ms:3d} | "
                                f"noise={noise_pct:5.1f}%  sil=NaN")
                pbar.update(1)

    return sil_mat, noise_mat, nclust_mat


def plot_dbscan_heatmap(sil_mat, noise_mat, nclust_mat,
                        eps_range, min_samples_range, out_dir: str):
    """
    Single heatmap:
      colour = noise percentage
      text   = silhouette score (NaN shown in grey)
      blue border = best silhouette cell
    """
    ne    = len(eps_range)
    nm    = len(min_samples_range)
    eps_labels = [f"{e:.3f}" for e in eps_range]
    ms_labels  = [str(m) for m in min_samples_range]

    cell_w  = max(1.0, 8.0 / ne)
    cell_h  = max(0.6, 6.0 / nm)
    figw    = max(10, ne * cell_w + 3)
    figh    = max(6,  nm * cell_h + 3)

    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_facecolor("white")

    noise_pct_mat = noise_mat * 100

    # mask NaN silhouette cells so they render as grey
    sil_display = np.where(np.isnan(sil_mat), -0.1, sil_mat)
    cmap = plt.cm.plasma.copy()
    cmap.set_under("lightgrey")

    im = ax.imshow(sil_display, aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=1.0, origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Silhouette Score")
    cbar.ax.tick_params()

    # ── text overlay: noise percentage ───────────────────────────────────────
    for i in range(nm):
        for j in range(ne):
            sil   = sil_mat[i, j]
            noise = noise_pct_mat[i, j]
            txt_col = "black" if (not np.isnan(sil) and sil > 0.6) else "white"
            if np.isnan(sil):
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        color="dimgrey", fontsize=9)
            else:
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        color=txt_col, fontweight="bold", fontsize=9)

    # ── find best silhouette ───────────────────────────────────────────────────
    best_eps, best_ms, best_sil = None, None, np.nan
    if not np.all(np.isnan(sil_mat)):
        bi       = np.unravel_index(np.nanargmax(sil_mat), sil_mat.shape)
        best_eps = eps_range[bi[1]]
        best_ms  = min_samples_range[bi[0]]
        best_sil = float(sil_mat[bi])

    ax.set_xticks(np.arange(ne))
    ax.set_xticklabels(eps_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(nm))
    ax.set_yticklabels(ms_labels)

    style_dbscan_param_heatmap_axes(ax, cbar)
    fig.text(
        0.5,
        0.02,
        dbscan_param_heatmap_figure_caption(best_eps, best_ms, best_sil),
        ha="center",
        fontsize=11,
        color="0.35",
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "dbscan_param_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return best_eps, best_ms, best_sil



# ─────────────────────────────────────────────────────────────────────────────
# 4.  HDBSCAN GRID SEARCH  →  HEATMAP  (matching DBSCAN visual style)
# ─────────────────────────────────────────────────────────────────────────────

def run_hdbscan_grid(X: np.ndarray, mcs_range, min_samples_range,
                     sample_size: int = 10_000):
    """
    2D grid: min_cluster_size (x) × min_samples (y).
    min_samples here controls noise sensitivity independently of mcs.
    """
    nc, nm = len(mcs_range), len(min_samples_range)
    sil_mat    = np.full((nm, nc), np.nan)
    noise_mat  = np.full((nm, nc), np.nan)
    nclust_mat = np.zeros((nm, nc), dtype=int)

    print(f"\nHDBSCAN grid: {nc} min_cluster_size × {nm} min_samples = "
          f"{nc*nm} combos")

    with tqdm(total=nc * nm, desc="HDBSCAN") as pbar:
        for i, ms in enumerate(min_samples_range):
            for j, mcs in enumerate(mcs_range):
                labels = HDBSCAN(
                    min_cluster_size=int(mcs),
                    min_samples=int(ms),
                    cluster_selection_method="eom",
                    n_jobs=-1,
                ).fit_predict(X)

                n_noise   = int(np.sum(labels == -1))
                n_clust   = len(set(labels)) - (1 if -1 in labels else 0)
                sil       = silhouette_denoising(X, labels, sample_size)
                noise_pct = n_noise / len(labels) * 100

                sil_mat[i, j]    = sil
                noise_mat[i, j]  = n_noise / len(labels)
                nclust_mat[i, j] = n_clust

                tqdm.write(f"  HDBSCAN mcs={mcs:5d} ms={ms:3d} | "
                           f"noise={noise_pct:5.1f}%  sil={sil:.4f}" if not np.isnan(sil)
                           else f"  HDBSCAN mcs={mcs:5d} ms={ms:3d} | "
                                f"noise={noise_pct:5.1f}%  sil=NaN")
                pbar.update(1)

    return sil_mat, noise_mat, nclust_mat


def _draw_heatmap(ax, sil_mat, noise_mat, x_labels, y_labels,
                  x_title, y_title, cmap_name="RdBu"):
    """Shared heatmap renderer used by both DBSCAN and HDBSCAN plots."""
    from matplotlib.colors import TwoSlopeNorm
    nm, ne = sil_mat.shape

    vmax = max(abs(np.nanmin(sil_mat)), abs(np.nanmax(sil_mat)), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")

    im = ax.imshow(sil_mat, aspect="auto", cmap=cmap, norm=norm, origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Silhouette Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.axhline(0, color="black", lw=0.8, alpha=0.6)

    noise_pct_mat = noise_mat * 100
    for i in range(nm):
        for j in range(ne):
            sil   = sil_mat[i, j]
            noise = noise_pct_mat[i, j]
            txt_col = "black" if (not np.isnan(sil) and sil > 0.6) else "white"
            if np.isnan(sil):
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        fontsize=9, color="dimgrey")
            else:
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        fontsize=9, color=txt_col, fontweight="bold")

    ax.set_xticks(np.arange(ne))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(nm))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(x_title, fontsize=11)
    ax.set_ylabel(y_title, fontsize=11)
    ax.set_title("colour = silhouette score   ·   text = noise removed (%)",
                 fontsize=10, pad=6)
    return im


def plot_hdbscan_heatmap(sil_mat, noise_mat, nclust_mat,
                          mcs_range, min_samples_range, out_dir: str,
                          cmap_name: str = "RdBu"):
    nc, nm = len(mcs_range), len(min_samples_range)
    cell_w = max(1.2, 9.0 / nc)
    cell_h = max(0.7, 7.0 / nm)
    figw   = max(10, nc * cell_w + 3)
    figh   = max(6,  nm * cell_h + 3)

    fig, ax = plt.subplots(figsize=(figw, figh))
    _draw_heatmap(
        ax, sil_mat, noise_mat,
        x_labels=[str(m) for m in mcs_range],
        y_labels=[str(m) for m in min_samples_range],
        x_title="min_cluster_size",
        y_title="min_samples  (noise sensitivity)",
        cmap_name=cmap_name,
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "hdbscan_param_heatmap.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SFVS METRIC  (Structure-Factor Validation Score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sfvs(labels: np.ndarray, df_raw: pd.DataFrame) -> float:
    """
    Structure-Factor Validation Score (SFVS).

    Evaluates how well the clustering separates the two physical water states
    using the order-parameter distributions rather than just cluster geometry.

    TODO: fill in formula once SFVS_metric.md is available.
          Placeholder returns NaN so downstream code stays robust.

    Parameters
    ----------
    labels  : int array of cluster labels (-1 = noise, 0 = DNLS, 1 = LFTS)
    df_raw  : DataFrame with raw (un-scaled) order parameters

    Returns
    -------
    score : float in [0, 1], or NaN if undefined
    """
    return np.nan


def evaluate_sfvs_grid(X: np.ndarray, df_raw: pd.DataFrame,
                        labels_list: list, param_labels: list) -> pd.DataFrame:
    """
    Compute SFVS for a list of (parameter, labels) pairs and return a summary
    DataFrame.  Intended to be called after the DBSCAN/HDBSCAN grid search.
    """
    rows = []
    for param, labels in zip(param_labels, labels_list):
        score = compute_sfvs(labels, df_raw)
        rows.append({**param, "sfvs": score})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DBSCAN / HDBSCAN denoising parameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-m", "--mat_file",  required=True,
                   help="Absolute path to OrderParam .mat file")
    p.add_argument("-z", "--zeta_file", required=True,
                   help="Absolute path to OrderParamZeta .mat file")
    p.add_argument("-n", "--n_runs",    type=int, default=1,
                   help="Number of MD runs (only used for old cell-array format)")

    p.add_argument("--features", nargs="+",
                   default=["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"],
                   help="Features to use (MinMax-scaled before clustering)")

    p.add_argument("--eps-list", type=float, nargs="+", default=None,
                   help="Explicit eps values to test (overrides --eps-min/max/steps). "
                        "E.g. --eps-list 0.02 0.04 0.08 0.16")
    p.add_argument("--eps-min",    type=float, default=0.05)
    p.add_argument("--eps-max",    type=float, default=0.50)
    p.add_argument("--eps-steps",  type=int,   default=10)
    p.add_argument("--min-samples-list", type=int, nargs="+",
                   default=[3, 5, 10, 20, 30, 50])

    p.add_argument("--hdbscan-mcs-list", type=int, nargs="+",
                   default=[10, 20, 50, 100, 200, 500],
                   help="min_cluster_size values for HDBSCAN grid")
    p.add_argument("--hdbscan-min-samples-list", type=int, nargs="+",
                   default=[3, 5, 10, 20, 50],
                   help="min_samples values for HDBSCAN grid")
    p.add_argument("--skip-dbscan",   action="store_true", default=False)
    p.add_argument("--skip-hdbscan",  action="store_true", default=False)

    p.add_argument("--fit-sample-size", type=int, default=20_000,
                   help="Points subsampled for fitting during the grid search. "
                        "Set 0 to use all data (slow for >50k points).")
    p.add_argument("--sample-size", type=int, default=10_000,
                   help="Points subsampled for silhouette_score computation.")

    p.add_argument("-o", "--out_dir", default="./param_search_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("DBSCAN DENOISING PARAMETER SEARCH")
    print("=" * 70)
    print(f"Output dir       : {args.out_dir}")
    print(f"Features         : {args.features}")
    print(f"Fit sample size  : {args.fit_sample_size:,}  (0 = full data)")
    print(f"Silhouette sample: {args.sample_size:,}")
    print(f"Stop on NaN      : yes")
    print("=" * 70)

    df_raw = load_order_params(args.mat_file, args.zeta_file, args.n_runs)

    invalid = [f for f in args.features if f not in df_raw.columns]
    if invalid:
        raise ValueError(f"Invalid feature(s): {invalid}.  "
                         f"Available: {list(df_raw.columns)}")

    X = scale_features(df_raw[args.features])
    print(f"\nClustering on {X.shape[1]} features, {X.shape[0]:,} total samples.")

    fit_n = args.fit_sample_size
    if fit_n > 0 and X.shape[0] > fit_n:
        idx   = np.random.default_rng(42).choice(X.shape[0], size=fit_n, replace=False)
        X_fit = X[idx]
        print(f"  Using subsample of {fit_n:,} points for grid search "
              f"(pass --fit-sample-size 0 to use all data).\n")
    else:
        X_fit = X
        print(f"  Using full {X.shape[0]:,} points for grid search.\n")

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    if not args.skip_dbscan:
        if args.eps_list:
            eps_range = np.array(sorted(args.eps_list))
        else:
            eps_range = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
        min_samples_range = sorted(set(args.min_samples_list))

        sil_db, noise_db, nclust_db = run_dbscan_grid(
            X_fit, eps_range, min_samples_range, args.sample_size)
        plot_dbscan_heatmap(
            sil_db, noise_db, nclust_db, eps_range, min_samples_range, args.out_dir)

        rows_db = []
        for i, ms in enumerate(min_samples_range):
            for j, eps in enumerate(eps_range):
                rows_db.append({
                    "eps": round(float(eps), 6),
                    "min_samples": ms,
                    "silhouette": sil_db[i, j],
                    "noise_pct": round(float(noise_db[i, j]) * 100, 2),
                    "n_clusters": nclust_db[i, j],
                })
        csv_db = os.path.join(args.out_dir, "param_search_summary.csv")
        pd.DataFrame(rows_db).to_csv(csv_db, index=False)
        print(f"  Saved CSV: {csv_db}")

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    if not args.skip_hdbscan:
        mcs_range = sorted(set(args.hdbscan_mcs_list))
        hms_range = sorted(set(args.hdbscan_min_samples_list))

        sil_hdb, noise_hdb, nclust_hdb = run_hdbscan_grid(
            X_fit, mcs_range, hms_range, args.sample_size)
        plot_hdbscan_heatmap(
            sil_hdb, noise_hdb, nclust_hdb, mcs_range, hms_range, args.out_dir)

        rows_hdb = []
        for i, ms in enumerate(hms_range):
            for j, mcs in enumerate(mcs_range):
                rows_hdb.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "silhouette": sil_hdb[i, j],
                    "noise_pct": round(float(noise_hdb[i, j]) * 100, 2),
                    "n_clusters": nclust_hdb[i, j],
                })
        csv_hdb = os.path.join(args.out_dir, "hdbscan_param_summary.csv")
        pd.DataFrame(rows_hdb).to_csv(csv_hdb, index=False)
        print(f"  Saved CSV: {csv_hdb}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
