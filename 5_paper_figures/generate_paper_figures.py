#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generate all figures needed to complete the four incomplete sections of
Paper_WaterMLClustering.md.

Sections
--------
  C.2  Minimal feature set: q and ζ only
         fig_c2_qzeta_sk_comparison.png   — per-cluster S(k): 5-feat vs q+ζ
         fig_c2_qzeta_sk_per_cluster.png  — q+ζ per-cluster S(k) alone

  C.3  Temperature dependence of cluster populations
         fig_c3_lfts_fraction_vs_temp.png — LFTS fraction s vs T
         fig_c3_sk_multitemp.png          — S(k) overlay across temperatures

  C.4  Optimal cluster number via BIC / AIC / silhouette
         fig_c4_cluster_number.png        — BIC, AIC, silhouette vs k

  C.5  Model comparison: TIP5P vs TIP4P/2005
         fig_c5_model_comparison_sk.png       — per-cluster S(k) side-by-side
         fig_c5_model_comparison_sk_zeta.png  — S(k,ζ) contour side-by-side

Usage
-----
  python generate_paper_figures.py                    # all sections
  python generate_paper_figures.py --sections c3 c4  # fast (no trajectories)
  python generate_paper_figures.py --no-cache         # recompute S(k) even if cached
  python generate_paper_figures.py --n-frames 5       # use fewer frames (faster)
"""

# ── thread limits — must come before numpy ────────────────────────────────────
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"]      = "8"
os.environ["OMP_NUM_THREADS"]      = "8"
os.environ["NUMEXPR_NUM_THREADS"]  = "8"

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from time import time

warnings.filterwarnings("ignore")

# ─── Resolve repo root (parent of this stage) and stage modules ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _sub in ("4_structure_factor", "3_clustering"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_style import set_default_plot, scale_density_plot_y, probability_density_ylabel

set_default_plot()

try:
    from structure_factor_bycluster import (
        load_trajectory,
        compute_per_cluster_structure_factor,
    )
    _HAS_SF = True
except ImportError as _e:
    _HAS_SF = False
    print(f"[WARN] structure_factor_bycluster not importable: {_e}")
    print("       Sections C.2 and C.5 require mdtraj — skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (edit these if your data lives elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

PARAM_DATA     = os.path.join(_ROOT, "data", "order_params")
SIM_DATA_4P    = os.path.join(_ROOT, "data", "simulations", "tip4p2005")
SIM_DATA_5P    = os.path.join(_ROOT, "data", "simulations", "tip5p")
SIM_DATA_SWM   = os.path.join(_ROOT, "data", "simulations", "swm4ndp")
CLUSTER_DIR    = os.path.join(_ROOT, "results", "clustering")
CS_DIR         = os.path.join(_ROOT, "results", "clustering", "cluster_labels_matrices")

# ── Cluster label flat CSVs (one row per molecule, all runs concatenated) ────
LABELS_4P_T10  = os.path.join(CLUSTER_DIR, "tip4p2005_T-10_dbscan_gmm",  "cluster_labels.csv")
LABELS_4P_T20  = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_dbscan_gmm",  "cluster_labels.csv")
LABELS_4P_T30  = os.path.join(CLUSTER_DIR, "tip4p2005_T-30_dbscan_gmm",  "cluster_labels.csv")
LABELS_4P_QZETA= os.path.join(CLUSTER_DIR, "tip4p2005_T-20_dbscan_gmm_q_zeta", "cluster_labels.csv")

# ── ADD NEW TEMPERATURES HERE (Step 2 outputs) ───────────────────────────────
LABELS_4P_T0   = os.path.join(CLUSTER_DIR, "tip4p2005_T0_dbscan_gmm",   "cluster_labels.csv")
LABELS_4P_TP10 = os.path.join(CLUSTER_DIR, "tip4p2005_T10_dbscan_gmm",  "cluster_labels.csv")
LABELS_4P_TP20 = os.path.join(CLUSTER_DIR, "tip4p2005_T20_dbscan_gmm",  "cluster_labels.csv")

# ── GMM-only (no DBSCAN noise removal) for C.3 LFTS fraction ─────────────────
LABELS_4P_T10_GMM  = os.path.join(CLUSTER_DIR, "tip4p2005_T-10_gmm",  "cluster_labels.csv")
LABELS_4P_T20_GMM  = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_gmm",  "cluster_labels.csv")
LABELS_4P_T30_GMM  = os.path.join(CLUSTER_DIR, "tip4p2005_T-30_gmm",  "cluster_labels.csv")
LABELS_4P_T0_GMM   = os.path.join(CLUSTER_DIR, "tip4p2005_T0_gmm",   "cluster_labels.csv")
LABELS_4P_TP10_GMM = os.path.join(CLUSTER_DIR, "tip4p2005_T10_gmm",  "cluster_labels.csv")
LABELS_4P_TP20_GMM = os.path.join(CLUSTER_DIR, "tip4p2005_T20_gmm",  "cluster_labels.csv")

# ── Cluster label matrices (one row per trajectory frame) ────────────────────
MAT_4P_T10     = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T-10_dbscan_gmm.csv")
MAT_4P_T20     = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv")
MAT_4P_T30     = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T-30_dbscan_gmm.csv")
MAT_4P_QZETA   = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm_q_zeta.csv")
MAT_5P_T20     = os.path.join(CS_DIR, "cluster_labels_matrix_tip5p_T-20_dbscan_gmm.csv")

# ── ADD NEW TEMPERATURE MATRICES HERE (Step 3 outputs) ───────────────────────
# Uncomment each line after running Steps 1–3 for that temperature:
MAT_4P_T0      = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T0_dbscan_gmm.csv")
MAT_4P_TP10    = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T10_dbscan_gmm.csv")
MAT_4P_TP20    = os.path.join(CS_DIR, "cluster_labels_matrix_tip4p2005_T20_dbscan_gmm.csv")

# ── Trajectory files ─────────────────────────────────────────────────────────
DCD_4P = {
    # Positive temperatures — uncomment after Steps 1-3 are complete:
    20: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T20_N1024_Run01_0.dcd"),
    10: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T10_N1024_Run01_0.dcd"),
     0: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T0_N1024_Run01_0.dcd"),
    # Negative temperatures — already available:
    -10: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T-10_N1024_Run01_0.dcd"),
    -20: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T-20_N1024_Run01_0.dcd"),
    -30: os.path.join(SIM_DATA_4P, "dcd_tip4p2005_T-30_N1024_Run01_0.dcd"),
}
PDB_4P = {
    # Positive temperatures — uncomment after Steps 1-3 are complete:
    20: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T20_N1024_Run01.pdb"),
    10: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T10_N1024_Run01.pdb"),
     0: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T0_N1024_Run01.pdb"),
    # Negative temperatures — already available:
    -10: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T-10_N1024_Run01.pdb"),
    -20: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T-20_N1024_Run01.pdb"),
    -30: os.path.join(SIM_DATA_4P, "inistate_tip4p2005_T-30_N1024_Run01.pdb"),
}
DCD_5P = {-20: os.path.join(SIM_DATA_5P, "dcd_tip5p_T-20_N1024_Run01_0.dcd")}
PDB_5P = {-20: os.path.join(SIM_DATA_5P, "inistate_tip5p_T-20_N1024_Run01.pdb")}

DCD_SWM = {-20: os.path.join(SIM_DATA_SWM, "dcd_swm4ndp_T-20_N1024_Run01_0.dcd")}
PDB_SWM = {-20: os.path.join(SIM_DATA_SWM, "inistate_swm4ndp_T-20_N1024_Run01.pdb")}
MAT_SWM_T20 = os.path.join(CLUSTER_DIR, "three_model_dbscan_gmm_T-20", "swm4ndp",
                            "cluster_labels_matrix_dbscan_gmm.csv")

# ── Order parameter data (.mat) ───────────────────────────────────────────────
OP_4P_T20_MAT  = os.path.join(PARAM_DATA, "OrderParam_tip4p2005_T-20_Run01.mat")
OP_4P_T20_ZETA = os.path.join(PARAM_DATA, "OrderParamZeta_tip4p2005_T-20_Run01.mat")

# ── Physics constants ─────────────────────────────────────────────────────────
R_OO_NM   = 0.285   # O-O nearest-neighbour distance (nm)
RC_CUTOFF = 1.5     # real-space cutoff (nm)
K_MAX     = 50.0    # max wavenumber (nm⁻¹)
K_NORM_RANGE = (0.6, 2.25)

# ── Colour palette (consistent with water_clustering.py) ─────────────────────
C_LFTS  = "#298c8c"   # teal  — LFTS / Cluster 0
C_DNLS  = "#800074"   # magenta — DNLS / Cluster 1
C_TOTAL = "#2196F3"   # blue   — total (unclustered)
C_QZETA = "#e67e22"   # orange — q+ζ only result

# ── Extended colour palette for main-text and appendix figures ────────────────
C_SK_LFTS      = "#5b9bd5"   # cornflower blue — LFTS S(k) (Fig 2)
C_SK_DNLS      = "#d9534f"   # salmon red      — DNLS S(k) (Fig 2)
C_KMEANS_LFTS  = "#1876aa"   # steel blue   — K-Means LFTS (Fig 1a)  [Color Comb2]
C_KMEANS_DNLS  = "#ff7876"   # coral salmon — K-Means DNLS (Fig 1a)  [Color Comb2]
C_DBGMM_NOISE  = "#9ca3af"   # grey           — DBSCAN-GMM noise (Fig 1c)
C_DBGMM_LFTS   = "#231760"   # dark navy      — DBSCAN-GMM LFTS (Fig 1c) [Color Comb2]
C_DBGMM_DNLS   = "#f4913a"   # orange         — DBSCAN-GMM DNLS (Fig 1c)
C_GMM_LFTS     = "#298c8c"   # mid teal   — standalone GMM LFTS (Figs 6, 7)
C_GMM_DNLS     = "#800074"   # magenta    — standalone GMM DNLS (Figs 6, 7)
C_HDBGMM_NOISE = "#b0b0b0"   # grey       — HDBSCAN noise (Fig 10)
C_HDBGMM_LFTS  = "#e6550d"   # orange     — HDBSCAN-GMM LFTS (Fig 10)
C_HDBGMM_DNLS  = "#3182bd"   # steel blue — HDBSCAN-GMM DNLS (Fig 10)

# Model colors for temperature- and model-comparison panels (Fig 4)
C_MODEL = {
    "TIP4P/2005": "#0f766e",  # deep teal
    "TIP5P":      "#c2410c",  # terracotta
    "SWM4-NDP":   "#6d28d9",  # violet
}
C_MODEL_MARKERS = {"TIP4P/2005": "o", "TIP5P": "s", "SWM4-NDP": "^"}

# ── Typography constants (applied consistently across ALL figures) ─────────
# Standalone / full-size axes inherit 16 pt from set_default_plot() rcParams.
# These constants fill in the gaps: small subpanels, legends, panel letters.
FS_SUPTITLE   = 16   # whole-figure suptitle
FS_TITLE      = 14   # subplot / panel title
FS_LABEL      = 14   # axis labels on full-size (non-compound) panels
FS_LABEL_SUB  = 13   # axis labels inside small subpanels of compound figures
FS_TICK       = 13   # explicit tick-label size (where overriding rcParams)
FS_TICK_SUB   = 11   # tick-label size inside small subpanels
FS_LEGEND     = 13   # legend — standalone or large single-panel figures
FS_LEGEND_SUB = 11   # legend — small subpanels in compound figures
FS_PANEL      = 16   # (a), (b), (c), (d) panel-letter annotations


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def k_norm(k_values):
    """Normalise raw k (nm⁻¹) to kr_OO/2π (dimensionless)."""
    return k_values * R_OO_NM / (2 * np.pi)


def style_sk_ax(ax, title="", xlim=K_NORM_RANGE, ylim=(0.5, 1.6), use_spans=False):
    """Apply standard formatting to an S(k) axes (FSDP / D$_1$ reference lines)."""
    if use_spans:
        ax.axvspan(0.75, 0.85, alpha=0.22, color="#5b9bd5", zorder=0,
                   label="LFTS region")
        ax.axvspan(0.95, 1.05, alpha=0.22, color="#d9534f", zorder=0,
                   label="DNLS region")
    else:
        ax.axvline(0.75, color="green", ls="--", lw=1.2, alpha=0.8,
                   label=r"$k_{\mathrm{T1}}$ (FSDP)", zorder=2)
        ax.axvline(1.00, color="red",   ls="--", lw=1.2, alpha=0.8,
                   label=r"$k_{\mathrm{D1}}$", zorder=2)
    ax.set_xlabel(r"$k\,r_{OO}/2\pi$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$S(k)$", fontsize=FS_LABEL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.25, ls="--", lw=0.6)
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=FS_TICK)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")


def load_flat_labels(csv_path):
    """
    Load flat cluster label CSV (one row per molecule).
    Returns pd.DataFrame with order-parameter columns + label column.
    """
    df = pd.read_csv(csv_path)
    return df


def identify_lfts_cluster(df, label_col):
    """
    Return the integer cluster label corresponding to LFTS
    (higher mean ζ among non-noise clusters).
    """
    labels = df[label_col]
    unique = sorted(set(labels[labels >= 0]))
    if "zeta_all" not in df.columns or len(unique) < 2:
        return unique[-1] if unique else 1
    means = {c: df.loc[labels == c, "zeta_all"].mean() for c in unique}
    return max(means, key=means.get)


def compute_sk_cached(dcd, pdb, labels_matrix_csv, out_npz,
                      n_frames_limit=None, force=False):
    """
    Compute per-cluster S(k) and cache result to NPZ.
    Returns (k_values, cluster_results) or (None, None) on failure.
    """
    if not _HAS_SF:
        print("  [SKIP] mdtraj not available.")
        return None, None

    # Load from cache if available
    if os.path.isfile(out_npz) and not force:
        print(f"  Loading cached S(k) from {os.path.basename(out_npz)}")
        d = np.load(out_npz, allow_pickle=True)
        k_values = d["k_values"]
        cluster_results = d["cluster_results"].item()
        return k_values, cluster_results

    # Check files exist
    for path, name in [(dcd, "DCD"), (pdb, "PDB"), (labels_matrix_csv, "Labels")]:
        if not os.path.isfile(path):
            print(f"  [SKIP] {name} not found: {path}")
            return None, None

    k_values = np.linspace(0.1, K_MAX, 300)

    print(f"  Loading trajectory: {os.path.basename(dcd)} …")
    traj = load_trajectory(dcd, pdb, n_frames=n_frames_limit)

    label_matrix = pd.read_csv(labels_matrix_csv, header=None).values.astype(int)
    n_use = min(traj.n_frames, label_matrix.shape[0])
    if n_use < traj.n_frames:
        traj = traj[:n_use]
    label_matrix = label_matrix[:n_use]

    print(f"  Computing S(k): {n_use} frames, k-points={len(k_values)} …")
    cluster_results = compute_per_cluster_structure_factor(
        traj, RC_CUTOFF, k_values, label_matrix)

    # Cache
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, k_values=k_values, cluster_results=cluster_results)
    print(f"  Cached → {out_npz}")
    return k_values, cluster_results


def load_order_params(mat_file, zeta_file):
    """Load five-dimensional order-parameter DataFrame."""
    water  = loadmat(mat_file)
    water1 = loadmat(zeta_file)

    def _flat(d, key):
        arr = d[key]
        if arr.dtype == object:
            return np.concatenate([np.asarray(arr[i]).ravel() for i in range(len(arr))])
        return arr.ravel().astype(float)

    df = pd.DataFrame({
        "q_all"    : _flat(water,  "q_all"),
        "Q6_all"   : _flat(water,  "Q6_all"),
        "LSI_all"  : _flat(water,  "LSI_all"),
        "Sk_all"   : _flat(water,  "Sk_all"),
        "zeta_all" : _flat(water1, "zeta_all"),
    })
    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C.2  —  Minimal feature set: q and ζ only
# ─────────────────────────────────────────────────────────────────────────────

def section_c2(out_dir, n_frames, force_recompute):
    """
    Fig C.2: Per-cluster S(k) for q+ζ-only DBSCAN-GMM vs full 5-feature,
    at TIP4P/2005 T = −20°C.
    """
    print("\n" + "="*65)
    print("SECTION C.2  —  Minimal feature set: q and ζ only")
    print("="*65)

    cache_5feat = os.path.join(out_dir, "_cache_sk_4p_T20_5feat.npz")
    cache_qzeta = os.path.join(out_dir, "_cache_sk_4p_T20_qzeta.npz")

    k_full,   res_full   = compute_sk_cached(
        DCD_4P[-20], PDB_4P[-20], MAT_4P_T20,   cache_5feat,
        n_frames_limit=n_frames, force=force_recompute)
    k_qzeta,  res_qzeta  = compute_sk_cached(
        DCD_4P[-20], PDB_4P[-20], MAT_4P_QZETA,  cache_qzeta,
        n_frames_limit=n_frames, force=force_recompute)

    if res_full is None or res_qzeta is None:
        print("  [SKIP C.2] Could not load S(k) data.")
        return

    # ── Identify LFTS cluster in each set ────────────────────────────────────
    def lfts_id_from_sk(cluster_results, k_values):
        """Cluster with higher S(k) in FSDP region is LFTS."""
        kn = k_norm(k_values)
        mask = (kn >= 0.65) & (kn <= 0.85)
        s0 = cluster_results[0]["S_k_avg"][mask].mean() if 0 in cluster_results else -np.inf
        s1 = cluster_results[1]["S_k_avg"][mask].mean() if 1 in cluster_results else -np.inf
        return 0 if s0 > s1 else 1

    lfts_full  = lfts_id_from_sk(res_full,  k_full)
    dnls_full  = 1 - lfts_full
    lfts_qz    = lfts_id_from_sk(res_qzeta, k_qzeta)
    dnls_qz    = 1 - lfts_qz

    kn_full  = k_norm(k_full)
    kn_qzeta = k_norm(k_qzeta)
    every    = max(1, len(kn_full) // 40)

    # ── Figure 1: side-by-side comparison ────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, k_n, res, lfts, tag in [
        (ax_l, kn_full,  res_full,  lfts_full,
         r"Five order parameters (DBSCAN$\to$GMM)"),
        (ax_r, kn_qzeta, res_qzeta, lfts_qz,
         r"$q$ and $\zeta$ only (DBSCAN$\to$GMM)"),
    ]:
        dnls = 1 - lfts
        for cid, col, marker, name in [
            (lfts, C_LFTS, "o", "LFTS"),
            (dnls, C_DNLS, "D", "DNLS"),
        ]:
            if cid not in res:
                continue
            r = res[cid]
            ax.plot(k_n, r["S_k_avg"],
                    color=col, lw=1.0,
                    marker=marker, markevery=every, ms=6,
                    markerfacecolor="none", markeredgewidth=1.3,
                    label=name, zorder=3)
            eb = np.arange(0, len(k_n), every)
            ax.errorbar(k_n[eb], r["S_k_avg"][eb], yerr=r["S_k_std"][eb],
                        fmt="none", ecolor=col, elinewidth=0.8,
                        capsize=2, capthick=0.8, alpha=0.6)
        style_sk_ax(ax, title=tag)
        ax.legend(frameon=True, edgecolor="black",
                  framealpha=0.9, loc="upper right")

    ax_l.set_ylabel(r"$S(k)$")
    ax_r.set_ylabel("")
    fig.suptitle(
        r"Per-cluster $S(k)$ at $T=-20\,^\circ\mathrm{C}$ (TIP4P/2005)" + "\n"
        r"Left: five order parameters; right: $q$ and $\zeta$ only",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "fig_c2_qzeta_sk_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")

    # ── Figure 2: q+ζ per-cluster S(k) alone (paper-ready) ──────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for cid, col, marker, name in [
        (lfts_qz, C_LFTS, "o", "LFTS (Cluster 0)"),
        (dnls_qz, C_DNLS, "D", "DNLS (Cluster 1)"),
    ]:
        if cid not in res_qzeta:
            continue
        r = res_qzeta[cid]
        ax2.plot(kn_qzeta, r["S_k_avg"],
                 color=col, lw=1.0,
                 marker=marker, markevery=every, ms=6,
                 markerfacecolor="none", markeredgewidth=1.3,
                 label=name, zorder=3)
        eb = np.arange(0, len(kn_qzeta), every)
        ax2.errorbar(kn_qzeta[eb], r["S_k_avg"][eb],
                     yerr=r["S_k_std"][eb],
                     fmt="none", ecolor=col, elinewidth=0.8,
                     capsize=2, capthick=0.8, alpha=0.6)
    style_sk_ax(
        ax2,
        title=r"Per-cluster $S(k)$ from $q$ and $\zeta$ only"
              + "\n"
              r"TIP4P/2005, $T=-20\,^\circ\mathrm{C}$",
    )
    ax2.legend(frameon=True, edgecolor="black", framealpha=0.9)
    plt.tight_layout()
    out2 = os.path.join(out_dir, "fig_c2_qzeta_sk_per_cluster.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"  Saved → {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C.3  —  Temperature dependence of cluster populations
# ─────────────────────────────────────────────────────────────────────────────

def _lfts_fraction_from_flat_csv(csv_path):
    """
    Return (s, s_std) — LFTS fraction and frame-to-frame std — from a
    flat cluster_labels.csv (one row per molecule per frame).

    The CSV has order-param columns + label_<method> column.
    LFTS = cluster with higher mean ζ among non-noise labels.
    """
    df = pd.read_csv(csv_path)
    label_col = [c for c in df.columns if c.startswith("label_")][0]
    labels = df[label_col].values.astype(int)

    lfts = identify_lfts_cluster(df, label_col)

    non_noise = labels[labels >= 0]
    if len(non_noise) == 0:
        return np.nan, np.nan

    s = float(np.mean(non_noise == lfts))

    # frame-to-frame variation: assume 1024 molecules per frame
    n_mol = 1024
    n_frames = max(1, len(labels) // n_mol)
    frame_s = []
    for fi in range(n_frames):
        seg = labels[fi * n_mol: (fi + 1) * n_mol]
        nn  = seg[seg >= 0]
        if len(nn) > 0:
            frame_s.append(float(np.mean(nn == lfts)))
    s_std = float(np.std(frame_s)) if len(frame_s) > 1 else 0.0
    return s, s_std


def section_c3(out_dir, n_frames, force_recompute):
    """
    Fig C.3: LFTS fraction vs temperature (bar + error bar) and
    multi-temperature per-cluster S(k) overlay.
    """
    print("\n" + "="*65)
    print("SECTION C.3  —  Temperature dependence of cluster populations")
    print("="*65)

    # ── Part A: LFTS fraction (GMM only — no DBSCAN noise removal) ─────────────
    temps_csv_gmm = {
        20: LABELS_4P_TP20_GMM,
        10: LABELS_4P_TP10_GMM,
         0: LABELS_4P_T0_GMM,
        -10: LABELS_4P_T10_GMM,
        -20: LABELS_4P_T20_GMM,
        -30: LABELS_4P_T30_GMM,
    }

    # Warm-to-cool palette covering the full +20 → -30°C range
    # (defined once so both Part A bars and Part B S(k) lines share the same colours)
    temp_colors = {
         20: "#d62728",  # deep red
         10: "#ff7f0e",  # orange
          0: "#bcbd22",  # olive-yellow
        -10: "#17becf",  # sky blue
        -20: "#1f77b4",  # medium blue
        -30: "#0d3b7a",  # dark navy
    }
    temp_markers = {20: "P", 10: "X", 0: "v", -10: "o", -20: "s", -30: "^"}

    temps_avail = {t: p for t, p in temps_csv_gmm.items() if os.path.isfile(p)}
    if not temps_avail:
        print("  [SKIP C.3] No GMM-only cluster label CSVs found.")
        return

    temps_sorted = sorted(temps_avail.keys(), reverse=True)   # warmest first
    s_vals   = []
    s_errs   = []

    for T in temps_sorted:
        s, se = _lfts_fraction_from_flat_csv(temps_avail[T])
        s_vals.append(s)
        s_errs.append(se)
        print(f"  T={T:+d}°C  s={s:.3f} ± {se:.3f}")

    x      = np.arange(len(temps_sorted))
    labels = [f"{t:+d}°C" for t in temps_sorted]
    bar_colors = [temp_colors.get(t, C_LFTS) for t in temps_sorted]

    fig, ax = plt.subplots(figsize=(max(6, len(temps_sorted) * 1.2), 5))
    bars = ax.bar(x, s_vals, width=0.55, color=bar_colors, alpha=0.88,
                  edgecolor="black", linewidth=0.8, zorder=3)
    ax.errorbar(x, s_vals, yerr=s_errs, fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5, zorder=4)

    from matplotlib.patches import Patch
    bar_proxy = Patch(facecolor=C_LFTS, edgecolor="black", alpha=0.88,
                      label="LFTS fraction $s$")
    schottky = ax.axhline(0.5, color="grey", ls="--", lw=1.2, alpha=0.7,
                          label=r"$s = 0.5$ (Schottky point)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Temperature ($^\circ\mathrm{C}$)")
    ax.set_ylabel(r"LFTS fraction $s$")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        r"LFTS fraction vs.\ temperature (TIP4P/2005)" + "\n"
        r"Two-component GMM on scaled order parameters (no DBSCAN pre-filter)"
    )

    ax.legend(handles=[bar_proxy, schottky],
              frameon=True, edgecolor="black", framealpha=0.9,
              loc="upper left")

    plt.tight_layout()
    out_a = os.path.join(out_dir, "fig_c3_lfts_fraction_vs_temp.png")
    fig.savefig(out_a)
    plt.close(fig)
    print(f"  Saved → {out_a}")

    # ── Part B: Multi-temperature S(k) overlay ────────────────────────────────
    if not _HAS_SF:
        print("  [SKIP C.3-B] mdtraj not available — skipping multi-temp S(k).")
        return

    k_values = np.linspace(0.1, K_MAX, 300)
    kn = k_norm(k_values)
    every = max(1, len(kn) // 40)

    # Load or compute S(k) per temperature
    # Matrix lookup — add new entries here once Steps 1-3 are complete:
    mat_lookup = {
        20: MAT_4P_TP20,
        10: MAT_4P_TP10,
         0: MAT_4P_T0,
        -10: MAT_4P_T10,
        -20: MAT_4P_T20,
        -30: MAT_4P_T30,
    }
    # S(k) overlay: -30°C to +10°C, using DBSCAN+GMM matrices (independent of Part A GMM)
    temps_for_sk = sorted(
        [t for t in mat_lookup if -30 <= t <= 10
         and os.path.isfile(DCD_4P.get(t, ""))
         and os.path.isfile(PDB_4P.get(t, ""))
         and os.path.isfile(mat_lookup[t])],
        reverse=True
    )
    if not temps_for_sk:
        print("  [SKIP C.3-B] No DCD/PDB/matrix found for temps in [-30, +10]°C.")
        return
    print(f"  S(k) overlay: T = {temps_for_sk}")
    all_res = {}
    for T in temps_for_sk:
        dcd_f = DCD_4P.get(T)
        pdb_f = PDB_4P.get(T)
        mat_f = mat_lookup.get(T)
        if not dcd_f or not pdb_f or not mat_f:
            continue
        cache = os.path.join(out_dir, f"_cache_sk_4p_T{T}_5feat.npz")
        k_v, res = compute_sk_cached(dcd_f, pdb_f, mat_f, cache,
                                     n_frames_limit=n_frames,
                                     force=force_recompute)
        if res is not None:
            all_res[T] = (k_v, res)

    if not all_res:
        print("  [SKIP C.3-B] No S(k) data loaded.")
        return

    # Enforce consistent labelling: cluster 1 = LFTS (higher FSDP)
    def lfts_from_sk(res, k_v):
        kn_ = k_norm(k_v)
        mask = (kn_ >= 0.65) & (kn_ <= 0.85)
        s0 = res[0]["S_k_avg"][mask].mean() if 0 in res else -np.inf
        s1 = res[1]["S_k_avg"][mask].mean() if 1 in res else -np.inf
        return 0 if s0 > s1 else 1

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cluster_names = ["LFTS (Cluster 0 / S-state)", "DNLS (Cluster 1 / ρ-state)"]

    for T in temps_for_sk:
        if T not in all_res:
            continue
        k_v, res = all_res[T]
        kn_ = k_norm(k_v)
        every_ = max(1, len(kn_) // 40)
        col = temp_colors.get(T, "#333")
        mkr = temp_markers.get(T, "o")
        lfts = lfts_from_sk(res, k_v)
        dnls = 1 - lfts

        for panel_idx, (cid, ax) in enumerate([(lfts, axes[0]), (dnls, axes[1])]):
            if cid not in res:
                continue
            r = res[cid]
            eb_ = np.arange(0, len(kn_), every_)
            ax.plot(kn_, r["S_k_avg"], color=col, lw=0.9,
                    marker=mkr, markevery=every_, ms=5,
                    markerfacecolor="none", markeredgewidth=1.2,
                    label=rf"$T = {T:+d}\,^\circ\mathrm{{C}}$", zorder=3)
            ax.errorbar(kn_[eb_], r["S_k_avg"][eb_], yerr=r["S_k_std"][eb_],
                        fmt="none", ecolor=col, elinewidth=0.7,
                        capsize=2, capthick=0.7, alpha=0.5)

    for ax, name in zip(axes, cluster_names):
        style_sk_ax(ax, title=f"{name}\n" + r"TIP4P/2005")
        ax.legend(frameon=True, edgecolor="black",
                  framealpha=0.9, loc="upper right")
    axes[1].set_ylabel("")

    fig2.suptitle(
        r"Per-cluster $S(k)$ vs.\ temperature (TIP4P/2005)" + "\n"
        r"Labels from DBSCAN$\to$GMM; $-30\,^\circ\mathrm{C}\leq T\leq +10\,^\circ\mathrm{C}$",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    plt.tight_layout()
    out_b = os.path.join(out_dir, "fig_c3_sk_multitemp.png")
    fig2.savefig(out_b)
    plt.close(fig2)
    print(f"  Saved → {out_b}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C.4  —  Optimal cluster number via information criteria
# ─────────────────────────────────────────────────────────────────────────────

def section_c4(out_dir, **kwargs):
    """
    Fig C.4: BIC, AIC, and silhouette score vs number of GMM components k.
    TIP4P/2005 at T = −20°C.
    """
    print("\n" + "="*65)
    print("SECTION C.4  —  Optimal cluster number via BIC / AIC / silhouette")
    print("="*65)

    if not os.path.isfile(OP_4P_T20_MAT) or not os.path.isfile(OP_4P_T20_ZETA):
        print(f"  [SKIP C.4] Order-parameter .mat files not found.")
        print(f"    Expected: {OP_4P_T20_MAT}")
        return

    print("  Loading order parameters …")
    df = load_order_params(OP_4P_T20_MAT, OP_4P_T20_ZETA)
    print(f"  {len(df):,} molecules loaded.")

    scaler   = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(df.values)

    # Sub-sample for silhouette (slow for large N)
    rng    = np.random.default_rng(42)
    n_sub  = min(len(X_scaled), 10_000)
    idx    = rng.choice(len(X_scaled), size=n_sub, replace=False)
    X_sub  = X_scaled[idx]

    k_range   = range(1, 5)    # k = 1, 2, 3, 4
    bic_vals  = []
    aic_vals  = []
    sil_vals  = []   # silhouette undefined for k=1

    for k in k_range:
        print(f"  Fitting GMM k={k} …", end=" ", flush=True)
        t0 = time()
        gm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=42, n_init=5, max_iter=300
        )
        gm.fit(X_scaled)
        bic_vals.append(gm.bic(X_scaled))
        aic_vals.append(gm.aic(X_scaled))

        if k >= 2:
            labels = gm.predict(X_sub)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X_sub, labels, random_state=42)
            else:
                sil = np.nan
            sil_vals.append(sil)
        else:
            sil_vals.append(np.nan)

        print(f"BIC={gm.bic(X_scaled):.0f}  AIC={gm.aic(X_scaled):.0f}  "
              f"time={time()-t0:.1f}s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ks = list(k_range)
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax_bic = fig.add_subplot(gs[0])
    ax_aic = fig.add_subplot(gs[1])
    ax_sil = fig.add_subplot(gs[2])

    def _plot_metric(ax, vals, ylabel, color):
        ax.plot(ks, vals, color=color, lw=1.5, marker="o", ms=7,
                markerfacecolor="white", markeredgewidth=2, zorder=3)
        ax.set_xlabel(r"Number of GMM components $k$", fontsize=FS_LABEL,
                      fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.25, ls="--", lw=0.6)
        ax.tick_params(
            direction="in", which="both", top=True, right=True, labelsize=FS_TICK,
        )

    # BIC — lower is better
    _plot_metric(ax_bic, bic_vals, "BIC", "#2c7bb6")

    # AIC — lower is better
    _plot_metric(ax_aic, aic_vals, "AIC", "#1a9641")

    # Silhouette — higher is better (defined for k ≥ 2)
    sil_clean = [(k, s) for k, s in zip(ks, sil_vals) if not np.isnan(s)]
    if sil_clean:
        ax_sil.plot([k for k, _ in sil_clean],
                    [s for _, s in sil_clean],
                    color="#d7191c", lw=1.5, marker="o", ms=7,
                    markerfacecolor="white", markeredgewidth=2, zorder=3)
    ax_sil.set_xlabel(r"Number of GMM components $k$", fontsize=FS_LABEL,
                      fontweight="bold")
    ax_sil.set_ylabel("Silhouette score", fontsize=FS_LABEL, fontweight="bold")
    ax_sil.set_xticks(ks)
    ax_sil.grid(True, alpha=0.25, ls="--", lw=0.6)
    ax_sil.tick_params(
        direction="in", which="both", top=True, right=True, labelsize=FS_TICK,
    )

    plt.tight_layout()
    out = os.path.join(out_dir, "fig_c4_cluster_number.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # Print summary table
    print("\n  ┌──────┬─────────────┬─────────────┬───────────┐")
    print("  │  k   │     BIC     │     AIC     │ Silhouette│")
    print("  ├──────┼─────────────┼─────────────┼───────────┤")
    for k, bic, aic, sil in zip(ks, bic_vals, aic_vals, sil_vals):
        sil_s  = f"{sil:9.4f}" if not np.isnan(sil) else "      —  "
        print(f"  │  {k}   │ {bic:11.0f}  │ {aic:11.0f}  │ {sil_s}  │")
    print("  └──────┴─────────────┴─────────────┴───────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C.5  —  Model comparison: TIP5P vs TIP4P/2005
# ─────────────────────────────────────────────────────────────────────────────

def _load_sk_zeta_png(cluster_id, model, temperature, results_3d_dir):
    """
    Attempt to load the pre-existing 2D S(k,ζ) contour PNG produced by
    sk_zeta_3d.py as a matplotlib image panel.
    Returns None if the file is not found.
    """
    fname = f"2d_sk_zeta_cluster{cluster_id}_{model}_T{temperature}.png"
    path  = os.path.join(results_3d_dir, fname)
    if os.path.isfile(path):
        return path
    return None


def section_c5(out_dir, n_frames, force_recompute):
    """
    Fig C.5: Side-by-side model comparison of per-cluster S(k) and S(k,ζ)
    for TIP4P/2005 vs TIP5P at T = −20°C.
    """
    print("\n" + "="*65)
    print("SECTION C.5  —  Model comparison: TIP5P vs TIP4P/2005")
    print("="*65)

    cache_5p = os.path.join(out_dir, "_cache_sk_5p_T-20_5feat.npz")
    cache_4p = os.path.join(out_dir, "_cache_sk_4p_T20_5feat.npz")

    k_4p, res_4p = compute_sk_cached(
        DCD_4P[-20], PDB_4P[-20], MAT_4P_T20,  cache_4p,
        n_frames_limit=n_frames, force=force_recompute)
    k_5p, res_5p = compute_sk_cached(
        DCD_5P[-20], PDB_5P[-20], MAT_5P_T20,  cache_5p,
        n_frames_limit=n_frames, force=force_recompute)

    if res_4p is None and res_5p is None:
        print("  [SKIP C.5] No S(k) data available for either model.")
        return

    def lfts_id(res, k_v):
        kn_ = k_norm(k_v)
        mask = (kn_ >= 0.65) & (kn_ <= 0.85)
        s0 = res[0]["S_k_avg"][mask].mean() if 0 in res else -np.inf
        s1 = res[1]["S_k_avg"][mask].mean() if 1 in res else -np.inf
        return 0 if s0 > s1 else 1

    # ── Figure 1: per-cluster S(k) side-by-side ──────────────────────────────
    n_panels  = sum([res_4p is not None, res_5p is not None])
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    panel_specs = []
    if res_4p is not None:
        panel_specs.append(("TIP4P/2005", k_4p, res_4p, lfts_id(res_4p, k_4p)))
    if res_5p is not None:
        panel_specs.append(("TIP5P",      k_5p, res_5p, lfts_id(res_5p, k_5p)))

    for ax, (model_name, k_v, res, lfts) in zip(axes, panel_specs):
        kn_  = k_norm(k_v)
        every_ = max(1, len(kn_) // 40)
        dnls = 1 - lfts
        for cid, col, marker, name in [
            (lfts, C_LFTS, "o", "LFTS"),
            (dnls, C_DNLS, "D", "DNLS"),
        ]:
            if cid not in res:
                continue
            r = res[cid]
            eb_ = np.arange(0, len(kn_), every_)
            ax.plot(kn_, r["S_k_avg"], color=col, lw=1.0,
                    marker=marker, markevery=every_, ms=6,
                    markerfacecolor="none", markeredgewidth=1.3,
                    label=name, zorder=3)
            ax.errorbar(kn_[eb_], r["S_k_avg"][eb_], yerr=r["S_k_std"][eb_],
                        fmt="none", ecolor=col, elinewidth=0.8,
                        capsize=2, capthick=0.8, alpha=0.6)
        style_sk_ax(
            ax,
            title=model_name + r", $T=-20\,^\circ\mathrm{C}$",
        )
        ax.legend(frameon=True, edgecolor="black",
                  framealpha=0.9, loc="upper right")

    axes[0].set_ylabel(r"$S(k)$")
    if len(axes) > 1:
        axes[1].set_ylabel("")

    fig.suptitle(
        r"Per-cluster $S(k)$: force-field comparison at $T=-20\,^\circ\mathrm{C}$"
        + "\n"
        r"Five order parameters; DBSCAN$\to$GMM labels",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    plt.tight_layout()
    out_sk = os.path.join(out_dir, "fig_c5_model_comparison_sk.png")
    fig.savefig(out_sk)
    plt.close(fig)
    print(f"  Saved → {out_sk}")

    # ── Figure 2: S(k,ζ) contour panel — assemble from pre-computed PNGs ─────
    results_3d_root = os.path.join(CS_DIR, "results_3d")
    configs = [
        ("TIP4P/2005", "tip4p2005", -20.0, "tip4p2005_T-20_dbscan_gmm"),
        ("TIP5P",      "tip5p",     -20.0, "tip5p_T-20_dbscan_gmm"),
    ]

    # Try to load pre-computed S(k,ζ) combined PNG for each model
    panels = []
    for model_label, model_key, T, subdir in configs:
        combined_png = os.path.join(results_3d_root, subdir,
                                    f"2d_sk_zeta_combined_{model_key}_T{T}.png")
        if os.path.isfile(combined_png):
            img = plt.imread(combined_png)
            panels.append((model_label, img))
        else:
            print(f"  [WARN] S(k,ζ) PNG not found: {combined_png}")

    if panels:
        n = len(panels)
        fig2, axes2 = plt.subplots(1, n, figsize=(9 * n, 5))
        if n == 1:
            axes2 = [axes2]
        for ax, (model_label, img) in zip(axes2, panels):
            ax.imshow(img)
            ax.set_title(model_label, fontsize=FS_TITLE, fontweight="bold", pad=8)
            ax.axis("off")
        fig2.suptitle(
            r"$S(k,\zeta)$: TIP4P/2005 vs.\ TIP5P at "
            r"$T=-20\,^\circ\mathrm{C}$",
            fontsize=FS_SUPTITLE, fontweight="bold",
        )
        plt.tight_layout()
        out_skz = os.path.join(out_dir, "fig_c5_model_comparison_sk_zeta.png")
        fig2.savefig(out_skz)
        plt.close(fig2)
        print(f"  Saved → {out_skz}")
    else:
        print("  [SKIP] No pre-computed S(k,ζ) PNGs found for model comparison.")

    # ── Figure 3: LFTS fractions side-by-side bar chart ──────────────────────
    model_data = {}
    for csv_path, model_label in [
        (LABELS_4P_T20, "TIP4P/2005"),
    ]:
        if os.path.isfile(csv_path):
            s, se = _lfts_fraction_from_flat_csv(csv_path)
            model_data[model_label] = (s, se)

    # TIP5P — try to find its flat cluster labels
    tip5p_label_csv = os.path.join(CLUSTER_DIR, "tip5p_T-20_dbscan_gmm", "cluster_labels.csv")
    if not os.path.isfile(tip5p_label_csv):
        # Try alternate naming
        for cand in [
            os.path.join(CLUSTER_DIR, "tip5p_T-20_dbscan_gmm", "cluster_labels.csv"),
            os.path.join(CLUSTER_DIR, "tip5p_dbscan_gmm", "cluster_labels.csv"),
        ]:
            if os.path.isfile(cand):
                tip5p_label_csv = cand
                break

    if os.path.isfile(tip5p_label_csv):
        s5, se5 = _lfts_fraction_from_flat_csv(tip5p_label_csv)
        model_data["TIP5P"] = (s5, se5)
    else:
        # Derive from matrix if flat CSV not found
        print("  TIP5P flat cluster labels not found — skipping TIP5P bar.")

    if len(model_data) >= 1:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        model_names = list(model_data.keys())
        s_vals_ = [model_data[m][0] for m in model_names]
        s_errs_ = [model_data[m][1] for m in model_names]
        x_ = np.arange(len(model_names))
        colors_ = [C_LFTS, C_DNLS][:len(model_names)]
        ax3.bar(x_, s_vals_, yerr=s_errs_, width=0.4,
                color=colors_, alpha=0.85, edgecolor="black", lw=0.8,
                capsize=6, zorder=3)
        ax3.axhline(0.5, color="grey", ls="--", lw=1.2, alpha=0.7)
        ax3.set_xticks(x_)
        ax3.set_xticklabels(model_names)
        ax3.set_ylabel(r"LFTS fraction $s$")
        ax3.set_ylim(0, 1)
        ax3.set_title(
            r"LFTS fraction at $T=-20\,^\circ\mathrm{C}$"
            + "\n"
            r"DBSCAN$\to$GMM (five order parameters)"
        )
        ax3.grid(True, axis="y", alpha=0.3, ls="--")
        ax3.tick_params(direction="in")
        plt.tight_layout()
        out3 = os.path.join(out_dir, "fig_c5_lfts_fraction_model_comparison.png")
        fig3.savefig(out3)
        plt.close(fig3)
        print(f"  Saved → {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS FOR MAIN-TEXT AND APPENDIX FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv_raw(csv_path):
    """Load cluster_labels.csv → (df, label_col_name)."""
    df = pd.read_csv(csv_path)
    label_col = next((c for c in df.columns if c.startswith("label_")), None)
    if label_col is None:
        raise ValueError(f"No label_* column in {csv_path}")
    return df, label_col


def _lfts_label(df, label_col):
    """Return cluster label integer for LFTS (higher mean ζ, noise excluded)."""
    labels = df[label_col].values
    unique = sorted(set(labels[labels >= 0]))
    if "zeta_all" not in df.columns or len(unique) < 2:
        return unique[-1] if unique else 1
    means = {c: df.loc[labels == c, "zeta_all"].mean() for c in unique}
    return max(means, key=means.get)


def _draw_scatter(ax, df, label_col, palette, display_names):
    """
    Draw q vs ζ scatter on ax.
    palette: {label_int → hex_color}
    display_names: {label_int → legend string}
    """
    labels = df[label_col].values
    unique = sorted(set(labels))
    for lbl in unique:
        mask = labels == lbl
        ax.scatter(
            df.loc[mask, "q_all"], df.loc[mask, "zeta_all"],
            s=4, alpha=0.3,
            color=palette.get(lbl, "#555"),
            label=f"{display_names.get(lbl, str(lbl))} ({mask.sum():,})",
            rasterized=True,
        )
    ax.set_xlabel(r"$q$", fontsize=FS_LABEL_SUB)
    ax.set_ylabel(r"$\zeta$ (Å)", fontsize=FS_LABEL_SUB)
    ax.tick_params(labelsize=FS_TICK_SUB)
    ax.legend(markerscale=4, fontsize=FS_LEGEND_SUB, frameon=True,
              edgecolor="black", framealpha=0.9)


def _draw_distrib_grid(axes_flat, df, label_col, palette, display_names,
                       features=("q_all", "LSI_all", "Sk_all", "zeta_all")):
    """
    Draw 2×2 histogram + KDE panels.
    common_norm=True: all groups normalized together so each cluster's bars
    represent its true fraction of the total data (noise ≠ signal-weight).
    Feature name shown as panel title; y-label = "Density".
    axes_flat: flat sequence of 4 Axes (row-major: q, LSI, Sk, ζ)
    """
    from matplotlib.patches import Patch

    FEAT_TITLES = {
        "q_all":    r"$q$",
        "LSI_all":  "LSI",
        "Sk_all":   r"$S_k$",
        "zeta_all": r"$\zeta$ (Å)",
    }
    labels = df[label_col].values
    unique = sorted(set(labels))           # -1 (noise) first → renders behind

    # Named palette and hue ordering
    palette_named = {display_names.get(lbl, str(lbl)): palette.get(lbl, "#555")
                     for lbl in unique}
    hue_order = [display_names.get(lbl, str(lbl)) for lbl in unique]

    df_plot = df[list(features) + [label_col]].copy()
    df_plot["_grp"] = [display_names.get(l, str(l)) for l in labels]

    axes_list = list(axes_flat)

    for ax, feat in zip(axes_list, features):
        ax.set_facecolor("white")

        # Single call with hue + common_norm=True: each cluster weighted by its
        # true fraction of the total data, so noise appears proportionally small
        sns.histplot(
            data=df_plot, x=feat, hue="_grp",
            hue_order=hue_order,
            ax=ax,
            kde=True, stat="probability",
            common_norm=True,
            palette=palette_named,
            bins=70, alpha=0.55,
            linewidth=0.35, edgecolor="#44475a",
            legend=False,              # we add a manual legend to the first panel
        )

        # Thicken KDE lines
        for line in ax.get_lines():
            line.set_linewidth(1.6)

        ax.set_ylim(bottom=0)
        ax.set_title(FEAT_TITLES.get(feat, feat),
                     fontsize=FS_LABEL_SUB, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=FS_LABEL_SUB)
        ax.tick_params(labelsize=FS_TICK_SUB, direction="in",
                       which="both", top=True, right=True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True, axis="y", alpha=0.18, ls="--", lw=0.5, zorder=0)

    # Manual legend on first panel only
    if axes_list:
        handles = [
            Patch(facecolor=palette.get(lbl, "#555"), alpha=0.70,
                  edgecolor="#231760", linewidth=0.8,
                  label=display_names.get(lbl, str(lbl)))
            for lbl in unique
        ]
        axes_list[0].legend(handles=handles, fontsize=FS_LEGEND_SUB,
                            frameon=True, edgecolor="#231760",
                            framealpha=0.92, fancybox=False)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  —  Main text: K-Means vs DBSCAN-GMM compound (panels a–d)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig1(out_dir, **kwargs):
    """
    Fig 1 (main text): 4-panel compound comparing K-Means and DBSCAN-GMM.
    (a) K-Means q-ζ scatter  |  (b) K-Means 2×2 distributions
    (c) DBSCAN-GMM scatter   |  (d) DBSCAN-GMM 2×2 distributions
    """
    print("\n" + "=" * 65)
    print("FIGURE 1  —  K-Means vs DBSCAN-GMM (4-panel compound)")
    print("=" * 65)

    csv_km = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_kmeans",     "cluster_labels.csv")
    csv_dg = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv")

    for path, name in [(csv_km, "K-Means"), (csv_dg, "DBSCAN-GMM")]:
        if not os.path.isfile(path):
            print(f"  [SKIP] {name} labels not found: {path}")
            return

    df_km, lbl_km = _load_csv_raw(csv_km)
    df_dg, lbl_dg = _load_csv_raw(csv_dg)

    lfts_km = _lfts_label(df_km, lbl_km)
    dnls_km = 1 - lfts_km
    lfts_dg = _lfts_label(df_dg, lbl_dg)
    dnls_dg = 1 - lfts_dg

    pal_km    = {lfts_km: C_KMEANS_LFTS, dnls_km: C_KMEANS_DNLS}
    pal_dg    = {-1: C_DBGMM_NOISE, lfts_dg: C_DBGMM_LFTS, dnls_dg: C_DBGMM_DNLS}
    names_km  = {lfts_km: "LFTS", dnls_km: "DNLS"}
    names_dg  = {-1: "Noise", lfts_dg: "LFTS", dnls_dg: "DNLS"}

    fig = plt.figure(figsize=(20, 14))
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.40, wspace=0.15,
        width_ratios=[1, 1.6],
    )

    specs = [
        (df_km, lbl_km, pal_km, names_km),
        (df_dg, lbl_dg, pal_dg, names_dg),
    ]

    for row, (df, label_col, pal, names) in enumerate(specs):
        # ── Left: q-ζ scatter (no title, no panel letter) ────────────────────
        ax_sc = fig.add_subplot(outer[row, 0])
        _draw_scatter(ax_sc, df, label_col, pal, names)

        # ── Right: 2×2 distribution grid ─────────────────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[row, 1],
            hspace=0.52, wspace=0.38,
        )
        axes_dist = [fig.add_subplot(inner[r, c])
                     for r in range(2) for c in range(2)]
        _draw_distrib_grid(axes_dist, df, label_col, pal, names)

    fig.suptitle(
        r"Clustering of TIP4P/2005 water at $T = -20\,^\circ\mathrm{C}$",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    out = os.path.join(out_dir, "fig1_kmeans_vs_dbscan_gmm.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  —  Main text: per-cluster S(k) with error bars
# ─────────────────────────────────────────────────────────────────────────────

def section_fig2(out_dir, **kwargs):
    """
    Fig 2 (main text): Per-cluster S(k) for TIP4P/2005 at T = -20°C.
    LFTS = cornflower blue circles; DNLS = salmon-red diamonds; error bars = SE.
    """
    print("\n" + "=" * 65)
    print("FIGURE 2  —  Per-cluster S(k) with error bars")
    print("=" * 65)

    cache = os.path.join(out_dir, "_cache_sk_4p_T-20_5feat.npz")
    if not os.path.isfile(cache):
        print(f"  [SKIP] S(k) cache not found: {cache}")
        print("  Run --sections c2 or c5 first to generate the cache.")
        return

    d = np.load(cache, allow_pickle=True)
    k_values = d["k_values"]
    res = d["cluster_results"].item()

    kn = k_norm(k_values)
    every = max(1, len(kn) // 40)

    # Identify LFTS by higher S(k) in FSDP window (k_norm ∈ [0.65, 0.85])
    fsdp_mask = (kn >= 0.65) & (kn <= 0.85)
    s0 = res[0]["S_k_avg"][fsdp_mask].mean() if 0 in res else -np.inf
    s1 = res[1]["S_k_avg"][fsdp_mask].mean() if 1 in res else -np.inf
    lfts_id = 0 if s0 > s1 else 1
    dnls_id = 1 - lfts_id

    fig, ax = plt.subplots(figsize=(14, 6))

    for cid, col, marker, name in [
        (lfts_id, C_SK_LFTS, "o", "LFTS"),
        (dnls_id, C_SK_DNLS, "D", "DNLS"),
    ]:
        if cid not in res:
            continue
        r = res[cid]
        eb = np.arange(0, len(kn), every)
        ax.plot(kn, r["S_k_avg"], color=col, lw=1.7,
                marker=marker, markevery=every, ms=7,
                markerfacecolor="none", markeredgewidth=1.8,
                label=name, zorder=3)
        ax.errorbar(kn[eb], r["S_k_avg"][eb], yerr=r["S_k_std"][eb],
                    fmt="none", ecolor=col, elinewidth=1.0,
                    capsize=3, capthick=1.0, alpha=0.7)

    # Shaded characteristic regions
    ax.axvspan(0.75, 0.85, alpha=0.15, color=C_SK_LFTS, zorder=0,
               label=r"LFTS region")
    ax.axvspan(0.95, 1.05, alpha=0.15, color=C_SK_DNLS, zorder=0,
               label=r"DNLS region")

    # Axis styling — no vertical reference lines (cleaner for main text)
    ax.set_xlabel(r"$k \cdot r_{OO}/2\pi$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$S(k)$", fontsize=FS_LABEL)
    ax.set_xlim(0.6, 2.0)
    ax.set_ylim(0.5, 1.6)
    ax.grid(True, alpha=0.25, ls="--", lw=0.6)
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=FS_TICK)
    ax.set_title(
        r"Per-cluster $S(k)$ — TIP4P/2005 — $-20\,^\circ\mathrm{C}$",
        fontsize=FS_TITLE, fontweight="bold",
    )
    ax.legend(frameon=True, edgecolor="black", framealpha=0.9, fontsize=15)

    out = os.path.join(out_dir, "fig2_sk_per_cluster.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  —  Main text: S(k,ζ) 3D surfaces + 2D contours (4-panel compound)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig3(out_dir, **kwargs):
    """
    Fig 3 (main text): S(k,ζ) compound — 4 panels assembled from pre-computed PNGs.
    (a) LFTS 3D surface  |  (b) DNLS 3D surface
    (c) LFTS 2D contour  |  (d) DNLS 2D contour
    """
    print("\n" + "=" * 65)
    print("FIGURE 3  —  S(k,ζ) surfaces + contour maps (4-panel compound)")
    print("=" * 65)

    # Identify LFTS/DNLS cluster indices from cluster labels
    csv_dg = LABELS_4P_T20
    lfts_id, dnls_id = 1, 0   # fallback
    if os.path.isfile(csv_dg):
        df_dg, lbl_dg = _load_csv_raw(csv_dg)
        lfts_id = _lfts_label(df_dg, lbl_dg)
        dnls_id = 1 - lfts_id

    results_3d_dir = os.path.join(
        CS_DIR, "results_3d", "tip4p2005_T-20_dbscan_gmm"
    )
    T_tag = "-20.0"
    model_tag = "tip4p2005"

    png_paths = {
        "a": os.path.join(results_3d_dir,
             f"3d_sk_zeta_cluster{lfts_id}_{model_tag}_T{T_tag}.png"),
        "b": os.path.join(results_3d_dir,
             f"3d_sk_zeta_cluster{dnls_id}_{model_tag}_T{T_tag}.png"),
        "c": os.path.join(results_3d_dir,
             f"2d_sk_zeta_cluster{lfts_id}_{model_tag}_T{T_tag}.png"),
        "d": os.path.join(results_3d_dir,
             f"2d_sk_zeta_cluster{dnls_id}_{model_tag}_T{T_tag}.png"),
    }

    missing = [f"({k})" for k, p in png_paths.items() if not os.path.isfile(p)]
    if missing:
        print(f"  [SKIP] Missing S(k,ζ) PNGs for panels {missing}")
        print(f"  Expected in: {results_3d_dir}")
        return

    # Crop the embedded title from the top of each PNG:
    #   3D PNGs: title at rows 20-50, then whitespace → crop 65px
    #   2D PNGs: no title is set in _plot_matplotlib_2d → no crop needed
    crop_top = {"a": 65, "b": 65, "c": 0, "d": 0}

    def _load(letter, path):
        img = plt.imread(path)
        return img[crop_top[letter]:, :, :]

    def _make_side_by_side(paths_letters, out_path):
        """Concatenate PNGs horizontally as raw pixel arrays — no matplotlib layout."""
        from PIL import Image as _PIL
        imgs_pil = []
        for l, p in paths_letters:
            raw = (plt.imread(p) * 255).astype(np.uint8)
            raw = raw[crop_top[l]:, :, :]           # crop title strip
            if raw.shape[2] == 4:                   # RGBA → RGB
                bg = np.ones_like(raw[:, :, :3]) * 255
                alpha = raw[:, :, 3:4] / 255.0
                raw = (raw[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
            imgs_pil.append(_PIL.fromarray(raw))

        # Pad images to same height then concatenate horizontally
        max_h = max(im.height for im in imgs_pil)
        padded = []
        for im in imgs_pil:
            if im.height < max_h:
                canvas = _PIL.new("RGB", (im.width, max_h), (255, 255, 255))
                canvas.paste(im, (0, 0))
                padded.append(canvas)
            else:
                padded.append(im)

        gap_px = 20   # white gap between panels
        total_w = sum(im.width for im in padded) + gap_px * (len(padded) - 1)
        combined = _PIL.new("RGB", (total_w, max_h), (255, 255, 255))
        x = 0
        for im in padded:
            combined.paste(im, (x, 0))
            x += im.width + gap_px

        combined.save(out_path, dpi=(600, 600))

    # ── Fig 3a: upper panel — 3D surfaces (LFTS left, DNLS right) ────────────
    out_top = os.path.join(out_dir, "fig3a_sk_zeta_3d.png")
    _make_side_by_side([("a", png_paths["a"]), ("b", png_paths["b"])], out_top)
    print(f"  Saved → {out_top}")

    # ── Fig 3b: lower panel — 2D contours (LFTS left, DNLS right) ───────────
    out_bot = os.path.join(out_dir, "fig3b_sk_zeta_2d.png")
    _make_side_by_side([("c", png_paths["c"]), ("d", png_paths["d"])], out_bot)
    print(f"  Saved → {out_bot}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4  —  Main text: temperature + model dependence of S(k) (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig4(out_dir, n_frames=None, force_recompute=False, **kwargs):
    """
    Fig 4 (main text): 4-panel compound.
    (a) LFTS S(k) at multiple temperatures (TIP4P/2005)
    (b) DNLS S(k) at multiple temperatures (TIP4P/2005)
    (c) LFTS S(k) for TIP4P/2005 and TIP5P at T = -20°C
    (d) DNLS S(k) for TIP4P/2005 and TIP5P at T = -20°C
    """
    print("\n" + "=" * 65)
    print("FIGURE 4  —  Temperature + model dependence of S(k)")
    print("=" * 65)

    # Low-to-high temperature palette: light yellow → dark red → brown → dark purple
    temp_colors = {
       -30: "#ffee55",   # light yellow  (coldest)
       -20: "#f5a800",   # golden orange
       -10: "#cc3300",   # orange-red
         0: "#990000",   # dark red
        10: "#5a1a00",   # brown
        20: "#4b0082",   # dark purple   (hottest)
    }
    temp_markers = {20: "P", 10: "X", 0: "v", -10: "o", -20: "s", -30: "^"}

    # ── Load multi-temperature S(k) from cache files ──────────────────────────
    temp_cache_map = {
        20: os.path.join(out_dir, "_cache_sk_4p_T20_5feat.npz"),
        10: os.path.join(out_dir, "_cache_sk_4p_T10_5feat.npz"),
         0: os.path.join(out_dir, "_cache_sk_4p_T0_5feat.npz"),
       -10: os.path.join(out_dir, "_cache_sk_4p_T-10_5feat.npz"),
       -20: os.path.join(out_dir, "_cache_sk_4p_T-20_5feat.npz"),
       -30: os.path.join(out_dir, "_cache_sk_4p_T-30_5feat.npz"),
    }
    all_res_temp = {}
    for T, cache in temp_cache_map.items():
        if os.path.isfile(cache):
            d = np.load(cache, allow_pickle=True)
            all_res_temp[T] = (d["k_values"], d["cluster_results"].item())
            print(f"  Loaded temp cache: T={T:+d}°C")
        else:
            print(f"  [WARN] Cache missing for T={T:+d}°C: {cache}")

    # ── Load model-comparison S(k) from cache files ───────────────────────────
    model_cache_map = {
        "TIP4P/2005": os.path.join(out_dir, "_cache_sk_4p_T-20_5feat.npz"),
        "TIP5P":      os.path.join(out_dir, "_cache_sk_5p_T-20_5feat.npz"),
    }
    all_res_model = {}
    for model_name, cache in model_cache_map.items():
        if os.path.isfile(cache):
            d = np.load(cache, allow_pickle=True)
            all_res_model[model_name] = (d["k_values"], d["cluster_results"].item())
            print(f"  Loaded model cache: {model_name}")
        else:
            print(f"  [WARN] Model cache missing: {cache}")

    if not all_res_temp and not all_res_model:
        print("  [SKIP] No cached S(k) data found.")
        return

    def _lfts_from_res(res, k_v):
        kn_ = k_norm(k_v)
        mask = (kn_ >= 0.65) & (kn_ <= 0.85)
        s0 = res[0]["S_k_avg"][mask].mean() if 0 in res else -np.inf
        s1 = res[1]["S_k_avg"][mask].mean() if 1 in res else -np.inf
        return 0 if s0 > s1 else 1

    # ── Hand-picked temperature palette: dark blue (cold) → dark crimson (hot)
    temp_colors = {
       -30: "#08306b",  # very dark navy   (coldest)
       -20: "#1f6fac",  # medium blue
       -10: "#41b6c4",  # teal-cyan
         0: "#f0a500",  # golden amber
        10: "#d9480f",  # dark orange-red
        20: "#7f0000",  # dark crimson     (hottest)
    }

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    # ── Panels a, b: temperature dependence ──────────────────────────────────
    temps_sorted = sorted(all_res_temp.keys(), reverse=True)   # hot→cold
    for T in temps_sorted:
        k_v, res = all_res_temp[T]
        kn_ = k_norm(k_v)
        every_ = max(1, len(kn_) // 40)
        col = temp_colors.get(T, "#333")
        mkr = temp_markers.get(T, "o")
        lfts = _lfts_from_res(res, k_v)
        dnls = 1 - lfts
        for panel_col, cid in [(0, lfts), (1, dnls)]:
            if cid not in res:
                continue
            r = res[cid]
            eb_ = np.arange(0, len(kn_), every_)
            axes[0][panel_col].plot(
                kn_, r["S_k_avg"], color=col, lw=1.5,
                marker=mkr, markevery=every_, ms=6,
                markerfacecolor=col, markeredgewidth=0.5,
                markeredgecolor="black",
                label=rf"$T={T:+d}\,^\circ\mathrm{{C}}$", zorder=3,
            )
            axes[0][panel_col].errorbar(
                kn_[eb_], r["S_k_avg"][eb_], yerr=r["S_k_std"][eb_],
                fmt="none", ecolor=col, elinewidth=0.9,
                capsize=2.5, capthick=0.9, alpha=0.55,
            )

    for col_idx, cname in enumerate(["LFTS", "DNLS"]):
        ax = axes[0][col_idx]
        style_sk_ax(ax, title=f"{cname}\nTIP4P/2005", use_spans=True)
        ax.legend(frameon=True, edgecolor="black", framealpha=0.9,
                  fontsize=9, loc="upper right", ncol=2,
                  columnspacing=0.8, handlelength=1.4, handletextpad=0.5)

    # ── Panels c, d: model comparison ────────────────────────────────────────
    model_marker_cycle = ["o", "s", "^"]
    for mi, (model_name, (k_v, res)) in enumerate(all_res_model.items()):
        kn_ = k_norm(k_v)
        every_ = max(1, len(kn_) // 40)
        col = C_MODEL.get(model_name, "#333")
        mkr = model_marker_cycle[mi % len(model_marker_cycle)]
        lfts = _lfts_from_res(res, k_v)
        dnls = 1 - lfts
        for panel_col, cid in [(0, lfts), (1, dnls)]:
            if cid not in res:
                continue
            r = res[cid]
            eb_ = np.arange(0, len(kn_), every_)
            axes[1][panel_col].plot(
                kn_, r["S_k_avg"], color=col, lw=1.5,
                marker=mkr, markevery=every_, ms=6,
                markerfacecolor=col, markeredgewidth=0.5,
                markeredgecolor="black",
                label=model_name, zorder=3,
            )
            axes[1][panel_col].errorbar(
                kn_[eb_], r["S_k_avg"][eb_], yerr=r["S_k_std"][eb_],
                fmt="none", ecolor=col, elinewidth=0.9,
                capsize=2.5, capthick=0.9, alpha=0.55,
            )

    for col_idx, cname in enumerate(["LFTS", "DNLS"]):
        ax = axes[1][col_idx]
        style_sk_ax(
            ax,
            title=f"{cname}\n" + r"$T=-20\,^\circ\mathrm{C}$",
            use_spans=True,
        )
        ax.legend(frameon=True, edgecolor="black", framealpha=0.9,
                  fontsize=FS_LEGEND_SUB, loc="upper right",
                  handlelength=1.4, handletextpad=0.5)

    out = os.path.join(out_dir, "fig4_sk_temp_model.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6  —  Appendix B: GMM per-cluster distributions (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig6(out_dir, **kwargs):
    """
    Fig 6 (Appendix B): Standalone GMM — per-cluster distributions, 2×2 grid.
    """
    print("\n" + "=" * 65)
    print("FIGURE 6  —  GMM per-cluster distributions (Appendix B)")
    print("=" * 65)

    csv_path = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_gmm", "cluster_labels.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] Not found: {csv_path}")
        return

    df, label_col = _load_csv_raw(csv_path)
    lfts = _lfts_label(df, label_col)
    dnls = 1 - lfts

    pal   = {lfts: C_GMM_LFTS, dnls: C_GMM_DNLS}
    names = {lfts: "LFTS", dnls: "DNLS"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _draw_distrib_grid(list(axes.flat), df, label_col, pal, names)

    fig.suptitle(
        r"GMM: per-cluster distributions — TIP4P/2005, $T=-20\,^\circ\mathrm{C}$",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "fig6_gmm_distributions.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7  —  Appendix B: GMM pairplot (left) + q-ζ scatter (right)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig7(out_dir, **kwargs):
    """
    Fig 7 (Appendix B): GMM — pairwise projections (left) + q-ζ scatter (right).
    """
    print("\n" + "=" * 65)
    print("FIGURE 7  —  GMM pairplot + q-ζ scatter (Appendix B)")
    print("=" * 65)

    csv_path = os.path.join(CLUSTER_DIR, "tip4p2005_T-20_gmm", "cluster_labels.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] Not found: {csv_path}")
        return

    df, label_col = _load_csv_raw(csv_path)
    lfts = _lfts_label(df, label_col)
    dnls = 1 - lfts

    pal_hex = {lfts: C_GMM_LFTS, dnls: C_GMM_DNLS}
    names   = {lfts: "LFTS", dnls: "DNLS"}

    # ── Pairplot saved to temp PNG ────────────────────────────────────────────
    feat_cols = ["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"]
    df_pair = df[feat_cols].copy()
    df_pair["cluster"] = [names.get(l, str(l)) for l in df[label_col].values]
    palette_named = {names[lfts]: C_GMM_LFTS, names[dnls]: C_GMM_DNLS}

    tmp_pp = os.path.join(out_dir, "_tmp_fig7_pairplot.png")
    g = sns.pairplot(df_pair, hue="cluster", palette=palette_named,
                     plot_kws={"s": 3, "alpha": 0.3}, diag_kind="kde")
    g.figure.suptitle("GMM — pairwise order-parameter projections",
                      y=1.01, fontsize=FS_TITLE)
    g.figure.savefig(tmp_pp, dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Assemble compound: pairplot left, scatter right ───────────────────────
    fig, axes2 = plt.subplots(1, 2, figsize=(18, 8),
                              gridspec_kw={"width_ratios": [3, 2]})

    pp_img = plt.imread(tmp_pp)
    axes2[0].imshow(pp_img)
    axes2[0].axis("off")
    axes2[0].text(0.02, 0.97, "(a)", transform=axes2[0].transAxes,
                  fontsize=FS_PANEL, fontweight="bold", va="top")

    _draw_scatter(axes2[1], df, label_col, pal_hex, names)
    axes2[1].set_title(r"GMM — $q$ vs $\zeta$", fontsize=FS_TITLE, fontweight="bold")
    axes2[1].text(0.02, 0.97, "(b)", transform=axes2[1].transAxes,
                  fontsize=FS_PANEL, fontweight="bold", va="top")

    plt.tight_layout()
    out = os.path.join(out_dir, "fig7_gmm_pairplot_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    if os.path.isfile(tmp_pp):
        os.remove(tmp_pp)

    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10  —  Appendix B3: HDBSCAN-GMM distributions (left) + scatter (right)
# ─────────────────────────────────────────────────────────────────────────────

def section_fig10(out_dir, **kwargs):
    """
    Fig 10 (Appendix B3): HDBSCAN-GMM — 2×2 distributions (left) + q-ζ scatter (right).
    """
    print("\n" + "=" * 65)
    print("FIGURE 10  —  HDBSCAN-GMM distributions + scatter (Appendix B3)")
    print("=" * 65)

    csv_path = os.path.join(
        CLUSTER_DIR, "tip4p2005_T-20_hdbscan_gmm", "cluster_labels.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] Not found: {csv_path}")
        return

    df, label_col = _load_csv_raw(csv_path)
    lfts = _lfts_label(df, label_col)
    dnls = 1 - lfts

    pal   = {-1: C_HDBGMM_NOISE, lfts: C_HDBGMM_LFTS, dnls: C_HDBGMM_DNLS}
    names = {-1: "Noise", lfts: "LFTS", dnls: "DNLS"}

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[2, 1], wspace=0.35)

    # Left: 2×2 distribution grid
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0], hspace=0.50, wspace=0.45,
    )
    axes_dist = [fig.add_subplot(inner[r, c])
                 for r in range(2) for c in range(2)]
    _draw_distrib_grid(axes_dist, df, label_col, pal, names)

    # Right: q-ζ scatter
    ax_sc = fig.add_subplot(gs[1])
    _draw_scatter(ax_sc, df, label_col, pal, names)
    ax_sc.set_title(r"HDBSCAN-GMM — $q$ vs $\zeta$", fontsize=FS_TITLE,
                    fontweight="bold")

    fig.suptitle(
        r"HDBSCAN-GMM — TIP4P/2005, $T=-20\,^\circ\mathrm{C}$",
        fontsize=FS_SUPTITLE, fontweight="bold",
    )
    out = os.path.join(out_dir, "fig10_hdbscan_gmm_dist_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5  —  Model comparison: TIP4P/2005 vs TIP5P at T = −20 °C
# ─────────────────────────────────────────────────────────────────────────────

def section_fig5(out_dir, **kwargs):
    """
    Fig 5 (main text): side-by-side per-cluster S(k) for TIP4P/2005 and TIP5P
    at T = -20°C.  Reads from the same S(k) caches used by section_fig4.
    Left panel: LFTS cluster.  Right panel: DNLS cluster.
    """
    print("\n" + "=" * 65)
    print("FIGURE 5  —  Model comparison S(k): TIP4P/2005 vs TIP5P")
    print("=" * 65)

    # TIP4P/2005 and TIP5P load from pre-built caches
    all_res_model = {}
    for model_name, cache in [
        ("TIP4P/2005", os.path.join(out_dir, "_cache_sk_4p_T-20_5feat.npz")),
        ("TIP5P",      os.path.join(out_dir, "_cache_sk_5p_T-20_5feat.npz")),
    ]:
        if os.path.isfile(cache):
            d = np.load(cache, allow_pickle=True)
            all_res_model[model_name] = (d["k_values"], d["cluster_results"].item())
            print(f"  Loaded: {model_name}")
        else:
            print(f"  [WARN] Cache missing: {cache}")

    # SWM4-NDP: compute cache if not present
    cache_swm = os.path.join(out_dir, "_cache_sk_swm_T-20_5feat.npz")
    k_swm, res_swm = compute_sk_cached(
        DCD_SWM[-20], PDB_SWM[-20], MAT_SWM_T20, cache_swm,
        n_frames_limit=kwargs.get("n_frames"), force=kwargs.get("force_recompute", False),
    )
    if res_swm is not None:
        all_res_model["SWM4-NDP"] = (k_swm, res_swm)

    if not all_res_model:
        print("  [SKIP] No model cache data found.")
        return

    def _lfts(res, k_v):
        kn_ = k_norm(k_v)
        mask = (kn_ >= 0.65) & (kn_ <= 0.85)
        s0 = res[0]["S_k_avg"][mask].mean() if 0 in res else -np.inf
        s1 = res[1]["S_k_avg"][mask].mean() if 1 in res else -np.inf
        return 0 if s0 > s1 else 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    model_marker_cycle = ["o", "s", "^"]
    for mi, (model_name, (k_v, res)) in enumerate(all_res_model.items()):
        kn_   = k_norm(k_v)
        every_ = max(1, len(kn_) // 40)
        col   = C_MODEL.get(model_name, "#333")
        mkr   = model_marker_cycle[mi % len(model_marker_cycle)]
        lfts  = _lfts(res, k_v)
        dnls  = 1 - lfts
        for panel_col, cid in [(0, lfts), (1, dnls)]:
            if cid not in res:
                continue
            r   = res[cid]
            eb_ = np.arange(0, len(kn_), every_)
            axes[panel_col].plot(
                kn_, r["S_k_avg"], color=col, lw=1.5,
                marker=mkr, markevery=every_, ms=6,
                markerfacecolor=col, markeredgewidth=0.5,
                markeredgecolor="black",
                label=model_name, zorder=3,
            )
            axes[panel_col].errorbar(
                kn_[eb_], r["S_k_avg"][eb_], yerr=r["S_k_std"][eb_],
                fmt="none", ecolor=col, elinewidth=0.9,
                capsize=2.5, capthick=0.9, alpha=0.55,
            )

    for col_idx, cname in enumerate(["LFTS", "DNLS"]):
        ax = axes[col_idx]
        style_sk_ax(ax, title=f"{cname}" + r" — $T=-20\,^\circ\mathrm{C}$",
                    use_spans=True)
        ax.legend(frameon=True, edgecolor="black", framealpha=0.9,
                  fontsize=FS_LEGEND, loc="upper right",
                  handlelength=1.4, handletextpad=0.5)

    axes[1].set_ylabel("")   # share y-axis, suppress duplicate label

    out = os.path.join(out_dir, "fig5_model_comparison_sk.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_ALL_SECTIONS = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig10",
                 "c2", "c3", "c4", "c5"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate all paper figures for Water ML Clustering paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--sections", nargs="+",
                   choices=_ALL_SECTIONS,
                   default=_ALL_SECTIONS,
                   help="Which sections to generate (default: all)")
    p.add_argument("--out-dir",
                   default=os.path.join(_ROOT, "results", "paper_figures"),
                   help="Output directory for generated figures")
    p.add_argument("--n-frames", type=int, default=None,
                   help="Limit trajectory frames loaded for S(k) (None=all; "
                        "use 5–10 for fast preview)")
    p.add_argument("--no-cache", action="store_true", default=False,
                   help="Force recompute S(k) even if cached NPZ exists")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("╔" + "═"*63 + "╗")
    print("║  PAPER FIGURE GENERATOR — Water ML Clustering              ║")
    print("╠" + "═"*63 + "╣")
    print(f"║  Sections  : {', '.join(s.upper() for s in args.sections):<48}║")
    print(f"║  Output    : {args.out_dir:<48}║")
    print(f"║  n-frames  : {str(args.n_frames):<48}║")
    print(f"║  Force     : {str(args.no_cache):<48}║")
    print("╚" + "═"*63 + "╝")

    kw = dict(out_dir=args.out_dir,
              n_frames=args.n_frames,
              force_recompute=args.no_cache)

    sections_requiring_traj = {"c2", "c3", "c5"}
    traj_sections = sections_requiring_traj & set(args.sections)
    if traj_sections and not _HAS_SF:
        print(f"\n[WARN] Sections {sorted(traj_sections)} need mdtraj "
              f"(structure_factor_bycluster.py).")
        print("       Install mdtraj or run only: --sections c4")

    # ── Phase 1: figures that need no S(k) cache ─────────────────────────────
    if "fig1" in args.sections:
        section_fig1(**kw)
    if "fig3" in args.sections:
        section_fig3(**kw)
    if "fig6" in args.sections:
        section_fig6(**kw)
    if "fig7" in args.sections:
        section_fig7(**kw)
    if "fig10" in args.sections:
        section_fig10(**kw)
    if "c4" in args.sections:
        section_c4(**kw)

    # ── Phase 2: appendix sections that build the S(k) cache files ───────────
    if "c2" in args.sections:
        section_c2(**kw)
    if "c3" in args.sections:
        section_c3(**kw)
    if "c5" in args.sections:
        section_c5(**kw)

    # ── Phase 3: main-text figures that read from the S(k) cache ─────────────
    if "fig2" in args.sections:
        section_fig2(**kw)
    if "fig4" in args.sections:
        section_fig4(**kw)
    if "fig5" in args.sections:
        section_fig5(**kw)

    print("\n" + "="*65)
    print(f"All requested figures saved to: {args.out_dir}/")
    print("="*65)

    # Print figure manifest (all PNGs in output dir)
    generated = sorted(f for f in os.listdir(args.out_dir)
                       if f.endswith(".png") and not f.startswith("_"))
    if generated:
        print("\nGenerated figures:")
        for f in generated:
            path = os.path.join(args.out_dir, f)
            size_kb = os.path.getsize(path) // 1024
            print(f"  {f:<52} {size_kb:5d} KB")


if __name__ == "__main__":
    main()
