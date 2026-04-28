#!/usr/bin/env python3
"""
run_sfvs.py
===========
Compute the Structure Factor Validation Score (SFVS) for one clustering result.

Uses the 3D Volume approach (Variant A):
  Integrates S(k, ζ) over two physical boxes in (k·r_OO/2π, ζ) space:
    LFTS box : k_norm ∈ [0.70, 0.85],  ζ ∈ [1.0, 1.5] Å   (tetrahedral window)
    DNLS box : k_norm ∈ [0.90, 1.05],  ζ ∈ [−1.0, 0.0] Å  (disordered window)

  For each cluster, the volume in the correct box is rewarded and
  the volume in the wrong box is penalized via Michelson contrast.

Pipeline:
  1. Load cluster_labels.csv       → flat per-molecule labels + zeta_all
  2. Load cluster_labels_matrix    → (frames × molecules) for S(k,ζ) computation
  3. Load DCD + PDB + zeta .mat    → trajectory + per-frame ζ values
  4. Compute S(k,ζ) per cluster    → 2D surface via sk_zeta_3d.py
  5. Compute SFVS-3D               → volume integrals + Michelson contrast
  6. Print breakdown + save CSV

Usage
-----
  python run_sfvs.py \\
      --labels-csv   /path/to/cluster_labels.csv \\
      --label-column label_dbscan_gmm \\
      --matrix-csv   /path/to/cluster_labels_matrix_dbscan_gmm.csv \\
      --dcd-file     /path/to/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \\
      --pdb-file     /path/to/inistate_tip4p2005_T-20_N1024_Run01.pdb \\
      --zeta-mat     /path/to/OrderParamZeta_tip4p2005_T-20_Run01.mat \\
      --output-dir   ./sfvs_results/tip4p2005_T-20_dbscan_gmm

Outputs
-------
  sfvs_score.csv      — all sub-scores in one row (for method comparison)
  sfvs_breakdown.txt  — human-readable report
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── make both sibling modules importable ─────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_SK    = os.path.join(os.path.dirname(_HERE), "cluster_structure")
for p in [_HERE, _SK]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sfvs import compute_sfvs_3d
from structure_factor_bycluster import load_trajectory
from sk_zeta_3d import compute_sk_zeta_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_labels_and_zeta(labels_csv: str,
                          label_col: str) -> tuple:
    """
    Load flat cluster_labels.csv.
    Returns (labels np.ndarray int, zeta np.ndarray float).
    """
    df = pd.read_csv(labels_csv)

    if label_col not in df.columns:
        available = [c for c in df.columns if c.startswith("label_")]
        raise ValueError(
            f"Column '{label_col}' not found in {labels_csv}.\n"
            f"Available label columns: {available}"
        )
    if "zeta_all" not in df.columns:
        raise ValueError(
            f"'zeta_all' column not found in {labels_csv}. "
            "Re-run clustering with the zeta .mat file."
        )

    labels = df[label_col].values.astype(int)
    zeta   = df["zeta_all"].values.astype(float)

    N0 = int((labels == 0).sum())
    N1 = int((labels == 1).sum())
    Nn = int((labels == -1).sum())
    print(f"  Labels loaded: N0={N0:,}  N1={N1:,}  noise={Nn:,}")
    return labels, zeta


def load_label_matrix(matrix_csv: str) -> np.ndarray:
    """Load (frames × molecules) cluster label matrix (no header)."""
    mat = pd.read_csv(matrix_csv, header=None).values.astype(int)
    print(f"  Label matrix: {mat.shape[0]} frames × {mat.shape[1]} molecules")
    return mat


def load_zeta_mat(zeta_mat_file: str) -> np.ndarray:
    """
    Load per-frame ζ values from a .mat file.
    Returns (n_frames, n_molecules) array in Å.
    sk_zeta_3d handles nm→Å conversion internally via _load_zeta.
    """
    from sk_zeta_3d import _load_zeta
    zeta = _load_zeta(zeta_mat_file)
    if zeta is None:
        raise RuntimeError(f"Failed to load zeta from: {zeta_mat_file}")
    return zeta


def compute_sk_zeta_per_cluster(dcd_file: str, pdb_file: str,
                                  label_matrix: np.ndarray,
                                  zeta_all: np.ndarray,
                                  k_values: np.ndarray,
                                  zeta_bins: np.ndarray,
                                  rc_cutoff: float,
                                  n_frames: int = None) -> dict:
    """
    Load trajectory and compute S(k, ζ) surface per cluster.
    Returns {cluster_id: (S_k_zeta array, zeta_centers array)}.
    """
    print(f"  Loading trajectory: {os.path.basename(dcd_file)}")
    traj = load_trajectory(dcd_file, pdb_file, n_frames)

    n_use = min(traj.n_frames, label_matrix.shape[0], zeta_all.shape[0])
    if n_use < traj.n_frames:
        print(f"  Frame alignment: using {n_use} frames")
        traj         = traj[:n_use]
        label_matrix = label_matrix[:n_use]
        zeta_all     = zeta_all[:n_use]

    results = {}
    for cid in sorted(set(label_matrix.ravel()) - {-1}):
        print(f"  Computing S(k,ζ) for cluster {cid} …")
        S_k_zeta, zeta_centers = compute_sk_zeta_matrix(
            traj, k_values, label_matrix, cid, zeta_all,
            zeta_bins, rc_cutoff=rc_cutoff
        )
        results[cid] = (S_k_zeta, zeta_centers)
        print(f"    Done: surface shape {S_k_zeta.shape}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect label column
    if args.label_column is None:
        df_tmp = pd.read_csv(args.labels_csv, nrows=0)
        label_cols = [c for c in df_tmp.columns if c.startswith("label_")]
        if not label_cols:
            raise ValueError("No 'label_*' columns found in labels CSV.")
        args.label_column = label_cols[0]
        print(f"  Auto-detected label column: {args.label_column}")

    method_tag = args.method_tag or args.label_column.replace("label_", "")

    print("=" * 60)
    print(f"  SFVS-3D — {method_tag}")
    print(f"  LFTS box : k_norm [{args.lfts_k_lo},{args.lfts_k_hi}]"
          f"  ζ [{args.lfts_z_lo},{args.lfts_z_hi}] Å")
    print(f"  DNLS box : k_norm [{args.dnls_k_lo},{args.dnls_k_hi}]"
          f"  ζ [{args.dnls_z_lo},{args.dnls_z_hi}] Å")
    print("=" * 60)

    # Step 1: flat labels (for population fractions only)
    print("\n── Step 1: Load cluster labels ─────────────────────────────")
    labels, _ = load_labels_and_zeta(args.labels_csv, args.label_column)

    # Step 2: label matrix
    print("\n── Step 2: Load label matrix ───────────────────────────────")
    label_matrix = load_label_matrix(args.matrix_csv)

    # Step 3: zeta from .mat file
    print("\n── Step 3: Load ζ from .mat file ───────────────────────────")
    zeta_all = load_zeta_mat(args.zeta_mat)

    # Step 4: S(k, ζ) per cluster
    print("\n── Step 4: Compute S(k,ζ) per cluster ─────────────────────")
    k_values  = np.linspace(0.1, args.k_max, args.k_points)
    zeta_bins = np.linspace(args.zeta_min, args.zeta_max, args.zeta_bins + 1)

    sk_zeta = compute_sk_zeta_per_cluster(
        args.dcd_file, args.pdb_file,
        label_matrix, zeta_all,
        k_values, zeta_bins,
        rc_cutoff=args.rc_cutoff,
        n_frames=args.n_frames,
    )

    if 0 not in sk_zeta or 1 not in sk_zeta:
        print("ERROR: S(k,ζ) missing for cluster 0 or 1.")
        sys.exit(1)

    S0_k_zeta, zeta_centers = sk_zeta[0]
    S1_k_zeta, _            = sk_zeta[1]

    # Step 5: override window constants if user changed them
    from sfvs import (LFTS_K_LO, LFTS_K_HI, LFTS_Z_LO, LFTS_Z_HI,
                      DNLS_K_LO, DNLS_K_HI, DNLS_Z_LO, DNLS_Z_HI)
    import sfvs as _sfvs_mod
    _sfvs_mod.LFTS_K_LO = args.lfts_k_lo; _sfvs_mod.LFTS_K_HI = args.lfts_k_hi
    _sfvs_mod.LFTS_Z_LO = args.lfts_z_lo; _sfvs_mod.LFTS_Z_HI = args.lfts_z_hi
    _sfvs_mod.DNLS_K_LO = args.dnls_k_lo; _sfvs_mod.DNLS_K_HI = args.dnls_k_hi
    _sfvs_mod.DNLS_Z_LO = args.dnls_z_lo; _sfvs_mod.DNLS_Z_HI = args.dnls_z_hi

    print("\n── Step 5: Compute SFVS-3D ─────────────────────────────────")
    sfvs, info = compute_sfvs_3d(
        S0_k_zeta, S1_k_zeta, k_values, zeta_centers,
        labels, r_oo=args.r_oo, verbose=True
    )

    # Step 6: save
    print("\n── Step 6: Save outputs ────────────────────────────────────")
    _save_csv(info, method_tag, args.output_dir)
    _save_txt(info, method_tag, args.output_dir)


def _save_csv(info: dict, method_tag: str, out_dir: str):
    nan = float("nan")
    row = {
        "method"   : method_tag,
        "sfvs"     : round(info.get("sfvs",    nan), 6),
        "C0"       : round(info.get("C0",      nan), 6),
        "C1"       : round(info.get("C1",      nan), 6),
        "V0_lfts"  : round(info.get("V0_lfts", nan), 6),
        "V0_dnls"  : round(info.get("V0_dnls", nan), 6),
        "V1_lfts"  : round(info.get("V1_lfts", nan), 6),
        "V1_dnls"  : round(info.get("V1_dnls", nan), 6),
        "f0"       : round(info.get("f0",      nan), 6),
        "f1"       : round(info.get("f1",      nan), 6),
        "N0"       : info.get("N0",      ""),
        "N1"       : info.get("N1",      ""),
        "N_noise"  : info.get("N_noise", ""),
    }
    path = os.path.join(out_dir, "sfvs_score.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"  Saved: {path}")


def _save_txt(info: dict, method_tag: str, out_dir: str):
    nan = float("nan")
    sfvs = info.get("sfvs", nan)
    lines = [
        "=" * 60,
        f"  SFVS-3D Report — {method_tag}",
        "=" * 60,
        f"  Populations  N0={info.get('N0','?'):,}  N1={info.get('N1','?'):,}  "
        f"noise={info.get('N_noise','?'):,}",
        f"  Fractions    f0={info.get('f0',nan):.4f}  f1={info.get('f1',nan):.4f}",
        "",
        "  Cluster 0 (LFTS)",
        f"    V_correct (LFTS box) = {info.get('V0_lfts',nan):.6f}",
        f"    V_penalty (DNLS box) = {info.get('V0_dnls',nan):.6f}",
        f"    C0 = {info.get('C0',nan):+.4f}",
        "",
        "  Cluster 1 (DNLS)",
        f"    V_correct (DNLS box) = {info.get('V1_dnls',nan):.6f}",
        f"    V_penalty (LFTS box) = {info.get('V1_lfts',nan):.6f}",
        f"    C1 = {info.get('C1',nan):+.4f}",
        "",
        f"  SFVS-3D = {sfvs:+.4f}",
        "=" * 60,
    ]
    path = os.path.join(out_dir, "sfvs_breakdown.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute SFVS-3D (volume score) for one clustering result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    p.add_argument("--labels-csv",   required=True,
                   help="cluster_labels.csv (contains label_* columns)")
    p.add_argument("--label-column", default=None,
                   help="Label column to score (auto-detected if omitted)")
    p.add_argument("--matrix-csv",   required=True,
                   help="cluster_labels_matrix_*.csv  (frames × molecules)")
    p.add_argument("--dcd-file",     required=True)
    p.add_argument("--pdb-file",     required=True)
    p.add_argument("--zeta-mat",     required=True,
                   help="OrderParamZeta_*.mat file for per-frame ζ values")

    p.add_argument("--method-tag",   default=None,
                   help="Name for output files (default: from label-column)")

    # S(k,ζ) computation
    p.add_argument("--rc-cutoff",   type=float, default=1.5)
    p.add_argument("--k-max",       type=float, default=50.0)
    p.add_argument("--k-points",    type=int,   default=500)
    p.add_argument("--zeta-min",    type=float, default=-2.0,
                   help="ζ bin range lower bound (Å)")
    p.add_argument("--zeta-max",    type=float, default= 3.0,
                   help="ζ bin range upper bound (Å)")
    p.add_argument("--zeta-bins",   type=int,   default=50,
                   help="Number of ζ bins")
    p.add_argument("--n-frames",    type=int,   default=None)
    p.add_argument("--r-oo",        type=float, default=0.285)

    # Integration windows (can override defaults)
    p.add_argument("--lfts-k-lo",   type=float, default=0.70)
    p.add_argument("--lfts-k-hi",   type=float, default=0.85)
    p.add_argument("--lfts-z-lo",   type=float, default=1.0)
    p.add_argument("--lfts-z-hi",   type=float, default=1.5)
    p.add_argument("--dnls-k-lo",   type=float, default=0.90)
    p.add_argument("--dnls-k-hi",   type=float, default=1.05)
    p.add_argument("--dnls-z-lo",   type=float, default=-1.0)
    p.add_argument("--dnls-z-hi",   type=float, default=0.0)

    p.add_argument("--output-dir",  default="./sfvs_results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
