#!/usr/bin/env python3
"""
run_sk_from_batch.py
====================
Post-process ONE batch clustering result folder into per-cluster S(k) plots.

Steps
-----
  1. Read cluster_labels.csv  (produced by auto_cluster_pipeline.py)
  2. For every label_* column (kmeans, gmm, dbscan_gmm, hdbscan_gmm …):
       a. Reshape flat labels → (frames × molecules) matrix CSV
       b. Compute per-cluster S(k) from the DCD trajectory
       c. Save PNG plots to  <output-dir>/<method>/
  3. Print a summary of all outputs.

Usage
-----
  python run_sk_from_batch.py \\
      --result-dir ./batch_results/tip4p2005_T-20_Run01_all \\
      --dcd-file   .../dcd_tip4p2005_T-20_N1024_Run01_0.dcd \\
      --pdb-file   .../inistate_tip4p2005_T-20_N1024_Run01.pdb

  # process only specific methods:
  python run_sk_from_batch.py ... --methods gmm dbscan_gmm

  # limit frames for speed:
  python run_sk_from_batch.py ... --n-frames 50
"""

import os
import sys
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from time import time

warnings.filterwarnings("ignore")

# ── add 4_structure_factor (sibling of pipeline/) to path ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SF_DIR = os.path.join(_ROOT, "4_structure_factor")
if os.path.isdir(_SF_DIR) and _SF_DIR not in sys.path:
    sys.path.insert(0, _SF_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PARSE METADATA FROM FOLDER NAME
# ─────────────────────────────────────────────────────────────────────────────

def parse_folder_metadata(folder: str) -> dict:
    """
    Extract model, temperature and run from a folder name such as:
      tip4p2005_T-20_Run01_all
    """
    name = Path(folder).name
    meta = {"model": "unknown", "temperature": "unknown", "run": "Run01"}

    m = re.search(r"(tip[45]p[0-9a-zA-Z]*)", name, re.IGNORECASE)
    if m:
        meta["model"] = m.group(1).lower()

    t = re.search(r"_(T-?\d+)(?:_|$)", name)
    if t:
        meta["temperature"] = t.group(1)

    r = re.search(r"_(Run\d+)", name, re.IGNORECASE)
    if r:
        meta["run"] = r.group(1)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONVERT FLAT LABELS → MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def labels_to_matrix(labels: np.ndarray, n_molecules: int) -> np.ndarray:
    """
    Reshape a flat label array of length (n_frames × n_molecules)
    into shape (n_frames, n_molecules).
    Auto-trims if not perfectly divisible.
    """
    n_total  = len(labels)
    n_frames = n_total // n_molecules
    if n_total % n_molecules != 0:
        print(f"  [convert] Trimming {n_total % n_molecules} extra rows to fit "
              f"{n_frames} complete frames.")
        labels = labels[: n_frames * n_molecules]
    return labels.reshape(n_frames, n_molecules)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — S(k) COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_sk_per_cluster(trajectory, label_matrix: np.ndarray,
                           k_values: np.ndarray, rc_cutoff: float,
                           n_frames_limit: int = None) -> dict:
    """
    Compute per-cluster S(k) using the Debye scattering equation.

    Returns
    -------
    dict  {cluster_id: {'S_k_avg': array, 'S_k_std': array}}
    """
    try:
        import mdtraj as md
    except ImportError:
        print("  [sk] ERROR: mdtraj not installed.  pip install mdtraj")
        return {}

    n_frames_use = min(trajectory.n_frames, label_matrix.shape[0])
    if n_frames_limit:
        n_frames_use = min(n_frames_use, n_frames_limit)
    traj = trajectory[:n_frames_use]

    residue_oxygen = {
        atom.residue.index: atom.index
        for atom in traj.topology.atoms if atom.name == "O"
    }

    cluster_ids = sorted(
        set(int(c) for c in np.unique(label_matrix[:n_frames_use]) if c >= 0)
    )
    n_k     = len(k_values)
    results = {}

    for cid in cluster_ids:
        print(f"\n  [sk] Cluster {cid}  ({n_frames_use} frames) …")
        Sk_frames = np.zeros((n_frames_use, n_k))
        t0 = time()

        for fi in range(n_frames_use):
            if (fi + 1) % 20 == 0:
                elapsed = time() - t0
                rate    = (fi + 1) / elapsed if elapsed > 0 else 1
                eta     = (n_frames_use - fi - 1) / rate
                print(f"       frame {fi+1}/{n_frames_use}  "
                      f"{rate:.1f} fps  ETA {eta:.0f}s")

            mol_idx  = np.where(label_matrix[fi] == cid)[0]
            atom_idx = np.array(
                [residue_oxygen[m] for m in mol_idx if m in residue_oxygen]
            )
            n_atoms = len(atom_idx)

            if n_atoms < 2:
                Sk_frames[fi] = 1.0
                continue

            frame = traj[fi]
            pairs = np.array(
                [[atom_idx[i], atom_idx[j]]
                 for i in range(n_atoms)
                 for j in range(i + 1, n_atoms)]
            )
            import mdtraj as md
            dists = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]

            for ki, k in enumerate(k_values):
                with np.errstate(divide="ignore", invalid="ignore"):
                    win  = np.nan_to_num(
                        np.sin(np.pi * dists / rc_cutoff)
                        / (np.pi * dists / rc_cutoff), nan=0.0)
                    sinc = np.nan_to_num(np.sin(k * dists) / (k * dists), nan=1.0)
                Sk_frames[fi, ki] = 1.0 + (2.0 / n_atoms) * np.sum(sinc * win)

        dt = time() - t0
        n_mol_avg = (label_matrix[:n_frames_use] == cid).sum(axis=1).mean()
        print(f"       Done {dt:.1f}s  |  avg {n_mol_avg:.0f} mol/frame")

        results[cid] = {
            "S_k_avg": Sk_frames.mean(axis=0),
            "S_k_std": Sk_frames.std(axis=0),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PLOTTING  (open-marker paper style)
# ─────────────────────────────────────────────────────────────────────────────

COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "D", "^", "s", "v"]
NAMES   = {
    0: r"Cluster 0 ($\rho$-state / DNLS)",
    1: r"Cluster 1 (S-state / LFTS / tetrahedral)",
}


def _style_ax(ax, title, xlim=(0.6, 3.0), ylim=(0.5, 1.5)):
    ax.axvline(0.75, color="green", ls="--", lw=1.2, alpha=0.7,
               label=r"$k_{T1}$ (FSDP)")
    ax.axvline(1.0,  color="red",   ls="--", lw=1.2, alpha=0.7,
               label=r"$k_{D1}$ (liquid)")
    ax.set_xlabel(r"$k \cdot r_{OO} / 2\pi$", fontsize=14)
    ax.set_ylabel(r"$S(k)$", fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, frameon=True, edgecolor="black", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def plot_sk(k_values: np.ndarray, sk_results: dict,
            out_dir: str, model_name: str, temperature: str, method: str):
    """Save individual + combined S(k) PNGs in open-marker style."""
    os.makedirs(out_dir, exist_ok=True)
    k_norm = k_values * 0.285 / (2 * np.pi)
    every  = max(1, len(k_norm) // 40)
    eb_idx = np.arange(0, len(k_norm), every)

    temp_label = temperature.lstrip("T").replace("_", "")
    saved = []

    # Individual cluster plots
    for cid, res in sorted(sk_results.items()):
        col    = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f"Cluster {cid}")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(k_norm, res["S_k_avg"],
                color=col, lw=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor="none", markeredgewidth=1.4, label=label)
        ax.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                    yerr=res["S_k_std"][eb_idx],
                    fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)
        _style_ax(ax,
                  f"{label}\n{model_name}  T={temp_label}°C  [{method}]")
        fig.tight_layout()
        fname = os.path.join(out_dir,
                             f"sk_cluster{cid}_{model_name}_T{temp_label}_{method}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"  Saved: {fname}")

    # Combined overlay
    fig, ax = plt.subplots(figsize=(12, 7))
    for cid, res in sorted(sk_results.items()):
        col    = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f"Cluster {cid}")
        ax.plot(k_norm, res["S_k_avg"],
                color=col, lw=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor="none", markeredgewidth=1.4, label=label)
        ax.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                    yerr=res["S_k_std"][eb_idx],
                    fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)
    _style_ax(ax,
              f"Per-cluster S(k) — {model_name}  T={temp_label}°C  [{method}]")
    fig.tight_layout()
    fname = os.path.join(out_dir,
                         f"sk_per_cluster_{model_name}_T{temp_label}_{method}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(fname)
    print(f"  Saved: {fname}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION  (importable)
# ─────────────────────────────────────────────────────────────────────────────

def run_sk_pipeline(result_dir: str, dcd_file: str, pdb_file: str,
                    n_molecules: int = 1024,
                    methods: list = None,
                    n_frames: int = None,
                    k_max: float = 50.0,
                    k_points: int = 500,
                    rc_cutoff: float = 1.5,
                    output_dir: str = None,
                    zeta_file: str = None) -> dict:
    """
    Full pipeline for one batch result folder.

    Parameters
    ----------
    result_dir   : folder containing cluster_labels.csv
    dcd_file     : path to DCD trajectory
    pdb_file     : path to PDB topology
    n_molecules  : molecules per frame (default 1024)
    methods      : list of method suffixes to process, e.g. ['gmm', 'dbscan_gmm']
                   None = all label_* columns found in cluster_labels.csv
    n_frames     : limit S(k) to first N frames (None = all)
    output_dir   : where to write sk_results/ (default: result_dir/sk_results)

    Returns
    -------
    dict  {method: [list of saved PNG paths]}
    """
    try:
        import mdtraj as md
    except ImportError:
        print("ERROR: mdtraj is required.  pip install mdtraj")
        return {}

    result_dir = str(result_dir)
    labels_csv = os.path.join(result_dir, "cluster_labels.csv")
    if not os.path.isfile(labels_csv):
        print(f"ERROR: cluster_labels.csv not found in {result_dir}")
        return {}

    if output_dir is None:
        output_dir = os.path.join(result_dir, "sk_results")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta        = parse_folder_metadata(result_dir)
    model_name  = meta["model"]
    temperature = meta["temperature"]
    print(f"\n{'='*65}")
    print(f"  S(k) PIPELINE: {model_name}  {temperature}")
    print(f"{'='*65}")
    print(f"  Result dir : {result_dir}")
    print(f"  DCD        : {dcd_file}")
    print(f"  PDB        : {pdb_file}")
    print(f"  Output     : {output_dir}")

    # ── Load labels CSV ───────────────────────────────────────────────────────
    df         = pd.read_csv(labels_csv)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    if not label_cols:
        print("ERROR: no label_* columns found in cluster_labels.csv")
        return {}

    # Filter to requested methods
    if methods is not None:
        label_cols = [c for c in label_cols if c.replace("label_", "") in methods]
        if not label_cols:
            print(f"ERROR: none of {methods} found in {list(df.columns)}")
            return {}

    print(f"  Methods    : {[c.replace('label_', '') for c in label_cols]}")
    print(f"  Rows       : {len(df):,}  →  {len(df)//n_molecules} frames × {n_molecules} mol")

    # ── Load trajectory once ─────────────────────────────────────────────────
    print(f"\n  Loading trajectory …")
    if not os.path.isfile(dcd_file):
        print(f"ERROR: DCD not found: {dcd_file}"); return {}
    if not os.path.isfile(pdb_file):
        print(f"ERROR: PDB not found: {pdb_file}"); return {}

    topology = md.load(pdb_file).topology
    traj     = md.load(dcd_file, top=topology)
    print(f"  Trajectory : {traj.n_frames} frames, {traj.n_atoms} atoms")

    k_values = np.linspace(1.0, k_max, k_points)
    all_saved = {}

    # ── Process each method ───────────────────────────────────────────────────
    for col in label_cols:
        method = col.replace("label_", "")
        print(f"\n{'─'*65}")
        print(f"  METHOD: {method.upper()}")
        print(f"{'─'*65}")

        # Convert flat → matrix
        raw_labels   = df[col].values
        label_matrix = labels_to_matrix(raw_labels, n_molecules)
        n_frames_mat = label_matrix.shape[0]

        # Print label statistics
        for lbl in sorted(np.unique(label_matrix)):
            cnt  = int(np.sum(label_matrix == lbl))
            pct  = 100 * cnt / label_matrix.size
            name = "Noise" if lbl == -1 else f"Cluster {lbl}"
            print(f"    {name:15s}: {cnt:8,} ({pct:5.2f}%)")

        # Save matrix CSV
        mat_csv = os.path.join(result_dir,
                               f"cluster_labels_matrix_{method}.csv")
        pd.DataFrame(label_matrix).to_csv(mat_csv, index=False, header=False)
        print(f"  Matrix saved → {mat_csv}  {label_matrix.shape}")

        # Align trajectory frame count
        n_use = min(traj.n_frames, n_frames_mat)
        if n_frames:
            n_use = min(n_use, n_frames)
        traj_use = traj[:n_use]
        mat_use  = label_matrix[:n_use]

        # Compute S(k)
        sk_out_dir = os.path.join(output_dir, method)
        sk_results = compute_sk_per_cluster(
            traj_use, mat_use, k_values, rc_cutoff
        )

        if not sk_results:
            print(f"  [WARN] No S(k) results for {method} — skipping plots.")
            continue

        # PNG plots
        saved = plot_sk(k_values, sk_results, sk_out_dir,
                        model_name, temperature, method)
        all_saved[method] = saved

        # HTML plots via sk_zeta_3d.py (only when --zeta-file is supplied)
        if zeta_file:
            try:
                from sk_zeta_3d import plot_sk_zeta_all_clusters
                temp_num = float(meta["temperature"].lstrip("T"))
                plot_sk_zeta_all_clusters(
                    trajectory            = traj_use,
                    k_values              = k_values,
                    cluster_labels_matrix = mat_use,
                    zeta_file             = zeta_file,
                    output_dir            = sk_out_dir,
                    model_name            = model_name,
                    temperature           = temp_num,
                    rc_cutoff             = rc_cutoff,
                )
            except ImportError:
                print("  [WARN] sk_zeta_3d.py not found — skipping HTML output.")
            except Exception as exc:
                print(f"  [WARN] sk_zeta_3d failed: {exc}")

    return all_saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-cluster S(k) from a batch result folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--result-dir",  required=True,
                   help="Batch result folder containing cluster_labels.csv")
    p.add_argument("--dcd-file",    required=True,
                   help="Path to DCD trajectory")
    p.add_argument("--pdb-file",    required=True,
                   help="Path to PDB topology")
    p.add_argument("--n-molecules", type=int, default=1024,
                   help="Molecules per simulation frame")
    p.add_argument("--methods",     nargs="+", default=None,
                   help="Which methods to process, e.g. --methods gmm dbscan_gmm. "
                        "Default: all label_* columns in cluster_labels.csv")
    p.add_argument("--n-frames",    type=int, default=None,
                   help="Limit S(k) computation to first N frames")
    p.add_argument("--k-max",       type=float, default=50.0)
    p.add_argument("--k-points",    type=int,   default=500)
    p.add_argument("--rc-cutoff",   type=float, default=1.5,
                   help="Real-space cutoff for S(k) window function (nm)")
    p.add_argument("--output-dir",  default=None,
                   help="Output directory (default: <result-dir>/sk_results)")
    p.add_argument("--zeta-file",   default=None,
                   help="Path to OrderParamZeta_*.mat — enables S(k,ζ) HTML output "
                        "via sk_zeta_3d.py (3D surface + 2D contour per cluster)")
    return p.parse_args()


def main():
    args = parse_args()
    saved = run_sk_pipeline(
        result_dir  = args.result_dir,
        dcd_file    = args.dcd_file,
        pdb_file    = args.pdb_file,
        n_molecules = args.n_molecules,
        methods     = args.methods,
        n_frames    = args.n_frames,
        k_max       = args.k_max,
        k_points    = args.k_points,
        rc_cutoff   = args.rc_cutoff,
        output_dir  = args.output_dir,
        zeta_file   = args.zeta_file,
    )

    print(f"\n{'='*65}")
    print(" DONE — files written:")
    for method, files in saved.items():
        print(f"  [{method}]")
        for f in files:
            print(f"    {f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
