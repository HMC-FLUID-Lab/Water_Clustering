#!/usr/bin/env python3
"""
auto_cluster_pipeline.py
========================
End-to-end automated clustering pipeline for water MD simulations.

Runs the full three-step workflow:
  Step 1 — Clustering        : load OrderParam + OrderParamZeta .mat files,
                               scale features, run GMM/DBSCAN/etc., save CSV.
  Step 2 — Label conversion  : reshape flat per-molecule labels into a
                               (frames × molecules) matrix CSV.
  Step 3 — Structure factor  : compute per-cluster S(k) from the DCD
                               trajectory and PDB topology, produce PNG plots.

Usage (minimal):
  python auto_cluster_pipeline.py \\
      --mat-file   /path/to/OrderParam_tip4p2005_T-20_Run01.mat \\
      --zeta-file  /path/to/OrderParamZeta_tip4p2005_T-20_Run01.mat \\
      --dcd-file   /path/to/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \\
      --pdb-file   /path/to/inistate_tip4p2005_T-20_N1024_Run01.pdb

Usage (explicit):
  python auto_cluster_pipeline.py \\
      --mat-file   OrderParam_tip5p_T-20_Run01.mat \\
      --zeta-file  OrderParamZeta_tip5p_T-20_Run01.mat \\
      --dcd-file   dcd_tip5p_T-20_N1024_Run01_0.dcd \\
      --pdb-file   inistate_tip5p_T-20_N1024_Run01.pdb \\
      --method     gmm \\
      --n-molecules 1024 \\
      --output-dir ./results/tip5p_T-20

Supported clustering methods:
  gmm, dbscan, kmeans, dbscan_gmm, hdbscan, hdbscan_gmm, all
"""

# ── thread limits must come before numpy/scipy ────────────────────────────────
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"]      = "8"
os.environ["OMP_NUM_THREADS"]      = "8"
os.environ["NUMEXPR_NUM_THREADS"]  = "8"

import sys
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from time import time
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: auto-detect metadata from filename
# ─────────────────────────────────────────────────────────────────────────────

def parse_filename_metadata(filepath: str) -> dict:
    """
    Attempt to extract model name, temperature, run number, and n_molecules
    from a standard filename such as:
      OrderParam_tip4p2005_T-20_Run01.mat
      dcd_tip5p_T-20_N1024_Run01_0.dcd
      inistate_tip4p2005_T-20_N1024_Run01.pdb

    Returns a dict with keys: model, temperature, run, n_molecules (ints/strs
    where available, None otherwise).
    """
    name = Path(filepath).stem
    meta = {"model": None, "temperature": None, "run": None, "n_molecules": None}

    model_match = re.search(r"(tip[45]p[0-9a-zA-Z]*)", name, re.IGNORECASE)
    if model_match:
        meta["model"] = model_match.group(1).lower()

    temp_match = re.search(r"_(T-?\d+)(?:_|$)", name)
    if temp_match:
        meta["temperature"] = temp_match.group(1)

    run_match = re.search(r"_(Run\d+)", name, re.IGNORECASE)
    if run_match:
        meta["run"] = run_match.group(1)

    nmol_match = re.search(r"_N(\d+)_", name)
    if nmol_match:
        meta["n_molecules"] = int(nmol_match.group(1))

    return meta


def infer_n_runs(mat_file: str) -> int:
    """
    Read the .mat file and return the number of MD runs (cells in the cell
    array), or 1 if the array is already 2-D float.
    """
    data = loadmat(mat_file)
    arr  = data.get("q_all")
    if arr is None:
        return 1
    if arr.dtype == object:
        return int(arr.shape[0])
    return 1


def infer_n_molecules(mat_file: str, n_runs: int) -> int:
    """
    Read the .mat file and return the number of molecules per frame.
    For a cell-array format each cell has shape (n_frames, n_molecules).
    For a 2-D float array the shape is (n_frames, n_molecules) directly.
    """
    data = loadmat(mat_file)
    arr  = data.get("q_all")
    if arr is None:
        return 1024  # safe default

    if arr.dtype == object:
        first_cell = np.asarray(arr[0])
        return int(first_cell.shape[1]) if first_cell.ndim == 2 else 1024
    else:
        return int(arr.shape[1]) if arr.ndim == 2 else 1024


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def load_order_params(mat_file: str, zeta_file: str, n_runs: int,
                      cluster_frames: int = None) -> pd.DataFrame:
    """
    Load structural order parameters from two MATLAB .mat files.
    Supports both old cell-array format and new 2-D float format.

    Parameters
    ----------
    cluster_frames : if set, use only the last N frames for clustering.
                     e.g. cluster_frames=1  →  1024 molecules (one frame only)
                          cluster_frames=10 →  10240 molecules (10 frames)
                     None (default) → use all frames
    """
    water  = loadmat(mat_file)
    water1 = loadmat(zeta_file)

    def extract(d, key):
        arr = d[key]
        if arr.dtype == object:
            # cell-array format: each cell is (n_frames, n_molecules)
            frames = []
            for i in range(n_runs):
                frames.append(np.asarray(arr[i], dtype=float))   # (n_frames, n_mol)
            mat = np.vstack(frames)                               # (total_frames, n_mol)
        else:
            mat = arr.astype(float)                               # (n_frames, n_mol)

        if cluster_frames is not None:
            n_use = min(cluster_frames, mat.shape[0])
            mat   = mat[-n_use:]   # take the LAST n frames
        return mat.ravel()

    df = pd.DataFrame({
        "q_all"    : extract(water,  "q_all"),
        "Q6_all"   : extract(water,  "Q6_all"),
        "LSI_all"  : extract(water,  "LSI_all"),
        "Sk_all"   : extract(water,  "Sk_all"),
        "zeta_all" : extract(water1, "zeta_all"),
    })

    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        print(f"  [data] Removed {n_before - len(df):,} rows with inf/NaN values.")

    frame_info = (f"last {cluster_frames} frame(s)" if cluster_frames
                  else f"all frames")
    print(f"  [data] Loaded {len(df):,} data points  ({frame_info}, {n_runs} run(s)).")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns)


def run_dbscan(X, eps=0.2, min_samples=5):
    t0 = time()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    dt = time() - t0
    n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  DBSCAN  eps={eps:.3f}  min_samples={min_samples}")
    print(f"    Clusters: {n_cl}  |  Noise: {n_noise:,} ({100*n_noise/len(labels):.1f}%)  |  {dt:.1f}s")
    if n_cl > 1:
        mask  = labels != -1
        score = silhouette_score(X[mask], labels[mask])
        print(f"    Silhouette (non-noise): {score:.4f}")
    return labels


def run_kmeans(X, n_clusters=2, random_state=42):
    t0 = time()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    dt = time() - t0
    score = silhouette_score(X, labels)
    print(f"  K-Means  n_clusters={n_clusters}  |  Silhouette: {score:.4f}  |  {dt:.1f}s")
    return labels


def run_gmm(X, n_components=2, random_state=42):
    t0 = time()
    gm = GaussianMixture(n_components=n_components, random_state=random_state,
                         covariance_type="full", n_init=5)
    gm.fit(X)
    labels = gm.predict(X)
    probs  = gm.predict_proba(X)
    dt = time() - t0

    score = silhouette_score(X, labels)
    print(f"  GMM  n_components={n_components}")
    print(f"    Silhouette: {score:.4f}  |  BIC: {gm.bic(X):.1f}  |  AIC: {gm.aic(X):.1f}  |  {dt:.1f}s")

    df_X = pd.DataFrame(X)
    for col_name in ("zeta_all", "q_all"):
        col_idx = _col_index(X, col_name)
        if col_idx is not None:
            means = [X[labels == c, col_idx].mean() for c in range(n_components)]
            lfts  = int(np.argmax(means))
            print(f"    Probable LFTS component: {lfts}  (ranked by '{col_name}')")
            print(f"    Estimated LFTS fraction s ≈ {np.mean(labels == lfts):.3f}")
            break

    return labels, probs


def _col_index(X, col_name):
    """Return column index if X came from a DataFrame with that column name, else None."""
    if isinstance(X, pd.DataFrame) and col_name in X.columns:
        return X.columns.get_loc(col_name)
    return None


def run_dbscan_gmm(X, eps=0.2, min_samples=5, n_components=2, random_state=42):
    print("  Stage 1 — DBSCAN noise removal …")
    db_labels  = run_dbscan(X, eps=eps, min_samples=min_samples)
    clean_mask = db_labels != -1
    n_noise    = int((~clean_mask).sum())
    print(f"  Stage 2 — GMM on {clean_mask.sum():,} clean points …")
    gmm_labels, _ = run_gmm(X[clean_mask], n_components=n_components,
                             random_state=random_state)
    final = np.full(len(X), -1, dtype=int)
    final[clean_mask] = gmm_labels
    if final[final != -1].size > 0 and len(np.unique(final[final != -1])) > 1:
        sil = silhouette_score(X[final != -1], final[final != -1])
        print(f"    Silhouette (clean, final): {sil:.4f}")
    return final


def run_hdbscan(X, min_cluster_size=50, min_samples=None,
                cluster_selection_epsilon=0.0, cluster_selection_method="eom"):
    t0 = time()
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                  cluster_selection_epsilon=cluster_selection_epsilon,
                  cluster_selection_method=cluster_selection_method, n_jobs=-1)
    labels = hdb.fit_predict(X)
    dt = time() - t0
    n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  HDBSCAN  mcs={min_cluster_size}  selection={cluster_selection_method}")
    print(f"    Clusters: {n_cl}  |  Noise: {n_noise:,} ({100*n_noise/len(labels):.1f}%)  |  {dt:.1f}s")
    if n_cl > 1:
        mask = labels != -1
        sil  = silhouette_score(X[mask], labels[mask])
        print(f"    Silhouette (non-noise): {sil:.4f}")
    return labels


def run_hdbscan_gmm(X, min_cluster_size=50, n_components=2, random_state=42):
    print("  Stage 1 — HDBSCAN noise removal …")
    hdb_labels = run_hdbscan(X, min_cluster_size=min_cluster_size)
    clean_mask = hdb_labels != -1
    if clean_mask.sum() < n_components * 10:
        print("  WARNING: too few clean points — returning HDBSCAN labels.")
        return hdb_labels
    print(f"  Stage 2 — GMM on {clean_mask.sum():,} clean points …")
    gmm_labels, _ = run_gmm(X[clean_mask], n_components=n_components,
                             random_state=random_state)
    final = np.full(len(X), -1, dtype=int)
    final[clean_mask] = gmm_labels
    return final


def run_clustering(df_scaled: pd.DataFrame, method: str, args) -> dict:
    """Dispatch to the requested clustering method(s). Returns {method: labels}."""
    X = df_scaled.values
    results = {}

    methods_to_run = (
        ["kmeans", "gmm", "dbscan_gmm", "hdbscan_gmm"] if method == "all" else [method]
    )

    for m in methods_to_run:
        print(f"\n{'─'*55}")
        print(f"  Running: {m.upper()}")
        print(f"{'─'*55}")

        if m == "dbscan":
            results[m] = run_dbscan(X, eps=args.eps, min_samples=args.min_samples)

        elif m == "kmeans":
            results[m] = run_kmeans(X, n_clusters=args.n_clusters)

        elif m == "gmm":
            labels, probs = run_gmm(X, n_components=args.n_clusters)
            if args.confidence is not None:
                max_p = probs.max(axis=1)
                noise = max_p < args.confidence
                labels[noise] = -1
                print(f"    Confidence threshold {args.confidence}: "
                      f"{noise.sum():,} points set to noise")
            results[m] = labels

        elif m == "dbscan_gmm":
            results[m] = run_dbscan_gmm(X, eps=args.eps,
                                         min_samples=args.min_samples,
                                         n_components=args.n_clusters)

        elif m == "hdbscan":
            results[m] = run_hdbscan(X,
                                      min_cluster_size=args.hdbscan_min_cluster_size,
                                      min_samples=args.hdbscan_min_samples,
                                      cluster_selection_epsilon=args.hdbscan_epsilon,
                                      cluster_selection_method=args.hdbscan_selection)

        elif m == "hdbscan_gmm":
            results[m] = run_hdbscan_gmm(X,
                                          min_cluster_size=args.hdbscan_min_cluster_size,
                                          n_components=args.n_clusters)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — VISUALISATION (clustering diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def save_cluster_plots(df_raw: pd.DataFrame, df_scaled: pd.DataFrame,
                       results: dict, out_dir: str):
    """Generate scatter, zeta-distribution, and pairplot for each method."""
    for method, labels in results.items():
        unique  = sorted(set(labels))
        palette = sns.color_palette("tab10", len(unique))

        # ── Scatter: q_all vs zeta_all ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, col in zip(unique, palette):
            mask = labels == lbl
            name = f"Noise ({mask.sum():,})" if lbl == -1 \
                   else f"Cluster {lbl} ({mask.sum():,})"
            ax.scatter(df_scaled.loc[mask, "q_all"],
                       df_scaled.loc[mask, "zeta_all"],
                       s=4, alpha=0.3, color=col, label=name)
        ax.set_xlabel("q_all (scaled)")
        ax.set_ylabel("zeta_all (scaled)")
        ax.set_title(f"{method.upper()} — q vs ζ")
        ax.legend(markerscale=4, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{method}_scatter.png"), dpi=150)
        plt.close(fig)

        # ── zeta distribution ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl, col in zip(unique, palette):
            mask  = labels == lbl
            label = f"Noise (n={mask.sum():,})" if lbl == -1 \
                    else f"Cluster {lbl} (n={mask.sum():,})"
            sns.histplot(df_raw.loc[mask, "zeta_all"], ax=ax, kde=True,
                         color=col, alpha=0.5, label=label, stat="density")
        ax.set_xlabel("ζ  [Å]")
        ax.set_ylabel("Density")
        ax.set_title(f"{method.upper()} — ζ distribution per cluster")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{method}_zeta_dist.png"), dpi=150)
        plt.close(fig)

        # ── Pairplot ──────────────────────────────────────────────────────────
        df_plot = df_scaled.copy()
        df_plot["cluster"] = labels.astype(str)
        g = sns.pairplot(df_plot, hue="cluster",
                         plot_kws={"s": 4, "alpha": 0.3}, diag_kind="kde")
        g.figure.suptitle(f"{method.upper()} — Pairplot", y=1.01, fontsize=10)
        g.figure.savefig(os.path.join(out_dir, f"{method}_pairplot.png"),
                         dpi=100, bbox_inches="tight")
        plt.close("all")

        print(f"  Saved plots for {method}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LABEL MATRIX CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def convert_labels_to_matrix(cluster_labels_csv: str, n_runs: int,
                              n_molecules: int, label_column: str,
                              output_csv: str) -> np.ndarray:
    """
    Reshape flat per-molecule cluster labels into a (frames × molecules) matrix.

    Parameters
    ----------
    cluster_labels_csv : output from Step 1
    n_runs             : number of MD runs concatenated
    n_molecules        : molecules per frame
    label_column       : column name for labels, e.g. 'label_gmm'
    output_csv         : path to write the matrix CSV

    Returns
    -------
    label_matrix : np.ndarray shape (n_frames_total, n_molecules)
    """
    df = pd.read_csv(cluster_labels_csv)

    # Auto-detect label column if not present
    if label_column not in df.columns:
        candidates = [c for c in df.columns if c.startswith("label_")]
        if not candidates:
            raise ValueError(
                f"No label column '{label_column}' found and no 'label_*' "
                f"columns detected. Available: {list(df.columns)}"
            )
        label_column = candidates[0]
        print(f"  [convert] Auto-detected label column: {label_column}")

    labels       = df[label_column].values
    n_total      = len(labels)
    n_frames_run = n_total // (n_runs * n_molecules)
    n_frames     = n_runs * n_frames_run
    expected     = n_frames * n_molecules

    if n_total != expected:
        print(f"  [convert] WARNING: size mismatch ({n_total:,} vs expected "
              f"{expected:,}). Using {n_total // n_molecules} frames.")
        n_frames = n_total // n_molecules
        labels   = labels[: n_frames * n_molecules]

    label_matrix = labels.reshape(n_frames, n_molecules)

    for lbl in sorted(np.unique(label_matrix)):
        cnt  = int(np.sum(label_matrix == lbl))
        pct  = 100 * cnt / label_matrix.size
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        print(f"    {name:15s}: {cnt:8,} ({pct:5.2f}%)")

    pd.DataFrame(label_matrix).to_csv(output_csv, index=False, header=False)
    print(f"  [convert] Saved matrix ({n_frames} × {n_molecules}) → {output_csv}")
    return label_matrix


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — STRUCTURE FACTOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_structure_factor_per_cluster(
        dcd_file: str, pdb_file: str, label_matrix: np.ndarray,
        k_max: float = 50.0, k_points: int = 500, rc_cutoff: float = 1.5,
        n_frames: int = None, cluster_ids: list = None):
    """
    Compute per-cluster S(k) using the Debye scattering function.

    Parameters
    ----------
    dcd_file      : path to DCD trajectory
    pdb_file      : path to PDB topology
    label_matrix  : (n_frames, n_molecules) cluster label array
    k_max         : maximum wave number in nm⁻¹
    k_points      : number of k-grid points
    rc_cutoff     : real-space cutoff (nm) for window function
    n_frames      : limit analysis to first n_frames (None = all)
    cluster_ids   : compute only specified clusters (None = all non-noise)

    Returns
    -------
    k_values      : 1-D array of k values (nm⁻¹)
    results       : dict {cluster_id: {'S_k_avg', 'S_k_std'}}
    """
    try:
        import mdtraj as md
    except ImportError:
        print("  [sk] mdtraj not installed — skipping structure factor step.")
        print("       Install with: pip install mdtraj")
        return None, None

    print(f"\n  [sk] Loading trajectory …")
    topology = md.load(pdb_file).topology
    traj     = md.load(dcd_file, top=topology)
    print(f"       {traj.n_frames} frames, {traj.n_atoms} atoms")

    if n_frames is not None and traj.n_frames > n_frames:
        traj = traj[:n_frames]

    n_frames_use = min(traj.n_frames, label_matrix.shape[0])
    traj = traj[:n_frames_use]

    k_values = np.linspace(1.0, k_max, k_points)

    residue_oxygen = {
        atom.residue.index: atom.index
        for atom in traj.topology.atoms if atom.name == "O"
    }

    unique_all = sorted(
        set(int(c) for c in np.unique(label_matrix[:n_frames_use]) if c >= 0)
    )
    if cluster_ids is not None:
        unique_all = [c for c in unique_all if c in cluster_ids]

    results = {}

    for cid in unique_all:
        print(f"\n  [sk] Computing S(k) for Cluster {cid} …")
        n_k        = len(k_values)
        Sk_frames  = np.zeros((n_frames_use, n_k))
        t0         = time()

        for fi in range(n_frames_use):
            if (fi + 1) % 20 == 0:
                elapsed = time() - t0
                rate    = (fi + 1) / elapsed if elapsed > 0 else 1
                eta     = (n_frames_use - fi - 1) / rate
                print(f"       frame {fi+1}/{n_frames_use}  "
                      f"— {rate:.1f} fps, ETA {eta:.0f}s")

            mol_indices  = np.where(label_matrix[fi] == cid)[0]
            atom_indices = np.array(
                [residue_oxygen[m] for m in mol_indices if m in residue_oxygen]
            )
            n_atoms = len(atom_indices)

            if n_atoms < 2:
                Sk_frames[fi] = 1.0
                continue

            frame = traj[fi]
            pairs = np.array(
                [[atom_indices[i], atom_indices[j]]
                 for i in range(n_atoms)
                 for j in range(i + 1, n_atoms)]
            )
            dists = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]

            for ki, k in enumerate(k_values):
                with np.errstate(divide="ignore", invalid="ignore"):
                    win   = np.nan_to_num(
                        np.sin(np.pi * dists / rc_cutoff) / (np.pi * dists / rc_cutoff),
                        nan=0.0)
                    sinc  = np.nan_to_num(np.sin(k * dists) / (k * dists), nan=1.0)
                Sk_frames[fi, ki] = 1.0 + (2.0 / n_atoms) * np.sum(sinc * win)

        dt = time() - t0
        print(f"       Done in {dt:.1f}s")
        results[cid] = {
            "S_k_avg": Sk_frames.mean(axis=0),
            "S_k_std": Sk_frames.std(axis=0),
        }

    return k_values, results


def plot_structure_factor(k_values: np.ndarray, sk_results: dict,
                          out_dir: str, model_name: str, temperature: str,
                          method: str):
    """
    Save PNG plots:
      - per-cluster S(k) comparison
      - individual cluster S(k) (normalised by r_OO)
    Uses open hollow markers with thin connecting lines (paper figure style).
    """
    COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    MARKERS = ["o", "D", "^", "s", "v"]
    NAMES   = {0: r"Cluster 0 ($\rho$-state)", 1: r"Cluster 1 (S-state / tetrahedral)"}
    k_norm  = k_values * 0.285 / (2 * np.pi)

    # Show one marker every ~40 visible points so symbols don't overlap
    every  = max(1, len(k_norm) // 40)
    eb_idx = np.arange(0, len(k_norm), every)

    def _style_ax(ax, title):
        ax.axvline(0.75, color="green", ls="--", lw=1.2, alpha=0.7,
                   label=r"$k_{T1}$ (FSDP)")
        ax.axvline(1.0,  color="red",   ls="--", lw=1.2, alpha=0.7,
                   label=r"$k_{D1}$ (liquid)")
        ax.set_xlabel(r"$k \cdot r_{OO} / 2\pi$", fontsize=13)
        ax.set_ylabel(r"$S(k)$", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, frameon=True, edgecolor="black", framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_xlim(0.6, 3.0)
        ax.set_ylim(0, 3.5)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    # Combined overlay
    fig, ax = plt.subplots(figsize=(12, 7))
    for cid, res in sorted(sk_results.items()):
        col    = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f"Cluster {cid}")
        ax.plot(k_norm, res["S_k_avg"],
                color=col, linewidth=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor="none", markeredgewidth=1.4,
                label=label)
        ax.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                    yerr=res["S_k_std"][eb_idx],
                    fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)
    _style_ax(ax, f"S(k) per cluster — {model_name} {temperature}°C  [{method}]")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"structure_factor_per_cluster_{method}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    # Individual plots
    for cid, res in sorted(sk_results.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        col    = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f"Cluster {cid}")
        ax.plot(k_norm, res["S_k_avg"],
                color=col, linewidth=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor="none", markeredgewidth=1.4,
                label=label)
        ax.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                    yerr=res["S_k_std"][eb_idx],
                    fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)
        _style_ax(ax, f"S(k) Cluster {cid} — {model_name} {temperature}°C  [{method}]")
        fig.tight_layout()
        fname = os.path.join(out_dir, f"structure_factor_cluster{cid}_{method}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, out_dir: str, model_name: str,
                  temperature: str, n_runs: int, n_molecules: int):
    width = 65
    print("\n" + "=" * width)
    print(" CLUSTERING PIPELINE — RESULTS SUMMARY")
    print("=" * width)
    print(f"  Model        : {model_name}")
    print(f"  Temperature  : {temperature}")
    print(f"  Runs         : {n_runs}")
    print(f"  Molecules/frame: {n_molecules}")
    print(f"  Output dir   : {out_dir}")
    print("─" * width)

    for method, labels in results.items():
        unique = sorted(set(labels))
        n_total = len(labels)
        print(f"\n  [{method.upper()}]")
        for lbl in unique:
            cnt  = int(np.sum(labels == lbl))
            pct  = 100 * cnt / n_total
            name = "Noise" if lbl == -1 else f"Cluster {lbl}"
            print(f"    {name:15s}: {cnt:8,} ({pct:5.2f}%)")

    print("\n" + "─" * width)
    print("  Output files:")
    for f in sorted(os.listdir(out_dir)):
        print(f"    {f}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Auto clustering pipeline for water MD simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Required inputs ───────────────────────────────────────────────────────
    p.add_argument("--mat-file",  required=True,
                   help="Path to OrderParam .mat file  "
                        "(contains q_all, Q6_all, LSI_all, Sk_all)")
    p.add_argument("--zeta-file", required=True,
                   help="Path to OrderParamZeta .mat file  "
                        "(contains zeta_all)")
    p.add_argument("--dcd-file",  required=True,
                   help="Path to DCD trajectory file")
    p.add_argument("--pdb-file",  required=True,
                   help="Path to PDB topology file")

    # ── Simulation metadata (auto-detected if not given) ──────────────────────
    p.add_argument("--model-name",   default=None,
                   help="Water model name (e.g. tip4p2005); auto-detected from filename")
    p.add_argument("--temperature",  default=None,
                   help="Temperature string (e.g. T-20); auto-detected from filename")
    p.add_argument("--n-runs",       type=int, default=None,
                   help="Number of MD runs in the .mat files; auto-detected if not set")
    p.add_argument("--n-molecules",  type=int, default=None,
                   help="Molecules per frame (default: 1024); auto-detected if not set")

    # ── Clustering ────────────────────────────────────────────────────────────
    p.add_argument("--method",
                   choices=["dbscan", "kmeans", "gmm", "dbscan_gmm",
                            "hdbscan", "hdbscan_gmm", "all"],
                   default="gmm",
                   help="Clustering method")
    p.add_argument("--n-clusters",   type=int,   default=2,
                   help="Number of clusters for K-Means / GMM")
    p.add_argument("--eps",          type=float, default=0.2,
                   help="DBSCAN epsilon radius in scaled space")
    p.add_argument("--min-samples",  type=int,   default=5,
                   help="DBSCAN / HDBSCAN min_samples")
    p.add_argument("--confidence",   type=float, default=None,
                   help="GMM confidence threshold for denoising [0–1]")

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    p.add_argument("--hdbscan-min-cluster-size", type=int,   default=50)
    p.add_argument("--hdbscan-min-samples",      type=int,   default=None)
    p.add_argument("--hdbscan-epsilon",          type=float, default=0.0)
    p.add_argument("--hdbscan-selection",
                   choices=["eom", "leaf"], default="eom")

    # ── Label column selection ────────────────────────────────────────────────
    p.add_argument("--label-column", default=None,
                   help="Which label column to use for Step 2 & 3. "
                        "Defaults to 'label_<method>' for single methods, "
                        "or 'label_gmm' when method=all.")

    # ── Structure factor ──────────────────────────────────────────────────────
    p.add_argument("--skip-structure-factor", action="store_true",
                   help="Skip Step 3 (structure factor computation)")
    p.add_argument("--rc-cutoff",  type=float, default=1.5,
                   help="Real-space cutoff in nm for S(k) window function")
    p.add_argument("--k-max",      type=float, default=50.0,
                   help="Maximum wave number (nm⁻¹)")
    p.add_argument("--k-points",   type=int,   default=500,
                   help="Number of k-grid points")
    p.add_argument("--n-frames",        type=int, default=None,
                   help="Limit structure factor to first N frames")
    p.add_argument("--cluster-frames",  type=int, default=None,
                   help="Use only the last N frames for clustering. "
                        "e.g. --cluster-frames 1  →  exactly 1024 points (one frame). "
                        "Default: all frames (1000 × 1024 = 1M points).")
    p.add_argument("--cluster-id", type=int,   nargs="+", default=None,
                   help="Compute S(k) only for these cluster IDs")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default=None,
                   help="Output directory (auto-generated from inputs if not set)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate input files ──────────────────────────────────────────────────
    for label, path in [("--mat-file",  args.mat_file),
                         ("--zeta-file", args.zeta_file),
                         ("--dcd-file",  args.dcd_file),
                         ("--pdb-file",  args.pdb_file)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # ── Auto-detect metadata ──────────────────────────────────────────────────
    meta = parse_filename_metadata(args.mat_file)
    model_name  = args.model_name  or meta.get("model")  or "unknown"
    temperature = args.temperature or meta.get("temperature") or "unknown"

    n_runs = args.n_runs
    if n_runs is None:
        n_runs = infer_n_runs(args.mat_file)
        print(f"  [auto] n_runs = {n_runs}  (detected from .mat file)")

    n_molecules = args.n_molecules
    if n_molecules is None:
        n_molecules = infer_n_molecules(args.mat_file, n_runs)
        if n_molecules is None:
            n_molecules = meta.get("n_molecules") or 1024
        print(f"  [auto] n_molecules = {n_molecules}  (detected from .mat file)")

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir is None:
        out_dir = os.path.join(
            os.path.dirname(args.mat_file),
            f"clustering_{model_name}_{temperature}_{args.method}"
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CLUSTERING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" STEP 1 — CLUSTERING")
    print("=" * 65)
    print(f"  Model      : {model_name}")
    print(f"  Temperature: {temperature}")
    print(f"  n_runs     : {n_runs}")
    print(f"  n_molecules: {n_molecules}")
    print(f"  Method     : {args.method}")
    print(f"  Output dir : {out_dir}")

    df_raw    = load_order_params(args.mat_file, args.zeta_file, n_runs,
                                  cluster_frames=args.cluster_frames)
    df_scaled = scale_features(df_raw)
    print(f"\n  Feature summary (raw):\n"
          f"{df_raw.describe().T[['mean','std','min','max']].to_string()}\n")

    results = run_clustering(df_scaled, args.method, args)

    save_cluster_plots(df_raw, df_scaled, results, out_dir)

    # Save cluster labels CSV
    out_df = df_raw.copy()
    for method, labels in results.items():
        out_df[f"label_{method}"] = labels
    labels_csv = os.path.join(out_dir, "cluster_labels.csv")
    out_df.to_csv(labels_csv, index=False)
    print(f"\n  Labels saved → {labels_csv}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — LABEL MATRIX CONVERSION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" STEP 2 — LABEL MATRIX CONVERSION (flat → frames × molecules)")
    print("=" * 65)

    # Generate one matrix CSV per clustering method so each has its own file.
    # If --label-column is explicitly set, generate only that one.
    label_cols_to_convert = []
    if args.label_column:
        label_cols_to_convert = [args.label_column]
    else:
        label_cols_to_convert = [f"label_{m}" for m in results.keys()]

    label_matrices = {}   # method_name -> np.ndarray
    for label_col in label_cols_to_convert:
        method_name = label_col.replace("label_", "")
        matrix_csv  = os.path.join(out_dir,
                                   f"cluster_labels_matrix_{method_name}.csv")
        print(f"  Converting {label_col}  →  {matrix_csv}")
        label_matrices[method_name] = convert_labels_to_matrix(
            cluster_labels_csv=labels_csv,
            n_runs=n_runs,
            n_molecules=n_molecules,
            label_column=label_col,
            output_csv=matrix_csv,
        )

    # Keep a convenience alias for the structure-factor step (first method)
    label_matrix = next(iter(label_matrices.values()))
    label_col    = label_cols_to_convert[0]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — STRUCTURE FACTOR
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_structure_factor:
        print("\n" + "=" * 65)
        print(" STEP 3 — STRUCTURE FACTOR S(k) PER CLUSTER")
        print("=" * 65)

        temp_str = temperature.replace("T", "").replace("_", "")
        try:
            temp_val = int(temp_str)
        except ValueError:
            temp_val = temp_str

        k_values, sk_results = compute_structure_factor_per_cluster(
            dcd_file    = args.dcd_file,
            pdb_file    = args.pdb_file,
            label_matrix = label_matrix,
            k_max       = args.k_max,
            k_points    = args.k_points,
            rc_cutoff   = args.rc_cutoff,
            n_frames    = args.n_frames,
            cluster_ids = args.cluster_id,
        )

        if sk_results is not None:
            plot_structure_factor(
                k_values    = k_values,
                sk_results  = sk_results,
                out_dir     = out_dir,
                model_name  = model_name,
                temperature = str(temp_val),
                method      = label_col.replace("label_", ""),
            )
    else:
        print("\n  [skipped] Structure factor step (--skip-structure-factor).")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print_summary(results, out_dir, model_name, temperature, n_runs, n_molecules)


if __name__ == "__main__":
    main()
