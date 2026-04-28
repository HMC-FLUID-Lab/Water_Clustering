#!/usr/bin/env python3
"""
prepare_positive_temps.py
=========================
End-to-end pipeline that processes TIP4P/2005 trajectories at positive
temperatures (T = 0, +10, +20°C) so they can be added to the multi-
temperature comparison figure in generate_paper_figures.py.

Pipeline steps per temperature
-------------------------------
  1. Compute q, Q6, LSI, Sk  → OrderParam_tip4p2005_T{T}_Run01.mat
  2. Compute ζ (zeta)         → OrderParamZeta_tip4p2005_T{T}_Run01.mat
  3. Run DBSCAN–GMM clustering → clustering/tip4p2005_T{T}_dbscan_gmm/
  4. Convert labels → matrix   → cluster_structure/cluster_labels_matrix_...csv
  5. Patch generate_paper_figures.py to include the new temperature

Usage
-----
  python prepare_positive_temps.py                   # all three temperatures
  python prepare_positive_temps.py --temps 0         # only T=0
  python prepare_positive_temps.py --temps 0 10 20
  python prepare_positive_temps.py --skip-op         # skip order params (already done)
  python prepare_positive_temps.py --skip-zeta       # skip zeta computation
  python prepare_positive_temps.py --n-frames 5      # use 5 frames (fast test)
  python prepare_positive_temps.py --n-frames 20     # default: matches clustering

Estimated run time
------------------
  Order params (q,Q6,LSI,Sk)  ~2–5 min per temperature
  Zeta                         ~30–90 min per temperature (parallelised)
  Clustering + conversion      < 1 min per temperature
"""

# ── thread limits — before numpy ─────────────────────────────────────────────
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"]      = "8"
os.environ["OMP_NUM_THREADS"]      = "8"
os.environ["NUMEXPR_NUM_THREADS"]  = "8"

import sys
import argparse
import warnings
import multiprocessing
import numpy as np
import scipy.io
import pandas as pd
from scipy.special import sph_harm_y as sph_harm
from joblib import Parallel, delayed
from time import time
import mdtraj as md

warnings.filterwarnings("ignore")

# ── resolve repo root and stage modules ──────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _sub in ("3_clustering", "4_structure_factor"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from water_clustering import (
    load_order_params,
    scale_features,
    run_dbscan_gmm,
)
from convert_cluster_labels import main as convert_labels_main


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DATA  = os.path.join(_ROOT, "data", "simulations", "tip4p2005")
PARAM_DIR = os.path.join(_ROOT, "data", "order_params")
CLUST_DIR = os.path.join(_ROOT, "results", "clustering")
CS_DIR    = os.path.join(_ROOT, "results", "clustering", "cluster_labels_matrices")
FIGURE_SCRIPT = os.path.join(_HERE, "generate_paper_figures.py")

# DBSCAN–GMM parameters (same as used for negative temperatures)
DBSCAN_EPS      = 0.05
DBSCAN_MIN_SAMP = 20
N_MOLECULES     = 1024
N_RUNS          = 20     # frames to read from the .mat file

# H-bond geometry criterion (Russo–Tanaka zeta)
R_OO_CUTOFF_NM  = 0.35   # O-O distance cutoff for H-bond candidates
COS_ANGLE_LIMIT = np.cos(np.pi / 6)   # cos(30°)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ORDER PARAMETERS  (q, Q6, LSI, Sk)
# ─────────────────────────────────────────────────────────────────────────────

def _cos_phi_jk(i_arr, j_arr, k_arr, pos_O):
    """
    Cosine of the i–j–k angle for each frame.

    Each of i_arr, j_arr, k_arr is length n_frames: the neighbour indices
    change frame by frame.  md.compute_distances(traj, pairs) returns
    (n_frames, n_pairs); with n_pairs == n_frames we take the diagonal
    to get one distance per frame — matching the original CosPhi_jk logic.
    """
    pij = np.stack([i_arr, j_arr], axis=1)   # (nf, 2)
    pik = np.stack([i_arr, k_arr], axis=1)
    pjk = np.stack([j_arr, k_arr], axis=1)
    b = md.compute_distances(pos_O, pij, periodic=True, opt=True).diagonal()
    c = md.compute_distances(pos_O, pik, periodic=True, opt=True).diagonal()
    a = md.compute_distances(pos_O, pjk, periodic=True, opt=True).diagonal()
    return (b**2 + c**2 - a**2) / (2 * b * c)


def _theta_phi(i_arr, j_arr, pos_O):
    """
    Spherical angles θ, φ of bond i→j for each frame.
    Same diagonal trick: pairs vary per frame, so we take diag of result.
    """
    pairs = np.stack([i_arr, j_arr], axis=1)   # (nf, 2)
    r_ij  = md.compute_displacements(pos_O, pairs, periodic=True, opt=True)
    # r_ij shape: (n_frames, n_pairs, 3) → diagonal along frame/pair axes
    nf    = r_ij.shape[0]
    r_ij  = r_ij[np.arange(nf), np.arange(nf), :]   # (nf, 3)
    norm  = np.linalg.norm(r_ij, axis=1)
    theta = np.arccos(np.clip(r_ij[:, 2] / norm, -1, 1))
    phi   = np.arctan2(r_ij[:, 1], r_ij[:, 0])
    return theta, phi


def compute_order_params(dcd_file, pdb_file, n_frames_limit=None):
    """
    Compute q, Q6, LSI, Sk for all molecules in the trajectory.
    Returns dict with arrays of shape (n_frames, N_mol).
    """
    traj = md.load(dcd_file, top=pdb_file)
    if n_frames_limit:
        traj = traj[:n_frames_limit]

    top  = traj.topology
    N    = top.n_residues
    nf   = traj.n_frames

    oxy_idx = [a.index for a in top.atoms if a.name == "O"]
    pos_O   = traj.atom_slice(oxy_idx)

    q_all   = np.zeros((nf, N))
    Sk_all  = np.zeros((nf, N))
    LSI_all = np.zeros((nf, N))
    Q6_all  = np.zeros((nf, N))

    print(f"  Computing q, Q6, LSI, Sk: {nf} frames, {N} molecules …")
    t0 = time()

    for i in range(N):
        if (i + 1) % 100 == 0 or i == N - 1:
            elapsed = time() - t0
            eta     = elapsed / (i + 1) * (N - i - 1)
            print(f"    molecule {i+1}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        # Pairwise distances from molecule i to all others
        others = [j for j in range(N) if j != i]
        pairs  = np.array([[i, j] for j in others])
        pdist  = md.compute_distances(pos_O, pairs, periodic=True, opt=True)
        # pdist shape: (n_frames, N-1)

        rank_idx  = np.argsort(pdist, axis=1)   # (nf, N-1) sorted indices
        pdist_s   = np.sort(pdist, axis=1)       # sorted distances

        # ── LSI ──────────────────────────────────────────────────────────────
        mask_37  = pdist_s < 0.37                # within 3.7 Å
        n_inner  = mask_37.sum(axis=1)           # n(frame) neighbours < 3.7 Å
        n_inner  = np.maximum(n_inner, 1)        # avoid /0
        Delta    = pdist_s[:, 1:] - pdist_s[:, :-1]    # (nf, N-2)
        cut_mask = mask_37[:, :-1]
        Delta_bar = (Delta * cut_mask).sum(axis=1) / n_inner
        diff      = (Delta - Delta_bar[:, np.newaxis]) * cut_mask
        LSI_all[:, i] = (diff**2).sum(axis=1) / n_inner

        # ── Sk ───────────────────────────────────────────────────────────────
        r_k    = pdist_s[:, :4]                  # 4 nearest (nf, 4)
        r_bar  = r_k.mean(axis=1, keepdims=True)
        Sk_all[:, i] = 1.0 - ((r_k - r_bar)**2 / (4 * r_bar**2)).sum(axis=1) / 3.0

        # ── Neighbours (index arrays) ─────────────────────────────────────────
        # rank_idx[:,k] is the k-th nearest OTHER molecule's index in `others`
        n1 = np.array([others[rank_idx[f, 0]] for f in range(nf)])
        n2 = np.array([others[rank_idx[f, 1]] for f in range(nf)])
        n3 = np.array([others[rank_idx[f, 2]] for f in range(nf)])
        n4 = np.array([others[rank_idx[f, 3]] for f in range(nf)])

        # ── q ────────────────────────────────────────────────────────────────
        ni = np.full(nf, i, dtype=int)
        pairs6 = [(ni, n1, n2), (ni, n1, n3), (ni, n1, n4),
                  (ni, n2, n3), (ni, n2, n4), (ni, n3, n4)]
        cos6   = np.column_stack([_cos_phi_jk(*p, pos_O) for p in pairs6])
        q_all[:, i] = 1.0 - (3.0 / 8.0) * ((cos6 + 1.0 / 3.0)**2).sum(axis=1)

        # ── Q6 ───────────────────────────────────────────────────────────────
        # 12 next nearest neighbours (indices 4-15 in sorted order)
        Y_lm = np.zeros((nf, 13), dtype=complex)
        for k in range(12):
            nk = np.array([others[rank_idx[f, k + 4]] for f in range(nf)])
            theta, phi = _theta_phi(ni, nk, pos_O)
            for m in range(-6, 7):
                Y_lm[:, m + 6] += sph_harm(m, 6, phi, theta)
        Y_lm_bar = Y_lm / 12.0
        Q6_all[:, i] = np.sqrt(4 * np.pi / 13.0 * (np.abs(Y_lm_bar)**2).sum(axis=1))

    print(f"  Done in {time()-t0:.1f}s")
    return {"q_all": q_all, "Q6_all": Q6_all, "LSI_all": LSI_all, "Sk_all": Sk_all}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ZETA
# ─────────────────────────────────────────────────────────────────────────────

def _h_bond(i, j, traj_frame, M, N):
    """Return True if molecules i and j are H-bonded in traj_frame."""
    def _disp(a, b):
        return md.compute_displacements(traj_frame, [[a, b]],
                                        periodic=True, opt=True)[0, 0]
    def _dist(a, b):
        return md.compute_distances(traj_frame,  [[a, b]],
                                    periodic=True, opt=True)[0, 0]

    d_ij = _dist(M * i, M * j)
    if d_ij >= R_OO_CUTOFF_NM:
        return False

    r_ij  = _disp(M * i, M * j)
    r_iH1 = _disp(M * i, M * i + 1)
    r_iH2 = _disp(M * i, M * i + 2)
    r_jH1 = _disp(M * j, M * j + 1)
    r_jH2 = _disp(M * j, M * j + 2)

    d_iH1 = np.linalg.norm(r_iH1)
    d_iH2 = np.linalg.norm(r_iH2)
    d_jH1 = np.linalg.norm(r_jH1)
    d_jH2 = np.linalg.norm(r_jH2)

    eps = 1e-12
    cos1 =  np.dot(r_ij,  r_iH1) / (d_ij * (d_iH1 + eps))
    cos2 =  np.dot(r_ij,  r_iH2) / (d_ij * (d_iH2 + eps))
    cos3 = -np.dot(r_ij,  r_jH1) / (d_ij * (d_jH1 + eps))
    cos4 = -np.dot(r_ij,  r_jH2) / (d_ij * (d_jH2 + eps))

    return max(cos1, cos2, cos3, cos4) > COS_ANGLE_LIMIT


def _molecule_zeta(i, traj_O, traj_frame, N, M):
    """Compute ζ for molecule i in a single frame."""
    others = [j for j in range(N) if j != i]
    pairs  = np.array([[i, j] for j in others])
    pdist  = md.compute_distances(traj_O, pairs, periodic=True, opt=True)[0]
    rank   = np.argsort(pdist)

    hb_neigh, non_hb = [], []
    for h in range(min(10, len(rank))):
        j = others[rank[h]]
        if _h_bond(i, j, traj_frame, M, N):
            hb_neigh.append(j)
        else:
            non_hb.append(j)

    if not non_hb:
        return 0.0

    n5 = non_hb[0]
    d5 = md.compute_distances(traj_O, [[i, n5]], periodic=True, opt=True)[0, 0]

    if hb_neigh:
        n4 = hb_neigh[-1]
        d4 = md.compute_distances(traj_O, [[i, n4]], periodic=True, opt=True)[0, 0]
    else:
        d4 = 0.0

    return float(d5 - d4)


def compute_zeta(dcd_file, pdb_file, n_frames_limit=None, checkpoint_mat=None):
    """
    Compute ζ for every molecule in every frame.
    Saves a checkpoint after each frame so you can resume interrupted runs.
    Returns array of shape (n_frames, N_mol).
    """
    traj = md.load(dcd_file, top=pdb_file)
    if n_frames_limit:
        traj = traj[:n_frames_limit]

    top     = traj.topology
    N       = top.n_residues
    M       = top.n_atoms // N
    nf      = traj.n_frames
    n_cores = multiprocessing.cpu_count()

    oxy_idx = [a.index for a in top.atoms if a.name == "O"]

    zeta_all   = np.zeros((nf, N))
    start_frame = 0

    # Resume from checkpoint if available
    if checkpoint_mat and os.path.isfile(checkpoint_mat):
        saved = scipy.io.loadmat(checkpoint_mat)["zeta_all"]
        saved_frames = saved.shape[0]
        if saved_frames <= nf:
            non_zero = np.any(saved != 0, axis=1)
            if non_zero.any():
                last_done = int(np.where(non_zero)[0][-1]) + 1
                zeta_all[:last_done] = saved[:last_done]
                start_frame = last_done
                print(f"  Resuming from frame {start_frame}/{nf}")

    print(f"  Computing ζ: {nf} frames, {N} molecules, {n_cores} cores …")
    t0 = time()

    for k in range(start_frame, nf):
        traj_frame = traj[k]
        traj_O     = traj_frame.atom_slice(oxy_idx)

        results = Parallel(n_jobs=n_cores)(
            delayed(_molecule_zeta)(i, traj_O, traj_frame, N, M)
            for i in range(N)
        )
        zeta_all[k] = results

        elapsed = time() - t0
        rate    = (k - start_frame + 1) / elapsed
        eta     = (nf - k - 1) / rate if rate > 0 else 0
        print(f"  Frame {k+1}/{nf}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
              flush=True)

        # Save checkpoint after every frame
        if checkpoint_mat:
            scipy.io.savemat(checkpoint_mat, {"zeta_all": zeta_all})

    print(f"  Zeta done in {time()-t0:.1f}s")
    return zeta_all


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(op_mat, zeta_mat, out_dir):
    """Run DBSCAN–GMM on the saved .mat files and write cluster_labels.csv."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Clustering → {out_dir}")

    df_raw = load_order_params(op_mat, zeta_mat, N_RUNS)
    df_sc  = scale_features(df_raw)
    labels = run_dbscan_gmm(df_sc, eps=DBSCAN_EPS,
                            min_samples=DBSCAN_MIN_SAMP, n_components=2)

    out_df = df_raw.copy()
    out_df["label_dbscan_gmm"] = labels
    csv_path = os.path.join(out_dir, "cluster_labels.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"  Saved cluster labels → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CONVERT LABELS TO MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_matrix(flat_csv, matrix_csv):
    """Reshape (n_frames*N, ) flat labels into (n_frames, N) matrix CSV."""
    print(f"  Converting labels to matrix → {matrix_csv}")
    df     = pd.read_csv(flat_csv)
    lc     = [c for c in df.columns if c.startswith("label_")][0]
    labels = df[lc].values

    n_total  = len(labels)
    n_frames = n_total // N_MOLECULES
    labels   = labels[: n_frames * N_MOLECULES]

    # Auto-orient: cluster 0 = LFTS (higher mean ζ)
    if "zeta_all" in df.columns:
        zeta = df["zeta_all"].values[: n_frames * N_MOLECULES]
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.any() and mask1.any():
            if zeta[mask0].mean() < zeta[mask1].mean():
                labels = labels.copy()
                labels[mask0], labels[mask1] = 1, 0
                labels[df[lc].values[: n_frames * N_MOLECULES] == 0] = 1
                labels[df[lc].values[: n_frames * N_MOLECULES] == 1] = 0
                print("  Auto-orient: swapped labels so cluster 0 = LFTS")

    matrix = labels.reshape(n_frames, N_MOLECULES)

    os.makedirs(os.path.dirname(matrix_csv) or ".", exist_ok=True)
    pd.DataFrame(matrix).to_csv(matrix_csv, index=False, header=False)

    for lbl in sorted(set(labels)):
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        pct  = 100 * np.mean(labels == lbl)
        print(f"    {name}: {pct:.1f}%")
    print(f"  Matrix shape: {matrix.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PATCH generate_paper_figures.py
# ─────────────────────────────────────────────────────────────────────────────

def patch_figure_script(temps_done):
    """
    Uncomment the relevant lines in generate_paper_figures.py for each
    newly processed temperature.
    """
    if not os.path.isfile(FIGURE_SCRIPT):
        print(f"  [WARN] {FIGURE_SCRIPT} not found — skipping patch.")
        return

    with open(FIGURE_SCRIPT) as fh:
        src = fh.read()

    orig = src
    for T in temps_done:
        tag = f"T{T}"   # e.g. T0, T10, T20

        # Lines that contain the tag and are still commented out
        patterns = [
            f"# {tag}: os.path.join(CLUSTER_DIR",   # labels flat CSV
            f"# MAT_4P_{tag.replace('-','_')}",       # matrix path
            f"# {T}: os.path.join(SIM_DATA_4P",      # DCD dict
            f"# {T}: os.path.join(SIM_DATA_4P",      # PDB dict
            f"# {T}: LABELS_4P",                      # temps_csv
            f"# {T}: MAT_4P",                         # mat_lookup
        ]
        for pat in patterns:
            src = src.replace(f"    # {T}: ",    f"    {T}: ")
            src = src.replace(f"    # {tag} ",   f"    {tag} ")
            src = src.replace(f"    # {tag}=",   f"    {tag}=")
            # Handle comment markers in the variable definition lines
            old = f"# LABELS_4P_{tag.upper()} "
            new = f"LABELS_4P_{tag.upper()} "
            src = src.replace(old, new)
            old = f"# MAT_4P_{tag.upper()} "
            new = f"MAT_4P_{tag.upper()} "
            src = src.replace(old, new)

    if src != orig:
        with open(FIGURE_SCRIPT, "w") as fh:
            fh.write(src)
        print(f"  Patched {FIGURE_SCRIPT} for temperatures {temps_done}")
    else:
        print(f"  No changes needed in {FIGURE_SCRIPT} "
              f"(lines may already be active or pattern not matched).")
        print("  → Manually uncomment the temperature entries in the script.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_temperature(T, args):
    """Run the full pipeline for one temperature."""
    tag = f"T{T}" if T >= 0 else f"T{T}"   # e.g. T0, T10, T20
    print("\n" + "═"*65)
    print(f"  TEMPERATURE  T = +{T}°C")
    print("═"*65)

    dcd_file = os.path.join(SIM_DATA, f"dcd_tip4p2005_T{T}_N1024_Run01_0.dcd")
    pdb_file = os.path.join(SIM_DATA, f"inistate_tip4p2005_T{T}_N1024_Run01.pdb")

    for path, label in [(dcd_file, "DCD"), (pdb_file, "PDB")]:
        if not os.path.isfile(path):
            print(f"  [SKIP T={T}] {label} not found: {path}")
            return False

    op_mat   = os.path.join(PARAM_DIR, f"OrderParam_tip4p2005_T{T}_Run01.mat")
    zeta_mat = os.path.join(PARAM_DIR, f"OrderParamZeta_tip4p2005_T{T}_Run01.mat")
    clust_dir = os.path.join(CLUST_DIR, f"tip4p2005_T{T}_dbscan_gmm")
    flat_csv  = os.path.join(clust_dir, "cluster_labels.csv")
    mat_csv   = os.path.join(CS_DIR,
                             f"cluster_labels_matrix_tip4p2005_T{T}_dbscan_gmm.csv")
    ckpt_mat  = os.path.join(PARAM_DIR, f"OrderParamZeta_tip4p2005_T{T}_Run01_ckpt.mat")

    # ── Step 1: Order parameters ─────────────────────────────────────────────
    if not args.skip_op:
        if os.path.isfile(op_mat):
            print(f"\nStep 1 — Order params already exist, skipping: {op_mat}")
        else:
            print(f"\nStep 1 — Computing order parameters for T={T}°C …")
            params = compute_order_params(dcd_file, pdb_file,
                                         n_frames_limit=args.n_frames)
            scipy.io.savemat(op_mat, params)
            print(f"  Saved → {op_mat}")
    else:
        print(f"\nStep 1 — Skipped (--skip-op).")
        if not os.path.isfile(op_mat):
            print(f"  [WARN] op_mat not found: {op_mat} — downstream steps may fail.")

    # ── Step 2: Zeta ─────────────────────────────────────────────────────────
    if not args.skip_zeta:
        if os.path.isfile(zeta_mat):
            print(f"\nStep 2 — Zeta already exists, skipping: {zeta_mat}")
        else:
            print(f"\nStep 2 — Computing ζ for T={T}°C …  (slow step)")
            zeta_all = compute_zeta(dcd_file, pdb_file,
                                    n_frames_limit=args.n_frames,
                                    checkpoint_mat=ckpt_mat)
            scipy.io.savemat(zeta_mat, {"zeta_all": zeta_all})
            # Remove checkpoint after successful completion
            if os.path.isfile(ckpt_mat):
                os.remove(ckpt_mat)
            print(f"  Saved → {zeta_mat}")
    else:
        print(f"\nStep 2 — Skipped (--skip-zeta).")
        if not os.path.isfile(zeta_mat):
            print(f"  [WARN] zeta_mat not found: {zeta_mat} — clustering may fail.")

    # ── Step 3: Clustering ───────────────────────────────────────────────────
    if not os.path.isfile(op_mat) or not os.path.isfile(zeta_mat):
        print(f"\nStep 3 — Skipped (missing .mat files).")
        return False

    if os.path.isfile(flat_csv):
        print(f"\nStep 3 — Cluster labels already exist, skipping: {flat_csv}")
    else:
        print(f"\nStep 3 — Running DBSCAN–GMM clustering …")
        run_clustering(op_mat, zeta_mat, clust_dir)

    # ── Step 4: Convert to matrix ────────────────────────────────────────────
    if os.path.isfile(mat_csv):
        print(f"\nStep 4 — Matrix already exists, skipping: {mat_csv}")
    else:
        print(f"\nStep 4 — Converting labels to matrix format …")
        convert_to_matrix(flat_csv, mat_csv)

    print(f"\n  ✓ T={T}°C complete.")
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare order params + cluster labels for T=0, +10, +20°C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--temps", nargs="+", type=int, default=[0, 10, 20],
                   help="Temperatures to process (°C, positive only)")
    p.add_argument("--skip-op",   action="store_true",
                   help="Skip order parameter computation (use existing .mat)")
    p.add_argument("--skip-zeta", action="store_true",
                   help="Skip zeta computation (use existing .mat)")
    p.add_argument("--n-frames",  type=int, default=N_RUNS,
                   help=f"Frames to load per DCD (default: {N_RUNS}, matching clustering)")
    p.add_argument("--no-patch",  action="store_true",
                   help="Do not modify generate_paper_figures.py when done")
    return p.parse_args()


def main():
    args = parse_args()

    print("╔" + "═"*63 + "╗")
    print("║  POSITIVE TEMPERATURE PIPELINE  (TIP4P/2005)             ║")
    print("╠" + "═"*63 + "╣")
    print(f"║  Temperatures : {str(args.temps):<46}║")
    print(f"║  Skip OP      : {str(args.skip_op):<46}║")
    print(f"║  Skip ζ       : {str(args.skip_zeta):<46}║")
    print(f"║  n-frames     : {str(args.n_frames):<46}║")
    print("╚" + "═"*63 + "╝")

    done = []
    for T in sorted(args.temps):
        if T < 0:
            print(f"  [SKIP] T={T}°C is negative — this script handles positive temps only.")
            continue
        ok = process_temperature(T, args)
        if ok:
            done.append(T)

    print("\n" + "═"*65)
    print(f"  Pipeline complete.  Processed temperatures: {done}")

    if done and not args.no_patch:
        print("\nStep 5 — Patching generate_paper_figures.py …")
        patch_figure_script(done)
        print("\n  Run the figure generator to see all temperatures:")
        print("    python generate_paper_figures.py --sections c3 --no-cache")
    elif done:
        print("\n  --no-patch specified. Manually uncomment these temperatures")
        print("  in generate_paper_figures.py:")
        for T in done:
            print(f"    T = +{T}°C")

    print("═"*65)


if __name__ == "__main__":
    main()
