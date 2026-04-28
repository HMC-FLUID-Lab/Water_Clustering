#!/usr/bin/env python3
"""
run_three_model_dbscan_gmm.py
=============================
Run **the same** DBSCAN → GMM pipeline on order-parameter data for up to three
water models (e.g. TIP4P/2005, TIP5P, SWM4-NDP): same ``eps``, same
``min_samples``, two GMM components.

Writes, per model subdirectory under ``--out-dir``::

  cluster_labels.csv              # flat table + label_dbscan_gmm column
  cluster_labels_matrix_dbscan_gmm.csv   # frames × molecules (for S(k))

Then run ``plot_sk_multimodel.py`` with DCD/PDB paths and the matrix CSVs.

Example
-------
  python run_three_model_dbscan_gmm.py \\
      --out-dir ./three_model_dbscan_gmm_T-20 \\
      --n-runs 1 --n-molecules 1024 \\
      --eps 0.15 --min-samples 20 \\
      --tip4p2005-mat /path/OrderParam_tip4p2005_T-20_Run01.mat \\
      --tip4p2005-zeta /path/OrderParamZeta_tip4p2005_T-20_Run01.mat \\
      --tip5p-mat /path/OrderParam_tip5p_T-20_Run01.mat \\
      --tip5p-zeta /path/OrderParamZeta_tip5p_T-20_Run01.mat \\
      --swm4ndp-mat /path/OrderParam_swm4ndp_T-20_Run01.mat \\
      --swm4ndp-zeta /path/OrderParamZeta_swm4ndp_T-20_Run01.mat
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from water_clustering import load_order_params, scale_features, run_dbscan_gmm


def _reshape_matrix(labels: np.ndarray, n_molecules: int) -> np.ndarray:
    n_total = len(labels)
    if n_total % n_molecules != 0:
        raise ValueError(
            f"Label count {n_total} is not divisible by n_molecules={n_molecules}"
        )
    n_frames = n_total // n_molecules
    return labels.reshape(n_frames, n_molecules)


def run_one_model(
    name: str,
    mat_file: str | None,
    zeta_file: str | None,
    out_root: str,
    n_runs: int,
    n_molecules: int,
    eps: float,
    min_samples: int,
) -> str | None:
    if not mat_file or not zeta_file:
        print(f"  [SKIP] {name}: mat or zeta path not provided")
        return None
    if not os.path.isfile(mat_file):
        print(f"  [SKIP] {name}: MAT not found: {mat_file}")
        return None
    if not os.path.isfile(zeta_file):
        print(f"  [SKIP] {name}: Zeta MAT not found: {zeta_file}")
        return None

    print(f"\n{'='*60}\n {name}\n{'='*60}")
    df_raw = load_order_params(mat_file, zeta_file, n_runs)
    df_scaled = scale_features(df_raw)
    labels = run_dbscan_gmm(
        df_scaled,
        eps=eps,
        min_samples=min_samples,
        n_components=2,
        random_state=42,
        min_cluster_size=None,
    )

    out_df = df_raw.copy()
    out_df["label_dbscan_gmm"] = labels

    sub = os.path.join(out_root, name)
    os.makedirs(sub, exist_ok=True)
    csv_flat = os.path.join(sub, "cluster_labels.csv")
    out_df.to_csv(csv_flat, index=False)
    print(f"  Saved: {csv_flat}")

    mat = _reshape_matrix(labels, n_molecules)
    mat_path = os.path.join(sub, "cluster_labels_matrix_dbscan_gmm.csv")
    pd.DataFrame(mat).to_csv(mat_path, index=False, header=False)
    print(f"  Saved: {mat_path}  shape={mat.shape}")

    return sub


def parse_args():
    p = argparse.ArgumentParser(
        description="DBSCAN→GMM for three models with shared eps/min_samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-o", "--out-dir", required=True, help="Output root directory")
    p.add_argument("--n-runs", type=int, default=1, help="Number of MD runs in MAT files")
    p.add_argument(
        "--n-molecules",
        type=int,
        default=1024,
        help="Molecules per frame (for matrix reshape)",
    )
    p.add_argument("--eps", type=float, default=0.15)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--tip4p2005-mat", default=None)
    p.add_argument("--tip4p2005-zeta", default=None)
    p.add_argument("--tip5p-mat", default=None)
    p.add_argument("--tip5p-zeta", default=None)
    p.add_argument("--swm4ndp-mat", default=None)
    p.add_argument("--swm4ndp-zeta", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("DBSCAN→GMM  |  eps={}  min_samples={}".format(args.eps, args.min_samples))

    models = [
        ("tip4p2005", args.tip4p2005_mat, args.tip4p2005_zeta),
        ("tip5p", args.tip5p_mat, args.tip5p_zeta),
        ("swm4ndp", args.swm4ndp_mat, args.swm4ndp_zeta),
    ]

    done = []
    for name, ma, ze in models:
        sub = run_one_model(
            name,
            ma,
            ze,
            args.out_dir,
            args.n_runs,
            args.n_molecules,
            args.eps,
            args.min_samples,
        )
        if sub:
            done.append((name, sub))

    print("\n" + "=" * 60)
    if not done:
        print("No models processed. Check MAT/Zeta paths.")
        sys.exit(1)

    print("Next: run plot_sk_multimodel.py with DCD/PDB and matrix CSV, e.g.\n")
    ann = f"ε={args.eps:g}  min_samples={args.min_samples}  DBSCAN→GMM"
    print(f'  python plot_sk_multimodel.py --output-dir ./multimodel_sk \\\n    --annotation "{ann}" \\\n    --conditions \\')
    for name, sub in done:
        print(
            f'      "{name} | /path/to/dcd_{name}_....dcd | '
            f'/path/to/inistate_....pdb | {sub}/cluster_labels_matrix_dbscan_gmm.csv" \\'
        )
    print("      ")
    print("=" * 60)


if __name__ == "__main__":
    main()
