#!/usr/bin/env python3
"""
convert_cluster_labels.py
==========================
Convert cluster labels from flat format (molecules concatenated across runs)
to matrix format (frames × molecules) for use with structure_factor_bycluster.py

Input:  cluster_labels.csv from water_clustering.py
        Shape: (n_runs × n_molecules × n_frames_per_run, n_features + 1)
        One row per molecule across all frames and runs

Output: cluster_labels_matrix.csv
        Shape: (total_frames, n_molecules)
        One row per frame, one column per molecule
"""

import argparse
import numpy as np
import pandas as pd
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert cluster labels from flat to matrix format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to cluster_labels.csv from water_clustering.py'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for matrix format CSV (frames × molecules)'
    )
    
    parser.add_argument(
        '--n-runs',
        type=int,
        required=True,
        help='Number of MD runs concatenated (must match clustering)'
    )
    
    parser.add_argument(
        '--n-molecules',
        type=int,
        required=True,
        help='Number of molecules per frame (typically 1024)'
    )
    
    parser.add_argument(
        '--label-column',
        type=str,
        default='label_dbscan_gmm',
        help='Column name for cluster labels (e.g., label_dbscan_gmm, label_gmm, label_dbscan)'
    )

    parser.add_argument(
        '--auto-orient',
        action='store_true',
        default=False,
        help='Auto-swap cluster 0 and 1 so that cluster 0 = LFTS (higher mean zeta). '
             'Requires zeta_all column in the input CSV.'
    )

    parser.add_argument(
        '--zeta-column',
        type=str,
        default='zeta_all',
        help='Column name for the zeta order parameter used by --auto-orient (default: zeta_all)'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("CONVERT CLUSTER LABELS: FLAT → MATRIX FORMAT")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Runs:   {args.n_runs}")
    print(f"Molecules per frame: {args.n_molecules}")
    print(f"Label column: {args.label_column}")
    print("="*70)
    
    # Load cluster labels
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nLoaded {len(df):,} rows from input file")
    print(f"Columns: {list(df.columns)}")
    
    # Check if label column exists, or auto-detect
    if args.label_column not in df.columns:
        # Try to auto-detect label column
        label_cols = [col for col in df.columns if col.startswith('label_')]
        
        if len(label_cols) == 0:
            print(f"\nERROR: Column '{args.label_column}' not found and no label columns detected.", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        elif len(label_cols) == 1:
            args.label_column = label_cols[0]
            print(f"\nAuto-detected label column: {args.label_column}")
        else:
            print(f"\nERROR: Column '{args.label_column}' not found.", file=sys.stderr)
            print(f"Available label columns: {label_cols}", file=sys.stderr)
            print(f"Please specify one with --label-column", file=sys.stderr)
            sys.exit(1)
    
    # Extract labels
    labels = df[args.label_column].values
    n_total = len(labels)
    
    # Calculate expected dimensions
    n_frames_per_run = n_total // (args.n_runs * args.n_molecules)
    n_frames_total = args.n_runs * n_frames_per_run
    expected_total = n_frames_total * args.n_molecules
    
    print(f"\nDimensions:")
    print(f"  Total data points: {n_total:,}")
    print(f"  Frames per run: {n_frames_per_run}")
    print(f"  Total frames: {n_frames_total}")
    print(f"  Expected total: {expected_total:,}")
    
    # Validate dimensions
    if n_total != expected_total:
        print(f"\nWARNING: Data size mismatch!", file=sys.stderr)
        print(f"  Got {n_total:,} rows, expected {expected_total:,}", file=sys.stderr)
        print(f"  This might cause incorrect reshaping.", file=sys.stderr)
        
        # Try to determine correct parameters
        possible_frames = n_total // args.n_molecules
        print(f"\nSuggested fix: Use --n-runs {possible_frames // 20} or check --n-molecules", file=sys.stderr)
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        
        # Use actual data size
        n_frames_total = n_total // args.n_molecules
        labels = labels[:n_frames_total * args.n_molecules]
    
    # Auto-orient: cluster 0 = LFTS (higher mean zeta)
    if args.auto_orient:
        if args.zeta_column not in df.columns:
            print(f"\nWARNING: --auto-orient requested but '{args.zeta_column}' column not found. Skipping.")
        else:
            zeta = df[args.zeta_column].values
            mask0 = labels == 0
            mask1 = labels == 1
            if mask0.any() and mask1.any():
                mean_zeta_0 = zeta[mask0].mean()
                mean_zeta_1 = zeta[mask1].mean()
                print(f"\nAuto-orient check:")
                print(f"  Mean zeta cluster 0 = {mean_zeta_0:.4f}")
                print(f"  Mean zeta cluster 1 = {mean_zeta_1:.4f}")
                if mean_zeta_0 < mean_zeta_1:
                    labels = labels.copy()
                    labels[mask0] = 1
                    labels[mask1] = 0
                    print("  → Swapped 0↔1 (cluster 1 had higher zeta, now assigned to cluster 0)")
                else:
                    print("  → No swap needed (cluster 0 already has higher zeta)")
            else:
                print("\nWARNING: --auto-orient: one of the clusters is empty, skipping swap.")

    # Reshape to matrix: (frames, molecules)
    print(f"\nReshaping to ({n_frames_total}, {args.n_molecules}) matrix...")
    
    try:
        label_matrix = labels.reshape(n_frames_total, args.n_molecules)
    except ValueError as e:
        print(f"\nERROR: Reshape failed: {e}", file=sys.stderr)
        print(f"Cannot reshape {len(labels)} elements into ({n_frames_total}, {args.n_molecules})", file=sys.stderr)
        sys.exit(1)
    
    # Convert to DataFrame for easy saving
    label_df = pd.DataFrame(label_matrix)
    
    # Print statistics
    unique_labels = np.unique(label_matrix)
    print(f"\nLabel statistics:")
    for label in sorted(unique_labels):
        count = np.sum(label_matrix == label)
        percentage = 100 * count / label_matrix.size
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        print(f"  {label_name:15s}: {count:8,} ({percentage:5.2f}%)")
    
    # Save to CSV
    label_df.to_csv(args.output, index=False, header=False)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Saved to: {args.output}")
    print(f"  Shape: {label_matrix.shape} (frames × molecules)")
    print("="*70)


if __name__ == "__main__":
    main()
