#!/usr/bin/env python3
"""
Structure Factor Computation for Water MD Trajectories
Converted from Compute_Structure_Factor_v3.ipynb for remote server execution
Adapted for Tanaka simulation outputs (TIP4P/2005 and TIP5P models)

Based on: J. Am. Chem. Soc. 2020, 142, 2868−2875 (Tanaka paper)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote server
import matplotlib.pyplot as plt
import mdtraj as md
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks
import os
import sys
import argparse
import glob
from time import time

# ============================================================================
# Configuration and Argument Parsing
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compute structure factor S(k) for water MD trajectories'
    )
    parser.add_argument(
        '--dcd-file',
        type=str,
        required=True,
        help='Path to DCD trajectory file (e.g., dcd_tip4p2005_T-40_Run01_0.dcd)'
    )
    parser.add_argument(
        '--pdb-file',
        type=str,
        required=True,
        help='Path to PDB topology file (e.g., inistate_tip4p2005_T-40_Run01.pdb)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./structure_factor_results',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='unknown',
        help='Water model name (e.g., tip4p2005, tip5p)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Simulation temperature in Celsius'
    )
    parser.add_argument(
        '--rc-cutoff',
        type=float,
        default=1.5,
        help='Cutoff distance rc in nanometers (default: 1.5 nm)'
    )
    parser.add_argument(
        '--k-max',
        type=float,
        default=50.0,
        help='Maximum wave number k in inverse nanometers (default: 50 nm^-1)'
    )
    parser.add_argument(
        '--k-points',
        type=int,
        default=500,
        help='Number of k points to compute (default: 500)'
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to analyze (default: all frames)'
    )
    parser.add_argument(
        '--atom-type',
        type=str,
        default='O',
        help='Atom type to analyze (default: O for oxygen)'
    )
    parser.add_argument(
        '--cluster-labels',
        type=str,
        default=None,
        metavar='CSV',
        help='Path to cluster labels CSV from cluster_v3.py (shape: frames x molecules). '
             'When provided, S(k) is computed separately for each cluster group. '
             'Molecules with label -1 (noise) are excluded.'
    )
    parser.add_argument(
        '--cluster-id',
        type=int,
        nargs='+',
        default=None,
        metavar='ID',
        help='Which cluster ID(s) to compute S(k) for (default: all clusters). '
             'E.g. --cluster-id 0 for only cluster 0, or --cluster-id 0 1 for both.'
    )
    
    return parser.parse_args()



### Calculate structure factor S(k)
def compute_structure_factor(trajectory, atom_indices, k_values, rc_cutoff):
    """
    Formula from Tanaka paper (Debye scattering function):
    S(k) = 1 + (1/N) * sum_i sum_{j≠i} [sin(k*r_ij) / (k*r_ij)] * W(r_ij)
    
    Where W(r_ij) = sin(π*r_ij/rc) / (π*r_ij/rc) is the window function
    """
    
    n_frames = trajectory.n_frames
    n_atoms = len(atom_indices)
    n_k = len(k_values)
    
    print(f"\nComputing structure factor...")
    print(f"  Frames: {n_frames}")
    print(f"  Atoms: {n_atoms}")
    print(f"  k points: {n_k}")
    print(f"  k range: [{k_values.min():.2f}, {k_values.max():.2f}] nm^-1")
    print(f"  Cutoff rc: {rc_cutoff:.3f} nm")
    
    # Store S(k) for each frame
    S_k_frames = np.zeros((n_frames, n_k))
    
    t0 = time()
    
    for frame_idx in range(n_frames):
        if (frame_idx + 1) % 10 == 0:
            elapsed = time() - t0
            rate = (frame_idx + 1) / elapsed
            eta = (n_frames - frame_idx - 1) / rate
            print(f"  Frame {frame_idx+1}/{n_frames} ({frame_idx/n_frames*100:.1f}%) "
                  f"- Rate: {rate:.1f} frames/s, ETA: {eta:.0f}s")
        
        # Get positions for this frame (only selected atoms)
        frame = trajectory[frame_idx]
        positions = frame.xyz[0, atom_indices, :]  # Shape: (n_atoms, 3), in nm
        
        # Compute all pairwise distances
        # Create pairs: (i, j) for all i != j
        pairs = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):  # Only upper triangle to avoid double counting
                pairs.append([atom_indices[i], atom_indices[j]])
        
        pairs = np.array(pairs)
        
        # Compute distances using mdtraj (handles periodic boundary conditions)
        distances = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]  # in nm
        
        # For each k value, compute S(k)
        for k_idx, k in enumerate(k_values):
            
            # Skip k=0 to avoid division by zero
            if k == 0:
                S_k_frames[frame_idx, k_idx] = 1.0
                continue
            
            # Compute window function: W(r) = sin(π*r/rc) / (π*r/rc)
            with np.errstate(divide='ignore', invalid='ignore'):
                window = np.sin(np.pi * distances / rc_cutoff) / (np.pi * distances / rc_cutoff)
                window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute sin(k*r) / (k*r) * W(r)
            kr = k * distances
            with np.errstate(divide='ignore', invalid='ignore'):
                sinc_kr = np.sin(kr) / kr
                sinc_kr = np.nan_to_num(sinc_kr, nan=1.0, posinf=0.0, neginf=0.0)
            
            # Structure factor: S(k) = 1 + (1/N) * sum_{pairs} [sin(kr)/(kr)] * W(r)
            # Note: multiply by 2 because we only computed upper triangle
            S_k_frames[frame_idx, k_idx] = 1.0 + (2.0 / n_atoms) * np.sum(sinc_kr * window)
    
    t1 = time()
    print(f"✓ Computation completed in {t1-t0:.1f} seconds")
    
    # Average over frames
    S_k_avg = S_k_frames.mean(axis=0)
    S_k_std = S_k_frames.std(axis=0)
    
    print(f"  S(k) range: [{S_k_avg.min():.3f}, {S_k_avg.max():.3f}]")
    
    return S_k_avg, S_k_std, S_k_frames


def compute_partial_structure_factor_OO(trajectory, rc_cutoff, k_values):
    """
    Compute O-O partial structure factor specifically for water
    
    Parameters:
    -----------
    trajectory : mdtraj.Trajectory
        Water MD trajectory
    rc_cutoff : float
        Cutoff distance (nm)
    k_values : numpy array
        Wave numbers
        
    Returns:
    --------
    S_k_avg : numpy array
        Average O-O partial structure factor
    """
    
    # Get oxygen atom indices
    topology = trajectory.topology
    oxygen_indices = [atom.index for atom in topology.atoms if atom.name == 'O']
    
    print(f"Found {len(oxygen_indices)} oxygen atoms")
    
    # Compute structure factor for oxygen atoms only
    S_k_avg, S_k_std, S_k_frames = compute_structure_factor(
        trajectory, oxygen_indices, k_values, rc_cutoff
    )
    
    return S_k_avg, S_k_std, S_k_frames


def compute_per_cluster_structure_factor(trajectory, rc_cutoff, k_values,
                                          cluster_labels_matrix):
    """
    Compute O-O partial structure factor separately for each cluster.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Water MD trajectory.
    rc_cutoff : float
        Cutoff distance (nm).
    k_values : numpy array
        Wave numbers to evaluate.
    cluster_labels_matrix : np.ndarray, shape (n_frames, n_molecules)
        Per-frame, per-molecule cluster labels.
        -1 = noise (excluded), 0/1/... = cluster IDs.

    Returns
    -------
    results : dict
        {cluster_id: {'S_k_avg': array, 'S_k_std': array, 'S_k_frames': array}}
    """
    topology = trajectory.topology
    # Build residue -> oxygen atom index mapping
    residue_oxygen = {}
    for atom in topology.atoms:
        if atom.name == 'O':
            residue_oxygen[atom.residue.index] = atom.index

    n_frames_traj = trajectory.n_frames
    n_frames_labels, n_molecules = cluster_labels_matrix.shape
    n_frames = min(n_frames_traj, n_frames_labels)
    n_k = len(k_values)

    if n_frames_traj != n_frames_labels:
        print(f"  WARNING: Trajectory has {n_frames_traj} frames but labels have "
              f"{n_frames_labels} frames. Using first {n_frames}.")

    # Find all unique cluster IDs (exclude noise = -1)
    unique_clusters = sorted(set(int(c) for c in np.unique(cluster_labels_matrix) if c >= 0))
    print(f"\n  Computing per-cluster S(k) for clusters: {unique_clusters}")
    print(f"  Frames: {n_frames}, Molecules: {n_molecules}")

    results = {}

    for cluster_id in unique_clusters:
        print(f"\n  --- Cluster {cluster_id} ---")

        S_k_frames = np.zeros((n_frames, n_k))
        t0 = time()

        for frame_idx in range(n_frames):
            if (frame_idx + 1) % 10 == 0:
                elapsed = time() - t0
                rate = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (n_frames - frame_idx - 1) / rate if rate > 0 else 0
                print(f"    Frame {frame_idx+1}/{n_frames} "
                      f"({frame_idx/n_frames*100:.1f}%) "
                      f"- Rate: {rate:.1f} frames/s, ETA: {eta:.0f}s")

            # Get oxygen indices for molecules in this cluster at this frame
            frame_labels = cluster_labels_matrix[frame_idx]
            mol_indices = np.where(frame_labels == cluster_id)[0]

            # Map molecule indices to oxygen atom indices
            atom_indices = []
            for mol_idx in mol_indices:
                if mol_idx in residue_oxygen:
                    atom_indices.append(residue_oxygen[mol_idx])

            atom_indices = np.array(atom_indices)
            n_atoms = len(atom_indices)

            if n_atoms < 2:
                S_k_frames[frame_idx, :] = 1.0
                continue

            # Get this frame
            frame = trajectory[frame_idx]

            # Compute all pairwise distances for these atoms
            pairs = []
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    pairs.append([atom_indices[i], atom_indices[j]])
            pairs = np.array(pairs)

            distances = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]

            # Compute S(k) for each k value
            for k_idx, k in enumerate(k_values):
                if k == 0:
                    S_k_frames[frame_idx, k_idx] = 1.0
                    continue

                # Window function
                with np.errstate(divide='ignore', invalid='ignore'):
                    window = np.sin(np.pi * distances / rc_cutoff) / \
                             (np.pi * distances / rc_cutoff)
                    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

                # Debye scattering function
                kr = k * distances
                with np.errstate(divide='ignore', invalid='ignore'):
                    sinc_kr = np.sin(kr) / kr
                    sinc_kr = np.nan_to_num(sinc_kr, nan=1.0, posinf=0.0, neginf=0.0)

                S_k_frames[frame_idx, k_idx] = 1.0 + \
                    (2.0 / n_atoms) * np.sum(sinc_kr * window)

        elapsed = time() - t0
        S_k_avg = S_k_frames.mean(axis=0)
        S_k_std = S_k_frames.std(axis=0)

        n_mol_avg = (cluster_labels_matrix[:n_frames] == cluster_id).sum(axis=1).mean()
        print(f"    Completed in {elapsed:.1f}s")
        print(f"    Avg molecules per frame: {n_mol_avg:.0f}")
        print(f"    S(k) range: [{S_k_avg.min():.3f}, {S_k_avg.max():.3f}]")

        results[cluster_id] = {
            'S_k_avg': S_k_avg,
            'S_k_std': S_k_std,
            'S_k_frames': S_k_frames,
        }

    return results


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_trajectory(dcd_file, pdb_file, n_frames=None):
    """
    Load MD trajectory from DCD and PDB files
    
    Parameters:
    -----------
    dcd_file : str
        Path to DCD trajectory file
    pdb_file : str
        Path to PDB topology file
    n_frames : int or None
        Number of frames to load (None = all)
        
    Returns:
    --------
    trajectory : mdtraj.Trajectory
        Loaded trajectory object
    """
    
    print(f"\nLoading trajectory...")
    print(f"  DCD file: {dcd_file}")
    print(f"  PDB file: {pdb_file}")
    
    if not os.path.exists(dcd_file):
        raise FileNotFoundError(f"DCD file not found: {dcd_file}")
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Load topology
    topology = md.load(pdb_file).topology
    
    # Load trajectory
    if n_frames is not None:
        # Load only specified number of frames
        traj = md.load(dcd_file, top=topology)
        if traj.n_frames > n_frames:
            indices = np.linspace(0, traj.n_frames-1, n_frames, dtype=int)
            traj = traj[indices]
    else:
        traj = md.load(dcd_file, top=topology)
    
    print(f"✓ Loaded trajectory:")
    print(f"    Frames: {traj.n_frames}")
    print(f"    Atoms: {traj.n_atoms}")
    print(f"    Residues: {traj.n_residues}")
    print(f"    Time: {traj.time[0]:.2f} - {traj.time[-1]:.2f} ps")
    
    return traj


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_structure_factor(k_values, S_k_avg, S_k_std, output_dir, model_name, temperature):
    """
    Plot structure factor S(k) vs k
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with error bars
    ax.plot(k_values, S_k_avg, 'b-', linewidth=2, label='S(k) average')
    ax.fill_between(k_values, S_k_avg - S_k_std, S_k_avg + S_k_std,
                    alpha=0.3, color='blue', label='±1 std')
    
    # Mark characteristic peaks from Tanaka paper
    # kT1 ≈ 3/4 * 2π/rOO (FSDP from LFTS)
    # kD1 ≈ 1 * 2π/rOO (normal liquid peak from DNLS)
    r_OO = 0.285  # nm, typical O-O distance
    k_T1 = 0.75 * 2 * np.pi / r_OO  # ~16.5 nm^-1
    k_D1 = 1.0 * 2 * np.pi / r_OO   # ~22 nm^-1
    
    ax.axvline(k_T1, color='green', linestyle='--', alpha=0.5, 
              label=f'kT1 (FSDP) ≈ {k_T1:.1f} nm⁻¹')
    ax.axvline(k_D1, color='red', linestyle='--', alpha=0.5,
              label=f'kD1 (normal) ≈ {k_D1:.1f} nm⁻¹')
    
    ax.set_xlabel('Wave number k (nm⁻¹)', fontsize=14)
    ax.set_ylabel('Structure factor S(k)', fontsize=14)
    ax.set_title(f'O-O Partial Structure Factor - {model_name} at {temperature}°C', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(k_values.min(), k_values.max())
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'structure_factor_{model_name}_T{temperature}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved structure factor plot: {output_file}")
    plt.close()


def plot_structure_factor_normalized(k_values, S_k_avg, output_dir, model_name, temperature):
    """
    Plot normalized structure factor (scaled by nearest-neighbor distance)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Normalize k by r_OO
    r_OO = 0.285  # nm
    k_norm = k_values * r_OO / (2 * np.pi)
    
    ax.plot(k_norm, S_k_avg, 'b-', linewidth=2)
    
    # Mark expected peak positions
    ax.axvline(0.75, color='green', linestyle='--', alpha=0.5, 
              label='kT1 ≈ 3/4 (FSDP, tetrahedral)')
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5,
              label='kD1 ≈ 1 (normal liquid)')
    
    ax.set_xlabel('Normalized wave number k*r_OO/(2π)', fontsize=14)
    ax.set_ylabel('Structure factor S(k)', fontsize=14)
    ax.set_title(f'Normalized Structure Factor - {model_name} at {temperature}°C', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(0, 3.0)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'structure_factor_normalized_{model_name}_T{temperature}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved normalized plot: {output_file}")
    plt.close()


def _find_main_peak(k_arr, sk_arr, k_min=None, k_max=None):
    """Find the position and height of the tallest peak in S(k) within [k_min, k_max]."""
    mask = np.ones(len(k_arr), dtype=bool)
    if k_min is not None:
        mask &= k_arr >= k_min
    if k_max is not None:
        mask &= k_arr <= k_max

    k_sub = k_arr[mask]
    sk_sub = sk_arr[mask]

    if len(sk_sub) < 3:
        return None, None

    peaks, props = find_peaks(sk_sub, prominence=0.05)
    if len(peaks) == 0:
        # Fallback: use the global maximum in the range
        idx = np.argmax(sk_sub)
        return k_sub[idx], sk_sub[idx]

    # Pick the tallest peak
    tallest = peaks[np.argmax(sk_sub[peaks])]
    return k_sub[tallest], sk_sub[tallest]


def plot_per_cluster_structure_factor(k_values, cluster_results, output_dir,
                                      model_name, temperature):
    """
    Plot S(k) for each cluster on the same axes for comparison.
    Includes auto-detected peak annotations and shaded structural zones.
    """
    CLUSTER_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    CLUSTER_NAMES = {
        0: r'Cluster 0 ($\rho$-state / normal liquid)',
        1: r'Cluster 1 (S-state / tetrahedral)',
    }

    r_OO = 0.285  # nm, typical O-O nearest-neighbor distance

    # =====================================================================
    # Absolute k plot
    # =====================================================================
    fig, ax = plt.subplots(figsize=(13, 7))

    for cid, res in sorted(cluster_results.items()):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        label = CLUSTER_NAMES.get(cid, f'Cluster {cid}')
        ax.plot(k_values, res['S_k_avg'], linewidth=2, color=color, label=label)
        ax.fill_between(k_values,
                        res['S_k_avg'] - res['S_k_std'],
                        res['S_k_avg'] + res['S_k_std'],
                        alpha=0.12, color=color)

        # Auto-detect and annotate the main peak
        k_peak, sk_peak = _find_main_peak(k_values, res['S_k_avg'],
                                           k_min=10.0, k_max=40.0)
        if k_peak is not None:
            ax.annotate(
                f'peak = {k_peak:.1f} nm$^{{-1}}$',
                xy=(k_peak, sk_peak),
                xytext=(k_peak + 2.0, sk_peak + 0.3),
                fontsize=10, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            )

    # Reference lines
    k_T1 = 0.75 * 2 * np.pi / r_OO
    k_D1 = 1.0 * 2 * np.pi / r_OO
    ax.axvline(k_T1, color='green', linestyle='--', alpha=0.4,
               label=f'$k_{{T1}}$ (FSDP) = {k_T1:.1f} nm$^{{-1}}$')
    ax.axvline(k_D1, color='gray', linestyle='--', alpha=0.4,
               label=f'$k_{{D1}}$ (normal) = {k_D1:.1f} nm$^{{-1}}$')

    ax.set_xlabel(r'Wave number $k$ (nm$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'Structure factor $S(k)$', fontsize=14)
    ax.set_title(f'Per-Cluster O-O Structure Factor: ML Clustering Validation\n'
                 f'{model_name} at {temperature}°C', fontsize=15)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(k_values.min(), k_values.max())

    plt.tight_layout()
    outfile = os.path.join(output_dir,
                           f'structure_factor_per_cluster_{model_name}_T{temperature}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-cluster S(k) plot: {outfile}")

    # =====================================================================
    # Normalized k plot (k * r_OO / 2pi)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(13, 7))
    k_norm = k_values * r_OO / (2 * np.pi)

    # Shaded structural zones
    ax.axvspan(0.65, 0.85, alpha=0.08, color='green',
               label='FSDP zone (tetrahedral)')
    ax.axvspan(0.90, 1.15, alpha=0.08, color='gray',
               label='Normal liquid zone')

    for cid, res in sorted(cluster_results.items()):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        label = CLUSTER_NAMES.get(cid, f'Cluster {cid}')
        ax.plot(k_norm, res['S_k_avg'], linewidth=2.5, color=color, label=label)
        ax.fill_between(k_norm,
                        res['S_k_avg'] - res['S_k_std'],
                        res['S_k_avg'] + res['S_k_std'],
                        alpha=0.12, color=color)

        # Auto-detect and annotate the main peak in normalized k space
        k_peak, sk_peak = _find_main_peak(k_norm, res['S_k_avg'],
                                           k_min=0.55, k_max=1.8)
        if k_peak is not None:
            # Position annotation to avoid overlap
            x_offset = 0.12 if cid == 0 else -0.12
            y_offset = 0.4 if cid == 0 else 0.6
            ax.annotate(
                f'$k \\cdot r_{{OO}} / 2\\pi$ = {k_peak:.2f}',
                xy=(k_peak, sk_peak),
                xytext=(k_peak + x_offset, sk_peak + y_offset),
                fontsize=11, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            )

    # Reference lines
    ax.axvline(0.75, color='green', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

    ax.set_xlabel(r'Normalized wave number $k \cdot r_{OO} / 2\pi$', fontsize=14)
    ax.set_ylabel(r'Structure factor $S(k)$', fontsize=14)
    ax.set_title(f'Per-Cluster S(k): ML Clustering Validation\n'
                 f'{model_name} at {temperature}°C — '
                 r'$\rho$-state vs S-state structural signatures',
                 fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 3.0)

    plt.tight_layout()
    outfile = os.path.join(output_dir,
                           f'structure_factor_per_cluster_norm_{model_name}_T{temperature}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved normalized per-cluster S(k) plot: {outfile}")


# ============================================================================
# Main Analysis Function
# ============================================================================

def main():
    """Main analysis workflow"""
    
    # Parse arguments
    args = parse_arguments()
    
    # ---- Validate all input files upfront (before any heavy computation) ----
    if not os.path.exists(args.dcd_file):
        print(f"ERROR: DCD file not found: {args.dcd_file}", file=sys.stderr)
        print(f"  (resolved to: {os.path.abspath(args.dcd_file)})", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.pdb_file):
        print(f"ERROR: PDB file not found: {args.pdb_file}", file=sys.stderr)
        print(f"  (resolved to: {os.path.abspath(args.pdb_file)})", file=sys.stderr)
        sys.exit(1)
    if args.cluster_labels is not None and not os.path.exists(args.cluster_labels):
        print(f"ERROR: Cluster labels file not found: {args.cluster_labels}", file=sys.stderr)
        print(f"  (resolved to: {os.path.abspath(args.cluster_labels)})", file=sys.stderr)
        sys.exit(1)

    print("="*70)
    print("STRUCTURE FACTOR COMPUTATION FOR WATER MD TRAJECTORIES")
    print("="*70)
    print(f"DCD file: {args.dcd_file}")
    print(f"PDB file: {args.pdb_file}")
    if args.cluster_labels:
        print(f"Cluster labels: {args.cluster_labels}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}°C")
    print(f"Cutoff rc: {args.rc_cutoff} nm")
    print(f"k range: [0, {args.k_max}] nm^-1")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trajectory
    trajectory = load_trajectory(args.dcd_file, args.pdb_file, args.n_frames)
    
    # Define k values
    k_values = np.linspace(0.1, args.k_max, args.k_points)  # Avoid k=0
    
    # Compute O-O partial structure factor (all atoms)
    S_k_avg, S_k_std, S_k_frames = compute_partial_structure_factor_OO(
        trajectory, args.rc_cutoff, k_values
    )
    
    # Generate visualizations (all atoms)
    print(f"\nGenerating visualizations...")
    plot_structure_factor(
        k_values, S_k_avg, S_k_std, args.output_dir,
        args.model_name, args.temperature
    )
    
    plot_structure_factor_normalized(
        k_values, S_k_avg, args.output_dir,
        args.model_name, args.temperature
    )
    
    # Save numerical results (all atoms)
    results_file = os.path.join(args.output_dir, 
                                f'structure_factor_data_{args.model_name}_T{args.temperature}.npz')
    np.savez(results_file,
             k_values=k_values,
             S_k_avg=S_k_avg,
             S_k_std=S_k_std,
             S_k_frames=S_k_frames)
    print(f"  Saved numerical data: {results_file}")
    
    # Also save as .mat for compatibility
    mat_file = os.path.join(args.output_dir,
                           f'structure_factor_data_{args.model_name}_T{args.temperature}.mat')
    savemat(mat_file, {
        'k_values': k_values,
        'S_k_avg': S_k_avg,
        'S_k_std': S_k_std,
        'S_k_frames': S_k_frames
    })
    print(f"  Saved .mat file: {mat_file}")

    # ---- Per-cluster S(k) computation ----
    if args.cluster_labels is not None:
        print(f"\n{'='*70}")
        print("PER-CLUSTER STRUCTURE FACTOR COMPUTATION")
        print(f"{'='*70}")

        import pandas as pd
        label_df = pd.read_csv(args.cluster_labels)
        cluster_labels_matrix = label_df.values.astype(int)
        n_label_frames, n_label_mols = cluster_labels_matrix.shape

        all_ids = sorted(set(int(c) for c in np.unique(cluster_labels_matrix) if c >= 0))
        noise_count = (cluster_labels_matrix == -1).sum()
        print(f"  Loaded cluster labels: {args.cluster_labels}")
        print(f"    Shape: {n_label_frames} frames x {n_label_mols} molecules")
        print(f"    Available clusters: {all_ids}")
        print(f"    Noise points: {noise_count:,}")

        # Filter to requested cluster IDs (if specified)
        if args.cluster_id is not None:
            selected_ids = [cid for cid in args.cluster_id if cid in all_ids]
            missing = [cid for cid in args.cluster_id if cid not in all_ids]
            if missing:
                print(f"  WARNING: Cluster ID(s) {missing} not found in labels. "
                      f"Available: {all_ids}")
            if not selected_ids:
                print(f"  ERROR: None of the requested cluster IDs exist.", file=sys.stderr)
                sys.exit(1)
            # Mask out non-selected clusters so they're treated as noise
            mask = np.isin(cluster_labels_matrix, selected_ids, invert=True) & \
                   (cluster_labels_matrix >= 0)
            filtered_matrix = cluster_labels_matrix.copy()
            filtered_matrix[mask] = -1
            print(f"  Computing S(k) for cluster(s): {selected_ids}")
        else:
            filtered_matrix = cluster_labels_matrix
            print(f"  Computing S(k) for all clusters: {all_ids}")

        # Compute per-cluster S(k)
        cluster_results = compute_per_cluster_structure_factor(
            trajectory, args.rc_cutoff, k_values, filtered_matrix
        )

        # Plot per-cluster comparison
        plot_per_cluster_structure_factor(
            k_values, cluster_results, args.output_dir,
            args.model_name, args.temperature
        )

        # Save per-cluster numerical results
        for cid, res in cluster_results.items():
            tag = f'{args.model_name}_T{args.temperature}_cluster{cid}'
            npz_path = os.path.join(args.output_dir, f'structure_factor_{tag}.npz')
            np.savez(npz_path,
                     k_values=k_values,
                     S_k_avg=res['S_k_avg'],
                     S_k_std=res['S_k_std'],
                     S_k_frames=res['S_k_frames'])
            print(f"  Saved: {npz_path}")

            mat_path = os.path.join(args.output_dir, f'structure_factor_{tag}.mat')
            savemat(mat_path, {
                'k_values': k_values,
                'S_k_avg': res['S_k_avg'],
                'S_k_std': res['S_k_std'],
                'S_k_frames': res['S_k_frames'],
            })
            print(f"  Saved: {mat_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Analyzed {trajectory.n_frames} frames")
    print(f"Structure factor computed at {len(k_values)} k points")
    print(f"S(k) range: [{S_k_avg.min():.3f}, {S_k_avg.max():.3f}]")
    if args.cluster_labels is not None:
        for cid, res in sorted(cluster_results.items()):
            print(f"  Cluster {cid} S(k) range: "
                  f"[{res['S_k_avg'].min():.3f}, {res['S_k_avg'].max():.3f}]")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
