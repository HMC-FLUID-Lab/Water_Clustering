#!/usr/bin/env python3
"""
Stage 2 — Order Parameter Computation
=====================================
Read DCD trajectories and produce two MAT files per condition:

  1. OrderParam_<model>_T<temp>_<run>.mat       (LSI, q, Sk, Q6, d5, r, g_r)
  2. OrderParamZeta_<model>_T<temp>_<run>.mat   (zeta_all)

Inputs  : data/simulations/<model>/dcd_<model>_T<temp>_N<n>_<run>_0.dcd
Outputs : data/order_params/OrderParam*<...>.mat
"""

import os
import glob
import numpy as np
import mdtraj as md
import scipy.io
import multiprocessing
from joblib import Parallel, delayed
from time import time
import argparse

try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y as sph_harm
import itertools

# ============================================================================
# Configuration — paths resolved relative to repo root (parent of this file)
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

DATA_DIR     = os.path.join(REPO_ROOT, "data", "simulations")
TIP5P_DATA_DIR     = os.path.join(DATA_DIR, "tip5p")
TIP4P2005_DATA_DIR = os.path.join(DATA_DIR, "tip4p2005")
SWM4NDP_DATA_DIR   = os.path.join(DATA_DIR, "swm4ndp")
OUTPUT_DIR   = os.path.join(REPO_ROOT, "data", "order_params")

# ============================================================================
# Order Parameter Functions (from ComputeOrderParameters.py)
# ============================================================================

def CosPhi_jk(i, j, k, traj):
    """
    Compute the cosine of the angle between atom 0 and its neighbors j and k
    """
    pairs_jk = np.vstack([j, k]).transpose()
    pairs_ij = np.vstack([i, j]).transpose()
    pairs_ik = np.vstack([i, k]).transpose()
    
    b = md.compute_distances(traj, pairs_ij, periodic=True, opt=True).diagonal()
    c = md.compute_distances(traj, pairs_ik, periodic=True, opt=True).diagonal()
    a = md.compute_distances(traj, pairs_jk, periodic=True, opt=True).diagonal()
    
    cosPhi_jk = (b**2 + c**2 - a**2) / (2 * b * c)
    
    return cosPhi_jk


def ThetaPhi(i, j, pos_O):
    """
    Compute theta and phi angles for spherical harmonics
    """
    pairs_ij = np.vstack([i, j]).transpose()
    r_ij = md.compute_displacements(pos_O, pairs_ij, periodic=True, opt=True).diagonal().transpose()
    theta = np.arccos(r_ij[:, 2] / np.linalg.norm(r_ij, ord=2, axis=1))
    phi = np.arctan(r_ij[:, 1] / r_ij[:, 0])
    
    return theta, phi


# ============================================================================
# Hydrogen Bond Functions (from Compute_Zeta_dcdList_continuous.py)
# ============================================================================

def H_bond(i, j, traj_frame, M):
    """
    Check if two O-O atoms are H-bonded
    """
    r_ij = md.compute_displacements(traj_frame, [[M*i, M*j]], periodic=True, opt=True).diagonal()
    r_iH1 = md.compute_displacements(traj_frame, [[M*i, M*i+1]], periodic=True, opt=True).diagonal()
    r_iH2 = md.compute_displacements(traj_frame, [[M*i, M*i+2]], periodic=True, opt=True).diagonal()
    r_jH1 = md.compute_displacements(traj_frame, [[M*j, M*j+1]], periodic=True, opt=True).diagonal()
    r_jH2 = md.compute_displacements(traj_frame, [[M*j, M*j+2]], periodic=True, opt=True).diagonal()
    
    d_ij = md.compute_distances(traj_frame, [[M*i, M*j]], periodic=True, opt=True).diagonal()
    d_iH1 = md.compute_distances(traj_frame, [[M*i, M*i+1]], periodic=True, opt=True).diagonal()
    d_iH2 = md.compute_distances(traj_frame, [[M*i, M*i+2]], periodic=True, opt=True).diagonal()
    d_jH1 = md.compute_distances(traj_frame, [[M*j, M*j+1]], periodic=True, opt=True).diagonal()
    d_jH2 = md.compute_distances(traj_frame, [[M*j, M*j+2]], periodic=True, opt=True).diagonal()
    
    r_ij = np.array(r_ij)[:, 0]
    r_iH1 = np.array(r_iH1)[:, 0]
    r_iH2 = np.array(r_iH2)[:, 0]
    r_jH1 = np.array(r_jH1)[:, 0]
    r_jH2 = np.array(r_jH2)[:, 0]
    
    cos_1 = np.dot(r_ij, r_iH1) / (d_ij * d_iH1)
    cos_2 = np.dot(r_ij, r_iH2) / (d_ij * d_iH2)
    cos_3 = -(np.dot(r_ij, r_jH1)) / (d_ij * d_jH1)
    cos_4 = -(np.dot(r_ij, r_jH2)) / (d_ij * d_jH2)
    
    if d_ij < 0.35:
        if (cos_1 > np.cos(np.pi/6) or cos_2 > np.cos(np.pi/6) or 
            cos_3 > np.cos(np.pi/6) or cos_4 > np.cos(np.pi/6)):
            return True
    
    return False


def Molecule_zeta(i, traj_O, traj_frame, N, M):
    """
    Calculate zeta order parameter for molecule i
    """
    pairs = [[i, j] for j in range(N)]
    pairs = np.array(pairs)
    
    pair_dist = md.compute_distances(traj_O, pairs, periodic=True, opt=True)
    pair_dist_rank = pair_dist.argsort(axis=1)
    
    H_neighbors = []
    Not_H = []
    
    for h in range(1, 10):
        if H_bond(i, pair_dist_rank[0][h], traj_frame, M):
            H_neighbors.append(pair_dist_rank[0][h])
        else:
            Not_H.append(pair_dist_rank[0][h])
    
    if len(H_neighbors) >= 1 and len(Not_H) >= 1:
        n4 = H_neighbors[-1]
        n5 = Not_H[0]
        
        pairs_n5 = np.array([[i, n5]])
        pairs_n4 = np.array([[i, n4]])
        
        d_n5 = md.compute_distances(traj_O, pairs_n5, periodic=True, opt=True)
        d_n4 = md.compute_distances(traj_O, pairs_n4, periodic=True, opt=True)
    else:
        n5 = Not_H[0]
        pairs_n5 = np.array([[i, n5]])
        d_n5 = md.compute_distances(traj_O, pairs_n5, periodic=True, opt=True)
        d_n4 = 0
    
    zeta = d_n5 - d_n4
    
    return zeta[0][0]




def compute_order_parameters(dcd_file, pdb_file, output_dir, model_name, temp_str, run_str):
    """
    Compute all order parameters (LSI, q, Sk, Q6, d5, r, g_r)
    Based on ComputeOrderParameters.py
    """
    print(f"\n{'='*70}")
    print(f"Computing Order Parameters: {os.path.basename(dcd_file)}")
    print(f"{'='*70}")
    
    t_start = time()
    
    # Load trajectory
    topology = md.load(pdb_file).topology
    traj_pos = md.load(dcd_file, top=pdb_file)
    
    N = topology.n_residues
    n_frames = traj_pos.n_frames
    
    print(f"  Molecules: {N}")
    print(f"  Frames: {n_frames}")
    
    # Get atom indices
    atom_to_consider_indices_O = [i for i in range(topology.n_atoms) 
                                  if topology.atom(i).name == 'O']
    
    # Initialize arrays
    q_all = np.array([]).reshape(0, N)
    Sk_all = np.array([]).reshape(0, N)
    LSI_all = np.array([]).reshape(0, N)
    Q6_all = np.array([]).reshape(0, N)
    d5_all = np.array([]).reshape(0, N)
    
    LSI_traj = np.zeros([n_frames, N])
    q_traj = np.zeros([n_frames, N])
    Sk_traj = np.zeros([n_frames, N])
    Q6_traj = np.zeros([n_frames, N])
    d5_traj = np.zeros([n_frames, N])
    next_neighbor_traj = np.zeros([n_frames, N, 12])
    
    # Atom trajectories
    pos_O = traj_pos.atom_slice(atom_to_consider_indices_O)
    
    print(f"\nComputing order parameters for each molecule...")
    
    for i in range(pos_O.n_atoms):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Molecule {i+1}/{pos_O.n_atoms} ({i/pos_O.n_atoms*100:.1f}%)")
        
        # Calculate pairwise oxygen distance
        pairs = np.array([[i] * (pos_O.n_atoms - 1), 
                         list(range(0, i)) + list(range(i + 1, pos_O.n_atoms))]).transpose()
        pair_dist = md.compute_distances(pos_O, pairs, periodic=True, opt=True)
        pair_dist_rank = pair_dist.argsort(axis=1)
        pair_dist_sort = np.sort(pair_dist, axis=1)
        
        # Calculate LSI
        n = np.sum(pair_dist < 0.37, axis=1)
        Delta_i = pair_dist_sort[:, 1:] - pair_dist_sort[:, :-1]
        cutoff_index = (pair_dist_sort < 0.37)[:, :-1]
        Delta_bar = np.sum(Delta_i * cutoff_index, axis=1) / n
        Delta_i_m_Delta_bar = Delta_i - Delta_bar.reshape(pos_O.n_frames, 1)
        LSI_i = np.sum((Delta_i_m_Delta_bar * cutoff_index)**2, axis=1) / n
        LSI_traj[:, i] = LSI_i
        
        # Calculate d5
        d5_i = pair_dist_sort[:, 4]
        d5_traj[:, i] = d5_i
        
        # Find five nearest neighbors
        n1 = pairs[pair_dist_rank[:, 0], 1]
        n2 = pairs[pair_dist_rank[:, 1], 1]
        n3 = pairs[pair_dist_rank[:, 2], 1]
        n4 = pairs[pair_dist_rank[:, 3], 1]
        
        # Calculate q
        cosphi_jk = np.zeros([pos_O.n_frames, 6])
        ni = np.array([i] * pos_O.n_frames)
        
        cosphi_jk[:, 0] = CosPhi_jk(ni, n1, n2, pos_O)
        cosphi_jk[:, 1] = CosPhi_jk(ni, n1, n3, pos_O)
        cosphi_jk[:, 2] = CosPhi_jk(ni, n1, n4, pos_O)
        cosphi_jk[:, 3] = CosPhi_jk(ni, n2, n3, pos_O)
        cosphi_jk[:, 4] = CosPhi_jk(ni, n2, n4, pos_O)
        cosphi_jk[:, 5] = CosPhi_jk(ni, n3, n4, pos_O)
        
        q_traj[:, i] = 1 - 3.0/8.0 * np.sum((cosphi_jk + 1.0/3.0)**2, axis=1)
        
        # Calculate Sk
        r_k = np.zeros([pos_O.n_frames, 4])
        r_k[:, 0] = pair_dist[range(pos_O.n_frames), pair_dist_rank[:, 0]]
        r_k[:, 1] = pair_dist[range(pos_O.n_frames), pair_dist_rank[:, 1]]
        r_k[:, 2] = pair_dist[range(pos_O.n_frames), pair_dist_rank[:, 2]]
        r_k[:, 3] = pair_dist[range(pos_O.n_frames), pair_dist_rank[:, 3]]
        
        r_bar = np.mean(r_k, axis=1).reshape(pos_O.n_frames, 1)
        r_factor = (r_k - r_bar)**2 / (4.0 * r_bar**2)
        Sk_traj[:, i] = 1 - np.sum(r_factor, axis=1) / 3.0
        
        # Find twelve next nearest neighbors
        for k in range(12):
            next_neighbor_traj[:, i, k] = pairs[pair_dist_rank[:, k + 4], 1]
        
        # Compute Q6
        ni = np.array([i] * pos_O.n_frames)
        Y_lm = np.zeros([pos_O.n_frames, 13], dtype=complex)
        for k in range(12):
            theta, phi = ThetaPhi(ni, next_neighbor_traj[:, i, k], pos_O)
            for m in range(-6, 7):
                Y_lm[:, m + 6] = sph_harm(m, 6, phi, theta) + Y_lm[:, m + 6]
        Y_lm_bar = Y_lm / 12.0
        Q_6i = np.sqrt(4 * np.pi / (2 * 6 + 1.0) * np.sum(np.abs(Y_lm_bar)**2, axis=1))
        Q6_traj[:, i] = Q_6i
    
    LSI_all = np.append(LSI_all, LSI_traj, axis=0)
    q_all = np.append(q_all, q_traj, axis=0)
    Sk_all = np.append(Sk_all, Sk_traj, axis=0)
    Q6_all = np.append(Q6_all, Q6_traj, axis=0)
    d5_all = np.append(d5_all, d5_traj, axis=0)
    
    # Calculate Radial Distribution Function
    print(f"\nComputing radial distribution function...")
    all_pairs = np.array(list(itertools.combinations(range(pos_O.n_atoms), 2)))
    r, g_r = md.compute_rdf(pos_O, all_pairs, r_range=(0, 1.2), bin_width=0.005, 
                            n_bins=None, periodic=True, opt=True)
    
    # Save to MAT file
    output_filename = f'OrderParam_{model_name}_{temp_str}_{run_str}.mat'
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nSaving results to: {output_filename}")
    scipy.io.savemat(output_path, dict(
        LSI_all=LSI_all,
        q_all=q_all,
        Sk_all=Sk_all,
        Q6_all=Q6_all,
        d5_all=d5_all,
        r=r,
        g_r=g_r
    ))
    
    t_end = time()
    print(f"✓ Completed in {t_end - t_start:.1f} seconds")
    
    return output_path


def compute_zeta_parameter(dcd_file, pdb_file, output_dir, model_name, temp_str, run_str):
    """
    Compute zeta order parameter
    Based on Compute_Zeta_dcdList_continuous.py
    """
    print(f"\n{'='*70}")
    print(f"Computing Zeta Parameter: {os.path.basename(dcd_file)}")
    print(f"{'='*70}")
    
    t_start = time()
    
    # Load topology and trajectory
    topology = md.load(pdb_file).topology
    trajectory = md.load(dcd_file, top=pdb_file)
    
    N = topology.n_residues
    M = int(topology.n_atoms / topology.n_residues)
    n_frames = trajectory.n_frames
    
    print(f"  Molecules: {N}")
    print(f"  Atoms per molecule: {M}")
    print(f"  Frames: {n_frames}")
    
    # Get oxygen atom indices
    atom_to_consider_indices = [i for i in range(topology.n_atoms) 
                               if topology.atom(i).name == 'O']
    
    # Create array to store data
    zeta_all = np.zeros([n_frames, N])
    
    print(f"\nComputing zeta order parameter...")
    
    # Process each frame
    for k in range(n_frames):
        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time() - t_start
            if k > 0:
                rate = k / elapsed
                eta = (n_frames - k) / rate
                print(f"  Frame {k+1}/{n_frames} ({k/n_frames*100:.1f}%) - "
                      f"Rate: {rate:.2f} frames/s, ETA: {eta:.0f}s")
            else:
                print(f"  Frame {k+1}/{n_frames}...")
        
        traj_frame = trajectory[k]
        traj_O = traj_frame.atom_slice(atom_to_consider_indices)
        
        # Compute zeta in parallel
        num_cores = multiprocessing.cpu_count()
        zeta_results = Parallel(n_jobs=num_cores)(
            delayed(Molecule_zeta)(i, traj_O, traj_frame, N, M) for i in range(N)
        )
        zeta_all[k, :] = zeta_results
    
    # Save to MAT file
    output_filename = f'OrderParamZeta_{model_name}_{temp_str}_{run_str}.mat'
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nSaving results to: {output_filename}")
    scipy.io.savemat(output_path, dict(zeta_all=zeta_all))
    
    t_end = time()
    print(f"✓ Completed in {t_end - t_start:.1f} seconds")
    
    return output_path


def process_single_dcd(dcd_file, pdb_file, output_dir, model_name):
    """
    Process a single DCD file and generate both MAT files
    """
    print(f"\n{'#'*70}")
    print(f"# Processing: {os.path.basename(dcd_file)}")
    print(f"{'#'*70}")
    
    total_start = time()
    
    # Extract run information from filename
    basename = os.path.basename(dcd_file)
    basename_noext = basename.replace('.dcd', '')
    parts = basename_noext.split('_')
    
    # Find temperature and run number
    temp_str = [p for p in parts if p.startswith('T')][0]
    run_str = [p for p in parts if p.startswith('Run')][0]
    
    try:
        # Generate OrderParam MAT file
        print(f"\n[1/2] Generating OrderParam MAT file...")
        orderparampath = compute_order_parameters(dcd_file, pdb_file, output_dir, 
                                                   model_name, temp_str, run_str)
        
        # Generate OrderParamZeta MAT file
        print(f"\n[2/2] Generating OrderParamZeta MAT file...")
        zeta_path = compute_zeta_parameter(dcd_file, pdb_file, output_dir, 
                                          model_name, temp_str, run_str)
        
        total_end = time()
        print(f"\n{'='*70}")
        print(f"✓ COMPLETED: {os.path.basename(dcd_file)}")
        print(f"{'='*70}")
        print(f"Total time: {total_end - total_start:.1f} seconds")
        print(f"Generated files:")
        print(f"  1. {os.path.basename(orderparampath)}")
        print(f"  2. {os.path.basename(zeta_path)}")
        
        return True
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR processing {os.path.basename(dcd_file)}")
        print(f"{'!'*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def find_matching_pdb(dcd_file):
    """
    Find the corresponding PDB file for a DCD file
    """
    basename = os.path.basename(dcd_file)
    basename_noext = basename.replace('.dcd', '')
    
    if basename_noext.endswith('_0'):
        basename_noext = basename_noext[:-2]
    
    pdb_basename = basename_noext.replace('dcd_', 'inistate_') + '.pdb'
    dcd_dir = os.path.dirname(dcd_file)
    pdb_file = os.path.join(dcd_dir, pdb_basename)
    
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    return pdb_file


def process_model(model_name, data_dir, output_dir):
    """
    Process all DCD files for a specific water model
    """
    print(f"\n{'#'*70}")
    print(f"# Processing {model_name.upper()} Water Model")
    print(f"{'#'*70}")
    
    # Find all DCD files
    dcd_pattern = os.path.join(data_dir, f'dcd_{model_name}_*.dcd')
    dcd_files = sorted(glob.glob(dcd_pattern))
    
    if len(dcd_files) == 0:
        print(f"WARNING: No DCD files found matching pattern: {dcd_pattern}")
        return
    
    print(f"\nFound {len(dcd_files)} DCD files to process")
    print(f"Each file will generate 2 MAT files (OrderParam and OrderParamZeta)")
    print(f"Total expected output: {len(dcd_files) * 2} MAT files")
    
    # Process each DCD file
    processed_count = 0
    failed_count = 0
    
    for i, dcd_file in enumerate(dcd_files):
        print(f"\n{'='*70}")
        print(f"Progress: [{i+1}/{len(dcd_files)}]")
        print(f"{'='*70}")
        
        try:
            pdb_file = find_matching_pdb(dcd_file)
            success = process_single_dcd(dcd_file, pdb_file, output_dir, model_name)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            failed_count += 1
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"MODEL {model_name.upper()} PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Total: {len(dcd_files)} files")
    print(f"Generated MAT files: {processed_count * 2}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate OrderParam and OrderParamZeta MAT files from DCD files'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['tip5p', 'tip4p2005', 'swm4ndp', 'all'],
        default='all',
        help='Water model to process (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for MAT files (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files to be processed without actually processing them'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("AUTOMATED ORDER PARAMETER MAT FILE GENERATION")
    print("="*70)
    print("Generates TWO types of MAT files:")
    print("  1. OrderParam_*.mat     - LSI, q, Sk, Q6, d5, r, g_r")
    print("  2. OrderParamZeta_*.mat - zeta_all")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Models to process: {args.model}")
    print(f"Dry run: {args.dry_run}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dry_run:
        print("\nDRY RUN MODE - No files will be processed")
        
        if args.model in ['tip5p', 'all']:
            dcd_files = sorted(glob.glob(os.path.join(TIP5P_DATA_DIR, 'dcd_tip5p_*.dcd')))
            print(f"\nTIP5P files to process ({len(dcd_files)}):")
            print(f"  Will generate {len(dcd_files) * 2} MAT files")
            for f in dcd_files[:5]:
                basename = os.path.basename(f).replace('.dcd', '')
                parts = basename.split('_')
                temp_str = [p for p in parts if p.startswith('T')][0]
                run_str = [p for p in parts if p.startswith('Run')][0]
                print(f"  - {os.path.basename(f)}")
                print(f"    -> OrderParam_tip5p_{temp_str}_{run_str}.mat")
                print(f"    -> OrderParamZeta_tip5p_{temp_str}_{run_str}.mat")
            if len(dcd_files) > 5:
                print(f"  ... and {len(dcd_files) - 5} more files")
        
        if args.model in ['tip4p2005', 'all']:
            dcd_files = sorted(glob.glob(os.path.join(TIP4P2005_DATA_DIR, 'dcd_tip4p2005_*.dcd')))
            print(f"\nTIP4P2005 files to process ({len(dcd_files)}):")
            print(f"  Will generate {len(dcd_files) * 2} MAT files")
            for f in dcd_files[:5]:
                basename = os.path.basename(f).replace('.dcd', '')
                parts = basename.split('_')
                temp_str = [p for p in parts if p.startswith('T')][0]
                run_str = [p for p in parts if p.startswith('Run')][0]
                print(f"  - {os.path.basename(f)}")
                print(f"    -> OrderParam_tip4p2005_{temp_str}_{run_str}.mat")
                print(f"    -> OrderParamZeta_tip4p2005_{temp_str}_{run_str}.mat")
            if len(dcd_files) > 5:
                print(f"  ... and {len(dcd_files) - 5} more files")
        
        return
    
    # Process models
    start_time = time()
    
    if args.model in ['tip5p', 'all']:
        process_model('tip5p', TIP5P_DATA_DIR, args.output_dir)

    if args.model in ['tip4p2005', 'all']:
        process_model('tip4p2005', TIP4P2005_DATA_DIR, args.output_dir)

    if args.model in ['swm4ndp', 'all']:
        process_model('swm4ndp', SWM4NDP_DATA_DIR, args.output_dir)

    end_time = time()
    
    # Final summary
    print(f"\n{'#'*70}")
    print("# ALL PROCESSING COMPLETE")
    print(f"{'#'*70}")
    print(f"Total execution time: {end_time - start_time:.1f} seconds")
    print(f"Output files saved to: {args.output_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
