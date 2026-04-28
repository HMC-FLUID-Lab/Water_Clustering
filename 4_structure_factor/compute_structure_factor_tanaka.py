
import numpy as np
import glob
import os
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compute structure factor for Tanaka simulation data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./output_data',
        help='Base directory containing simulation outputs'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='tip4p2005',
        choices=['tip4p2005', 'tip5p'],
        help='Water model (tip4p2005 or tip5p)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        required=True,
        help='Temperature in Celsius (e.g., -30, -20, 0, 10)'
    )
    parser.add_argument(
        '--run-number',
        type=str,
        default='Run01',
        help='Run number (e.g., Run01, Run02)'
    )
    parser.add_argument(
        '--n-molecules',
        type=int,
        default=1024,
        help='Number of molecules (default: 1024)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./structure_factor_results',
        help='Output directory for results'
    )
    
    return parser.parse_args()


def find_simulation_files(data_dir, model, temperature, run_number, n_molecules):
    """
    Find DCD and PDB files matching the Tanaka simulation naming pattern
    
    File pattern:
    - DCD: dcd_{model}_T{temp}_N{nmol}_{run}_0.dcd
    - PDB: inistate_{model}_T{temp}_N{nmol}_{run}.pdb
    
    Parameters:
    -----------
    data_dir : str
        Base directory (e.g., './output_data/tip4p2005_runs')
    model : str
        Model name (tip4p2005 or tip5p)
    temperature : float
        Temperature in Celsius
    run_number : str
        Run identifier (e.g., 'Run01')
    n_molecules : int
        Number of molecules
        
    Returns:
    --------
    dcd_files : list
        List of DCD trajectory files
    pdb_file : str
        PDB topology file
    """
    
    # Construct search patterns
    # Temperature formatting: -30 stays as T-30, 0 becomes T0, 10 becomes T10
    if temperature < 0:
        temp_str = f"T{int(temperature)}"
    else:
        temp_str = f"T{int(temperature)}"
    
    # Pattern for DCD files: dcd_tip4p2005_T-30_N1024_Run01_*.dcd
    dcd_pattern = os.path.join(
        data_dir, 
        f"dcd_{model}_{temp_str}_N{n_molecules}_{run_number}_*.dcd"
    )
    
    # Pattern for PDB file: inistate_tip4p2005_T-30_N1024_Run01.pdb
    pdb_pattern = os.path.join(
        data_dir,
        f"inistate_{model}_{temp_str}_N{n_molecules}_{run_number}.pdb"
    )
    
    print(f"\nSearching for files...")
    print(f"  Directory: {data_dir}")
    print(f"  DCD pattern: dcd_{model}_{temp_str}_N{n_molecules}_{run_number}_*.dcd")
    print(f"  PDB pattern: inistate_{model}_{temp_str}_N{n_molecules}_{run_number}.pdb")
    
    # Find DCD files
    dcd_files = glob.glob(dcd_pattern)
    dcd_files = sorted(dcd_files)  # Sort for consistent ordering
    
    # Find PDB file
    pdb_files = glob.glob(pdb_pattern)
    
    if len(dcd_files) == 0:
        raise FileNotFoundError(
            f"No DCD files found matching pattern: {dcd_pattern}\n"
            f"Available files in {data_dir}:\n" + 
            "\n".join(sorted(glob.glob(os.path.join(data_dir, "*.dcd")))[:10])
        )
    
    if len(pdb_files) == 0:
        raise FileNotFoundError(
            f"No PDB file found matching pattern: {pdb_pattern}\n"
            f"Available PDB files in {data_dir}:\n" +
            "\n".join(sorted(glob.glob(os.path.join(data_dir, "*.pdb")))[:10])
        )
    
    pdb_file = pdb_files[0]
    
    print(f"\n✓ Found {len(dcd_files)} DCD file(s):")
    for dcd in dcd_files:
        print(f"    {os.path.basename(dcd)}")
    print(f"✓ Found PDB file:")
    print(f"    {os.path.basename(pdb_file)}")
    
    return dcd_files, pdb_file


def main():
    args = parse_arguments()

    
    # Determine subdirectory based on model
    if args.model == 'tip4p2005':
        subdir = 'tip4p2005_runs'
    elif args.model == 'tip5p':
        subdir = 'tip5p_runs'
    else:
        subdir = ''
    
    # Construct full path
    data_directory = os.path.join(args.data_dir, subdir) if subdir else args.data_dir
    
    # Find simulation files
    try:
        dcd_files, pdb_file = find_simulation_files(
            data_directory,
            args.model,
            args.temperature,
            args.run_number,
            args.n_molecules
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nTip: Make sure you've organized your files using organize_files.sh")
        print(f"     or check that the data directory is correct: {data_directory}")
        return 1
    
    # Now call the structure factor computation with the found files
    print(f"\n{'='*70}")
    print("COMPUTING STRUCTURE FACTOR...")
    print(f"{'='*70}\n")
    
    # Import and run the structure factor computation
    from compute_structure_factor import load_trajectory, compute_partial_structure_factor_OO
    from compute_structure_factor import plot_structure_factor, plot_structure_factor_normalized
    import matplotlib
    matplotlib.use('Agg')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trajectory (use first DCD file)
    trajectory = load_trajectory(dcd_files[0], pdb_file, n_frames=None)
    
    # Define k values
    # k_max = 50.0 nm^-1 to ensure normalized k reaches 2.0
    # (k_norm = k * 0.285 / (2π), so k_max = 50 gives k_norm_max ≈ 2.25)
    k_values = np.linspace(0.1, 50.0, 500)
    
    # Compute structure factor
    rc_cutoff = 1.5  # nm
    S_k_avg, S_k_std, S_k_frames = compute_partial_structure_factor_OO(
        trajectory, rc_cutoff, k_values
    )
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_structure_factor(
        k_values, S_k_avg, S_k_std, args.output_dir,
        args.model, args.temperature
    )
    plot_structure_factor_normalized(
        k_values, S_k_avg, args.output_dir,
        args.model, args.temperature
    )
    
    # Save results
    from scipy.io import savemat
    
    results_file = os.path.join(
        args.output_dir,
        f'structure_factor_data_{args.model}_T{args.temperature}.npz'
    )
    np.savez(results_file,
             k_values=k_values,
             S_k_avg=S_k_avg,
             S_k_std=S_k_std,
             S_k_frames=S_k_frames)
    print(f"✓ Saved NumPy data: {results_file}")
    
    mat_file = os.path.join(
        args.output_dir,
        f'structure_factor_data_{args.model}_T{args.temperature}.mat'
    )
    savemat(mat_file, {
        'k_values': k_values,
        'S_k_avg': S_k_avg,
        'S_k_std': S_k_std,
        'S_k_frames': S_k_frames
    })
    print(f"✓ Saved MATLAB data: {mat_file}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
