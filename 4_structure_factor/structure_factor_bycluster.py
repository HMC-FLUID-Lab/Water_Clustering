import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add 3_clustering for plot_style (bold Arial 16pt, 600 DPI)
import sys as _sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_clustering_dir = os.path.normpath(os.path.join(_script_dir, "..", "3_clustering"))
if _clustering_dir not in _sys.path:
    _sys.path.insert(0, _clustering_dir)
try:
    from plot_style import set_default_plot
    set_default_plot()
except ImportError:
    pass
import mdtraj as md
from scipy.io import loadmat
import sys
import argparse
from time import time

try:
    from sk_zeta_3d import plot_sk_zeta_all_clusters
    _HAS_SK_ZETA = True
except ImportError:
    plot_sk_zeta_all_clusters = None
    _HAS_SK_ZETA = False
    print("  [WARNING] sk_zeta_3d.py not found — 3D S(k,zeta) plots will be skipped.")


def compute_structure_factor(trajectory, atom_indices, k_values, rc_cutoff):
    """
    Debye scattering function (Tanaka Eq. 2):
    S(k) = 1 + (1/N) * sum_i sum_{j!=i} [sin(k*r_ij)/(k*r_ij)] * W(r_ij)
    W(r_ij) = sin(pi*r_ij/rc) / (pi*r_ij/rc)
    """
    n_frames = trajectory.n_frames
    n_atoms  = len(atom_indices)
    n_k      = len(k_values)

    print(f"\nComputing S(k): {n_frames} frames, {n_atoms} atoms, {n_k} k-points")

    S_k_frames = np.zeros((n_frames, n_k))
    t0 = time()

    for frame_idx in range(n_frames):
        if (frame_idx + 1) % 10 == 0:
            elapsed = time() - t0
            rate    = (frame_idx + 1) / elapsed
            eta     = (n_frames - frame_idx - 1) / rate
            print(f"  Frame {frame_idx+1}/{n_frames} — {rate:.1f} fps, ETA {eta:.0f}s")

        frame     = trajectory[frame_idx]
        pairs     = np.array([[atom_indices[i], atom_indices[j]]
                               for i in range(n_atoms)
                               for j in range(i + 1, n_atoms)])
        distances = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]

        for k_idx, k in enumerate(k_values):
            if k == 0:
                S_k_frames[frame_idx, k_idx] = 1.0
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                window  = np.nan_to_num(
                    np.sin(np.pi * distances / rc_cutoff) / (np.pi * distances / rc_cutoff),
                    nan=0.0)
                kr      = k * distances
                sinc_kr = np.nan_to_num(np.sin(kr) / kr, nan=1.0)
            # x2 because we used upper triangle only
            S_k_frames[frame_idx, k_idx] = 1.0 + (2.0 / n_atoms) * np.sum(sinc_kr * window)

    print(f"  Done in {time()-t0:.1f}s")
    return S_k_frames.mean(axis=0), S_k_frames.std(axis=0), S_k_frames


def compute_partial_structure_factor_OO(trajectory, rc_cutoff, k_values):
    """O-O partial structure factor (oxygen atoms only)."""
    oxygen_indices = [atom.index for atom in trajectory.topology.atoms if atom.name == 'O']
    print(f"Found {len(oxygen_indices)} oxygen atoms")
    return compute_structure_factor(trajectory, oxygen_indices, k_values, rc_cutoff)


def compute_per_cluster_structure_factor(trajectory, rc_cutoff, k_values, cluster_labels_matrix):
    """
    S(k) computed separately for each cluster.
    Returns {cluster_id: {'S_k_avg', 'S_k_std', 'S_k_frames'}}
    """
    residue_oxygen = {atom.residue.index: atom.index
                      for atom in trajectory.topology.atoms if atom.name == 'O'}

    n_frames = min(trajectory.n_frames, cluster_labels_matrix.shape[0])
    n_k      = len(k_values)

    if trajectory.n_frames != cluster_labels_matrix.shape[0]:
        print(f"  WARNING: frame count mismatch — using first {n_frames} frames")

    unique_clusters = sorted(set(int(c) for c in np.unique(cluster_labels_matrix) if c >= 0))
    print(f"\n  Clusters: {unique_clusters}  |  Frames: {n_frames}")

    results = {}

    for cluster_id in unique_clusters:
        print(f"\n  --- Cluster {cluster_id} ---")
        S_k_frames = np.zeros((n_frames, n_k))
        t0 = time()

        for frame_idx in range(n_frames):
            if (frame_idx + 1) % 10 == 0:
                elapsed = time() - t0
                rate    = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                eta     = (n_frames - frame_idx - 1) / rate if rate > 0 else 0
                print(f"    Frame {frame_idx+1}/{n_frames} — {rate:.1f} fps, ETA {eta:.0f}s")

            mol_indices  = np.where(cluster_labels_matrix[frame_idx] == cluster_id)[0]
            atom_indices = np.array([residue_oxygen[m] for m in mol_indices if m in residue_oxygen])
            n_atoms      = len(atom_indices)

            if n_atoms < 2:
                S_k_frames[frame_idx, :] = 1.0
                continue

            frame     = trajectory[frame_idx]
            pairs     = np.array([[atom_indices[i], atom_indices[j]]
                                   for i in range(n_atoms)
                                   for j in range(i + 1, n_atoms)])
            distances = md.compute_distances(frame, pairs, periodic=True, opt=True)[0]

            for k_idx, k in enumerate(k_values):
                if k == 0:
                    S_k_frames[frame_idx, k_idx] = 1.0
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    window  = np.nan_to_num(
                        np.sin(np.pi * distances / rc_cutoff) / (np.pi * distances / rc_cutoff),
                        nan=0.0)
                    sinc_kr = np.nan_to_num(np.sin(k * distances) / (k * distances), nan=1.0)
                S_k_frames[frame_idx, k_idx] = 1.0 + (2.0 / n_atoms) * np.sum(sinc_kr * window)

        n_mol_avg = (cluster_labels_matrix[:n_frames] == cluster_id).sum(axis=1).mean()
        print(f"    Done in {time()-t0:.1f}s  |  avg {n_mol_avg:.0f} mol/frame")

        results[cluster_id] = {
            'S_k_avg': S_k_frames.mean(axis=0),
            'S_k_std': S_k_frames.std(axis=0),
            'S_k_frames': S_k_frames,
        }

    return results


def load_trajectory(dcd_file, pdb_file, n_frames=None):
    """Load DCD trajectory with PDB topology."""
    print(f"\nLoading: {dcd_file}")
    if not os.path.exists(dcd_file):
        raise FileNotFoundError(f"DCD not found: {dcd_file}")
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB not found: {pdb_file}")

    topology = md.load(pdb_file).topology
    traj     = md.load(dcd_file, top=topology)

    if n_frames is not None and traj.n_frames > n_frames:
        traj = traj[np.linspace(0, traj.n_frames - 1, n_frames, dtype=int)]

    print(f"  {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues")
    return traj


def _display_model_name(model_name: str) -> str:
    """Pretty-print model string for figure titles (matches publication style)."""
    key = (model_name or "").strip().lower()
    if key == "tip4p2005":
        return "TIP4P/2005"
    if key == "tip5p":
        return "TIP5P"
    return model_name


def _temperature_title_math(temp: float) -> str:
    """Matplotlib mathtext fragment for temperature in °C (for use inside titles)."""
    if abs(temp - round(temp)) < 1e-9:
        t = int(round(temp))
    else:
        t = temp
    return rf"${t}\,^{{\circ}}\mathrm{{C}}$"


def plot_structure_factor_normalized(k_values, S_k_avg, output_dir, model_name, temperature):
    """Plot all-atom S(k) normalised by rOO with kT1 and kD1 reference lines."""
    k_norm = k_values * 0.285 / (2 * np.pi)

    # Thin markers every N points so individual symbols are visible
    every = max(1, len(k_norm) // 40)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(k_norm, S_k_avg,
            color='#2196F3', linewidth=0.8,
            marker='o', markevery=every, markersize=7,
            markerfacecolor='none', markeredgewidth=1.4,
            label=model_name)
    ax.axvline(0.75, color='green', linestyle='--', linewidth=1.2,
               alpha=0.7, label=r'$k_{T1}$ (FSDP, tetrahedral)')
    ax.axvline(1.0,  color='red',   linestyle='--', linewidth=1.2,
               alpha=0.7, label=r'$k_{D1}$ (DNLS)')
    ax.set_xlabel(r'$k \cdot r_{OO} / 2\pi$', fontsize=14)
    ax.set_ylabel(r'$S(k)$', fontsize=14)
    ax.set_title(f'Structure Factor — {model_name} at {temperature}°C', fontsize=16)
    ax.legend(fontsize=11, frameon=True, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    ax.set_xlim(0.6, 2.25)
    ax.set_ylim(0.5, 1.5)
    ax.tick_params(direction='in', which='both', top=True, right=True)
    plt.tight_layout()

    outfile = os.path.join(output_dir, f'structure_factor_normalized_{model_name}_T{temperature}.png')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_per_cluster_structure_factor(k_values, cluster_results, output_dir,
                                      model_name, temperature,
                                      cluster_labels_matrix=None,
                                      zeta_file=None,
                                      trajectory=None,
                                      rc_cutoff=None):
    """
    Individual and combined S(k) plots per cluster.
    If zeta_file + trajectory + rc_cutoff are all provided, also generates
    3D S(k,zeta) surfaces via sk_zeta_3d.
    """
    # Marker styles: LFTS (0), DNLS (1) for dbscan_gmm
    COLORS  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    MARKERS = ['o', 'D', '^', 's', 'v']
    NAMES   = {0: 'LFTS', 1: 'DNLS'}
    k_norm  = k_values * 0.285 / (2 * np.pi)

    # Thin markers every N points so individual symbols don't overlap
    every = max(1, len(k_norm) // 40)

    if (cluster_labels_matrix is not None and zeta_file is not None
            and trajectory is not None and rc_cutoff is not None and _HAS_SK_ZETA):
        plot_sk_zeta_all_clusters(
            trajectory=trajectory, k_values=k_values,
            cluster_labels_matrix=cluster_labels_matrix, zeta_file=zeta_file,
            output_dir=output_dir, model_name=model_name, temperature=temperature,
            rc_cutoff=rc_cutoff,
        )
    elif zeta_file is not None and not _HAS_SK_ZETA:
        print("  [WARNING] sk_zeta_3d.py not found — skipping 3D S(k,zeta) plots.")

    _dm = _display_model_name(model_name)
    _tmath = _temperature_title_math(temperature)

    def _style_ax(ax, title):
        ax.set_facecolor("white")
        ax.set_xlabel(r"$k \cdot r_{OO} / 2\pi$", fontsize=16, fontweight="bold")
        ax.set_ylabel(r"$S(k)$", fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
        ax.legend(
            frameon=True,
            edgecolor="black",
            framealpha=0.9,
            fontsize=15,
        )
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_xlim(0.6, 2.0)
        ax.set_ylim(0.5, 1.5)
        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
            labelsize=14,
        )

    # Individual plots — one per cluster
    for cid, res in sorted(cluster_results.items()):
        fig, ax = plt.subplots(figsize=(13, 7))
        color  = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f'Cluster {cid}')

        ax.plot(k_norm, res['S_k_avg'],
                color=color, linewidth=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor='none', markeredgewidth=1.4,
                label=label)

        # Error as sparse vertical errorbars instead of shaded band
        eb_idx = np.arange(0, len(k_norm), every)
        ax.errorbar(k_norm[eb_idx], res['S_k_avg'][eb_idx],
                    yerr=res['S_k_std'][eb_idx],
                    fmt='none', ecolor=color, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)

        _style_ax(
            ax,
            rf"{label} — {_dm} — {_tmath}",
        )
        plt.tight_layout()
        outfile = os.path.join(output_dir,
                               f'structure_factor_cluster{cid}_norm_{model_name}_T{temperature}.png')
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outfile}")

    # Combined overlay — all clusters on one plot
    fig, ax = plt.subplots(figsize=(13, 7))
    for cid, res in sorted(cluster_results.items()):
        color  = COLORS[cid % len(COLORS)]
        marker = MARKERS[cid % len(MARKERS)]
        label  = NAMES.get(cid, f'Cluster {cid}')

        ax.plot(k_norm, res['S_k_avg'],
                color=color, linewidth=0.8,
                marker=marker, markevery=every, markersize=7,
                markerfacecolor='none', markeredgewidth=1.4,
                label=label)

        eb_idx = np.arange(0, len(k_norm), every)
        ax.errorbar(k_norm[eb_idx], res['S_k_avg'][eb_idx],
                    yerr=res['S_k_std'][eb_idx],
                    fmt='none', ecolor=color, elinewidth=0.8,
                    capsize=2, capthick=0.8, alpha=0.6)

    _style_ax(
        ax,
        rf"Per-cluster $S(k)$ — {_dm} — {_tmath}",
    )
    plt.tight_layout()
    outfile = os.path.join(output_dir,
                           f'structure_factor_per_cluster_norm_{model_name}_T{temperature}.png')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='S(k) for water MD trajectories')
    parser.add_argument('--dcd-file',       required=True)
    parser.add_argument('--pdb-file',       required=True)
    parser.add_argument('--output-dir',     default='./structure_factor_results')
    parser.add_argument('--model-name',     default='unknown')
    parser.add_argument('--temperature',    type=float, default=None)
    parser.add_argument('--rc-cutoff',      type=float, default=1.5,
                        help='Cutoff rc in nm (default: 1.5)')
    parser.add_argument('--k-max',          type=float, default=50.0,
                        help='Max k in nm^-1 (default: 50)')
    parser.add_argument('--k-points',       type=int,   default=500)
    parser.add_argument('--n-frames',       type=int,   default=None)
    parser.add_argument('--cluster-labels', type=str,   default=None, metavar='CSV')
    parser.add_argument('--cluster-id',     type=int,   nargs='+', default=None, metavar='ID')
    parser.add_argument('--cluster-only',   action='store_true')
    parser.add_argument('--zeta-file',      type=str,   default=None, metavar='MAT')
    return parser.parse_args()


def main():
    args = parse_arguments()

    for path, label in [(args.dcd_file, 'DCD'), (args.pdb_file, 'PDB')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr); sys.exit(1)
    if args.cluster_labels and not os.path.exists(args.cluster_labels):
        print(f"ERROR: cluster labels not found: {args.cluster_labels}", file=sys.stderr); sys.exit(1)
    if args.cluster_only and not args.cluster_labels:
        print("ERROR: --cluster-only requires --cluster-labels", file=sys.stderr); sys.exit(1)

    print("=" * 70)
    print(f"S(k) COMPUTATION  |  {args.model_name}  |  T={args.temperature}°C  |  rc={args.rc_cutoff} nm")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    trajectory = load_trajectory(args.dcd_file, args.pdb_file, args.n_frames)

    # Align trajectory to label frame count
    if args.cluster_labels:
        import pandas as pd
        n_label_frames = len(pd.read_csv(args.cluster_labels, header=None))
        if trajectory.n_frames > n_label_frames:
            print(f"  Truncating trajectory to {n_label_frames} frames")
            trajectory = trajectory[:n_label_frames]
        elif trajectory.n_frames < n_label_frames:
            print(f"  WARNING: trajectory shorter than labels")

    k_values = np.linspace(0.1, args.k_max, args.k_points)

    if not args.cluster_only:
        S_k_avg, _, _ = compute_partial_structure_factor_OO(trajectory, args.rc_cutoff, k_values)
        plot_structure_factor_normalized(
            k_values, S_k_avg, args.output_dir, args.model_name, args.temperature)
    else:
        print("  Skipping all-atoms S(k) (--cluster-only)")

    if args.cluster_labels:
        import pandas as pd
        print(f"\n{'='*70}\nPER-CLUSTER S(k)\n{'='*70}")

        cluster_labels_matrix = pd.read_csv(args.cluster_labels, header=None).values.astype(int)
        all_ids     = sorted(set(int(c) for c in np.unique(cluster_labels_matrix) if c >= 0))
        noise_count = (cluster_labels_matrix == -1).sum()
        print(f"  Shape: {cluster_labels_matrix.shape}  |  Clusters: {all_ids}  |  Noise: {noise_count:,}")

        if args.cluster_id is not None:
            selected_ids = [c for c in args.cluster_id if c in all_ids]
            missing      = [c for c in args.cluster_id if c not in all_ids]
            if missing:
                print(f"  WARNING: cluster IDs {missing} not found")
            if not selected_ids:
                print("ERROR: no valid cluster IDs", file=sys.stderr); sys.exit(1)
            filtered_matrix = cluster_labels_matrix.copy()
            filtered_matrix[
                np.isin(cluster_labels_matrix, selected_ids, invert=True)
                & (cluster_labels_matrix >= 0)] = -1
        else:
            filtered_matrix = cluster_labels_matrix

        cluster_results = compute_per_cluster_structure_factor(
            trajectory, args.rc_cutoff, k_values, filtered_matrix)

        plot_per_cluster_structure_factor(
            k_values, cluster_results, args.output_dir,
            args.model_name, args.temperature,
            cluster_labels_matrix=filtered_matrix,
            zeta_file=args.zeta_file,
            trajectory=trajectory,
            rc_cutoff=args.rc_cutoff,
        )

    print(f"\n{'='*70}")
    print(f"DONE — {trajectory.n_frames} frames | output: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()