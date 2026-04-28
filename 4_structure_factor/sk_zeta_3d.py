#!/usr/bin/env python3
"""
sk_zeta_3d.py
=============
Compute and plot S(k, ζ): structure factor resolved by individual molecule
ζ order parameter, replicating Tanaka Figure 2D (3D surface) and 2E (2D contour).

Tanaka, R. Shi & H. Tanaka, J. Am. Chem. Soc. 2020, 142, 2868-2875

Method:
    For each ζ bin, per-molecule partial S_i(k) is computed using the Debye
    sum over neighbours within rc_cutoff. S(k, ζ) = average of S_i(k) for
    molecules whose ζ falls in that bin (across all frames).

Usage:
    from sk_zeta_3d import plot_sk_zeta_all_clusters

    plot_sk_zeta_all_clusters(
        trajectory            = trajectory,           # mdtraj.Trajectory
        k_values              = k_values,             # nm^-1
        cluster_labels_matrix = filtered_matrix,      # (n_frames, n_molecules)
        zeta_file             = args.zeta_file,       # path to .mat with zeta_all
        output_dir            = args.output_dir,
        model_name            = args.model_name,
        temperature           = args.temperature,
        rc_cutoff             = args.rc_cutoff,
    )

Outputs per cluster:
    3d_sk_zeta_cluster{id}_{model}_T{temp}.html    Plotly 3D surface
    3d_sk_zeta_cluster{id}_{model}_T{temp}.png     Matplotlib 3D surface
    2d_sk_zeta_cluster{id}_{model}_T{temp}.html    Plotly 2D contour
    2d_sk_zeta_cluster{id}_{model}_T{temp}.png     Matplotlib 2D contour
    2d_sk_zeta_combined_{model}_T{temp}.html       Plotly side-by-side contour
    2d_sk_zeta_combined_{model}_T{temp}.png        Matplotlib side-by-side contour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_zeta(zeta_file):
    """
    Load zeta_all from a .mat file.
    Returns np.ndarray shape (n_frames, n_molecules) in Å, or None on failure.
    """
    try:
        raw = loadmat(zeta_file)
        zeta_all = np.asarray(raw['zeta_all']).squeeze()
    except Exception as exc:
        print(f"  [sk_zeta_3d] WARNING: Cannot load '{zeta_file}': {exc}")
        return None

    if zeta_all.ndim == 1:
        zeta_all = zeta_all.reshape(1, -1)

    # Auto-convert nm → Å (Tanaka uses Å; typical nm range < 0.3)
    if np.abs(zeta_all).max() < 2.0:
        print("  [sk_zeta_3d] Converting ζ from nm to Å (×10).")
        zeta_all = zeta_all * 10.0

    print(f"  [sk_zeta_3d] ζ loaded: shape={zeta_all.shape}, "
          f"range=[{zeta_all.min():.3f}, {zeta_all.max():.3f}] Å")
    return zeta_all


# ---------------------------------------------------------------------------
# S(k, ζ) computation
# ---------------------------------------------------------------------------

def compute_sk_zeta_matrix(trajectory, k_values, cluster_labels_matrix,
                            cluster_id, zeta_all, zeta_bins,
                            rc_cutoff=1.5):
    """
    Compute S(k, ζ) for one cluster by binning individual molecule ζ values.

    For each molecule i in the cluster (across all frames):
        - identify its ζ bin
        - compute its per-molecule partial S_i(k) via the Debye sum
          over all neighbours within rc_cutoff
        - accumulate into S(k, ζ) matrix

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
    k_values : np.ndarray, nm^-1
    cluster_labels_matrix : np.ndarray (n_frames, n_molecules)
    cluster_id : int
    zeta_all : np.ndarray (n_frames, n_molecules), Å
    zeta_bins : np.ndarray  — bin edges in Å
    rc_cutoff : float, nm

    Returns
    -------
    S_k_zeta : np.ndarray (n_zeta_bins, n_k)  — NaN where no data
    zeta_centers : np.ndarray (n_zeta_bins,)
    """
    import mdtraj as md

    topology = trajectory.topology
    residue_oxygen = {atom.residue.index: atom.index
                      for atom in topology.atoms if atom.name == 'O'}

    zeta_centers = 0.5 * (zeta_bins[:-1] + zeta_bins[1:])
    n_zeta = len(zeta_centers)
    n_k    = len(k_values)
    k_arr  = k_values                                     # alias

    S_sum  = np.zeros((n_zeta, n_k))
    counts = np.zeros(n_zeta)

    n_frames = min(trajectory.n_frames,
                   cluster_labels_matrix.shape[0],
                   zeta_all.shape[0])

    all_mol_indices = np.array(sorted(residue_oxygen.keys()))

    for frame_idx in range(n_frames):
        if (frame_idx + 1) % 5 == 0 or frame_idx == n_frames - 1:
            print(f"    [sk_zeta_3d] Frame {frame_idx+1}/{n_frames}", flush=True)

        frame        = trajectory[frame_idx]
        frame_labels = cluster_labels_matrix[frame_idx]
        frame_zeta   = zeta_all[frame_idx]
        box          = frame.unitcell_lengths[0]          # (3,) nm

        # All oxygen positions for minimum-image calculation
        all_oxy_atoms = np.array([residue_oxygen[m] for m in all_mol_indices
                                   if m in residue_oxygen])
        positions     = frame.xyz[0, all_oxy_atoms, :]    # (N_mol, 3) nm

        # Molecules belonging to this cluster in this frame
        mol_in_cluster = np.where(frame_labels == cluster_id)[0]
        if len(mol_in_cluster) < 2:
            continue

        for mol_idx in mol_in_cluster:
            zeta_val = float(frame_zeta[mol_idx])
            bin_idx  = int(np.searchsorted(zeta_bins, zeta_val, side='right')) - 1
            if not (0 <= bin_idx < n_zeta):
                continue

            # Distances from mol_idx to all others (minimum-image)
            diff    = positions - positions[mol_idx]      # (N_mol, 3)
            diff   -= box * np.round(diff / box)
            r_all   = np.linalg.norm(diff, axis=1)        # (N_mol,)

            # Neighbours: exclude self, apply cutoff
            mask        = (r_all > 1e-6) & (r_all < rc_cutoff)
            r_neighbors = r_all[mask]
            if len(r_neighbors) == 0:
                continue

            # Window function W(r) = sin(π r/rc) / (π r/rc)
            pi_r_rc = np.pi * r_neighbors / rc_cutoff
            W = np.where(pi_r_rc > 1e-9,
                         np.sin(pi_r_rc) / pi_r_rc,
                         1.0)

            # Vectorised over all k: shape (n_k, n_neighbours)
            kr      = k_arr[:, np.newaxis] * r_neighbors[np.newaxis, :]  # (n_k, n_nb)
            sinc_kr = np.where(kr > 1e-9, np.sin(kr) / kr, 1.0)

            # S_i(k) = 1 + Σ_j sinc(kr_ij) * W(r_ij)
            S_i = 1.0 + np.sum(sinc_kr * W[np.newaxis, :], axis=1)  # (n_k,)

            S_sum[bin_idx]  += S_i
            counts[bin_idx] += 1

    # Average over (frame, molecule) contributions
    S_k_zeta = np.full((n_zeta, n_k), np.nan)
    valid = counts > 0
    S_k_zeta[valid] = S_sum[valid] / counts[valid, np.newaxis]

    n_filled = valid.sum()
    print(f"    [sk_zeta_3d] Cluster {cluster_id}: "
          f"{n_filled}/{n_zeta} ζ bins filled  "
          f"(S range [{np.nanmin(S_k_zeta):.3f}, {np.nanmax(S_k_zeta):.3f}])")
    return S_k_zeta, zeta_centers


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def _plot_matplotlib_3d(k_norm, zeta_centers, S_k_zeta,
                         cluster_id, model_name, temperature,
                         output_dir, k_norm_range, zeta_range, s_k_range,
                         cmap='jet'):
    """Matplotlib 3D surface (Panel D style)."""
    from matplotlib.colors import Normalize
    from scipy.ndimage import gaussian_filter

    S_plot = np.copy(S_k_zeta)

    # --- Step 0: slice k_norm to the requested range -----------------------
    # set_xlim only changes axis ticks; the surface is still drawn for ALL k
    # columns. We must filter the data itself so no out-of-range geometry exists.
    k_mask  = (k_norm >= k_norm_range[0]) & (k_norm <= k_norm_range[1])
    k_plot  = k_norm[k_mask]
    S_plot  = S_plot[:, k_mask]

    # --- Step 1: trim leading/trailing all-NaN zeta rows -------------------
    # Empty edge bins cause a filled "wall" polygon at the plot boundary.
    row_has_data = ~np.all(np.isnan(S_plot), axis=1)
    if row_has_data.any():
        first = int(np.argmax(row_has_data))
        last  = int(len(row_has_data) - np.argmax(row_has_data[::-1]))
        S_plot        = S_plot[first:last]
        zeta_trimmed  = zeta_centers[first:last]
    else:
        zeta_trimmed = zeta_centers

    # --- Step 2: fill isolated interior NaN bins by column interpolation ---
    # Isolated NaN (surrounded by valid data) would still create spikes.
    for ki in range(S_plot.shape[1]):
        col      = S_plot[:, ki]
        nan_mask = np.isnan(col)
        if nan_mask.any() and not nan_mask.all():
            idx          = np.arange(len(col))
            col[nan_mask] = np.interp(idx[nan_mask],
                                      idx[~nan_mask], col[~nan_mask])
            S_plot[:, ki] = col

    # --- Step 3: clip to display range ------------------------------------
    S_plot = np.clip(S_plot, s_k_range[0], s_k_range[1])

    # --- Step 4: Gaussian smoothing to suppress noise spikes ---------------
    S_plot = gaussian_filter(S_plot, sigma=1.2)
    S_plot = np.clip(S_plot, s_k_range[0], s_k_range[1])   # re-clip after blur

    K, Z = np.meshgrid(k_plot, zeta_trimmed)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    norm    = Normalize(vmin=s_k_range[0], vmax=s_k_range[1])
    cmap_obj = plt.get_cmap(cmap)
    fcolors = cmap_obj(norm(S_plot))

    ax.plot_surface(K, Z, S_plot,
                    facecolors=fcolors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, label='S(k, ζ)')

    ax.set_xlabel('$kr_{OO}/2\\pi$', labelpad=10, fontsize=13)
    ax.set_ylabel('ζ (Å)',          labelpad=10, fontsize=13)
    ax.set_zlabel('S(k, ζ)',        labelpad=5,  fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim((float(k_plot[0]), float(k_plot[-1])))
    ax.set_ylim((float(zeta_trimmed[0]), float(zeta_trimmed[-1])))
    ax.set_zlim(s_k_range)
    ax.view_init(elev=28, azim=-115)     # ~Tanaka Panel D angle; -115 reduces left-wall exposure

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'3d_sk_zeta_cluster{cluster_id}_{model_name}_T{temperature}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved matplotlib 3D  → {out}")


def _plot_matplotlib_2d(k_norm, zeta_centers, S_k_zeta,
                         cluster_id, model_name, temperature,
                         output_dir, k_norm_range, zeta_range, s_k_range,
                         cmap='jet'):
    """Matplotlib 2D filled contour (Panel E style)."""
    S_plot = np.copy(S_k_zeta)
    S_plot = np.clip(S_plot, s_k_range[0], s_k_range[1])

    levels = np.linspace(s_k_range[0], s_k_range[1], 40)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(k_norm, zeta_centers, S_plot,
                     levels=levels, cmap=cmap,
                     vmin=s_k_range[0], vmax=s_k_range[1],
                     extend='both')
    ax.contour(k_norm, zeta_centers, S_plot,
               levels=levels[::4], colors='k', linewidths=0.4, alpha=0.4)

    # Tanaka reference lines
    ax.axvline(0.75, color='blue', linestyle='--', linewidth=1.5,
               label='$k_{T1}$ (FSDP)')
    ax.axvline(1.00, color='red',  linestyle='--', linewidth=1.5,
               label='$k_{D1}$ (DNLS)')

    cbar = fig.colorbar(cf, ax=ax, label='S(k, ζ)')
    cbar.set_ticks(np.linspace(s_k_range[0], s_k_range[1], 5))

    ax.set_xlabel('$kr_{OO}/2\\pi$', fontsize=13)
    ax.set_ylabel('ζ (Å)',          fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(k_norm_range)
    ax.set_ylim(zeta_range)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'2d_sk_zeta_cluster{cluster_id}_{model_name}_T{temperature}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved matplotlib 2D  → {out}")


def _plot_matplotlib_combined(k_norm, zeta_centers, matrices, cluster_ids,
                               model_name, temperature, output_dir,
                               k_norm_range, zeta_range, s_k_range, cmap='jet',
                               swap_cluster_order=False):
    """Side-by-side 2D contour for all clusters (matplotlib)."""
    n = len(cluster_ids)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6),
                              sharey=True, squeeze=False)
    levels = np.linspace(s_k_range[0], s_k_range[1], 40)
    display_order = list(reversed(cluster_ids)) if swap_cluster_order else list(cluster_ids)

    for col, cid in enumerate(display_order):
        ax    = axes[0, col]
        S_plot = np.clip(matrices[cid], s_k_range[0], s_k_range[1])
        cf = ax.contourf(k_norm, zeta_centers, S_plot,
                         levels=levels, cmap=cmap,
                         vmin=s_k_range[0], vmax=s_k_range[1], extend='both')
        ax.contour(k_norm, zeta_centers, S_plot,
                   levels=levels[::4], colors='k', linewidths=0.4, alpha=0.4)
        ax.axvline(0.75, color='blue', linestyle='--', linewidth=1.5)
        ax.axvline(1.00, color='red',  linestyle='--', linewidth=1.5)
        ax.set_xlim(k_norm_range)
        ax.set_ylim(zeta_range)
        ax.set_xlabel('$kr_{OO}/2\\pi$', fontsize=13)
        ax.set_title(f'Cluster {cid}')
        if col == 0:
            ax.set_ylabel('ζ (Å)', fontsize=13)
        if col == n - 1:
            cbar = fig.colorbar(cf, ax=ax, label='S(k, ζ)')
            cbar.set_ticks(np.linspace(s_k_range[0], s_k_range[1], 5))

    fig.suptitle(f'S(k,ζ) by Cluster | {model_name} {temperature}°C', fontsize=14)
    plt.tight_layout()
    out = os.path.join(output_dir,
                       f'2d_sk_zeta_combined_{model_name}_T{temperature}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved matplotlib combined → {out}")


# ---------------------------------------------------------------------------
# Plotly plots
# ---------------------------------------------------------------------------

def _plot_plotly_3d(k_norm, zeta_centers, S_k_zeta,
                    cluster_id, model_name, temperature,
                    output_dir, k_norm_range, zeta_range, s_k_range):
    """Plotly interactive 3D surface."""
    s_min, s_max = s_k_range
    S_plot = np.clip(S_k_zeta, s_min, s_max)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=k_norm,
        y=zeta_centers,
        z=S_plot,
        colorscale='Jet',
        cmin=s_min, cmax=s_max,
        colorbar=dict(
            title=dict(text='S(k,ζ)', side='right'),
            tickvals=np.round(np.linspace(s_min, s_max, 5), 2).tolist(),
            len=0.7,
        ),
        contours=dict(
            z=dict(show=True, usecolormap=False,
                   highlightcolor='black', project=dict(z=False))
        ),
    ))
    fig.update_layout(
        title=f'S(k,ζ) — Cluster {cluster_id} | {model_name} {temperature}°C',
        scene=dict(
            xaxis_title='kr<sub>OO</sub> / 2π',
            yaxis_title='ζ (Å)',
            zaxis_title='S(k,ζ)',
            xaxis=dict(range=list(k_norm_range)),
            yaxis=dict(range=list(zeta_range)),
            zaxis=dict(range=[s_min, s_max]),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=0.8, z=0.6),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out = os.path.join(output_dir,
                       f'3d_sk_zeta_cluster{cluster_id}_{model_name}_T{temperature}.html')
    fig.write_html(out)
    print(f"    Saved Plotly 3D      → {out}")


def _plot_plotly_2d(k_norm, zeta_centers, S_k_zeta,
                    cluster_id, model_name, temperature,
                    output_dir, k_norm_range, zeta_range, s_k_range):
    """Plotly 2D contour — HTML + identical PNG."""
    s_min, s_max = s_k_range
    S_plot = np.clip(S_k_zeta, s_min, s_max)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=k_norm, y=zeta_centers, z=S_plot,
        colorscale='Jet',
        zmin=s_min, zmax=s_max,
        contours=dict(
            start=s_min, end=s_max,
            size=(s_max - s_min) / 25,
            showlabels=False,
        ),
        colorbar=dict(title=dict(text='S(k,ζ)', side='right'), len=0.7),
        line=dict(smoothing=0.85),
    ))
    for x_ref, label, color in [(0.75, 'k<sub>T1</sub>', 'blue'),
                                 (1.00, 'k<sub>D1</sub> (DNLS)', 'red')]:
        fig.add_vline(x=x_ref, line=dict(color=color, dash='dash', width=1.5),
                      annotation_text=label, annotation_position='top',
                      annotation_font=dict(color=color))
    fig.update_layout(
        title=f'S(k,ζ) Contour — Cluster {cluster_id} | {model_name} {temperature}°C',
        xaxis=dict(title='kr<sub>OO</sub> / 2π', range=list(k_norm_range)),
        yaxis=dict(title='ζ (Å)',               range=list(zeta_range)),
        width=800, height=600,
    )

    out_html = os.path.join(output_dir,
                            f'2d_sk_zeta_cluster{cluster_id}_{model_name}_T{temperature}.html')
    fig.write_html(out_html)
    print(f"    Saved Plotly 2D HTML → {out_html}")

    out_png = os.path.join(output_dir,
                           f'2d_sk_zeta_cluster{cluster_id}_{model_name}_T{temperature}.png')
    try:
        fig.write_image(out_png, scale=2)
        print(f"    Saved Plotly 2D PNG  → {out_png}")
    except Exception as exc:
        print(f"    [WARNING] PNG export failed ({exc}). pip install kaleido")


def _plot_plotly_combined(k_norm, zeta_centers, matrices, cluster_ids,
                           model_name, temperature, output_dir,
                           k_norm_range, zeta_range, s_k_range, cmap='jet',
                           swap_cluster_order=False):
    """Plotly side-by-side 2D contour for all clusters — HTML + identical PNG."""
    s_min, s_max = s_k_range
    n = len(cluster_ids)
    display_order = list(reversed(cluster_ids)) if swap_cluster_order else list(cluster_ids)
    # Plotly colorscale names (capitalize first letter)
    plotly_cmap = cmap.capitalize() if cmap in ('jet', 'plasma', 'viridis') else 'Jet'

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f'Cluster {c}' for c in display_order],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col_idx, cid in enumerate(display_order, start=1):
        S_plot     = np.clip(matrices[cid], s_min, s_max)
        show_scale = (col_idx == n)
        fig.add_trace(
            go.Contour(
                x=k_norm, y=zeta_centers, z=S_plot,
                colorscale=plotly_cmap, zmin=s_min, zmax=s_max,
                showscale=show_scale,
                colorbar=dict(title=dict(text='S(k,ζ)', side='right'),
                              len=0.85) if show_scale else {},
                contours=dict(start=s_min, end=s_max,
                              size=(s_max - s_min) / 25, showlabels=False),
                line=dict(smoothing=0.85),
            ),
            row=1, col=col_idx,
        )
        for x_ref, lbl, color in [(0.75, 'k<sub>T1</sub>', 'blue'),
                                   (1.00, 'k<sub>D1</sub> (DNLS)', 'red')]:
            fig.add_vline(
                x=x_ref, line=dict(color=color, dash='dash', width=1.5),
                annotation_text=lbl, annotation_position='top',
                annotation_font=dict(color=color),
                row=1, col=col_idx,
            )

    fig.update_xaxes(title_text='kr<sub>OO</sub> / 2π', range=list(k_norm_range))
    fig.update_yaxes(title_text='ζ (Å)',               range=list(zeta_range))
    fig.update_layout(
        title=f'S(k,ζ) — {model_name} {temperature}°C',
        width=700 * n, height=600,
    )

    out_html = os.path.join(output_dir,
                            f'2d_sk_zeta_combined_{model_name}_T{temperature}.html')
    fig.write_html(out_html)
    print(f"    Saved Plotly HTML    → {out_html}")

    out_png = os.path.join(output_dir,
                           f'2d_sk_zeta_combined_{model_name}_T{temperature}.png')
    try:
        fig.write_image(out_png, scale=2)
        print(f"    Saved Plotly PNG     → {out_png}")
    except Exception as exc:
        print(f"    [WARNING] PNG export failed ({exc}). "
              f"Install kaleido: pip install kaleido")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_sk_zeta_all_clusters(
        trajectory,
        k_values,
        cluster_labels_matrix,
        zeta_file,
        output_dir,
        model_name,
        temperature,
        rc_cutoff         = 1.5,
        zeta_range        = (-1.0, 1.5),
        k_norm_range      = (0.6, 2.25),
        n_zeta_bins       = 40,
        s_k_range         = None,
        cmap              = None,
        swap_cluster_order= True,
        r_OO_nm           = 0.285,
):
    """
    Compute S(k,ζ) per cluster and produce matplotlib + Plotly plots.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
    k_values : np.ndarray  raw k in nm^-1
    cluster_labels_matrix : np.ndarray (n_frames, n_molecules)
    zeta_file : str  path to .mat file with zeta_all
    output_dir : str
    model_name : str
    temperature : float  °C
    rc_cutoff : float  nm
    zeta_range : (min, max) Å
    k_norm_range : (min, max) for kr_OO/2π axis
    n_zeta_bins : int
    s_k_range : (min, max) S(k) colour/z limits; None = auto per model
    cmap : str  matplotlib colormap; None = auto per model
    swap_cluster_order : bool  if True, left=Cluster 1, right=Cluster 0
    r_OO_nm : float  O-O nearest-neighbour distance in nm
    """
    # TIP5P: narrower range and different colormap for better contrast
    if s_k_range is None:
        s_k_range = (0.0, 1.5) if 'tip5p' in str(model_name).lower() else (0.0, 2.5)
    if cmap is None:
        cmap = 'plasma' if 'tip5p' in str(model_name).lower() else 'jet'

    if zeta_file is None or not os.path.exists(str(zeta_file)):
        print(f"  [sk_zeta_3d] Skipping — zeta file not found: {zeta_file}")
        return

    os.makedirs(output_dir, exist_ok=True)

    zeta_all = _load_zeta(zeta_file)
    if zeta_all is None:
        return

    k_norm    = k_values * r_OO_nm / (2 * np.pi)
    zeta_bins = np.linspace(zeta_range[0], zeta_range[1], n_zeta_bins + 1)

    cluster_ids = sorted(set(int(c)
                             for c in np.unique(cluster_labels_matrix)
                             if c >= 0))
    print(f"\n  [sk_zeta_3d] Clusters: {cluster_ids}  "
          f"ζ bins: {n_zeta_bins}  "
          f"ζ range: {zeta_range} Å")

    matrices = {}    # cluster_id -> S_k_zeta (n_zeta, n_k)

    for cid in cluster_ids:
        print(f"\n  [sk_zeta_3d] --- Cluster {cid} ---")

        S_k_zeta, zeta_centers = compute_sk_zeta_matrix(
            trajectory=trajectory,
            k_values=k_values,
            cluster_labels_matrix=cluster_labels_matrix,
            cluster_id=cid,
            zeta_all=zeta_all,
            zeta_bins=zeta_bins,
            rc_cutoff=rc_cutoff,
        )
        matrices[cid] = S_k_zeta

        kw = dict(k_norm=k_norm, zeta_centers=zeta_centers, S_k_zeta=S_k_zeta,
                  cluster_id=cid, model_name=model_name, temperature=temperature,
                  output_dir=output_dir, k_norm_range=k_norm_range,
                  zeta_range=zeta_range, s_k_range=s_k_range, cmap=cmap)

        _plot_matplotlib_3d(**kw)
        _plot_matplotlib_2d(**kw)
        _plot_plotly_3d(**kw)
        _plot_plotly_2d(**kw)

    # Combined side-by-side (swap order so LFTS/DNLS match paper convention)
    _plot_matplotlib_combined(k_norm, zeta_centers, matrices, cluster_ids,
                               model_name, temperature, output_dir,
                               k_norm_range, zeta_range, s_k_range, cmap,
                               swap_cluster_order=swap_cluster_order)
    _plot_plotly_combined(k_norm, zeta_centers, matrices, cluster_ids,
                          model_name, temperature, output_dir,
                          k_norm_range, zeta_range, s_k_range, cmap,
                          swap_cluster_order=swap_cluster_order)

    print(f"\n  [sk_zeta_3d] Done. All outputs → {output_dir}")
