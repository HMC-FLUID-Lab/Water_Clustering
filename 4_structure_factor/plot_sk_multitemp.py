#!/usr/bin/env python3
"""
plot_sk_multitemp.py
====================
Overlay S(k) curves for multiple temperatures on one graph, one panel per
cluster.  Reuses compute_per_cluster_structure_factor() from
structure_factor_bycluster.py so the physics is identical.

Usage
-----
  python plot_sk_multitemp.py \\
      --model tip4p2005 \\
      --conditions \\
          "-10 | dcd_tip4p2005_T-10_N1024_Run01_0.dcd | inistate_tip4p2005_T-10_N1024_Run01.pdb | cluster_labels_matrix_tip4p2005_T-10_dbscan_gmm.csv" \\
          "-20 | dcd_tip4p2005_T-20_N1024_Run01_0.dcd | inistate_tip4p2005_T-20_N1024_Run01.pdb | cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv" \\
          "-30 | dcd_tip4p2005_T-30_N1024_Run01_0.dcd | inistate_tip4p2005_T-30_N1024_Run01.pdb | cluster_labels_matrix_tip4p2005_T-30_dbscan_gmm.csv" \\
      --output-dir ./multitemp_results \\
      --layout side-by-side

  # Or specify a config file instead of --conditions (see example below)
  python plot_sk_multitemp.py --config my_conditions.txt --model tip4p2005

Config file format (one condition per line, fields separated by |):
  temperature | dcd_file | pdb_file | cluster_labels_csv

Layout options:
  side-by-side   one panel per cluster, placed side by side  (default)
  rows           one panel per cluster, stacked vertically
  overlay        all clusters on a single panel

Example config file (tip4p2005_conditions.txt):
  -10 | /path/to/dcd_T-10.dcd | /path/to/inistate_T-10.pdb | /path/to/labels_T-10.csv
  -20 | /path/to/dcd_T-20.dcd | /path/to/inistate_T-20.pdb | /path/to/labels_T-20.csv
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors

# ── make sure structure_factor_bycluster.py is importable ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from structure_factor_bycluster import (
    load_trajectory,
    compute_per_cluster_structure_factor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_NAMES = {
    0: r"Cluster 0 ($\rho$-state / DNLS)",
    1: r"Cluster 1 (S-state / LFTS / tetrahedral)",
}

# One distinct marker per temperature (coldest → warmest)
_TEMP_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Fixed high-contrast colors: coldest → warmest
_TEMP_COLORS = ["#2166ac", "#4dac26", "#d6604d"]   # blue, green, red


def _temp_colormap(temps):
    """Return {temp: color}, cold→warm mapped to blue, green, red."""
    temps_sorted = sorted(temps)
    n = len(temps_sorted)
    colors = (_TEMP_COLORS * ((n // len(_TEMP_COLORS)) + 1))[:n]
    return {t: colors[i] for i, t in enumerate(temps_sorted)}


def enforce_consistent_labels(results_by_temp, k_values):
    """
    Ensure cluster 0 = DNLS and cluster 1 = LFTS consistently across
    all temperatures by inspecting S(k) in the FSDP region (0.65–0.82 in
    kr_OO/2pi units).  LFTS has higher S(k) there; DNLS has lower S(k)
    but a higher main peak.  If a temperature has them swapped, exchange
    the two cluster entries.
    """
    k_norm = k_values * 0.285 / (2 * np.pi)
    fsdp_mask = (k_norm >= 0.65) & (k_norm <= 0.82)

    corrected = {}
    for temp, res in results_by_temp.items():
        if 0 not in res or 1 not in res:
            corrected[temp] = res
            continue

        sk0_fsdp = res[0]["S_k_avg"][fsdp_mask].mean()
        sk1_fsdp = res[1]["S_k_avg"][fsdp_mask].mean()

        # cluster with HIGHER FSDP S(k) is LFTS → should be cluster 1
        if sk0_fsdp > sk1_fsdp:
            # labels are swapped — exchange them
            print(f"  [{_condition_label_str(temp)}] Labels swapped "
                  f"(FSDP: c0={sk0_fsdp:.4f} > c1={sk1_fsdp:.4f}) — correcting.")
            corrected[temp] = {0: res[1], 1: res[0]}
        else:
            corrected[temp] = res

    return corrected


def _style_ax(ax, title):
    ax.axvline(0.75, color="green", ls="--", lw=1.2, alpha=0.7,
               label=r"$k_{T1}$ (FSDP)")
    ax.axvline(1.0,  color="red",   ls="--", lw=1.2, alpha=0.7,
               label=r"$k_{D1}$ (DNLS)")
    ax.set_xlabel(r"$k \cdot r_{OO} / 2\pi$", fontsize=13)
    ax.set_ylabel(r"$S(k)$", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.5, 2.2)
    ax.set_ylim(0.5, 1.4)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(direction="in", which="both", top=True, right=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core: load one condition
# ─────────────────────────────────────────────────────────────────────────────

def _condition_label_str(condition_label):
    """Format temperature (float) or arbitrary tag (str) for log lines."""
    if isinstance(condition_label, (int, float)):
        return f"T={condition_label}"
    return str(condition_label)


def load_condition(condition_label, dcd_file, pdb_file, labels_csv,
                   k_values, rc_cutoff, n_frames=None):
    """
    Load trajectory + cluster labels and compute S(k).
    ``condition_label`` may be a temperature (float/int) or a model tag (str).
    Returns {cluster_id: {'S_k_avg', 'S_k_std'}} or None on failure.
    """
    import pandas as pd

    _lab = _condition_label_str(condition_label)

    for path, name in [(dcd_file, "DCD"), (pdb_file, "PDB"), (labels_csv, "Labels")]:
        if not os.path.isfile(path):
            print(f"  [SKIP {_lab}] {name} not found: {path}")
            return None

    traj = load_trajectory(dcd_file, pdb_file, n_frames)

    label_matrix = pd.read_csv(labels_csv, header=None).values.astype(int)
    n_use = min(traj.n_frames, label_matrix.shape[0])
    if traj.n_frames != label_matrix.shape[0]:
        print(f"  [{_lab}] Frame mismatch — using {n_use} frames")
        traj         = traj[:n_use]
        label_matrix = label_matrix[:n_use]

    results = compute_per_cluster_structure_factor(
        traj, rc_cutoff, k_values, label_matrix
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_multitemp(all_results, k_values, model_name, layout, output_dir):
    """
    all_results: dict  {temp (float): {cluster_id: {'S_k_avg', 'S_k_std'}}}
    """
    k_norm = k_values * 0.285 / (2 * np.pi)
    every  = max(1, len(k_norm) // 40)
    eb_idx = np.arange(0, len(k_norm), every)

    temps = sorted(all_results.keys())

    # Enforce physical consistency: cluster 1 = LFTS (higher FSDP) across all temps
    print("\n  Checking cluster label consistency across temperatures …")
    all_results = enforce_consistent_labels(all_results, k_values)

    color_map = _temp_colormap(temps)

    # Collect all cluster IDs present across conditions
    cluster_ids = sorted(set(
        cid for res in all_results.values() for cid in res.keys()
    ))

    os.makedirs(output_dir, exist_ok=True)

    # ── Layout setup ────────────────────────────────────────────────────────
    n_clusters = len(cluster_ids)

    if layout == "overlay":
        fig, axes = plt.subplots(1, 1, figsize=(13, 7))
        axes = {cid: axes for cid in cluster_ids}   # all clusters → same ax
    elif layout == "rows":
        fig, axarr = plt.subplots(n_clusters, 1,
                                  figsize=(13, 6 * n_clusters), squeeze=False)
        axes = {cid: axarr[i, 0] for i, cid in enumerate(cluster_ids)}
    else:  # side-by-side (default)
        fig, axarr = plt.subplots(1, n_clusters,
                                  figsize=(13 * n_clusters, 7), squeeze=False)
        axes = {cid: axarr[0, i] for i, cid in enumerate(cluster_ids)}

    # ── Draw curves ─────────────────────────────────────────────────────────
    for ti, temp in enumerate(temps):
        res_dict = all_results[temp]
        color    = color_map[temp]
        marker   = _TEMP_MARKERS[ti % len(_TEMP_MARKERS)]

        for cid in cluster_ids:
            if cid not in res_dict:
                continue
            ax  = axes[cid]
            res = res_dict[cid]

            ax.plot(k_norm, res["S_k_avg"],
                    color=color, linewidth=0.9,
                    marker=marker, markevery=every, markersize=6,
                    markerfacecolor="none", markeredgewidth=1.3,
                    label=f"T = {temp:.0f}°C")
            ax.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                        yerr=res["S_k_std"][eb_idx],
                        fmt="none", ecolor=color, elinewidth=0.7,
                        capsize=2, capthick=0.7, alpha=0.5)

    # ── Style each panel ────────────────────────────────────────────────────
    for cid in cluster_ids:
        ax    = axes[cid]
        cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        if layout == "overlay":
            title = f"All Clusters — {model_name}"
        else:
            title = f"{cname}\n{model_name}"
        _style_ax(ax, title)
        # Legend: temperatures + reference lines
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=9, frameon=True,
                  edgecolor="black", framealpha=0.9,
                  ncol=1, loc="upper right")


    fig.suptitle(f"Per-Cluster S(k) — {model_name} — multiple temperatures",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out = os.path.join(output_dir, f"sk_multitemp_{model_name}_{layout}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # ── Also save one PNG per cluster ────────────────────────────────────────
    for cid in cluster_ids:
        cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        fig2, ax2 = plt.subplots(figsize=(13, 7))
        for ti, temp in enumerate(temps):
            if cid not in all_results[temp]:
                continue
            res    = all_results[temp][cid]
            color  = color_map[temp]
            marker = _TEMP_MARKERS[ti % len(_TEMP_MARKERS)]
            ax2.plot(k_norm, res["S_k_avg"],
                     color=color, linewidth=0.9,
                     marker=marker, markevery=every, markersize=6,
                     markerfacecolor="none", markeredgewidth=1.3,
                     label=f"T = {temp:.0f}°C")
            ax2.errorbar(k_norm[eb_idx], res["S_k_avg"][eb_idx],
                         yerr=res["S_k_std"][eb_idx],
                         fmt="none", ecolor=color, elinewidth=0.7,
                         capsize=2, capthick=0.7, alpha=0.5)
        _style_ax(ax2, f"{cname} — {model_name}")
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels, fontsize=9, frameon=True,
                   edgecolor="black", framealpha=0.9)
        plt.tight_layout()
        out2 = os.path.join(output_dir,
                            f"sk_multitemp_{model_name}_cluster{cid}.png")
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-temperature per-cluster S(k) comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",       required=True,
                   help="Water model name for plot titles (e.g. tip4p2005)")
    p.add_argument("--output-dir",  default="./multitemp_sk_results",
                   help="Directory for output PNGs")
    p.add_argument("--layout",
                   choices=["side-by-side", "rows", "overlay"],
                   default="side-by-side",
                   help="Panel arrangement (default: side-by-side)")
    p.add_argument("--rc-cutoff",   type=float, default=1.5,
                   help="Real-space cutoff (nm, default 1.5)")
    p.add_argument("--k-max",       type=float, default=50.0)
    p.add_argument("--k-points",    type=int,   default=500)
    p.add_argument("--n-frames",    type=int,   default=None,
                   help="Limit frames per condition (None = all)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--conditions", nargs="+", metavar="COND",
                     help='Each condition as "temp | dcd | pdb | labels_csv"')
    src.add_argument("--config",     metavar="FILE",
                     help="Text file with one condition per line "
                          "(fields: temp | dcd | pdb | labels_csv)")
    return p.parse_args()


def parse_conditions(raw_lines):
    """Parse list of 'temp | dcd | pdb | csv' strings into list of dicts."""
    conditions = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            print(f"  [WARN] Skipping malformed condition line: {line!r}")
            continue
        temp, dcd, pdb, csv = parts
        conditions.append({"temp": float(temp), "dcd": dcd,
                            "pdb": pdb, "csv": csv})
    return conditions


def main():
    args = parse_args()

    # Load condition list
    if args.config:
        with open(args.config) as fh:
            raw = fh.readlines()
    else:
        raw = args.conditions

    conditions = parse_conditions(raw)
    if not conditions:
        print("ERROR: no valid conditions found."); sys.exit(1)

    print("=" * 65)
    print(f" MULTI-TEMPERATURE S(k)  |  model: {args.model}")
    print(f" Conditions: {len(conditions)}")
    for c in conditions:
        print(f"   T={c['temp']}°C  |  {os.path.basename(c['dcd'])}")
    print("=" * 65)

    k_values = np.linspace(0.1, args.k_max, args.k_points)

    all_results = {}
    for c in conditions:
        print(f"\n── T = {c['temp']}°C ──────────────────────────────────────")
        res = load_condition(
            temp      = c["temp"],
            dcd_file  = c["dcd"],
            pdb_file  = c["pdb"],
            labels_csv= c["csv"],
            k_values  = k_values,
            rc_cutoff = args.rc_cutoff,
            n_frames  = args.n_frames,
        )
        if res is not None:
            all_results[c["temp"]] = res

    if not all_results:
        print("ERROR: no conditions loaded successfully."); sys.exit(1)

    print(f"\n── Plotting ({len(all_results)} temperatures) ──────────────────")
    plot_multitemp(all_results, k_values, args.model,
                   args.layout, args.output_dir)

    print("\n" + "=" * 65)
    print(f" DONE — outputs in: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
