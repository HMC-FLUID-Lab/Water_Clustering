#!/usr/bin/env python3
"""
plot_sk_multimodel.py
=====================
Overlay per-cluster S(k) for **several water models** on shared axes (one panel
per cluster), using the same workflow as ``plot_sk_multitemp.py``.

Typical use: TIP4P/2005, TIP5P, and SWM4-NDP at one temperature, all clustered
with DBSCAN→GMM at the same ``eps`` and ``min_samples``, then compared here.

Each condition line (config or ``--conditions``) is::

  model_slug | path/to.dcd | path/to.pdb | path/to/cluster_labels_matrix.csv

``model_slug`` is a short name for the legend, e.g. ``tip4p2005``, ``tip5p``,
``swm4ndp``.

Example
-------
  python plot_sk_multimodel.py \\
      --output-dir ./multimodel_sk \\
      --annotation "DBSCAN→GMM  ε=0.15  min_samples=20  T=-20°C" \\
      --conditions \\
        "tip4p2005 | .../dcd_tip4p2005_T-20_...dcd | .../inistate_....pdb | .../cluster_labels_matrix_dbscan_gmm.csv" \\
        "tip5p | ... | ... | ..." \\
        "swm4ndp | ... | ... | ..."
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from plot_sk_multitemp import (  # noqa: E402
    load_condition,
    enforce_consistent_labels,
    _style_ax,
    CLUSTER_NAMES,
)

try:
    _clustering = os.path.normpath(os.path.join(_HERE, "..", "clustering"))
    if _clustering not in sys.path:
        sys.path.insert(0, _clustering)
    from plot_style import set_default_plot
    set_default_plot()
except ImportError:
    pass

_MODEL_DISPLAY = {
    "tip4p2005": "TIP4P/2005",
    "tip5p": "TIP5P",
    "swm4ndp": "SWM4-NDP",
}

# Distinct, print-friendly trio: cool teal / warm clay / soft violet (harmonious on white)
_MODEL_COLORS = {
    "tip4p2005": "#0f766e",   # deep lagoon teal
    "tip5p": "#c2410c",       # burnt sienna / terracotta
    "swm4ndp": "#6d28d9",     # rich violet (muted enough for thin lines)
}
_MODEL_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _display_model(slug: str) -> str:
    return _MODEL_DISPLAY.get(slug.lower(), slug)


def _model_color(slug: str, fallback_index: int) -> str:
    key = slug.lower()
    if key in _MODEL_COLORS:
        return _MODEL_COLORS[key]
    palette = ["#0f766e", "#c2410c", "#6d28d9", "#0369a1", "#b45309"]
    return palette[fallback_index % len(palette)]


def parse_model_conditions(raw_lines):
    """Lines: model_slug | dcd | pdb | csv"""
    conditions = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            print(f"  [WARN] Skipping malformed line: {line!r}")
            continue
        slug, dcd, pdb, csv = parts
        conditions.append({"model": slug.strip(), "dcd": dcd, "pdb": pdb, "csv": csv})
    return conditions


def plot_multimodel(all_results, k_values, output_dir, annotation=None):
    """
    all_results: dict  {model_slug (str): {cluster_id: {'S_k_avg', 'S_k_std'}}}
    """
    k_norm = k_values * 0.285 / (2 * np.pi)
    every = max(1, len(k_norm) // 40)
    eb_idx = np.arange(0, len(k_norm), every)

    models = sorted(all_results.keys(), key=lambda s: s.lower())

    print("\n  Checking cluster label consistency across models (FSDP criterion) …")
    all_results = enforce_consistent_labels(all_results, k_values)

    cluster_ids = sorted(
        set(cid for res in all_results.values() for cid in res.keys())
    )
    os.makedirs(output_dir, exist_ok=True)

    n_clusters = len(cluster_ids)
    fig, axarr = plt.subplots(n_clusters, 1, figsize=(13, 6 * n_clusters), squeeze=False)
    axes = {cid: axarr[i, 0] for i, cid in enumerate(cluster_ids)}

    for mi, slug in enumerate(models):
        res_dict = all_results[slug]
        color = _model_color(slug, mi)
        marker = _MODEL_MARKERS[mi % len(_MODEL_MARKERS)]
        disp = _display_model(slug)

        for cid in cluster_ids:
            if cid not in res_dict:
                continue
            ax = axes[cid]
            res = res_dict[cid]
            ax.plot(
                k_norm,
                res["S_k_avg"],
                color=color,
                linewidth=0.9,
                marker=marker,
                markevery=every,
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.3,
                label=disp,
            )
            ax.errorbar(
                k_norm[eb_idx],
                res["S_k_avg"][eb_idx],
                yerr=res["S_k_std"][eb_idx],
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=2,
                capthick=0.7,
                alpha=0.5,
            )

    for cid in cluster_ids:
        ax = axes[cid]
        cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        title = cname
        if annotation:
            title = f"{cname}\n{annotation}"
        _style_ax(ax, title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            fontsize=9,
            frameon=True,
            edgecolor="black",
            framealpha=0.9,
            loc="upper right",
        )

    fig.suptitle(
        r"Per-cluster $S(k)$ — model comparison (DBSCAN→GMM labels)",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "sk_multimodel_dbscan_gmm_panels.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # One PNG per cluster (all models on one axis)
    for cid in cluster_ids:
        cname = CLUSTER_NAMES.get(cid, f"Cluster {cid}")
        fig2, ax2 = plt.subplots(figsize=(13, 7))
        for mi, slug in enumerate(models):
            if cid not in all_results[slug]:
                continue
            res = all_results[slug][cid]
            color = _model_color(slug, mi)
            marker = _MODEL_MARKERS[mi % len(_MODEL_MARKERS)]
            disp = _display_model(slug)
            ax2.plot(
                k_norm,
                res["S_k_avg"],
                color=color,
                linewidth=0.9,
                marker=marker,
                markevery=every,
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.3,
                label=disp,
            )
            ax2.errorbar(
                k_norm[eb_idx],
                res["S_k_avg"][eb_idx],
                yerr=res["S_k_std"][eb_idx],
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=2,
                capthick=0.7,
                alpha=0.5,
            )
        title2 = f"{cname} — all models"
        if annotation:
            title2 = f"{cname}\n{annotation}"
        _style_ax(ax2, title2)
        ax2.legend(
            fontsize=9,
            frameon=True,
            edgecolor="black",
            framealpha=0.9,
            loc="upper right",
        )
        plt.tight_layout()
        out2 = os.path.join(output_dir, f"sk_multimodel_dbscan_gmm_cluster{cid}.png")
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out2}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-model per-cluster S(k) comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir",
        default="./multimodel_sk_results",
        help="Directory for output PNGs",
    )
    p.add_argument(
        "--annotation",
        default=None,
        help="Optional subtitle (e.g. ε, min_samples, T)",
    )
    p.add_argument("--rc-cutoff", type=float, default=1.5)
    p.add_argument("--k-max", type=float, default=50.0)
    p.add_argument("--k-points", type=int, default=500)
    p.add_argument("--n-frames", type=int, default=None)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--conditions",
        nargs="+",
        metavar="COND",
        help='Each line: "model_slug | dcd | pdb | labels_csv"',
    )
    src.add_argument("--config", metavar="FILE", help="One condition per line")
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        with open(args.config) as fh:
            raw = fh.readlines()
    else:
        raw = args.conditions

    conditions = parse_model_conditions(raw)
    if not conditions:
        print("ERROR: no valid conditions."); sys.exit(1)

    print("=" * 65)
    print(" MULTI-MODEL S(k)  |  DBSCAN→GMM comparison")
    for c in conditions:
        print(f"   {_display_model(c['model']):12s}  |  {os.path.basename(c['dcd'])}")
    print("=" * 65)

    k_values = np.linspace(0.1, args.k_max, args.k_points)
    all_results = {}

    for c in conditions:
        slug = c["model"]
        print(f"\n── {slug} ────────────────────────────────────────")
        res = load_condition(
            slug,
            c["dcd"],
            c["pdb"],
            c["csv"],
            k_values,
            args.rc_cutoff,
            args.n_frames,
        )
        if res is not None:
            all_results[slug] = res

    if not all_results:
        print("ERROR: no conditions loaded successfully."); sys.exit(1)

    print(f"\n── Plotting ({len(all_results)} models) ─────────────────────")
    plot_multimodel(all_results, k_values, args.output_dir, args.annotation)

    print("\n" + "=" * 65)
    print(f" DONE — outputs in: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
