#!/usr/bin/env python3
"""
replot_heatmap.py
=================
Regenerate the DBSCAN parameter heatmap from an existing
param_search_summary.csv — no DBSCAN re-run required.

Usage
-----
  python replot_heatmap.py \
      --csv  param_search_results/tip4p2005_T-20/param_search_summary.csv \
      --out  param_search_results/tip4p2005_T-20 \
      --cmap RdBu
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# Custom diverging colormap: deep pink → white → steel blue
_PINK_BLUE = LinearSegmentedColormap.from_list(
    "pink_blue",
    ["#c2185b", "#e75480", "#f8bbd9", "#ffffff", "#bbdefb", "#4a90d9", "#1565c0"],
    N=256,
)


def replot(csv_path: str, out_dir: str, cmap_name: str, algorithm: str = "dbscan"):
    df = pd.read_csv(csv_path)

    if algorithm == "hdbscan":
        x_col, x_title = "min_cluster_size", "min_cluster_size"
        y_title = "min_samples  (noise sensitivity)"
        out_name = "hdbscan_param_heatmap.png"
    else:
        x_col, x_title = "eps", "eps  (neighbourhood radius in scaled feature space)"
        y_title = "Minimum Sample"
        out_name = "dbscan_param_heatmap.png"

    x_vals = sorted(df[x_col].unique())
    ms_vals = sorted(df["min_samples"].unique())
    ne, nm  = len(x_vals), len(ms_vals)

    sil_mat   = np.full((nm, ne), np.nan)
    noise_mat = np.full((nm, ne), np.nan)

    for _, row in df.iterrows():
        i = ms_vals.index(row["min_samples"])
        j = x_vals.index(row[x_col])
        sil_mat[i, j]   = row["silhouette"]
        noise_mat[i, j] = row["noise_pct"]

    # ── diverging norm: white at 0, full colour at ±max ──────────────────────
    vmax = max(abs(np.nanmin(sil_mat)), abs(np.nanmax(sil_mat)), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    if cmap_name == "pink_blue":
        cmap = _PINK_BLUE.copy()
    else:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad("#cccccc")          # NaN cells → mid grey

    cell_w = max(1.4, 9.0 / ne)
    cell_h = max(0.8, 7.0 / nm)
    figw   = max(10, ne * cell_w + 3)
    figh   = max(6,  nm * cell_h + 3)

    fig, ax = plt.subplots(figsize=(figw, figh))

    im = ax.imshow(sil_mat, aspect="auto", cmap=cmap, norm=norm,
                   origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Silhouette Score", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.axhline(0, color="black", lw=1.2, alpha=0.7)   # zero line on bar

    # ── cell text: noise % with luminance-aware colour ───────────────────────
    for i in range(nm):
        for j in range(ne):
            sil   = sil_mat[i, j]
            noise = noise_mat[i, j]
            style = "bold" if not np.isnan(sil) else "normal"
            ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                    fontsize=9.5, color="black", fontweight=style)

    # ── axes ─────────────────────────────────────────────────────────────────
    # x-axis labels: decimals for eps, integers for mcs
    if algorithm == "hdbscan":
        x_tick_labels = [str(int(v)) for v in x_vals]
    else:
        x_tick_labels = [f"{v:.2f}" for v in x_vals]

    ax.set_xticks(np.arange(ne))
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(nm))
    ax.set_yticklabels([str(m) for m in ms_vals], fontsize=9)
    ax.set_xlabel(x_title, fontsize=12)
    ax.set_ylabel(y_title, fontsize=12)

    ax.set_title("colour = silhouette score   ·   text = noise removed (%)",
                 fontsize=11, pad=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Replot DBSCAN heatmap from existing CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",  required=True, help="Path to param_search_summary.csv")
    p.add_argument("--out",  required=True, help="Output directory for the PNG")
    p.add_argument("--algorithm", default="dbscan", choices=["dbscan", "hdbscan"],
                   help="Which algorithm's CSV to replot")
    p.add_argument("--cmap", default="RdBu",
                   help="Matplotlib colormap name. "
                        "Good diverging options: RdBu, coolwarm, PiYG, PRGn, BrBG. "
                        "Sequential: magma, viridis, plasma, turbo.")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    replot(args.csv, args.out, args.cmap, args.algorithm)


if __name__ == "__main__":
    main()
