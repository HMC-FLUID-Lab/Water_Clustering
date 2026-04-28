#!/usr/bin/env python3
"""
Replot DBSCAN param heatmap from existing param_search_summary.csv
(no rerun of the parameter search).

Usage
-----
  python replot_param_heatmap.py param_search_results/tip4p2005_T-20/param_search_summary.csv
  python replot_param_heatmap.py path/to/param_search_summary.csv --out ./figures
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import (
    set_default_plot,
    style_dbscan_param_heatmap_axes,
    dbscan_param_heatmap_figure_caption,
)

set_default_plot()


def plot_dbscan_heatmap_from_csv(csv_path: str, out_dir: str | None = None) -> None:
    """Load param_search_summary.csv and redraw the heatmap with publication style."""
    df = pd.read_csv(csv_path)
    if "eps" not in df.columns or "min_samples" not in df.columns:
        raise ValueError(
            f"Expected columns 'eps', 'min_samples', 'silhouette', 'noise_pct' in {csv_path}"
        )

    eps_range = np.array(sorted(df["eps"].unique()))
    ms_range = sorted(df["min_samples"].unique())
    ne, nm = len(eps_range), len(ms_range)

    sil_mat = np.full((nm, ne), np.nan)
    noise_mat = np.full((nm, ne), np.nan)
    for _, row in df.iterrows():
        i = ms_range.index(int(row["min_samples"]))
        j = np.searchsorted(eps_range, row["eps"])
        if j < ne and abs(eps_range[j] - row["eps"]) < 1e-9:
            sil_mat[i, j] = row["silhouette"]
            noise_mat[i, j] = row["noise_pct"] / 100.0

    nclust_mat = np.zeros((nm, ne), dtype=int)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(csv_path))

    # Reuse param_search plotting logic (style-aligned)
    eps_labels = [f"{e:.3f}" for e in eps_range]
    ms_labels = [str(m) for m in ms_range]
    cell_w = max(1.0, 8.0 / ne)
    cell_h = max(0.6, 6.0 / nm)
    figw = max(10, ne * cell_w + 3)
    figh = max(6, nm * cell_h + 3)
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_facecolor("white")

    noise_pct_mat = noise_mat * 100
    sil_display = np.where(np.isnan(sil_mat), -0.1, sil_mat)
    cmap = plt.cm.plasma.copy()
    cmap.set_under("lightgrey")
    im = ax.imshow(sil_display, aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=1.0, origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    for i in range(nm):
        for j in range(ne):
            sil = sil_mat[i, j]
            noise = noise_pct_mat[i, j]
            txt_col = "black" if (not np.isnan(sil) and sil > 0.6) else "white"
            if np.isnan(sil):
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        color="dimgrey", fontsize=9)
            else:
                ax.text(j, i, f"{noise:.1f}%", ha="center", va="center",
                        color=txt_col, fontweight="bold", fontsize=9)

    best_eps, best_ms, best_sil = None, None, np.nan
    if not np.all(np.isnan(sil_mat)):
        bi = np.unravel_index(np.nanargmax(sil_mat), sil_mat.shape)
        best_eps = float(eps_range[bi[1]])
        best_ms = ms_range[bi[0]]
        best_sil = float(sil_mat[bi])

    ax.set_xticks(np.arange(ne))
    ax.set_xticklabels(eps_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(nm))
    ax.set_yticklabels(ms_labels)

    style_dbscan_param_heatmap_axes(ax, cbar)
    fig.text(
        0.5,
        0.02,
        dbscan_param_heatmap_figure_caption(best_eps, best_ms, best_sil),
        ha="center",
        fontsize=11,
        color="0.35",
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "dbscan_param_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Replot DBSCAN param heatmap from CSV")
    p.add_argument("csv_file", help="path to param_search_summary.csv")
    p.add_argument("--out", "-o", default=None, help="output directory (default: same as CSV)")
    args = p.parse_args()

    csv_path = os.path.abspath(args.csv_file)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"File not found: {csv_path}")

    plot_dbscan_heatmap_from_csv(csv_path, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
