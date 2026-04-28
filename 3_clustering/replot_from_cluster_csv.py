#!/usr/bin/env python3
"""
Regenerate clustering figures from an existing cluster_labels.csv
(no re-clustering — uses saved labels + raw features in the CSV).

Distribution figures (*_all_distributions*, *_zeta_distribution*) use this
module's own plotting: **raw x**; **y** is probability density (PDF, area = 1)
**rescaled per panel to [0, 1]** so panels are visually comparable. White
background. ``plot_style.set_default_plot()``
is applied. Other figures call ``water_clustering`` unchanged.

Example
-------
  python 3_clustering/replot_from_cluster_csv.py \\
      results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels.csv

  python 3_clustering/replot_from_cluster_csv.py path/to/cluster_labels.csv \\
      --label-column label_dbscan_gmm \\
      --out-dir ./my_figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Same thread env as water_clustering
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

from plot_style import set_default_plot

from water_clustering import (
    scale_features,
    plot_scatter,
    plot_pairplot,
    plot_umap_embedding,
    run_umap,
    HAS_UMAP,
    _palette_for_method,
    _cluster_display_name,
)

# ── Replot-only: axis labels (x = raw data) ─────────────────────────────────
FEATURE_AXIS = {
    "q_all": r"$q$",
    "Q6_all": r"$Q_6$",
    "LSI_all": r"LSI",
    "Sk_all": r"$S_k$",
    "zeta_all": r"$\zeta$ (Å)",
}

# Per-panel: y rescaled to [0, 1]; underlying PDF has area = 1.
DENSITY_YLABEL = "Density"
EXCLUDE_Q6 = frozenset({"Q6_all"})

def _feature_list(columns, exclude_q6: bool) -> list[str]:
    cols = [c for c in columns if not str(c).startswith("label_")]
    if exclude_q6:
        cols = [c for c in cols if c not in EXCLUDE_Q6]
    return cols


def _normalize_y_0_to_1(ax) -> None:
    """
    After ``sns.histplot(..., stat='density', kde=True)``, map all y values
    in this panel to [0, 1]. Underlying PDF has area = 1; display is rescaled
    for comparable axes across panels.
    """
    ys: list[float] = []
    for p in ax.patches:
        h = p.get_height()
        if h is not None and np.isfinite(h):
            ys.append(float(h))
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        ys.extend(float(v) for v in y[np.isfinite(y)])
    if not ys:
        ax.set_ylim(0, 1)
        return
    ymin, ymax = min(ys), max(ys)
    if ymax <= ymin:
        for p in ax.patches:
            if (h := p.get_height()) is not None and np.isfinite(h):
                p.set_height(0.5)
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            line.set_ydata(np.where(np.isfinite(y), 0.5, np.nan))
        ax.set_ylim(0, 1)
        return
    scale = 1.0 / (ymax - ymin)
    for p in ax.patches:
        h = p.get_height()
        if h is not None and np.isfinite(h):
            p.set_height((float(h) - ymin) * scale)
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        line.set_ydata((y - ymin) * scale)
    ax.set_ylim(0, 1)


def _apply_density_plot_style(ax) -> None:
    """White background, no grid."""
    ax.set_facecolor("white")


def plot_zeta_distribution_replot(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    out_dir: str,
) -> None:
    if "zeta_all" not in df_raw.columns:
        print("  [replot] Skipping zeta distribution (no zeta_all)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    unique = sorted(set(labels))
    palette = _palette_for_method(method, unique)
    for lbl, col in zip(unique, palette):
        mask = labels == lbl
        leg = f"{_cluster_display_name(lbl, method)} (n={np.sum(mask):,})"
        sns.histplot(
            df_raw.loc[mask, "zeta_all"],
            ax=ax,
            kde=True,
            color=col,
            alpha=0.5,
            label=leg,
            stat="density",
            edgecolor="black",
            linewidth=0.3,
        )
    _normalize_y_0_to_1(ax)
    _apply_density_plot_style(ax)
    ax.set_xlabel(FEATURE_AXIS.get("zeta_all", r"$\zeta$ (Å)"))
    ax.set_ylabel(DENSITY_YLABEL)
    ax.set_title(
        f"{method.upper()} — ζ distribution per cluster\n"
        "(bimodal separation validates LFTS / DNLS assignment)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{method}_zeta_distribution.png"))
    plt.close(fig)


def plot_all_distributions_replot(
    df_raw: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    out_dir: str,
    *,
    exclude_q6: bool = True,
) -> None:
    """
    Raw x per feature; PDF (area=1) rescaled per panel to [0, 1].
    Default: omit Q6 → 2×2 grid with q, LSI, Sk, ζ.
    """
    features = _feature_list(df_raw.columns, exclude_q6=exclude_q6)
    n_features = len(features)
    if n_features == 0:
        print("  [replot] Skipping all_distributions (no features)")
        return

    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes_arr = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), layout="constrained"
    )
    axes = np.atleast_1d(axes_arr).ravel()

    unique = sorted(set(labels))
    palette = _palette_for_method(method, unique)

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for lbl, col_c in zip(unique, palette):
            mask = labels == lbl
            name = _cluster_display_name(lbl, method)
            sns.histplot(
                df_raw.loc[mask, feat],
                ax=ax,
                color=col_c,
                alpha=0.5,
                kde=True,
                label=name,
                stat="density",
                bins=40,
                edgecolor="black",
                linewidth=0.3,
            )
        _normalize_y_0_to_1(ax)
        _apply_density_plot_style(ax)
        ax.set_title(FEATURE_AXIS.get(feat, feat))
        ax.set_xlabel(FEATURE_AXIS.get(feat, feat))
        ax.set_ylabel(DENSITY_YLABEL if idx % n_cols == 0 else "")
        ax.tick_params(axis="y", labelleft=(idx % n_cols == 0))

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    axes[0].legend()
    note = "Q6 omitted; " if exclude_q6 else ""
    fig.suptitle(
        f"{method.upper()} — Distribution of raw order parameters\n"
        f"({note}PDF, area=1; y rescaled per panel to 0–1)",
    )
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
    fig.savefig(os.path.join(out_dir, f"{method}_all_distributions.png"))
    plt.close(fig)


def _label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_")]


def _method_from_label_column(col: str) -> str:
    if col.startswith("label_"):
        return col[len("label_") :]
    return col


def load_csv_for_plots(
    csv_path: str,
    label_column: str | None,
    feature_columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]:
    """
    Returns (df_raw_features, df_scaled, labels, method_tag, out_dir_default).
    """
    df = pd.read_csv(csv_path)
    labels_avail = _label_columns(df)
    if not labels_avail:
        raise SystemExit(
            f"No column starting with 'label_' found in {csv_path}. "
            "Expected e.g. label_dbscan_gmm, label_gmm."
        )

    if label_column is None:
        label_column = labels_avail[0]
        if len(labels_avail) > 1:
            print(f"  Multiple label columns {labels_avail}; using '{label_column}'")
    elif label_column not in df.columns:
        raise SystemExit(f"Column '{label_column}' not in CSV. Available: {list(df.columns)}")

    if feature_columns is None:
        feature_columns = [c for c in df.columns if not c.startswith("label_")]
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise SystemExit(f"Unknown feature column(s): {missing}")

    if not feature_columns:
        raise SystemExit("No feature columns to plot.")

    df_raw = df[feature_columns].copy()
    labels = df[label_column].values
    if labels.dtype == object:
        labels = pd.to_numeric(labels, errors="coerce").fillna(-1).astype(int)
    else:
        labels = np.asarray(labels, dtype=int)

    df_scaled = scale_features(df_raw)
    method = _method_from_label_column(label_column)
    out_default = os.path.dirname(os.path.abspath(csv_path))
    return df_raw, df_scaled, labels, method, out_default


def main():
    p = argparse.ArgumentParser(
        description="Plot clustering figures from existing cluster_labels.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "csv_file",
        help="Path to cluster_labels.csv (order parameters + label_* column(s))",
    )
    p.add_argument(
        "--label-column",
        default=None,
        help="e.g. label_dbscan_gmm (default: first label_* column)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Figure output directory (default: same folder as the CSV)",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Subset of feature columns (default: all non-label columns)",
    )
    p.add_argument(
        "--include-q6",
        action="store_true",
        help="Include Q6 in replot (default: omit Q6, show q LSI Sk ζ)",
    )
    p.add_argument(
        "--umap",
        action="store_true",
        help="Also compute UMAP on scaled features and save umap_embedding plot",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (only with --umap)",
    )
    args = p.parse_args()

    csv_path = os.path.abspath(args.csv_file)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"File not found: {csv_path}")

    set_default_plot()

    df_raw, df_scaled, labels, method, out_default = load_csv_for_plots(
        csv_path, args.label_column, args.features
    )
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else out_default
    os.makedirs(out_dir, exist_ok=True)

    print(f"  CSV      : {csv_path}")
    print(f"  Rows     : {len(df_raw):,}")
    print(f"  Features : {list(df_raw.columns)}")
    print(f"  Labels   : {args.label_column or '(auto)'} → method tag '{method}'")
    print(f"  Out dir  : {out_dir}")
    print("\nGenerating plots …")

    plot_scatter(df_scaled, labels, method, out_dir, df_raw=df_raw)
    plot_zeta_distribution_replot(df_raw, labels, method, out_dir)
    plot_all_distributions_replot(
        df_raw, labels, method, out_dir, exclude_q6=not args.include_q6
    )
    df_plot = df_scaled.drop(columns=["Q6_all"], errors="ignore")
    plot_pairplot(df_plot, labels, method, out_dir)

    if args.umap:
        if not HAS_UMAP:
            print("  [SKIP] --umap requested but umap-learn is not installed.")
        else:
            df_umap = run_umap(
                df_scaled,
                n_components=2,
                n_neighbors=args.umap_n_neighbors,
            )
            plot_umap_embedding(df_umap, labels, df_raw, method, out_dir)

    print(f"\nDone. Figures saved under: {out_dir}")


if __name__ == "__main__":
    main()