"""
Publication-style matplotlib defaults (MATLAB defaultPlot–like).
Import and call set_default_plot() once before creating figures.
"""
import matplotlib.pyplot as plt


def set_default_plot():
    """Global plot parameters (font sizes, linewidths, Arial, bold, high-DPI export)."""
    plt.rcParams.update({
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "figure.titlesize": 16,
        "font.size": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "font.style": "normal",
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 100,
    })


# Column names in DataFrame → axis / title labels (publication math)
FEATURE_LABELS = {
    "q_all": r"$q$",
    "Q6_all": r"$Q_6$",
    "LSI_all": r"LSI",
    "Sk_all": r"$S_k$",
    "zeta_all": r"$\zeta$ (Å)",
}

# Features to omit from distribution / pair grids (omit Q6, keep LSI)
EXCLUDE_FROM_DISTRIBUTION_PLOTS = frozenset({"Q6_all"})


def feature_axis_label(column_name: str) -> str:
    """Human-readable math label for a raw feature column."""
    if column_name in FEATURE_LABELS:
        return FEATURE_LABELS[column_name]
    # Fallback: no snake_case / machine-style names on axes
    return str(column_name).replace("_", " ")


def distribution_feature_columns(df_columns) -> list:
    """Order-parameter columns for histogram grids, excluding Q6."""
    return [c for c in df_columns if c not in EXCLUDE_FROM_DISTRIBUTION_PLOTS]


def scale_density_plot_y(ax, factor: float = 100.0) -> None:
    """
    Multiply histogram bar heights and KDE line y-values by ``factor``
    so probability density reads on a 0–100-like scale (×100 convention).
    Call after sns.histplot(..., stat='density', kde=True).
    """
    import numpy as np

    for p in ax.patches:
        h = p.get_height()
        if h is not None and np.isfinite(h) and h > 0:
            p.set_height(h * factor)
    for line in ax.lines:
        y = line.get_ydata()
        line.set_ydata(y * factor)


def probability_density_ylabel() -> str:
    """Y-axis label after scale_density_plot_y(factor=100)."""
    return r"Probability density (×$100$)"


# ── DBSCAN parameter-search heatmap (param_search.py, replot_param_heatmap.py) ─

DBSCAN_HEATMAP_XLABEL = r"$\varepsilon$ (neighbourhood radius, scaled features)"
DBSCAN_HEATMAP_YLABEL = r"Minimum samples ($N_{\mathrm{min}}$)"
DBSCAN_HEATMAP_CBAR_LABEL = "Silhouette score"


def style_dbscan_param_heatmap_axes(ax, cbar):
    """Bold axis labels + colorbar; clears axes title."""
    ax.set_xlabel(DBSCAN_HEATMAP_XLABEL, fontsize=16, fontweight="bold")
    ax.set_ylabel(DBSCAN_HEATMAP_YLABEL, fontsize=16, fontweight="bold")
    ax.set_title("")
    cbar.set_label(DBSCAN_HEATMAP_CBAR_LABEL, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis="both", labelsize=14, direction="in", top=True, right=True)


def dbscan_param_heatmap_figure_caption(best_eps, best_ms, best_sil):
    """Caption below figure: colour key and optional best-parameter line."""
    base = "Colour = silhouette; cell labels = noise removed (%)."
    if best_eps is None:
        return base
    return (
        f"{base}  Best: ε = {best_eps:.3f}, min_samples = {best_ms}, "
        f"silhouette = {best_sil:.3f}"
    )
