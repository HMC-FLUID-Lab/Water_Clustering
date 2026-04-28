# Stage 3 — Clustering

Read order-parameter MAT files from `../data/order_params/`, scale features,
and run an unsupervised clustering algorithm to label every molecule as
**LFTS** or **DNLS** (or noise). Each run writes a directory under
`../results/clustering/<run_name>/` containing `cluster_labels.csv` plus
diagnostic plots.

## Methods

| Method           | When to use                                                  |
|------------------|--------------------------------------------------------------|
| `dbscan`         | Density-based; identifies noise + clusters.                  |
| `kmeans`         | Forces exactly *N* clusters; no noise label.                 |
| `gmm`            | Gaussian Mixture; closest match to the Tanaka two-state model. |
| `dbscan_gmm`     | Production: DBSCAN denoising → GMM. Used in the paper.       |
| `hdbscan`        | Adaptive density; no `eps` to tune.                          |
| `hdbscan_gmm`    | HDBSCAN denoising → GMM.                                     |

Optional `--umap` reduces features to a low-D embedding before clustering.

## Files

| File | Purpose |
|------|---------|
| `water_clustering.py`              | Main entry point: every method above. |
| `run_three_model_dbscan_gmm.py`    | Apply same DBSCAN→GMM to three water models. |
| `param_search.py`                  | DBSCAN `eps` / `min_samples` grid search → silhouette heatmap. |
| `plot_style.py`                    | Shared matplotlib style (16 pt Arial, 600 dpi). |
| `plot_umap_figure1.py`             | Standalone UMAP scatter for paper Figure 1. |
| `replot_from_cluster_csv.py`       | Re-render plots from an existing `cluster_labels.csv`. |
| `replot_heatmap.py`, `replot_param_heatmap.py` | Re-plot DBSCAN-search heatmaps. |
| `sfvs.py`                          | Structure-Factor Validation Score (spec: [`SFVS_metric.md`](SFVS_metric.md)). |
| `run_sfvs.py`                      | CLI wrapper around `sfvs.py`. |
| `test_sfvs.py`                     | Unit tests (`pytest` or `python test_sfvs.py`). |

## Usage

DBSCAN→GMM (production):

```bash
python water_clustering.py \
    --mat_file  ../data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta_file ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --n_runs 1 \
    --method dbscan_gmm \
    --eps 0.05 --min_samples 30 \
    --features zeta_all \
    --out_dir ../results/clustering/tip4p2005_T-20_dbscan_gmm
```

DBSCAN parameter sweep:

```bash
python param_search.py \
    -m ../data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    -z ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    -n 1 \
    -o ../results/clustering/param_search_results/tip4p2005_T-20
```

→ Next: [Stage 4 — Structure Factor](../4_structure_factor/README.md)
