# Stage 4 — Structure Factor

Compute the Debye structure factor $S(k)$ per cluster (and across all atoms)
to verify that the clusters from Stage 3 correspond to physically distinct
ordered/disordered populations.

Two pipelines:

* **Per-cluster S(k)** — driven by Stage 3 cluster labels.
* **All-atoms S(k)** — independent of clustering; reproduces the Tanaka
  reference curves.

## Files

| File | Purpose |
|------|---------|
| `convert_cluster_labels.py`     | Reshape flat `cluster_labels.csv` → (frames × molecules) matrix CSV. |
| `structure_factor_bycluster.py` | Main entry: per-cluster S(k) from DCD + label matrix. |
| `sk_zeta_3d.py`                 | 3D S(k, ζ) surfaces (called by `structure_factor_bycluster.py`). |
| `plot_sk_multimodel.py`         | Multi-model panel plot of S(k) per cluster. |
| `plot_sk_multitemp.py`          | Multi-temperature panel plot of S(k). |
| `compute_structure_factor.py`         | All-atoms S(k) (Tanaka reference). |
| `compute_structure_factor_tanaka.py`  | Variant using Tanaka conditions only. |
| `batch_quick.sh`                | Quick all-atoms S(k) batch over a list of conditions. |

## Usage

Step 1 — reshape cluster labels:

```bash
python convert_cluster_labels.py \
    --input  ../results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels.csv \
    --output ../results/clustering/cluster_labels_matrices/cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv \
    --n-runs 1 --n-molecules 1024 --label-column label_dbscan_gmm
```

Step 2 — per-cluster S(k):

```bash
python structure_factor_bycluster.py \
    --dcd-file       ../data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file       ../data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --zeta-file      ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --cluster-labels ../results/clustering/cluster_labels_matrices/cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv \
    --cluster-only \
    --model-name tip4p2005 --temperature -20 \
    --output-dir ../results/structure_factor/tip4p2005_T-20_dbscan_gmm
```

→ Next: [Stage 5 — Paper Figures](../5_paper_figures/README.md)
