# Water ML-Clustering Pipeline

End-to-end pipeline for identifying **Locally Favored Tetrahedral Structures
(LFTS)** and **Disordered Normal-Liquid Structures (DNLS)** in water MD
simulations, following the Shi & Tanaka two-state framework
(*JACS* **142**, 2868 — 2020).

The custom Structure-Factor Validation Score is documented in
[`3_clustering/SFVS_metric.md`](3_clustering/SFVS_metric.md), next to its
implementation in `3_clustering/sfvs.py`.

---

## Pipeline (run in order)

```
  1_simulate           MD trajectories                        → data/simulations/*
  2_order_params       DCD → q, Q6, LSI, Sk, ζ                → data/order_params/*.mat
  3_clustering         MAT → cluster labels                   → results/clustering/<run>/
  4_structure_factor   labels + DCD → per-cluster S(k)        → results/structure_factor/<run>/
  5_paper_figures      composite figures used in the paper    → results/paper_figures/
```

The numbered prefix is the running order: stage *N* consumes the output of
stage *N − 1*. Each directory has its own `README.md` documenting the stage.

---

## Quickstart

Single condition, all five stages:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01
```

Skip the (expensive) MD step and start from existing DCDs:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --skip 2
```

Run only one stage:

```bash
bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --only 3
```

Batch every (model × temperature) at once:

```bash
bash pipeline/run_batch.sh        # clustering + S(k) sweep
bash pipeline/run_sk_batch.sh     # S(k) only, post-process existing batches
```

---

## Repository layout

```
.
├── README.md                  ← this file
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── 1_simulate/                ← OpenMM drivers + simulation engine
├── 2_order_params/            ← DCD → MAT (LSI, q, Sk, Q6, ζ)
├── 3_clustering/              ← clustering methods + SFVS metric
│   └── SFVS_metric.md         ← spec for sfvs.py
├── 4_structure_factor/        ← per-cluster S(k), label-matrix conversion
├── 5_paper_figures/           ← combined composite figures
│
├── pipeline/                  ← orchestration scripts
│   ├── run_pipeline.sh        ← top-level driver (stages 1 → 5)
│   ├── auto_cluster_pipeline.py
│   ├── run_sk_from_batch.py
│   ├── run_batch.sh           ← batch clustering + S(k)
│   └── run_sk_batch.sh        ← batch S(k) post-process
│
├── data/                      ← inputs (gitignored — large binaries)
│   ├── simulations/{tip4p2005, tip5p, swm4ndp}/
│   └── order_params/
│
├── results/                   ← outputs (gitignored)
│   ├── clustering/
│   ├── structure_factor/
│   └── paper_figures/
│
└── _archive/                  ← legacy code kept for reference (gitignored)
```

**Code is committed; data and results are not.** See [.gitignore](.gitignore).

---

## Stage cheat sheet

### Stage 1 — Simulate

```bash
python 1_simulate/runWater_tip4p2005.py
python 1_simulate/runWater_tip5p.py
# Drude-polarizable SWM4-NDP needs swm4ndp.xml in the run dir:
mkdir -p data/simulations/swm4ndp && cd data/simulations/swm4ndp
python ../../../1_simulate/runWater_swm4ndp_multitemp.py
```

### Stage 2 — Order parameters

```bash
# One condition
python 2_order_params/run_single_condition.py tip4p2005 T-20 Run01

# Every DCD for a model
python 2_order_params/compute_order_params.py --model tip4p2005
```

### Stage 3 — Clustering

```bash
python 3_clustering/water_clustering.py \
    --mat_file  data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta_file data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --n_runs 1 --method dbscan_gmm --eps 0.05 --min_samples 30 \
    --features zeta_all \
    --out_dir results/clustering/tip4p2005_T-20_dbscan_gmm
```

DBSCAN parameter sweep:

```bash
python 3_clustering/param_search.py \
    -m data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    -z data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    -n 1 \
    -o results/clustering/param_search_results/tip4p2005_T-20
```

### Stage 4 — Per-cluster structure factor

```bash
# 4a: reshape flat labels → (frames × molecules) matrix
python 4_structure_factor/convert_cluster_labels.py \
    --input  results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels.csv \
    --output results/clustering/cluster_labels_matrices/cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv \
    --n-runs 1 --n-molecules 1024 --label-column label_dbscan_gmm

# 4b: compute per-cluster S(k) from DCD + label matrix
python 4_structure_factor/structure_factor_bycluster.py \
    --dcd-file       data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file       data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --zeta-file      data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --cluster-labels results/clustering/cluster_labels_matrices/cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv \
    --cluster-only --model-name tip4p2005 --temperature -20 \
    --output-dir results/structure_factor/tip4p2005_T-20_dbscan_gmm
```

### Stage 5 — Paper figures

```bash
python 5_paper_figures/generate_paper_figures.py
python 5_paper_figures/generate_paper_figures.py --sections c3 c4   # subset
```

---

## Dependencies

Python 3.11+. Install:

```
numpy scipy pandas scikit-learn matplotlib seaborn tqdm joblib
mdtraj openmm
umap-learn  # optional — only used by --umap flag in 3_clustering
```

`pip install -r requirements.txt` if you want to use the included list.

---

## License

Released under the [MIT License](LICENSE) — see the `LICENSE` file for full text.
