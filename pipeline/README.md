# Pipeline — Orchestration

Top-level entry points for running the pipeline end-to-end or in batches.

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.sh`          | **Top-level driver** — runs stages 1 → 5 for one (model, temperature, run). Supports `--only` and `--skip`. |
| `auto_cluster_pipeline.py` | One condition: clustering + label conversion + per-cluster S(k). Self-contained — does not import from any stage directory. |
| `run_sk_from_batch.py`     | Post-process an `auto_cluster_pipeline` batch directory: turn every `label_*` column into per-cluster S(k) plots. |
| `run_batch.sh`             | Driver around `auto_cluster_pipeline.py` over a hard-coded job list. |
| `run_sk_batch.sh`          | Driver around `run_sk_from_batch.py` over every batch sub-folder. |

## Usage

End-to-end for a single condition:

```bash
bash run_pipeline.sh tip4p2005 T-20 Run01
bash run_pipeline.sh tip5p     T-10 Run01 --skip 2
bash run_pipeline.sh tip4p2005 T-20 Run01 --only 3
```

Single condition with the all-in-one Python wrapper:

```bash
python auto_cluster_pipeline.py \
    --mat-file  ../data/order_params/OrderParam_tip4p2005_T-20_Run01.mat \
    --zeta-file ../data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat \
    --dcd-file  ../data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd \
    --pdb-file  ../data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb \
    --method    dbscan_gmm \
    --output-dir ../results/clustering/batch/tip4p2005_T-20
```

Whole sweep:

```bash
bash run_batch.sh                 # do clustering + S(k) for every job
bash run_batch.sh --dry-run       # just print the commands
bash run_sk_batch.sh              # only the S(k) post-process
```

Both shells resolve paths relative to the repo root automatically.
