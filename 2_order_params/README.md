# Stage 2 — Order Parameters

Read DCD trajectories from `../data/simulations/` and compute structural order
parameters per molecule per frame. Two MAT files are written per condition to
`../data/order_params/`:

| File                              | Contents |
|-----------------------------------|----------|
| `OrderParam_<model>_T<temp>_<run>.mat`     | `q`, `Q6`, `LSI`, `Sk`, `d5`, `r`, `g_r` |
| `OrderParamZeta_<model>_T<temp>_<run>.mat` | `zeta_all` (Russo–Tanaka ζ)              |

## Files

| File | Purpose |
|------|---------|
| `compute_order_params.py`  | Batch processor over every DCD for one or more models. |
| `run_single_condition.py`  | Process ONE (model, temperature, run) tuple. |

## Usage

```bash
# Single condition
python run_single_condition.py tip4p2005 T-20 Run01
python run_single_condition.py tip5p     T-10 Run01
python run_single_condition.py swm4ndp   T-20 Run01

# Whole-model batch
python compute_order_params.py --model tip4p2005
python compute_order_params.py --model all --dry-run
```

Override the output directory:

```bash
ORDER_PARAM_OUT_DIR=/some/where python run_single_condition.py tip5p T-20 Run01
```

→ Next: [Stage 3 — Clustering](../3_clustering/README.md)
