# Data (gitignored)

Inputs to the pipeline. Directory structure is committed; **contents are
not** — they're large binary files (DCD trajectories, MAT order parameters)
that should live outside source control.

```
data/
├── simulations/        ← outputs of Stage 1 (DCD + PDB + CHK)
│   ├── tip4p2005/
│   ├── tip5p/
│   └── swm4ndp/
└── order_params/       ← outputs of Stage 2 (.mat)
```

Place trajectories from your own simulation runs (or copy from a shared
location) into `simulations/<model>/`. Stage 2 then writes
`OrderParam_*.mat` and `OrderParamZeta_*.mat` into `order_params/`.
