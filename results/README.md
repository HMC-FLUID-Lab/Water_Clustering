# Results (gitignored)

Generated outputs from stages 3, 4, 5. Layout:

```
results/
├── clustering/                          ← Stage 3 outputs
│   ├── <run_name>/                      ← one per (model, temp, method)
│   │   ├── cluster_labels.csv           ← flat CSV: rows = molecules
│   │   ├── *_scatter.png
│   │   ├── *_pairplot.png
│   │   └── *_zeta_distribution.png
│   ├── cluster_labels_matrices/         ← reshaped (frames × molecules) CSVs
│   └── param_search_results/            ← param sweeps (Stage 3.5)
│
├── structure_factor/                    ← Stage 4 outputs
│   ├── <run_name>/                      ← per-cluster S(k) plots
│   ├── per_cluster_3d/                  ← 3D S(k, ζ) renders
│   ├── multimodel/                      ← cross-model comparisons
│   ├── multitemp/                       ← cross-temperature comparisons
│   └── all_atoms/                       ← Tanaka-style reference S(k)
│
└── paper_figures/                       ← Stage 5 outputs
    ├── intermediate/                    ← in-progress drafts
    └── final/                           ← figures for the paper
```

Everything in this tree is regenerable by re-running the pipeline.
