# Stage 5 — Paper Figures

Generate the composite figures used in
[`../Paper_WaterMLClustering.md`](../Paper_WaterMLClustering.md). Reuses
plotting utilities from `3_clustering/` and `4_structure_factor/`, then
saves PNGs to `../results/paper_figures/`.

## Files

| File | Purpose |
|------|---------|
| `generate_paper_figures.py`  | Render every paper figure (sections C.2 — C.5). |
| `prepare_positive_temps.py`  | Build T = 0, +10, +20 °C results so the multi-temp figure can be regenerated. |

## Usage

```bash
# All sections
python generate_paper_figures.py

# Subset (no trajectory loading — fast)
python generate_paper_figures.py --sections c3 c4

# Force recompute even if cached NPZ exists
python generate_paper_figures.py --no-cache

# Quick preview (5 frames)
python generate_paper_figures.py --n-frames 5
```

`prepare_positive_temps.py` is run once per new positive-temperature condition
to produce the order parameters, cluster labels, and label matrices that
`generate_paper_figures.py` then consumes:

```bash
python prepare_positive_temps.py --temps 0 10 20
python prepare_positive_temps.py --temps 0 --skip-op   # already have order params
```
