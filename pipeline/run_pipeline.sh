#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# ===============
# End-to-end driver for the Water ML-clustering pipeline.
# Each stage lives in its own numbered directory; this script runs them in
# order for ONE (model, temperature, run) tuple.
#
# Pipeline:
#   1_simulate          DCD trajectories         (OpenMM)
#   2_order_params      DCD  →  OrderParam*.mat  (q, Q6, LSI, Sk, ζ)
#   3_clustering        MAT  →  cluster_labels.csv
#   4_structure_factor  labels + DCD  →  per-cluster S(k) plots
#   5_paper_figures     composite figures used in the paper
#
# Usage (run from anywhere):
#   bash pipeline/run_pipeline.sh                                # default: tip4p2005 T-20 Run01
#   bash pipeline/run_pipeline.sh tip5p T-10 Run01
#   bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --skip 1  # skip MD, start at OPs
#   bash pipeline/run_pipeline.sh tip4p2005 T-20 Run01 --only 3  # only run clustering
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:-tip4p2005}"
TEMP="${2:-T-20}"
RUN="${3:-Run01}"
shift $(( $# >= 3 ? 3 : $# ))

ONLY=""
SKIP_BEFORE=0
while (( $# )); do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --skip) SKIP_BEFORE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

want_stage() {
    local s=$1
    [[ -n "$ONLY"   && "$s" != "$ONLY" ]] && return 1
    [[ -z "$ONLY"   && "$s" -lt "$SKIP_BEFORE" ]] && return 1
    return 0
}

print_stage() {
    echo
    printf '═%.0s' {1..70}; echo
    echo " Stage $1 — $2"
    printf '═%.0s' {1..70}; echo
}

# ─── Stage 1: MD simulation ──────────────────────────────────────────────────
if want_stage 1; then
    print_stage 1 "MD simulation  (skip if DCDs already exist)"
    case "$MODEL" in
        tip4p2005) python 1_simulate/runWater_tip4p2005.py ;;
        tip5p)     python 1_simulate/runWater_tip5p.py ;;
        swm4ndp)
            (cd data/simulations/swm4ndp && \
             python "$SCRIPT_DIR/1_simulate/runWater_swm4ndp_multitemp.py")
            ;;
        *) echo "Unknown model: $MODEL"; exit 1 ;;
    esac
fi

# ─── Stage 2: order parameters ───────────────────────────────────────────────
if want_stage 2; then
    print_stage 2 "Order parameters  ($MODEL $TEMP $RUN)"
    python 2_order_params/run_single_condition.py "$MODEL" "$TEMP" "$RUN"
fi

# ─── Stage 3: clustering ─────────────────────────────────────────────────────
if want_stage 3; then
    print_stage 3 "Clustering  (DBSCAN → GMM)"
    OUT_CLUST="results/clustering/${MODEL}_${TEMP}_dbscan_gmm"
    python 3_clustering/water_clustering.py \
        --mat_file  "data/order_params/OrderParam_${MODEL}_${TEMP}_${RUN}.mat" \
        --zeta_file "data/order_params/OrderParamZeta_${MODEL}_${TEMP}_${RUN}.mat" \
        --n_runs 1 \
        --method dbscan_gmm \
        --eps 0.05 --min_samples 30 \
        --features zeta_all \
        --out_dir "$OUT_CLUST"
fi

# ─── Stage 4: structure factor ───────────────────────────────────────────────
if want_stage 4; then
    print_stage 4 "Per-cluster structure factor"
    OUT_CLUST="results/clustering/${MODEL}_${TEMP}_dbscan_gmm"
    MATRIX_DIR="results/clustering/cluster_labels_matrices"
    mkdir -p "$MATRIX_DIR"
    MATRIX_CSV="$MATRIX_DIR/cluster_labels_matrix_${MODEL}_${TEMP}_dbscan_gmm.csv"

    python 4_structure_factor/convert_cluster_labels.py \
        --input  "$OUT_CLUST/cluster_labels.csv" \
        --output "$MATRIX_CSV" \
        --n-runs 1 --n-molecules 1024 \
        --label-column label_dbscan_gmm

    python 4_structure_factor/structure_factor_bycluster.py \
        --dcd-file       "data/simulations/${MODEL}/dcd_${MODEL}_${TEMP}_N1024_${RUN}_0.dcd" \
        --pdb-file       "data/simulations/${MODEL}/inistate_${MODEL}_${TEMP}_N1024_${RUN}.pdb" \
        --zeta-file      "data/order_params/OrderParamZeta_${MODEL}_${TEMP}_${RUN}.mat" \
        --cluster-labels "$MATRIX_CSV" \
        --cluster-only \
        --model-name "$MODEL" \
        --temperature "${TEMP#T}" \
        --output-dir "results/structure_factor/${MODEL}_${TEMP}_dbscan_gmm"
fi

# ─── Stage 5: paper figures ──────────────────────────────────────────────────
if want_stage 5; then
    print_stage 5 "Paper figures"
    python 5_paper_figures/generate_paper_figures.py \
        --out-dir "results/paper_figures"
fi

echo
echo "Pipeline complete."
