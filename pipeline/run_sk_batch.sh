#!/usr/bin/env bash
# =============================================================================
# run_sk_batch.sh
# ===============
# Batch S(k) pipeline: process every clustering result folder and compute
# per-cluster structure factor for all methods (gmm, dbscan_gmm, hdbscan_gmm, …)
#
# Usage:
#   bash run_sk_batch.sh              # run all
#   bash run_sk_batch.sh --dry-run    # print commands without executing
# =============================================================================

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Resolve paths relative to repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIPELINE="$SCRIPT_DIR/run_sk_from_batch.py"

# Folder that contains all the clustering sub-folders (e.g. tip4p2005_T-20_Run01_all)
BATCH_DIR="$REPO_ROOT/results/clustering/batch"

# Simulation data directories
TIP4P_DIR="$REPO_ROOT/data/simulations/tip4p2005"
TIP5P_DIR="$REPO_ROOT/data/simulations/tip5p"

# Order parameter zeta directory (set to "" to skip HTML output)
ZETA_DIR="$REPO_ROOT/data/order_params"

# Which clustering methods to compute S(k) for.
# Leave empty ("") to process ALL label_* columns found in cluster_labels.csv.
METHODS="gmm dbscan_gmm hdbscan_gmm"   # e.g. "gmm dbscan_gmm"   or ""

# S(k) settings
N_MOLECULES=1024
N_FRAMES=""        # limit frames per job, e.g. "50"  (leave empty = all)
K_MAX=50.0
K_POINTS=500
RC_CUTOFF=1.5

# ── INTERNALS ─────────────────────────────────────────────────────────────────

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

PASS=0; FAIL=0; SKIP=0
FAILED_JOBS=()

print_sep() { printf '%0.s─' {1..65}; echo; }

echo "============================================================="
echo " BATCH S(k) PIPELINE"
echo "============================================================="
echo " Batch dir  : $BATCH_DIR"
echo " Methods    : ${METHODS:-<all>}"
echo " N frames   : ${N_FRAMES:-all}"
[[ "$DRY_RUN" == true ]] && echo " *** DRY RUN ***"
echo "============================================================="

# Collect all sub-folders
mapfile -t RESULT_DIRS < <(find "$BATCH_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL=${#RESULT_DIRS[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: no sub-folders found in $BATCH_DIR"
    exit 1
fi

echo " Jobs found : $TOTAL"
echo "============================================================="

for i in "${!RESULT_DIRS[@]}"; do
    RESULT_DIR="${RESULT_DIRS[$i]}"
    FOLDER=$(basename "$RESULT_DIR")
    JOB_NUM=$((i + 1))

    print_sep
    echo " Job $JOB_NUM / $TOTAL  →  $FOLDER"

    # ── Check cluster_labels.csv exists ───────────────────────────────────────
    if [[ ! -f "$RESULT_DIR/cluster_labels.csv" ]]; then
        echo "  [SKIP] cluster_labels.csv not found"
        (( SKIP++ ))
        continue
    fi

    # ── Auto-detect model, temperature, run from folder name ──────────────────
    MODEL=""
    TEMP=""
    RUN="Run01"

    [[ "$FOLDER" =~ (tip[45]p[0-9a-zA-Z]*) ]] && MODEL="${BASH_REMATCH[1]}"
    [[ "$FOLDER" =~ _(T-?[0-9]+)(_|$)      ]] && TEMP="${BASH_REMATCH[1]}"
    [[ "$FOLDER" =~ _(Run[0-9]+)            ]] && RUN="${BASH_REMATCH[1]}"

    if [[ -z "$MODEL" || -z "$TEMP" ]]; then
        echo "  [SKIP] Cannot parse model/temperature from folder name: $FOLDER"
        (( SKIP++ ))
        continue
    fi

    # ── Locate DCD and PDB files ───────────────────────────────────────────────
    if [[ "$MODEL" == *"tip4p"* ]]; then
        SIM_DIR="$TIP4P_DIR"
    else
        SIM_DIR="$TIP5P_DIR"
    fi

    DCD_FILE="$SIM_DIR/dcd_${MODEL}_${TEMP}_N${N_MOLECULES}_${RUN}_0.dcd"
    PDB_FILE="$SIM_DIR/inistate_${MODEL}_${TEMP}_N${N_MOLECULES}_${RUN}.pdb"

    # ── Locate zeta file (optional — enables HTML output) ─────────────────────
    ZETA_FILE=""
    if [[ -n "$ZETA_DIR" ]]; then
        ZETA_CANDIDATE="$ZETA_DIR/OrderParamZeta_${MODEL}_${TEMP}_${RUN}.mat"
        [[ -f "$ZETA_CANDIDATE" ]] && ZETA_FILE="$ZETA_CANDIDATE"
    fi

    echo "  model  : $MODEL   temp : $TEMP   run : $RUN"
    echo "  DCD    : $DCD_FILE"
    echo "  PDB    : $PDB_FILE"
    echo "  Zeta   : ${ZETA_FILE:-<not found — HTML skipped>}"

    # ── Validate files ────────────────────────────────────────────────────────
    MISSING=false
    [[ ! -f "$DCD_FILE" ]] && { echo "  [SKIP] DCD not found"; MISSING=true; }
    [[ ! -f "$PDB_FILE" ]] && { echo "  [SKIP] PDB not found"; MISSING=true; }
    if [[ "$MISSING" == true ]]; then
        (( SKIP++ ))
        continue
    fi

    # ── Build command ─────────────────────────────────────────────────────────
    CMD=(python "$PIPELINE"
        --result-dir   "$RESULT_DIR"
        --dcd-file     "$DCD_FILE"
        --pdb-file     "$PDB_FILE"
        --n-molecules  "$N_MOLECULES"
        --k-max        "$K_MAX"
        --k-points     "$K_POINTS"
        --rc-cutoff    "$RC_CUTOFF"
    )

    # Optional flags
    if [[ -n "$METHODS" ]]; then
        # shellcheck disable=SC2206
        CMD+=(--methods $METHODS)
    fi
    [[ -n "$N_FRAMES"   ]] && CMD+=(--n-frames  "$N_FRAMES")
    [[ -n "$ZETA_FILE"  ]] && CMD+=(--zeta-file "$ZETA_FILE")

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] ${CMD[*]}"
        (( PASS++ ))
        continue
    fi

    # ── Run ───────────────────────────────────────────────────────────────────
    LOG="$RESULT_DIR/sk_pipeline.log"
    echo "  Log    : $LOG"
    START=$(date +%s)

    "${CMD[@]}" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    END=$(date +%s)
    ELAPSED=$((END - START))

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "  [DONE] ${ELAPSED}s"
        (( PASS++ ))
    else
        echo "  [FAILED] exit=$EXIT_CODE  log=$LOG"
        (( FAIL++ ))
        FAILED_JOBS+=("$FOLDER")
    fi
done

# ── SUMMARY ───────────────────────────────────────────────────────────────────
echo
echo "============================================================="
echo " S(k) BATCH COMPLETE"
echo "============================================================="
printf "  Passed  : %d\n  Failed  : %d\n  Skipped : %d\n" $PASS $FAIL $SKIP

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo "  Failed jobs:"
    for J in "${FAILED_JOBS[@]}"; do echo "    ✗ $J"; done
fi
echo "============================================================="
