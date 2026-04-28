# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Resolve paths relative to repo root so the script is portable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIPELINE="$SCRIPT_DIR/auto_cluster_pipeline.py"
OUTPUT_BASE="$REPO_ROOT/results/clustering/batch"

# Clustering settings (applied to every job)
METHOD="all"                     # gmm | dbscan | kmeans | dbscan_gmm | hdbscan | hdbscan_gmm | all
N_CLUSTERS=2
EPS=0.05                         # DBSCAN eps
MIN_SAMPLES=15                   # DBSCAN min_samples
HDBSCAN_MIN_CLUSTER_SIZE=10       # HDBSCAN min group size
CONFIDENCE=""                    # GMM confidence threshold, e.g. 0.8  (leave empty to disable)
CLUSTER_FRAMES="5"             # frames used for clustering: "1" = 1024 pts, "" = all frames (1M pts)

# Structure factor settings
SKIP_SK=false                    # set to true to skip Step 3 (much faster)
N_FRAMES=""                      # limit S(k) to first N frames, e.g. 50  (leave empty = all)
K_MAX=50.0
K_POINTS=500
RC_CUTOFF=1.5

# ── JOB LIST ──────────────────────────────────────────────────────────────────
# Format per line:  "mat_file | zeta_file | dcd_file | pdb_file"
# Lines starting with # are skipped.

PARAM_DIR="$REPO_ROOT/data/order_params"
TIP4P_DIR="$REPO_ROOT/data/simulations/tip4p2005"
TIP5P_DIR="$REPO_ROOT/data/simulations/tip5p"

JOBS=(
    # ── TIP4P/2005 ─────────────────────────────────────────────────────────
    "$PARAM_DIR/OrderParam_tip4p2005_T-10_Run01.mat | $PARAM_DIR/OrderParamZeta_tip4p2005_T-10_Run01.mat | $TIP4P_DIR/dcd_tip4p2005_T-10_N1024_Run01_0.dcd | $TIP4P_DIR/inistate_tip4p2005_T-10_N1024_Run01.pdb"
    "$PARAM_DIR/OrderParam_tip4p2005_T-20_Run01.mat | $PARAM_DIR/OrderParamZeta_tip4p2005_T-20_Run01.mat | $TIP4P_DIR/dcd_tip4p2005_T-20_N1024_Run01_0.dcd | $TIP4P_DIR/inistate_tip4p2005_T-20_N1024_Run01.pdb"
    "$PARAM_DIR/OrderParam_tip4p2005_T-30_Run01.mat | $PARAM_DIR/OrderParamZeta_tip4p2005_T-30_Run01.mat | $TIP4P_DIR/dcd_tip4p2005_T-30_N1024_Run01_0.dcd | $TIP4P_DIR/inistate_tip4p2005_T-30_N1024_Run01.pdb"

    # ── TIP5P ───────────────────────────────────────────────────────────────
    "$PARAM_DIR/OrderParam_tip5p_T-10_Run01.mat | $PARAM_DIR/OrderParamZeta_tip5p_T-10_Run01.mat | $TIP5P_DIR/dcd_tip5p_T-10_N1024_Run01_0.dcd | $TIP5P_DIR/inistate_tip5p_T-10_N1024_Run01.pdb"
    "$PARAM_DIR/OrderParam_tip5p_T-15_Run01.mat | $PARAM_DIR/OrderParamZeta_tip5p_T-15_Run01.mat | $TIP5P_DIR/dcd_tip5p_T-15_N1024_Run01_0.dcd | $TIP5P_DIR/inistate_tip5p_T-15_N1024_Run01.pdb"
    "$PARAM_DIR/OrderParam_tip5p_T-20_Run01.mat | $PARAM_DIR/OrderParamZeta_tip5p_T-20_Run01.mat | $TIP5P_DIR/dcd_tip5p_T-20_N1024_Run01_0.dcd | $TIP5P_DIR/inistate_tip5p_T-20_N1024_Run01.pdb"
    "$PARAM_DIR/OrderParam_tip5p_T-25_Run01.mat | $PARAM_DIR/OrderParamZeta_tip5p_T-25_Run01.mat | $TIP5P_DIR/dcd_tip5p_T-25_N1024_Run01_0.dcd | $TIP5P_DIR/inistate_tip5p_T-25_N1024_Run01.pdb"

    # ── Add more jobs here in the same format ───────────────────────────────
    # "$PARAM_DIR/OrderParam_tip5p_T-18_Run01.mat | ... | ... | ..."
)

# ── INTERNALS (no need to edit below) ─────────────────────────────────────────

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

TOTAL=${#JOBS[@]}
PASS=0
FAIL=0
SKIP=0
FAILED_JOBS=()

print_separator() { printf '%0.s─' {1..65}; echo; }

echo "============================================================="
echo " BATCH CLUSTERING PIPELINE"
echo "============================================================="
echo " Method         : $METHOD"
echo " Jobs           : $TOTAL"
echo " Output base    : $OUTPUT_BASE"
echo " Skip S(k)      : $SKIP_SK"
[[ "$DRY_RUN" == true ]] && echo " *** DRY RUN — commands will be printed, not executed ***"
echo "============================================================="

mkdir -p "$OUTPUT_BASE"

for i in "${!JOBS[@]}"; do
    JOB="${JOBS[$i]}"

    # Skip comment lines
    [[ "$JOB" =~ ^[[:space:]]*# ]] && { (( SKIP++ )); continue; }
    [[ -z "${JOB// }" ]]           && { (( SKIP++ )); continue; }

    # Parse the four fields
    IFS='|' read -r MAT ZETA DCD PDB <<< "$JOB"
    MAT="${MAT// /}"
    ZETA="${ZETA// /}"
    DCD="${DCD// /}"
    PDB="${PDB// /}"

    JOB_NUM=$((i + 1))
    print_separator
    echo " Job $JOB_NUM / $TOTAL"
    echo "   mat  : $MAT"
    echo "   zeta : $ZETA"
    echo "   dcd  : $DCD"
    echo "   pdb  : $PDB"

    # Check all files exist before running
    MISSING=false
    for F in "$MAT" "$ZETA" "$DCD" "$PDB"; do
        if [[ ! -f "$F" ]]; then
            echo "   [SKIP] File not found: $F"
            MISSING=true
        fi
    done
    if [[ "$MISSING" == true ]]; then
        (( SKIP++ ))
        continue
    fi

    # Build output dir from mat filename
    BASENAME=$(basename "$MAT" .mat)
    BASENAME="${BASENAME#OrderParam_}"      # strip "OrderParam_" prefix
    OUT_DIR="$OUTPUT_BASE/${BASENAME}_${METHOD}"

    # Build the command
    CMD=(python "$PIPELINE"
        --mat-file  "$MAT"
        --zeta-file "$ZETA"
        --dcd-file  "$DCD"
        --pdb-file  "$PDB"
        --method    "$METHOD"
        --n-clusters "$N_CLUSTERS"
        --eps        "$EPS"
        --min-samples "$MIN_SAMPLES"
        --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE"
        --k-max      "$K_MAX"
        --k-points   "$K_POINTS"
        --rc-cutoff  "$RC_CUTOFF"
        --output-dir "$OUT_DIR"
    )

    # Optional flags
    [[ -n "$CONFIDENCE"     ]] && CMD+=(--confidence      "$CONFIDENCE")
    [[ -n "$N_FRAMES"       ]] && CMD+=(--n-frames        "$N_FRAMES")
    [[ -n "$CLUSTER_FRAMES" ]] && CMD+=(--cluster-frames  "$CLUSTER_FRAMES")
    [[ "$SKIP_SK" == true ]] && CMD+=(--skip-structure-factor)

    if [[ "$DRY_RUN" == true ]]; then
        echo "   [DRY RUN] ${CMD[*]}"
        (( PASS++ ))
        continue
    fi

    echo "   Output → $OUT_DIR"
    echo "   Running …"
    START=$(date +%s)

    # Run and tee output to a log file
    LOG="$OUT_DIR/pipeline.log"
    mkdir -p "$OUT_DIR"
    "${CMD[@]}" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    END=$(date +%s)
    ELAPSED=$((END - START))

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "   [DONE] Job $JOB_NUM finished in ${ELAPSED}s"
        (( PASS++ ))
    else
        echo "   [FAILED] Job $JOB_NUM — exit code $EXIT_CODE — log: $LOG"
        (( FAIL++ ))
        FAILED_JOBS+=("Job $JOB_NUM: $MAT")
    fi
done

# ── SUMMARY ───────────────────────────────────────────────────────────────────
echo
echo "============================================================="
echo " BATCH COMPLETE"
echo "============================================================="
echo "  Passed  : $PASS"
echo "  Failed  : $FAIL"
echo "  Skipped : $SKIP"
echo "  Results : $OUTPUT_BASE"

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo
    echo "  Failed jobs:"
    for J in "${FAILED_JOBS[@]}"; do
        echo "    ✗ $J"
    done
fi
echo "============================================================="
