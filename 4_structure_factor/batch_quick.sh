# ============================================================================
# QUICK CONFIGURATION - EDIT HERE
# ============================================================================


RUNS=(
    "tip5p -20 Run01"
    "tip5p -14 Run01"
    "tip5p -10 Run01"
    "tip5p -08 Run01"
)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/simulations"
PYTHON_SCRIPT="${SCRIPT_DIR}/compute_structure_factor_tanaka.py"

echo "========================================================================"
echo "QUICK BATCH STRUCTURE FACTOR COMPUTATION"
echo "========================================================================"
echo "Processing ${#RUNS[@]} condition(s)"
if [ "$QUICK_MODE" = true ]; then
    echo "Mode: QUICK TEST ($N_FRAMES frames)"
else
    echo "Mode: FULL ANALYSIS (all frames)"
fi
echo "========================================================================"

TOTAL=${#RUNS[@]}
COUNT=0
SUCCESS=0
FAIL=0

for RUN_CONFIG in "${RUNS[@]}"; do
    COUNT=$((COUNT + 1))
    
    # Parse configuration: "model temperature run_number"
    read -r MODEL TEMP RUN <<< "$RUN_CONFIG"
    
    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    echo "[$COUNT/$TOTAL] Processing: $MODEL at T=${TEMP}°C, $RUN"
    echo "──────────────────────────────────────────────────────────────────"
    
    OUTPUT_DIR="structure_factor_results/${MODEL}_T${TEMP}_${RUN}"
    
    # Build command
    CMD="python3 $PYTHON_SCRIPT --data-dir $DATA_DIR --model $MODEL --temperature $TEMP --run-number $RUN --output-dir $OUTPUT_DIR"
    
    if [ "$QUICK_MODE" = true ]; then
        CMD="$CMD --n-frames $N_FRAMES"
    fi
    
    # Run computation
    eval $CMD
    
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ Success"
    else
        FAIL=$((FAIL + 1))
        echo "✗ Failed"
    fi
done

echo ""
echo "========================================================================"
echo "QUICK BATCH COMPLETE!"
echo "========================================================================"
echo "Successful: $SUCCESS/$TOTAL"
echo "Failed: $FAIL/$TOTAL"
echo "Results in: structure_factor_results/"
echo "========================================================================"
