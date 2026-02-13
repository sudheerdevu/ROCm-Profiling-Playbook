#!/bin/bash
#
# Quick profile script for rocprof
# Usage: ./profile-quick.sh <application> [args...]
#

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <application> [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 ./my_kernel"
    echo "  $0 ./my_kernel --size 1024"
    exit 1
fi

APP="$1"
shift
ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$PROJECT_DIR/configs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "==================================="
echo "  Quick ROCm Profiling Session"
echo "==================================="
echo ""
echo "Application: $APP $ARGS"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Basic Statistics
echo "[1/4] Collecting basic statistics..."
rocprof --stats "$APP" $ARGS 2>&1 | tee "$OUTPUT_DIR/stats.txt"

# Step 2: Memory Counters
echo ""
echo "[2/4] Collecting memory counters..."
rocprof --input "$CONFIGS_DIR/counters-memory.txt" \
    -o "$OUTPUT_DIR/memory_counters.csv" \
    "$APP" $ARGS 2>/dev/null

# Step 3: Compute Counters
echo "[3/4] Collecting compute counters..."
rocprof --input "$CONFIGS_DIR/counters-compute.txt" \
    -o "$OUTPUT_DIR/compute_counters.csv" \
    "$APP" $ARGS 2>/dev/null

# Step 4: Timeline Trace
echo "[4/4] Generating timeline trace..."
rocprof --hsa-trace --hip-trace \
    -o "$OUTPUT_DIR/trace.json" \
    "$APP" $ARGS 2>/dev/null

echo ""
echo "==================================="
echo "  Profiling Complete!"
echo "==================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR"
echo ""
echo "Quick Analysis:"
echo "---------------"

# Parse and display key metrics
if [ -f "$OUTPUT_DIR/memory_counters.csv" ]; then
    echo ""
    echo "Memory Bandwidth:"
    python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/memory_counters.csv')
    if 'FETCH_SIZE' in df.columns and 'WRITE_SIZE' in df.columns:
        total_bytes = df['FETCH_SIZE'].sum() + df['WRITE_SIZE'].sum()
        total_time = df['EndNs'].max() - df['BeginNs'].min()
        bw = total_bytes / total_time
        print(f'  Total data moved: {total_bytes/1e9:.2f} GB')
        print(f'  Effective bandwidth: {bw:.2f} GB/s')
except Exception as e:
    print(f'  Could not analyze: {e}')
" 2>/dev/null || echo "  (Install pandas for analysis)"
fi

echo ""
echo "To view timeline trace:"
echo "  1. Open chrome://tracing in Chrome"
echo "  2. Load: $OUTPUT_DIR/trace.json"
echo ""
