# 05. Omniperf Guide

## What is Omniperf?

Omniperf is AMD's comprehensive profiling tool that provides:
- Automated bottleneck detection
- Roofline analysis
- Hardware counter collection
- Performance recommendations

## Installation

```bash
# Install via pip
pip install omniperf

# Or from ROCm repository
sudo apt install omniperf

# Verify installation
omniperf --version
```

## Basic Workflow

### 1. Profile Your Application

```bash
# Basic profiling
omniperf profile -n my_profile -- ./my_app

# With specific workload
omniperf profile -n bert_profile -- python run_bert.py --batch-size 32

# With custom kernel filter
omniperf profile -n specific_kernel \
    --kernel-name "matmul" \
    -- ./my_app
```

### 2. Analyze Results

```bash
# Interactive analysis
omniperf analyze -p my_profile/

# Generate report
omniperf analyze -p my_profile/ --report-format html -o report.html

# Compare two profiles
omniperf analyze -p baseline/ -p optimized/ --compare
```

## Profile Output Structure

```
my_profile/
├── rocprof_data/
│   ├── timestamps.csv
│   ├── pmc_perf.csv
│   └── SQ_*.csv
├── roofline/
│   └── roofline_data.json
├── config.yaml
└── analysis/
    └── report.json
```

## Understanding Omniperf Output

### Overview Panel

```
================================================================================
                              OMNIPERF RESULTS
================================================================================
Execution Time: 125.3 ms
GPU Utilization: 78.4%
Peak Bandwidth: 456.2 GB/s (72% of theoretical)
Arithmetic Intensity: 12.3 FLOPs/byte

Primary Bottleneck: Memory Bound (L2 Cache)
================================================================================
```

### Workload Characterization

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKLOAD ANALYSIS                        │
├─────────────────────────────────────────────────────────────┤
│ Category              │ Value       │ Assessment           │
├───────────────────────┼─────────────┼──────────────────────┤
│ Compute Utilization   │ 45.2%       │ ⚠️ Moderate          │
│ Memory Bandwidth      │ 78.9%       │ ✓ Good               │
│ L1 Cache Hit Rate     │ 92.3%       │ ✓ Excellent          │
│ L2 Cache Hit Rate     │ 67.8%       │ ⚠️ Needs improvement │
│ Occupancy            │ 62.5%       │ ✓ Good               │
└───────────────────────┴─────────────┴──────────────────────┘
```

## Key Metrics Explained

### Speed of Light (SOL)

```
SOL metrics show percentage of theoretical peak:

VALU SOL: 45%  → Using 45% of peak vector ALU throughput
MFMA SOL: 0%   → Matrix units not used
LDS SOL:  23%  → Using 23% of peak LDS bandwidth
L2 SOL:   78%  → Using 78% of peak L2 bandwidth
HBM SOL:  65%  → Using 65% of peak HBM bandwidth
```

### Roofline Model

```
                    Performance (TFLOPS)
                           │
         Compute Ceiling ──┤████████████████████████
                           │              ╱
                           │            ╱
                           │          ╱ Memory Ceiling
                           │        ╱
                      •────┼──────•──── Your Kernel
                           │    ╱
                           │  ╱
                           │╱
                           └──────────────────────────
                                Arithmetic Intensity
                                   (FLOPs/byte)
                                   
If kernel below memory ceiling: Memory-bound
If kernel below compute ceiling: Compute-bound
```

## Using Omniperf for Optimization

### Step 1: Identify Bottleneck

```bash
omniperf analyze -p my_profile/ --block-size 256
```

Look for:
- Highest SOL percentage (your limit)
- Lowest SOL percentage (underutilized)
- Stall reasons

### Step 2: Drill Down

```bash
# Focus on specific kernel
omniperf analyze -p my_profile/ --kernel-id 2

# Focus on specific metric
omniperf analyze -p my_profile/ --metric-group memory
```

### Step 3: Compare After Optimization

```bash
omniperf analyze \
    -p baseline_profile/ \
    -p optimized_profile/ \
    --compare \
    --report-type diff
```

## Common Bottleneck Recommendations

### Memory-Bound

```
Symptoms:
- High HBM SOL, low Compute SOL
- High memory stall cycles

Omniperf suggests:
✓ Increase arithmetic intensity
✓ Use shared memory for data reuse
✓ Coalesce memory accesses
✓ Consider compression/quantization
```

### Compute-Bound

```
Symptoms:
- High VALU SOL, lower memory SOL
- Minimal memory stalls

Omniperf suggests:
✓ Reduce instruction count
✓ Use faster intrinsics
✓ Consider matrix acceleration (MFMA)
✓ Optimize instruction mix
```

### Latency-Bound

```
Symptoms:
- Low SOL across the board
- High stall cycles
- Low occupancy

Omniperf suggests:
✓ Increase parallelism
✓ Reduce register pressure
✓ Check synchronization overhead
✓ Analyze divergent branches
```

## Advanced Features

### Custom Counter Collection

```bash
# Specify counter groups
omniperf profile -n detailed \
    --metrics SQ,TA,TD,TCP,TCC \
    -- ./my_app
```

### Kernel Filtering

```bash
# Profile specific kernels
omniperf profile -n matmul_only \
    --kernel-filter "name:matmul" \
    -- ./my_app

# Skip warmup iterations
omniperf profile -n skip_warmup \
    --kernel-skip 5 \
    -- ./my_app
```

### Dispatch Analysis

```bash
# Analyze individual kernel dispatches
omniperf analyze -p my_profile/ \
    --dispatch 1,2,5 \
    --detail
```

## Integration with Development Flow

```
Development Cycle with Omniperf:
┌─────────────────────────────────────────────┐
│ 1. Write kernel                             │
│         ↓                                   │
│ 2. Profile: omniperf profile                │
│         ↓                                   │
│ 3. Analyze: omniperf analyze                │
│         ↓                                   │
│ 4. Identify bottleneck                      │
│         ↓                                   │
│ 5. Apply optimization                       │
│         ↓                                   │
│ 6. Compare: omniperf analyze --compare      │
│         ↓                                   │
│ 7. If improved: commit. Else: goto 4        │
└─────────────────────────────────────────────┘
```

## Best Practices

1. **Profile release builds**: Debug builds have extra overhead
2. **Use representative data**: Profile with production-like inputs
3. **Warm up before profiling**: Let JIT compilation finish
4. **Profile individual kernels**: Isolate for clearer analysis
5. **Track metrics over time**: Build performance regression tests
6. **Trust the recommendations**: Omniperf's suggestions are data-driven
