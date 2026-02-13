# rocprof Basics

## What is rocprof?

`rocprof` is ROCm's command-line profiling tool that provides:
- Kernel execution statistics
- Hardware performance counter collection
- HSA API tracing
- Timeline traces for visualization

## Installation

rocprof is included with ROCm. Verify installation:

```bash
which rocprof
rocprof --version
```

## Basic Usage

### Kernel Statistics

```bash
# Basic stats - shows kernel names, time, and call count
rocprof --stats ./my_application
```

Output example:
```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"matmul_kernel",100,15234567,152345,45.2
"activation_kernel",100,8234567,82345,24.4
"softmax_kernel",100,10234567,102345,30.4
```

### Understanding the Output

| Column | Description |
|--------|-------------|
| Name | Kernel function name |
| Calls | Number of times kernel was executed |
| TotalDurationNs | Total time in nanoseconds |
| AverageNs | Average execution time |
| Percentage | Percentage of total GPU time |

## Command-Line Options

### Essential Options

```bash
# Basic statistics
rocprof --stats ./app

# Output to specific file
rocprof -o results.csv ./app

# Collect hardware counters
rocprof --input counters.txt -o results.csv ./app

# HSA API trace
rocprof --hsa-trace ./app

# Hip API trace  
rocprof --hip-trace ./app

# Combined trace (recommended for visualization)
rocprof --hsa-trace --hip-trace -o trace.json ./app
```

### Useful Flags

```bash
# Timestamp all output
rocprof --timestamp on ./app

# Flush output frequently (for long runs)
rocprof --flush-rate 10 ./app

# Verbose mode
rocprof -v ./app

# Specify output directory
rocprof -d ./profile_output ./app
```

## Profiling Workflow

### Step 1: Quick Overview

```bash
rocprof --stats ./my_app > stats.txt
cat stats.txt
```

### Step 2: Identify Hot Kernels

Top kernels by time:
```bash
rocprof --stats ./my_app 2>/dev/null | \
  tail -n +2 | \
  sort -t',' -k4 -rn | \
  head -5
```

### Step 3: Detailed Analysis

Create counter input file:
```bash
# counters.txt
pmc: GRBM_COUNT, GRBM_GUI_ACTIVE
pmc: SQ_WAVES, SQ_INSTS_VALU
```

Run with counters:
```bash
rocprof --input counters.txt -o detailed.csv ./my_app
```

## Example Session

```bash
# 1. Build with debug info (optional but helpful)
hipcc -g -O3 -o my_kernel my_kernel.hip

# 2. Get basic stats
$ rocprof --stats ./my_kernel
Application: ./my_kernel
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"vecAdd",1000,5234567,5234,100.0

# 3. Collect detailed counters
$ cat > counters.txt << EOF
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: FETCH_SIZE
EOF

$ rocprof --input counters.txt -o results.csv ./my_kernel

# 4. Analyze results
$ cat results.csv
```

## Output Formats

### CSV (Default)
Best for programmatic analysis:
```csv
"Index","KernelName","gpu-id","queue-id","queue-index","pid","tid","grd","wgr","lds","scr","vgpr","sgpr","fbar","sig","obj","DispatchNs","BeginNs","EndNs","CompleteNs"
0,"vecAdd",0,0,1,12345,12345,65536,256,0,0,16,8,0,"0x0","0x...",1234567890,1234567900,1234568000,1234568100
```

### JSON Trace
Best for visualization in Chrome tracing or Perfetto:
```json
{
  "traceEvents": [
    {"name":"vecAdd","cat":"Kernel","ph":"X","ts":1234567,"dur":100,"pid":1,"tid":1}
  ]
}
```

## Visualization

### Chrome Tracing

1. Generate JSON trace:
```bash
rocprof --hsa-trace --hip-trace -o trace.json ./my_app
```

2. Open `chrome://tracing` in Chrome
3. Load `trace.json`

### Perfetto

1. Generate trace:
```bash
rocprof --hsa-trace -o trace.json ./my_app
```

2. Open https://ui.perfetto.dev/
3. Load trace file

## Common Issues

### "No permission to access performance counters"

```bash
# Run with sudo (not recommended for production)
sudo rocprof --stats ./my_app

# Or add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### "ROCProfiler Error"

Check ROCm installation:
```bash
rocm-smi
hipconfig --full
```

### Output File Empty

Ensure application actually runs GPU kernels:
```bash
rocprof -v --stats ./my_app
```

## Best Practices

1. **Warm up** - Run a few iterations before profiling
2. **Multiple runs** - Average results for accuracy
3. **Minimize noise** - Close other GPU applications
4. **Use specific counters** - Don't collect everything at once
5. **Profile release builds** - Debug builds may differ significantly

## Next Steps

- Learn about [Hardware Counters](03-hardware-counters.md)
- Explore [Trace Analysis](04-trace-analysis.md)

---

## Quick Reference Card

```bash
# Basic stats
rocprof --stats ./app

# With counters
rocprof --input counters.txt -o out.csv ./app

# JSON trace
rocprof --hsa-trace --hip-trace -o trace.json ./app

# Verbose
rocprof -v --stats ./app

# Specific output directory
rocprof -d ./output --stats ./app
```
