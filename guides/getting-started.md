# Getting Started with ROCm Profiling

This guide walks you through the basics of profiling HIP applications using ROCm tools.

## Prerequisites

- ROCm 5.0 or later installed
- AMD GPU (MI series, Radeon RX 6000/7000 series)
- HIP application to profile

## Quick Start

### 1. Check ROCm Installation

```bash
# Verify ROCm is installed
rocm-smi --showallinfo

# Check rocprof is available
which rocprof
rocprof --version
```

### 2. Basic Profiling with rocprof

The simplest way to profile a HIP application:

```bash
# Basic profiling
rocprof ./my_hip_app

# This creates results.csv with kernel execution times
```

### 3. Collecting Performance Counters

To get detailed performance metrics:

```bash
# Create an input file listing desired metrics
cat > metrics.txt << EOF
pmc: GPUBusy
pmc: VALUInsts
pmc: VFetchInsts
pmc: VWriteInsts
pmc: VALUUtilization
pmc: VALUBusy
pmc: MemUnitBusy
pmc: WriteSize
pmc: FetchSize
EOF

# Run with metrics
rocprof -i metrics.txt -o detailed_results.csv ./my_hip_app
```

### 4. Tracing HIP API Calls

To trace HIP runtime calls:

```bash
# Enable HIP API tracing
rocprof --hip-trace ./my_hip_app

# This creates hip_api_trace.csv
```

### 5. Full System Trace

For comprehensive tracing including HSA:

```bash
# Full trace with all components
rocprof --hip-trace --hsa-trace --sys-trace ./my_hip_app
```

## Understanding Results

### Kernel Timing (results.csv)

| Column | Description |
|--------|-------------|
| `Name` | Kernel function name |
| `grd0/grd1/grd2` | Grid dimensions |
| `wgr0/wgr1/wgr2` | Workgroup (block) dimensions |
| `DurationNs` | Execution time in nanoseconds |
| `BeginNs` | Start timestamp |
| `EndNs` | End timestamp |

### Performance Counters

| Metric | Description | Target |
|--------|-------------|--------|
| `GPUBusy` | GPU utilization % | >90% |
| `VALUUtilization` | Vector ALU utilization | >80% |
| `MemUnitBusy` | Memory unit utilization | varies |
| `FetchSize` | Bytes read from memory | minimize |
| `WriteSize` | Bytes written to memory | minimize |

## Common Profiling Commands

```bash
# Profile specific kernel
rocprof --stats ./my_app

# Profile with timestamps
rocprof --timestamp on ./my_app

# Profile to specific output directory
rocprof -d ./profile_output ./my_app

# Profile with application arguments
rocprof ./my_app arg1 arg2
```

## Troubleshooting

### Permission Denied

```bash
# May need to set HSA permissions
sudo chmod 666 /dev/kfd
sudo chmod 666 /dev/dri/render*

# Or add user to render group
sudo usermod -a -G render $USER
# Then logout and login
```

### Missing Counters

Some counters may not be available on all GPUs. Check available counters:

```bash
rocprof --list-basic
rocprof --list-derived
```

## Next Steps

- Read the [Kernel Optimization Guide](kernel-optimization.md)
- Learn about [Memory Profiling](memory-profiling.md)
- Explore [Advanced Tracing](advanced-tracing.md)
