# Introduction to GPU Profiling

## Why Profile?

GPU profiling helps answer critical questions:
- **Where** is my application spending time?
- **Why** is a kernel slow?
- **How** can I improve performance?

## The GPU Performance Model

### Key Metrics

1. **Execution Time** - How long kernels take
2. **Occupancy** - How many waves/threads are active
3. **Memory Bandwidth** - Data movement rate
4. **Cache Efficiency** - Hit rates for L1/L2 caches
5. **Compute Utilization** - ALU unit usage

### Bottleneck Categories

```
┌──────────────────────────────────────────────────────┐
│                 GPU Bottlenecks                      │
├──────────────────┬───────────────────────────────────┤
│   Memory-Bound   │         Compute-Bound            │
├──────────────────┼───────────────────────────────────┤
│ • Low cache hits │ • High ALU utilization           │
│ • High BW usage  │ • Low memory stalls              │
│ • Memory stalls  │ • Limited by FP/INT units        │
│ • Bank conflicts │ • Register pressure              │
└──────────────────┴───────────────────────────────────┘
```

## AMD GPU Architecture Basics

### Compute Unit (CU) Structure

```
┌────────────────────────────────────────────────────┐
│                   Compute Unit                      │
├────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  SIMD 0  │  │  SIMD 1  │  │  SIMD 2  │  ...    │
│  │ (64 ALUs)│  │ (64 ALUs)│  │ (64 ALUs)│         │
│  └──────────┘  └──────────┘  └──────────┘         │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │     Local Data Share (LDS) - 64 KB          │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │           L1 Cache - 16 KB                   │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

### Memory Hierarchy

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| Registers | ~256 KB/CU | 0 cycles | Instant |
| LDS | 64 KB/CU | ~20 cycles | ~12 TB/s |
| L1 Cache | 16 KB/CU | ~80 cycles | ~8 TB/s |
| L2 Cache | 4-8 MB | ~200 cycles | ~6 TB/s |
| HBM/GDDR | 16-128 GB | ~300+ cycles | 400-3200 GB/s |

## When to Profile

1. **After initial implementation** - Establish baseline
2. **After optimization** - Measure improvement
3. **When performance degrades** - Find regressions
4. **Before production** - Final validation

## Profiling Tools Overview

### rocprof
The primary ROCm profiling tool

**Strengths:**
- Low overhead
- Hardware counter access
- Kernel-level metrics
- Built into ROCm

**Best for:**
- Quick performance checks
- Detailed kernel analysis
- Counter collection

### omniperf
Advanced performance analysis tool

**Strengths:**
- Pre-built analysis workloads
- Web-based dashboard
- Automatic bottleneck detection

**Best for:**
- Deep-dive analysis
- Performance optimization
- Team collaboration

### omnitrace
Full application tracing

**Strengths:**
- CPU + GPU combined view
- API call tracing
- Memory operations
- Thread activity

**Best for:**
- Understanding full picture
- Identifying host-device issues
- Complex applications

## Next Steps

1. Install ROCm profiling tools
2. Run your first profile with rocprof
3. Learn to interpret results

Continue to: [rocprof Basics](02-rocprof-basics.md)
