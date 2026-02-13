# Hardware Performance Counters

## Understanding Hardware Counters

Hardware counters are special registers that count specific events in the GPU:
- Memory accesses
- Cache hits/misses
- Instructions executed
- Wavefront activity
- And much more

## Counter Categories

### Memory Counters

Track data movement and memory subsystem performance.

```
# counters-memory.txt
pmc: FETCH_SIZE, WRITE_SIZE
pmc: TCP_TCC_READ_REQ_sum, TCP_TCC_WRITE_REQ_sum
pmc: TCC_HIT_sum, TCC_MISS_sum
pmc: TCC_EA_RDREQ_sum, TCC_EA_WRREQ_sum
```

| Counter | Description |
|---------|-------------|
| FETCH_SIZE | Bytes read from memory |
| WRITE_SIZE | Bytes written to memory |
| TCP_TCC_* | L1 to L2 cache requests |
| TCC_HIT/MISS | L2 cache hit/miss |
| TCC_EA_* | L2 to external memory |

### Compute Counters

Track ALU utilization and instruction mix.

```
# counters-compute.txt
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU, SQ_INSTS_SALU
pmc: SQ_INSTS_LDS
pmc: SQ_INSTS_FLAT
pmc: SQ_INSTS_VMEM_WR, SQ_INSTS_VMEM_RD
```

| Counter | Description |
|---------|-------------|
| SQ_WAVES | Total wavefronts launched |
| SQ_INSTS_VALU | Vector ALU instructions |
| SQ_INSTS_SALU | Scalar ALU instructions |
| SQ_INSTS_LDS | LDS (shared memory) instructions |
| SQ_INSTS_VMEM | Vector memory instructions |

### Occupancy Counters

Track how well the GPU is utilized.

```
# counters-occupancy.txt
pmc: GRBM_COUNT, GRBM_GUI_ACTIVE
pmc: GRBM_CP_BUSY, GRBM_SPI_BUSY
pmc: SQ_BUSY_CYCLES, SQ_WAIT_INST_VMEM
```

| Counter | Description |
|---------|-------------|
| GRBM_COUNT | Total cycles |
| GRBM_GUI_ACTIVE | Cycles GPU is active |
| SQ_BUSY_CYCLES | Cycles shader is busy |
| SQ_WAIT_INST_VMEM | Cycles waiting for memory |

### LDS (Local Data Share) Counters

Track shared memory performance.

```
# counters-lds.txt  
pmc: SQ_INSTS_LDS
pmc: SQ_LDS_BANK_CONFLICT
pmc: LDS_MEM_VIOLATIONS
```

| Counter | Description |
|---------|-------------|
| SQ_INSTS_LDS | LDS instruction count |
| SQ_LDS_BANK_CONFLICT | Bank conflicts detected |
| LDS_MEM_VIOLATIONS | LDS memory violations |

## Using Counters

### Creating Counter Files

Format:
```
pmc: COUNTER1, COUNTER2, COUNTER3
pmc: COUNTER4, COUNTER5
```

Rules:
- Multiple counters per line (limited by HW)
- rocprof automatically handles multi-pass collection
- Comments start with #

### Running Collection

```bash
# Single counter file
rocprof --input counters.txt -o results.csv ./my_app

# Collect all results in directory
rocprof -d ./profile_run --input counters.txt ./my_app
```

### Example: Complete Collection

```bash
# counters-all.txt
# Memory
pmc: FETCH_SIZE, WRITE_SIZE
pmc: TCC_HIT_sum, TCC_MISS_sum

# Compute
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU, SQ_INSTS_SALU

# Occupancy
pmc: GRBM_COUNT, GRBM_GUI_ACTIVE
```

Run:
```bash
rocprof --input counters-all.txt -o all_counters.csv ./my_app
```

## Interpreting Results

### Memory Bandwidth Calculation

```python
# From counter results
fetch_bytes = FETCH_SIZE  
write_bytes = WRITE_SIZE
time_ns = EndNs - BeginNs

# Calculate bandwidth
bandwidth_gbps = (fetch_bytes + write_bytes) / time_ns  # GB/s
```

### Cache Hit Rate

```python
hit_rate = TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum) * 100
print(f"L2 Cache Hit Rate: {hit_rate:.1f}%")
```

### Compute Efficiency

```python
# Waves per CU
waves_per_cu = SQ_WAVES / num_cus

# Instructions per wave
insts_per_wave = SQ_INSTS_VALU / SQ_WAVES

# ALU utilization  
alu_util = GRBM_GUI_ACTIVE / GRBM_COUNT * 100
```

### Memory vs Compute Bound

```python
memory_stall_ratio = SQ_WAIT_INST_VMEM / SQ_BUSY_CYCLES

if memory_stall_ratio > 0.5:
    print("Kernel is MEMORY-BOUND")
elif memory_stall_ratio < 0.2:
    print("Kernel is COMPUTE-BOUND")
else:
    print("Kernel is BALANCED")
```

## Counter Availability by Architecture

Different GPU architectures have different counter sets:

| Counter | gfx906 | gfx908 | gfx90a | gfx1030 |
|---------|--------|--------|--------|---------|
| SQ_WAVES | ✓ | ✓ | ✓ | ✓ |
| TCC_HIT_sum | ✓ | ✓ | ✓ | ✓ |
| FETCH_SIZE | ✓ | ✓ | ✓ | ✓ |
| SQ_LDS_BANK_CONFLICT | ✓ | ✓ | ✓ | - |

Check available counters:
```bash
rocprof --list-counters
```

## Advanced Usage

### Derived Metrics

Create derived metrics from raw counters:

```python
#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('results.csv')

# Derived metrics
df['bandwidth_gbps'] = (df['FETCH_SIZE'] + df['WRITE_SIZE']) / df['DurationNs']
df['cache_hit_rate'] = df['TCC_HIT_sum'] / (df['TCC_HIT_sum'] + df['TCC_MISS_sum']) * 100
df['valu_per_wave'] = df['SQ_INSTS_VALU'] / df['SQ_WAVES']

print(df[['KernelName', 'bandwidth_gbps', 'cache_hit_rate', 'valu_per_wave']])
```

### Per-Kernel Analysis

```bash
# Collect for specific kernel (by name match)
rocprof --input counters.txt -k "matmul" -o matmul_counters.csv ./my_app
```

### Multi-Pass Optimization

When collecting many counters, minimize overhead:

```bash
# Separate into logical groups
rocprof --input counters-memory.txt -o mem.csv ./my_app
rocprof --input counters-compute.txt -o compute.csv ./my_app

# Combine results
python merge_results.py mem.csv compute.csv > combined.csv
```

## Counter Reference

### Essential Counters (Start Here)

```
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU
pmc: FETCH_SIZE, WRITE_SIZE
pmc: TCC_HIT_sum, TCC_MISS_sum
pmc: GRBM_GUI_ACTIVE, GRBM_COUNT
```

### Memory Debug

```
pmc: TCC_EA_RDREQ_sum, TCC_EA_WRREQ_sum
pmc: TA_TA_BUSY_sum, TA_FLAT_READ_WAVEFRONTS_sum
pmc: TCP_TA_DATA_STALL_CYCLES_sum
```

### Compute Debug

```
pmc: SQ_INSTS_SALU, SQ_INSTS_SMEM
pmc: SQ_ACTIVE_INST_VALU, SQ_ACTIVE_INST_SCA
pmc: SQ_VALU_MFMA_BUSY_CYCLES (for matrix cores)
```

## Next Steps

- Practice with [Example Profiling Session](../examples/basic-profiling/)
- Learn [Trace Analysis](04-trace-analysis.md)
