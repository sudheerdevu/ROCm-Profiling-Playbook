# 06. Common GPU Bottlenecks

## Quick Reference: Bottleneck Identification

| Symptom | Likely Bottleneck | Key Counters |
|---------|------------------|--------------|
| High HBM utilization | Global memory bandwidth | TCC_EA_RDREQ, TCC_EA_WRREQ |
| High L2 miss rate | Cache thrashing | TCC_MISS, TCC_HIT |
| Low occupancy | Register/LDS pressure | VGPR, SGPR, LDS_ALLOC |
| High stall cycles | Memory latency | SQ_WAIT_INST_* |
| Low IPC | Instruction dependencies | SQ_INSTS_*/SQ_BUSY_CYCLES |

## 1. Memory Bandwidth Bottleneck

### Symptoms

```
- HBM SOL > 70%
- Low compute utilization
- Long kernel execution despite simple math
```

### Counter Evidence

```
TCC_EA_RDREQ:     High read requests to DRAM
TCC_EA_WRREQ:     High write requests to DRAM
TCC_HIT:          Low (data not in cache)
SQ_INSTS_VMEM:    High (many memory instructions)
```

### Common Causes

1. **Streaming access patterns**
   ```cpp
   // Bad: Pure streaming, no reuse
   for (int i = 0; i < N; i++) {
       output[i] = input[i] * 2.0f;
   }
   ```

2. **Large working sets**
   ```cpp
   // Working set exceeds cache
   // 100MB array won't fit in L2
   ```

3. **Low arithmetic intensity**
   ```
   FLOPs/byte < 10: Almost certainly memory-bound
   ```

### Solutions

✓ Increase arithmetic intensity via tiling
✓ Use shared memory for data reuse
✓ Quantize to reduce data size (FP32→FP16→INT8)
✓ Operator fusion to reduce intermediate storage

## 2. Cache Inefficiency

### 2a. L1 Cache (TCP) Issues

```
Symptoms:
- High TCP_READ_TAGCONFLICT_STALL_CYCLES
- Low L1 hit rate

Causes:
- Non-coalesced access
- Access stride = cache line size
```

### 2b. L2 Cache (TCC) Issues

```
Symptoms:
- TCC_MISS >> TCC_HIT
- High TCC_EA_* (evictions to DRAM)

Causes:
- Working set > L2 size
- Poor locality
- Excessive evictions
```

### Counter Analysis

```python
def analyze_cache_efficiency(counters):
    l1_hit_rate = counters['TCP_HIT'] / (counters['TCP_HIT'] + counters['TCP_MISS'])
    l2_hit_rate = counters['TCC_HIT'] / (counters['TCC_HIT'] + counters['TCC_MISS'])
    
    print(f"L1 Hit Rate: {l1_hit_rate:.2%}")
    print(f"L2 Hit Rate: {l2_hit_rate:.2%}")
    
    if l1_hit_rate < 0.8:
        print("⚠️  L1 cache issues - check memory coalescing")
    if l2_hit_rate < 0.5:
        print("⚠️  L2 cache issues - consider tiling or blocking")
```

### Solutions

✓ Ensure coalesced access (consecutive threads → consecutive addresses)
✓ Use blocking/tiling to fit in cache
✓ Prefetch data when possible
✓ Structure of Arrays (SoA) instead of Array of Structures (AoS)

## 3. Occupancy Limitations

### Symptoms

```
- Waves per CU < 4
- Low GPU utilization despite parallelism
- Resources not fully utilized
```

### Resource Constraints

```
Limited by:              Solution:
─────────────────────────────────────────
VGPRs per kernel        Reduce register usage
SGPRs per kernel        Use memory instead of registers
LDS per workgroup       Smaller LDS allocation
Waves per CU            Launch more wavefronts
```

### Counter Evidence

```
SQ_LEVEL_WAVES:        Low average waves
SQ_OCCUPANCY:          Low percentage
SQ_BUSY:               Low cycles utilized
```

### Diagnosing Register Pressure

```bash
# Check register usage
hipcc --dump-resource-usage kernel.hip

# Output:
# SGPRs: 28
# VGPRs: 64
# LDS: 8192 bytes

# Max waves calculation:
# VGPRs: 256 / ceil(64/4) = 16 waves → Limited!
# LDS: 65536 / 8192 = 8 waves → OK
# Result: Limited by VGPRs to 16 waves
```

### Solutions

✓ Reduce register usage with compiler flags
✓ Use __launch_bounds__ to hint occupancy
✓ Spill to LDS or global memory (trade-off)
✓ Process more elements per thread

```cpp
// Hint for occupancy
__launch_bounds__(256, 4)  // 256 threads, target 4 waves
__global__ void my_kernel(...) {
    // Compiler will try to limit registers
}
```

## 4. Instruction Latency

### Symptoms

```
- Low IPC
- High stall cycles
- Serial dependency chains
```

### Counter Evidence

```
SQ_WAIT_INST_LDS:      Waiting for LDS operations
SQ_WAIT_INST_VALU:     Waiting for VALU completion
SQ_WAIT_INST_VMEM:     Waiting for memory operations
SQ_INSTS / SQ_CYCLES:  IPC < 0.5 indicates stalls
```

### Common Causes

1. **Long dependency chains**
   ```cpp
   // Bad: Serial dependency
   float a = x * y;
   float b = a + z;    // Depends on a
   float c = b * w;    // Depends on b
   float d = c + v;    // Depends on c
   ```

2. **Insufficient instruction-level parallelism**
   ```cpp
   // Good: Independent operations interleaved
   float a0 = x0 * y0;
   float a1 = x1 * y1;  // Independent
   float b0 = a0 + z0;
   float b1 = a1 + z1;  // Independent
   ```

### Solutions

✓ Increase ILP with loop unrolling
✓ Process multiple elements per thread
✓ Use independent operations between dependent ones
✓ Prefetch memory to hide latency

## 5. Control Flow Divergence

### Symptoms

```
- Threads executing different paths
- Serialized execution within warp/wavefront
- Low SIMD efficiency
```

### Counter Evidence

```
SQ_INSTS_BRANCH:       High branch instructions
SIMD_EFFICIENCY:       < 100% (threads disabled)
```

### Common Causes

```cpp
// Bad: Divergent branch
if (threadIdx.x % 2 == 0) {
    // Half the threads take this path
    result = heavy_computation_a();
} else {
    // Other half takes this path
    result = heavy_computation_b();
}
// Both paths execute serially!
```

### Solutions

✓ Reorganize data to avoid divergence
✓ Use predication instead of branches
✓ Group similar work items together
✓ Use warp-level primitives

```cpp
// Better: Predication (no divergence)
float result_a = heavy_computation_a();
float result_b = heavy_computation_b();
result = (threadIdx.x % 2 == 0) ? result_a : result_b;
// Both computed, one selected - no serialization
```

## 6. Atomic Contention

### Symptoms

```
- High atomic operation count
- Serialized updates to shared locations
- Poor scaling with thread count
```

### Counter Evidence

```
SQ_INSTS_LDS:          High LDS atomics
L2_ATOMIC_*:           High L2 atomics
LDS_BANK_CONFLICT:     Contention on same location
```

### Solutions

✓ Use thread-local accumulation, then reduce
✓ Privatization (per-thread/per-warp copies)
✓ Hierarchical reduction
✓ Avoid atomics when possible

```cpp
// Bad: All threads atomic to same location
atomicAdd(&global_sum, local_value);

// Better: Hierarchical reduction
__shared__ float warp_sums[32];
// First reduce within warp
float warp_sum = warp_reduce(local_value);
if (lane == 0) warp_sums[warp_id] = warp_sum;
__syncthreads();
// Then block reduces warp sums
if (warp_id == 0) {
    float block_sum = warp_reduce(warp_sums[lane]);
    if (lane == 0) atomicAdd(&global_sum, block_sum);
}
// One atomic per block instead of per thread!
```

## Summary Decision Tree

```
Low Performance
      │
      ├─► High Memory Traffic? ───► Memory Bandwidth Bound
      │                              → Tiling, fusion, quantization
      │
      ├─► High Cache Misses? ─────► Cache Bound
      │                              → Blocking, coalescing
      │
      ├─► Low Occupancy? ─────────► Resource Bound
      │                              → Reduce registers/LDS
      │
      ├─► High Stalls? ───────────► Latency Bound
      │                              → More ILP, prefetching
      │
      ├─► Low SIMD Efficiency? ───► Divergence
      │                              → Predication, reorganize
      │
      └─► High Atomic Count? ─────► Contention
                                    → Hierarchical reduction
```
