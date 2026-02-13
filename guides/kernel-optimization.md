# Kernel Optimization Guide

Practical techniques for optimizing HIP kernels on AMD GPUs.

## Understanding GPU Architecture

### AMD CDNA Architecture (MI Series)

- **Compute Units (CUs)**: Basic execution units
- **Wavefronts**: 64 threads executing in lockstep
- **VGPRs**: Vector registers (per wavefront)
- **SGPRs**: Scalar registers (per wavefront)
- **LDS**: Local Data Share (shared memory)

### Key Specifications

| GPU | CUs | VGPRs/SIMD | LDS/CU | Max Waves/CU |
|-----|-----|------------|--------|--------------|
| MI100 | 120 | 256 | 64KB | 40 |
| MI210 | 110 | 512 | 64KB | 40 |
| MI300X | 304 | 512 | 64KB | 40 |

## Occupancy Optimization

Occupancy = (Active Waves / Max Waves per CU) × 100%

### Factors Affecting Occupancy

1. **Register Usage**
   - Each wave needs VGPRs
   - More VGPRs = fewer concurrent waves

2. **LDS Usage**
   - Shared among blocks on same CU
   - High LDS = fewer concurrent blocks

3. **Block Size**
   - Should be multiple of wave size (64)
   - Typical: 256 or 512 threads

### Checking Occupancy

```cpp
// Get occupancy in code
int blockSize = 256;
int minGridSize;
int maxBlocksPerSM;

hipOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize,
    myKernel, 0, 0);

hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocksPerSM,
    myKernel, blockSize, 0);
```

## Memory Optimization

### Global Memory Access

✅ **Do**: Coalesced accesses
```cpp
// Good: Threads access consecutive addresses
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx] * 2;
```

❌ **Don't**: Strided accesses
```cpp
// Bad: Threads access non-consecutive addresses
int idx = threadIdx.x * stride;  // if stride > 1
output[idx] = input[idx];
```

### Use Vector Types

```cpp
// Load 4 floats at once
float4 data = *reinterpret_cast<float4*>(&input[idx * 4]);
```

### LDS Usage

```cpp
__shared__ float lds_buffer[256];

// Load to LDS
lds_buffer[threadIdx.x] = input[global_idx];
__syncthreads();

// Multiple accesses from LDS (fast)
float sum = lds_buffer[threadIdx.x];
sum += lds_buffer[threadIdx.x + 1];
```

### Avoiding Bank Conflicts

```cpp
// Potential bank conflicts
__shared__ float matrix[32][32];
// Access pattern: matrix[row][col] with consecutive threads
// Solution: Pad the array
__shared__ float matrix[32][33];  // +1 padding
```

## Instruction Optimization

### Use Intrinsics

```cpp
// Fast math functions
float result = __fmaf_rn(a, b, c);  // FMA: a*b + c
float inv = __frcp_rn(x);           // Fast reciprocal
float sqrt = __fsqrt_rn(x);         // Fast square root
```

### Loop Unrolling

```cpp
// Manual unrolling hint
#pragma unroll 4
for (int i = 0; i < N; i++) {
    result += data[i];
}
```

### Minimize Divergence

✅ **Do**: Uniform branching
```cpp
if (blockIdx.x == 0) {
    // All threads in block take same path
}
```

❌ **Don't**: Thread-dependent branching
```cpp
if (threadIdx.x % 2 == 0) {
    // Half the threads take different paths
}
```

## Profiling Workflow

### 1. Baseline Profile

```bash
rocprof --stats -o baseline.csv ./my_app
```

### 2. Identify Hotspots

Look for kernels with:
- Longest total time
- Highest call count
- Low utilization

### 3. Collect Detailed Metrics

```bash
# Create metrics file
cat > optimize_metrics.txt << EOF
pmc: VALUUtilization
pmc: VALUBusy
pmc: MemUnitBusy
pmc: LDSBankConflict
pmc: FetchSize
pmc: WriteSize
EOF

rocprof -i optimize_metrics.txt -o detailed.csv ./my_app
```

### 4. Analyze Bottleneck

| Symptom | Bottleneck | Action |
|---------|------------|--------|
| High MemUnitBusy, Low VALUUtil | Memory | Improve access patterns |
| High VALUUtil | Compute | Use faster math |
| High LDSBankConflict | LDS | Add padding |
| Low Occupancy | Resources | Reduce registers/LDS |

### 5. Iterate

Make one change at a time and re-profile.

## Common Optimizations Checklist

- [ ] Block size is multiple of 64
- [ ] Global memory accesses are coalesced
- [ ] Using vector loads where possible
- [ ] LDS arrays are padded to avoid bank conflicts
- [ ] Loop unrolling is applied
- [ ] Fast math functions are used
- [ ] Branch divergence is minimized
- [ ] Occupancy is reasonable (>50%)

## Tools Reference

| Tool | Purpose |
|------|---------|
| `rocprof` | Kernel profiling and metrics |
| `rocm-smi` | GPU monitoring |
| `rocgdb` | Debugging |
| `omniperf` | Advanced profiling |
| `roctracer` | API tracing |
