# 04. Trace Analysis with rocprof

## What is Trace Analysis?

Trace analysis captures a timeline of GPU activities, showing exactly when kernels execute, when memory transfers occur, and how different operations overlap.

## Generating Traces

### Basic Trace Collection

```bash
# Generate JSON trace
rocprof --hip-trace --hsa-trace -o trace.csv ./my_app

# Output files:
# - trace.csv         : Summary statistics
# - trace.json        : Chrome tracing format
# - trace.stats.csv   : Per-kernel statistics
```

### Trace Options

```bash
# HIP API trace only
rocprof --hip-trace ./my_app

# HSA (low-level) trace
rocprof --hsa-trace ./my_app

# Combined traces
rocprof --hip-trace --hsa-trace ./my_app

# With timestamps
rocprof --timestamp on --hip-trace ./my_app
```

## Viewing Traces

### Chrome Tracing Viewer

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select `trace.json`
4. Navigate with WASD keys, zoom with mouse wheel

### Trace Timeline Layout

```
┌────────────────────────────────────────────────────────────┐
│ CPU Thread 0                                               │
│ ├──► hipMalloc  ├──► hipMemcpy  ├──► hipLaunchKernel      │
├────────────────────────────────────────────────────────────┤
│ GPU Queue 0                                                │
│               ├──────► MemCopy H2D  ├────► Kernel ────►    │
├────────────────────────────────────────────────────────────┤
│ GPU Queue 1                                                │
│                                    ├──► Kernel2 ────►      │
└────────────────────────────────────────────────────────────┘
         0ms        5ms        10ms        15ms        20ms
```

## Key Metrics in Traces

### Kernel Execution

```
Kernel Duration: Time from start to end
Grid Size:       Total number of work-groups
Block Size:      Threads per work-group
Register Usage:  VGPRs and SGPRs used
LDS Usage:       Local Data Share memory
```

### Memory Operations

```
H2D Transfer:    Host to Device copy
D2H Transfer:    Device to Host copy
D2D Transfer:    Device to Device copy
Transfer Size:   Bytes transferred
Bandwidth:       Size / Duration
```

### Overlap Analysis

```
Good: Kernels overlap with transfers
       ├── Kernel A ──►
       └── H2D Transfer ──►

Bad: Serial execution
       ├── Kernel A ──►├── H2D Transfer ──►
```

## Analyzing Trace Data

### Python Script for Analysis

```python
import json
import pandas as pd

def load_trace(filename):
    with open(filename) as f:
        data = json.load(f)
    return data['traceEvents']

def analyze_kernels(events):
    kernels = [e for e in events if e.get('cat') == 'kernel']
    
    df = pd.DataFrame(kernels)
    df['dur_ms'] = df['dur'] / 1000.0
    
    print("Kernel Summary:")
    print(f"  Total kernels: {len(df)}")
    print(f"  Total time: {df['dur_ms'].sum():.2f} ms")
    print(f"  Average duration: {df['dur_ms'].mean():.3f} ms")
    
    print("\nTop 5 Slowest Kernels:")
    top5 = df.nlargest(5, 'dur_ms')[['name', 'dur_ms']]
    print(top5.to_string(index=False))
    
    return df

def analyze_transfers(events):
    transfers = [e for e in events if 'copy' in e.get('name', '').lower()]
    
    df = pd.DataFrame(transfers)
    if len(df) == 0:
        return None
        
    df['dur_ms'] = df['dur'] / 1000.0
    
    print("\nMemory Transfer Summary:")
    print(f"  Total transfers: {len(df)}")
    print(f"  Total time: {df['dur_ms'].sum():.2f} ms")
    
    return df

def compute_overlap(events):
    # Find overlapping kernel and transfer regions
    kernels = [(e['ts'], e['ts'] + e['dur']) 
               for e in events if e.get('cat') == 'kernel']
    transfers = [(e['ts'], e['ts'] + e['dur']) 
                 for e in events if 'copy' in e.get('name', '').lower()]
    
    overlap_time = 0
    for k_start, k_end in kernels:
        for t_start, t_end in transfers:
            overlap_start = max(k_start, t_start)
            overlap_end = min(k_end, t_end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
    
    print(f"\nCompute-Transfer Overlap: {overlap_time / 1000:.2f} ms")

# Usage
events = load_trace('trace.json')
analyze_kernels(events)
analyze_transfers(events)
compute_overlap(events)
```

## Common Trace Patterns

### Pattern 1: Serialized Execution (Bad)

```
CPU:  ├─malloc─►├─copy─►├─launch─►├─sync─►├─copy─►
GPU:                     ├─kernel─►        ├─copy─►

Problem: GPU idle during CPU operations
Fix: Async operations, multiple streams
```

### Pattern 2: Pipelined Execution (Good)

```
Stream 0: ├─copy─►├─kernel─►├─copy─►
Stream 1:         ├─copy─►├─kernel─►├─copy─►
GPU:      ├──────────────────────────────────────►

Benefit: Continuous GPU utilization
```

### Pattern 3: Memory Bottleneck

```
Kernel:   ├─╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌─► (long, low intensity)
Transfer: ├─────────────────────► (saturated bandwidth)

Problem: Too much data movement
Fix: Fusion, compression, caching
```

### Pattern 4: Kernel Launch Overhead

```
CPU:  ├─launch─►├─launch─►├─launch─►├─launch─►
GPU:     ├─►      ├─►      ├─►      ├─►
         ^^       ^^       ^^       ^^ (tiny kernels)

Problem: Launch overhead dominates
Fix: Kernel fusion, batching
```

## Automated Trace Analysis

```bash
# rocprof provides summary statistics
rocprof --stats ./my_app

# Output includes:
# - Kernel execution counts
# - Min/Max/Average durations
# - Total time per kernel
```

### Example stats.csv Output

```csv
Name,Calls,TotalDurationNs,AverageNs,Percentage
conv2d_kernel,100,50000000,500000,45.5
relu_kernel,100,5000000,50000,4.5
matmul_kernel,50,55000000,1100000,50.0
```

## Best Practices

1. **Warmup before tracing**: First iterations have JIT overhead
2. **Trace representative workload**: Not too short, not too long
3. **Compare before/after**: Trace after each optimization
4. **Look for gaps**: Idle time indicates opportunity
5. **Check overlap**: Maximize compute-transfer overlap
6. **Watch kernel granularity**: Too many small kernels is bad

## Integration with Other Tools

```bash
# Combined hardware counters + trace
rocprof --hip-trace -i counters.txt -o full_profile.csv ./my_app

# Use with Omniperf for detailed analysis
omniperf profile -n my_trace --hip-trace -- ./my_app
omniperf analyze -p my_trace/
```
