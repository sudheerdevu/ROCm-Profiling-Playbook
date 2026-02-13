# ROCm Profiling Playbook

> A comprehensive guide to profiling GPU applications on AMD hardware using ROCm tools.

## 🎯 What This Repository Contains

This playbook provides hands-on examples and detailed explanations for:
- **rocprof** - Hardware counter profiling and kernel tracing
- **roctracer** - API and runtime call tracing
- **omniperf** - Advanced performance analysis
- **omnitrace** - Full application tracing
- **Radeon GPU Analyzer (RGA)** - Shader/kernel analysis

## 📊 Why GPU Profiling Matters

Understanding GPU performance requires visibility into:
- Kernel execution time and occupancy
- Memory bandwidth utilization
- Cache hit/miss rates
- Instruction throughput
- Resource bottlenecks

## 🚀 Quick Start

```bash
# Basic kernel profiling
rocprof --stats ./my_application

# Collect hardware counters
rocprof --input counters.txt -o results.csv ./my_application

# Generate trace for visualization
rocprof --hsa-trace --roctx-trace -o trace.json ./my_application
```

## 📁 Repository Structure

```
ROCm-Profiling-Playbook/
├── README.md                    # This file
├── docs/
│   ├── 01-intro-to-profiling.md
│   ├── 02-rocprof-basics.md
│   ├── 03-hardware-counters.md
│   ├── 04-trace-analysis.md
│   ├── 05-omniperf-guide.md
│   └── 06-common-bottlenecks.md
├── examples/
│   ├── basic-profiling/
│   ├── counter-collection/
│   ├── trace-visualization/
│   └── advanced-analysis/
├── configs/
│   ├── counters-memory.txt
│   ├── counters-compute.txt
│   ├── counters-cache.txt
│   └── counters-all.txt
└── scripts/
    ├── profile-quick.sh
    ├── analyze-trace.py
    └── generate-report.py
```

## 🔧 Tool Summary

| Tool | Use Case | Output |
|------|----------|--------|
| rocprof | Kernel stats, HW counters | CSV, JSON |
| roctracer | API call tracing | Text, JSON |
| omniperf | Deep performance analysis | Web dashboard |
| omnitrace | Full application profiling | Perfetto trace |
| RGA | Static shader analysis | Text report |

## 🎓 Learning Path

1. **Beginner**: Start with [Introduction to Profiling](docs/01-intro-to-profiling.md)
2. **Intermediate**: Learn [rocprof Basics](docs/02-rocprof-basics.md)
3. **Advanced**: Explore [Hardware Counters](docs/03-hardware-counters.md)

## 📈 Example Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    Profiling Workflow                        │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 1. Quick Stats  │────▶│ Identify Hot    │
│    rocprof      │     │ Kernels         │
└─────────────────┘     └────────┬────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────┐                             ┌─────────────────┐
│ 2a. Memory      │                             │ 2b. Compute     │
│     Analysis    │                             │     Analysis    │
│ (cache/BW)      │                             │ (occupancy/ALU) │
└────────┬────────┘                             └────────┬────────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │ 3. Targeted     │
                        │    Optimization │
                        └─────────────────┘
```

## 🛠️ Prerequisites

- ROCm 5.0+ installed
- AMD GPU (MI, Radeon series)
- Linux (Ubuntu 20.04/22.04, RHEL 8/9)

## 📚 Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD GPU Architecture Guide](https://gpuopen.com/learn/)
- [ROCm GitHub](https://github.com/ROCm/ROCm)

## License

MIT License - See [LICENSE](LICENSE)
