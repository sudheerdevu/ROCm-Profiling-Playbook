"""
ROCm Profiling Analysis Module

Tools for analyzing rocprof traces and kernel performance.
"""

from .trace_analyzer import (
    KernelExecution,
    MemoryTransfer,
    TraceAnalysis,
    RocprofTraceParser,
    TraceAnalyzer,
    analyze_trace,
)

from .kernel_analyzer import (
    OccupancyIssue,
    KernelProfile,
    KernelAnalyzer,
    quick_occupancy_check,
)

__all__ = [
    # Trace analysis
    'KernelExecution',
    'MemoryTransfer',
    'TraceAnalysis',
    'RocprofTraceParser',
    'TraceAnalyzer',
    'analyze_trace',
    
    # Kernel analysis
    'OccupancyIssue',
    'KernelProfile',
    'KernelAnalyzer',
    'quick_occupancy_check',
]

__version__ = '1.0.0'
