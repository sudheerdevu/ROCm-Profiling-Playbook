"""
ROCm Trace Analyzer

Comprehensive tools for analyzing rocprof, roctracer, and other ROCm traces.
"""

import re
import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class KernelExecution:
    """Represents a single kernel execution"""
    name: str
    duration_ns: int
    start_time_ns: int
    end_time_ns: int
    grid_dims: Tuple[int, int, int]
    block_dims: Tuple[int, int, int]
    device_id: int
    stream_id: int
    queue_index: int = 0
    correlation_id: int = 0
    
    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000
    
    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000
    
    @property
    def total_threads(self) -> int:
        grid_total = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
        block_total = self.block_dims[0] * self.block_dims[1] * self.block_dims[2]
        return grid_total * block_total


@dataclass
class MemoryTransfer:
    """Represents a memory copy operation"""
    direction: str  # "HtoD", "DtoH", "DtoD", "HtoH"
    size_bytes: int
    duration_ns: int
    start_time_ns: int
    end_time_ns: int
    src_device: int
    dst_device: int
    stream_id: int
    
    @property
    def bandwidth_gbps(self) -> float:
        if self.duration_ns > 0:
            return (self.size_bytes / 1e9) / (self.duration_ns / 1e9)
        return 0.0


@dataclass
class TraceAnalysis:
    """Complete analysis of a profiling trace"""
    total_runtime_ms: float
    kernel_time_ms: float
    memory_time_ms: float
    idle_time_ms: float
    num_kernels: int
    num_memory_ops: int
    
    # Kernel statistics
    kernel_breakdown: Dict[str, Dict[str, float]]  # kernel_name -> {time, count, avg}
    top_kernels: List[Tuple[str, float]]  # (name, total_time_ms)
    
    # Memory statistics
    total_memory_transferred_mb: float
    avg_hod_bandwidth_gbps: float
    avg_doh_bandwidth_gbps: float
    
    # Efficiency metrics
    gpu_utilization: float
    kernel_efficiency: float


class RocprofTraceParser:
    """
    Parser for rocprof output traces (CSV format).
    """
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.kernels: List[KernelExecution] = []
        self.memory_ops: List[MemoryTransfer] = []
        self.raw_data: List[Dict[str, Any]] = []
    
    def parse(self) -> bool:
        """
        Parse the rocprof trace file.
        
        Returns:
            True if parsing succeeded
        """
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                self.raw_data = list(reader)
            
            # Process each row
            for row in self.raw_data:
                if 'KernelName' in row or 'Name' in row:
                    kernel = self._parse_kernel_row(row)
                    if kernel:
                        self.kernels.append(kernel)
                elif 'Kind' in row and 'copy' in row.get('Kind', '').lower():
                    mem_op = self._parse_memory_row(row)
                    if mem_op:
                        self.memory_ops.append(mem_op)
            
            logger.info(f"Parsed {len(self.kernels)} kernels, {len(self.memory_ops)} memory ops")
            return True
            
        except FileNotFoundError:
            logger.error(f"Trace file not found: {self.filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to parse trace: {e}")
            return False
    
    def _parse_kernel_row(self, row: Dict[str, Any]) -> Optional[KernelExecution]:
        """Parse a kernel execution row"""
        try:
            name = row.get('KernelName') or row.get('Name', 'Unknown')
            
            # Parse duration
            duration = int(row.get('DurationNs', 0) or row.get('Duration', 0))
            
            # Parse start/end times
            start = int(row.get('BeginNs', 0) or row.get('Start', 0))
            end = int(row.get('EndNs', 0) or row.get('End', 0))
            
            if duration == 0 and start > 0 and end > 0:
                duration = end - start
            
            # Parse grid dimensions
            grid_x = int(row.get('grd0', 1) or row.get('GridX', 1))
            grid_y = int(row.get('grd1', 1) or row.get('GridY', 1))
            grid_z = int(row.get('grd2', 1) or row.get('GridZ', 1))
            
            # Parse block dimensions
            block_x = int(row.get('wgr0', 1) or row.get('BlockX', 1))
            block_y = int(row.get('wgr1', 1) or row.get('BlockY', 1))
            block_z = int(row.get('wgr2', 1) or row.get('BlockZ', 1))
            
            return KernelExecution(
                name=name,
                duration_ns=duration,
                start_time_ns=start,
                end_time_ns=end,
                grid_dims=(grid_x, grid_y, grid_z),
                block_dims=(block_x, block_y, block_z),
                device_id=int(row.get('gpu-id', 0) or row.get('DeviceId', 0)),
                stream_id=int(row.get('queue-id', 0) or row.get('StreamId', 0)),
                correlation_id=int(row.get('CorrelationId', 0)),
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse kernel row: {e}")
            return None
    
    def _parse_memory_row(self, row: Dict[str, Any]) -> Optional[MemoryTransfer]:
        """Parse a memory transfer row"""
        try:
            kind = row.get('Kind', '')
            
            # Determine direction
            if 'HtoD' in kind or 'HostToDevice' in kind:
                direction = 'HtoD'
            elif 'DtoH' in kind or 'DeviceToHost' in kind:
                direction = 'DtoH'
            elif 'DtoD' in kind or 'DeviceToDevice' in kind:
                direction = 'DtoD'
            else:
                direction = 'unknown'
            
            return MemoryTransfer(
                direction=direction,
                size_bytes=int(row.get('Bytes', 0) or row.get('Size', 0)),
                duration_ns=int(row.get('DurationNs', 0) or row.get('Duration', 0)),
                start_time_ns=int(row.get('BeginNs', 0) or row.get('Start', 0)),
                end_time_ns=int(row.get('EndNs', 0) or row.get('End', 0)),
                src_device=int(row.get('SrcDeviceId', 0)),
                dst_device=int(row.get('DstDeviceId', 0)),
                stream_id=int(row.get('StreamId', 0)),
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse memory row: {e}")
            return None


class TraceAnalyzer:
    """
    Analyzes parsed trace data to extract performance insights.
    """
    
    def __init__(self, parser: RocprofTraceParser):
        self.parser = parser
        self.kernels = parser.kernels
        self.memory_ops = parser.memory_ops
    
    def analyze(self) -> TraceAnalysis:
        """
        Perform comprehensive trace analysis.
        
        Returns:
            TraceAnalysis with all computed metrics
        """
        # Calculate total runtime
        all_times = []
        for k in self.kernels:
            all_times.extend([k.start_time_ns, k.end_time_ns])
        for m in self.memory_ops:
            all_times.extend([m.start_time_ns, m.end_time_ns])
        
        if all_times:
            total_runtime_ns = max(all_times) - min(all_times)
        else:
            total_runtime_ns = 0
        
        # Calculate kernel time
        kernel_time_ns = sum(k.duration_ns for k in self.kernels)
        
        # Calculate memory time
        memory_time_ns = sum(m.duration_ns for m in self.memory_ops)
        
        # Calculate kernel breakdown
        kernel_breakdown = {}
        for kernel in self.kernels:
            if kernel.name not in kernel_breakdown:
                kernel_breakdown[kernel.name] = {
                    'total_time_ms': 0,
                    'count': 0,
                    'min_ms': float('inf'),
                    'max_ms': 0,
                }
            
            stats = kernel_breakdown[kernel.name]
            stats['total_time_ms'] += kernel.duration_ms
            stats['count'] += 1
            stats['min_ms'] = min(stats['min_ms'], kernel.duration_ms)
            stats['max_ms'] = max(stats['max_ms'], kernel.duration_ms)
        
        # Calculate averages
        for name, stats in kernel_breakdown.items():
            stats['avg_ms'] = stats['total_time_ms'] / stats['count'] if stats['count'] > 0 else 0
            if stats['min_ms'] == float('inf'):
                stats['min_ms'] = 0
        
        # Top kernels by time
        top_kernels = sorted(
            [(name, stats['total_time_ms']) for name, stats in kernel_breakdown.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Memory statistics
        total_memory_bytes = sum(m.size_bytes for m in self.memory_ops)
        
        htod_ops = [m for m in self.memory_ops if m.direction == 'HtoD']
        dtoh_ops = [m for m in self.memory_ops if m.direction == 'DtoH']
        
        avg_htod_bw = (
            sum(m.bandwidth_gbps for m in htod_ops) / len(htod_ops)
            if htod_ops else 0
        )
        avg_dtoh_bw = (
            sum(m.bandwidth_gbps for m in dtoh_ops) / len(dtoh_ops)
            if dtoh_ops else 0
        )
        
        # GPU utilization
        gpu_util = (kernel_time_ns / total_runtime_ns * 100) if total_runtime_ns > 0 else 0
        
        return TraceAnalysis(
            total_runtime_ms=total_runtime_ns / 1_000_000,
            kernel_time_ms=kernel_time_ns / 1_000_000,
            memory_time_ms=memory_time_ns / 1_000_000,
            idle_time_ms=(total_runtime_ns - kernel_time_ns - memory_time_ns) / 1_000_000,
            num_kernels=len(self.kernels),
            num_memory_ops=len(self.memory_ops),
            kernel_breakdown=kernel_breakdown,
            top_kernels=top_kernels,
            total_memory_transferred_mb=total_memory_bytes / (1024 * 1024),
            avg_hod_bandwidth_gbps=avg_htod_bw,
            avg_doh_bandwidth_gbps=avg_dtoh_bw,
            gpu_utilization=gpu_util,
            kernel_efficiency=gpu_util,  # Simplified
        )
    
    def generate_report(self, format: str = 'text') -> str:
        """
        Generate analysis report.
        
        Args:
            format: Output format ('text', 'markdown', 'json')
            
        Returns:
            Formatted report string
        """
        analysis = self.analyze()
        
        if format == 'json':
            return json.dumps({
                'total_runtime_ms': analysis.total_runtime_ms,
                'kernel_time_ms': analysis.kernel_time_ms,
                'memory_time_ms': analysis.memory_time_ms,
                'gpu_utilization': analysis.gpu_utilization,
                'num_kernels': analysis.num_kernels,
                'top_kernels': analysis.top_kernels,
                'kernel_breakdown': analysis.kernel_breakdown,
            }, indent=2)
        
        elif format == 'markdown':
            return self._generate_markdown_report(analysis)
        
        else:
            return self._generate_text_report(analysis)
    
    def _generate_text_report(self, analysis: TraceAnalysis) -> str:
        """Generate plain text report"""
        lines = [
            "=" * 60,
            "ROCm Trace Analysis Report",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 30,
            f"Total Runtime:      {analysis.total_runtime_ms:.2f} ms",
            f"Kernel Time:        {analysis.kernel_time_ms:.2f} ms",
            f"Memory Time:        {analysis.memory_time_ms:.2f} ms",
            f"Idle Time:          {analysis.idle_time_ms:.2f} ms",
            f"GPU Utilization:    {analysis.gpu_utilization:.1f}%",
            "",
            f"Total Kernels:      {analysis.num_kernels}",
            f"Total Memory Ops:   {analysis.num_memory_ops}",
            f"Data Transferred:   {analysis.total_memory_transferred_mb:.2f} MB",
            "",
            "TOP KERNELS BY TIME",
            "-" * 30,
        ]
        
        for name, time_ms in analysis.top_kernels[:10]:
            pct = (time_ms / analysis.kernel_time_ms * 100) if analysis.kernel_time_ms > 0 else 0
            lines.append(f"  {name[:40]:<40} {time_ms:>8.2f} ms ({pct:.1f}%)")
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, analysis: TraceAnalysis) -> str:
        """Generate markdown report"""
        lines = [
            "# ROCm Trace Analysis Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Runtime | {analysis.total_runtime_ms:.2f} ms |",
            f"| Kernel Time | {analysis.kernel_time_ms:.2f} ms |",
            f"| Memory Time | {analysis.memory_time_ms:.2f} ms |",
            f"| GPU Utilization | {analysis.gpu_utilization:.1f}% |",
            f"| Total Kernels | {analysis.num_kernels} |",
            f"| Data Transferred | {analysis.total_memory_transferred_mb:.2f} MB |",
            "",
            "## Top Kernels",
            "",
            "| Kernel | Time (ms) | % of Total |",
            "|--------|-----------|------------|",
        ]
        
        for name, time_ms in analysis.top_kernels[:10]:
            pct = (time_ms / analysis.kernel_time_ms * 100) if analysis.kernel_time_ms > 0 else 0
            lines.append(f"| {name[:30]} | {time_ms:.2f} | {pct:.1f}% |")
        
        return "\n".join(lines)


def analyze_trace(filepath: str, format: str = 'text') -> str:
    """
    Convenience function to analyze a trace file.
    
    Args:
        filepath: Path to rocprof trace CSV
        format: Output format
        
    Returns:
        Analysis report string
    """
    parser = RocprofTraceParser(filepath)
    if not parser.parse():
        return "Failed to parse trace file"
    
    analyzer = TraceAnalyzer(parser)
    return analyzer.generate_report(format)
