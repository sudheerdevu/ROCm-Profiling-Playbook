#!/usr/bin/env python3
"""
Analyze rocprof counter results and generate insights.

Usage:
    python analyze-trace.py --memory mem.csv --compute compute.csv
    python analyze-trace.py --stats stats.txt
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

class ProfileAnalyzer:
    """Analyze rocprof output files."""
    
    def __init__(self):
        self.memory_df: Optional[pd.DataFrame] = None
        self.compute_df: Optional[pd.DataFrame] = None
        self.stats: List[Dict] = []
    
    def load_memory_counters(self, path: str) -> None:
        """Load memory counter CSV."""
        self.memory_df = pd.read_csv(path)
        print(f"Loaded {len(self.memory_df)} kernel records from {path}")
    
    def load_compute_counters(self, path: str) -> None:
        """Load compute counter CSV."""
        self.compute_df = pd.read_csv(path)
        print(f"Loaded {len(self.compute_df)} kernel records from {path}")
    
    def load_stats(self, path: str) -> None:
        """Load stats output."""
        with open(path) as f:
            lines = f.readlines()
        
        # Parse CSV-like stats
        for line in lines:
            if line.startswith('"Name"'):
                continue
            parts = line.strip().strip('"').split('","')
            if len(parts) >= 4:
                self.stats.append({
                    'name': parts[0],
                    'calls': int(parts[1]) if parts[1].isdigit() else 0,
                    'total_ns': int(parts[2]) if parts[2].isdigit() else 0,
                    'avg_ns': int(parts[3]) if parts[3].isdigit() else 0,
                })
    
    def analyze_memory(self) -> Dict:
        """Analyze memory performance."""
        if self.memory_df is None:
            return {}
        
        results = {}
        
        # Total data movement
        if 'FETCH_SIZE' in self.memory_df.columns:
            total_fetch = self.memory_df['FETCH_SIZE'].sum()
            results['total_fetch_gb'] = total_fetch / 1e9
        
        if 'WRITE_SIZE' in self.memory_df.columns:
            total_write = self.memory_df['WRITE_SIZE'].sum()
            results['total_write_gb'] = total_write / 1e9
        
        # Cache hit rate
        if 'TCC_HIT_sum' in self.memory_df.columns and 'TCC_MISS_sum' in self.memory_df.columns:
            hits = self.memory_df['TCC_HIT_sum'].sum()
            misses = self.memory_df['TCC_MISS_sum'].sum()
            if hits + misses > 0:
                results['l2_cache_hit_rate'] = hits / (hits + misses) * 100
        
        # Bandwidth calculation
        if 'BeginNs' in self.memory_df.columns and 'EndNs' in self.memory_df.columns:
            total_time_ns = self.memory_df['EndNs'].max() - self.memory_df['BeginNs'].min()
            if total_time_ns > 0 and 'total_fetch_gb' in results:
                total_data = results.get('total_fetch_gb', 0) + results.get('total_write_gb', 0)
                results['bandwidth_gbps'] = total_data / (total_time_ns / 1e9)
        
        return results
    
    def analyze_compute(self) -> Dict:
        """Analyze compute performance."""
        if self.compute_df is None:
            return {}
        
        results = {}
        
        # Wavefronts
        if 'SQ_WAVES' in self.compute_df.columns:
            results['total_waves'] = self.compute_df['SQ_WAVES'].sum()
        
        # Instruction counts
        if 'SQ_INSTS_VALU' in self.compute_df.columns:
            results['valu_insts'] = self.compute_df['SQ_INSTS_VALU'].sum()
        
        if 'SQ_INSTS_SALU' in self.compute_df.columns:
            results['salu_insts'] = self.compute_df['SQ_INSTS_SALU'].sum()
        
        # Instructions per wave
        if 'total_waves' in results and results['total_waves'] > 0:
            if 'valu_insts' in results:
                results['valu_per_wave'] = results['valu_insts'] / results['total_waves']
        
        # Memory vs compute bound estimation
        if 'SQ_BUSY_CYCLES' in self.compute_df.columns and 'SQ_WAIT_INST_VMEM' in self.compute_df.columns:
            busy = self.compute_df['SQ_BUSY_CYCLES'].sum()
            wait = self.compute_df['SQ_WAIT_INST_VMEM'].sum()
            if busy > 0:
                wait_ratio = wait / busy
                results['memory_stall_ratio'] = wait_ratio
                if wait_ratio > 0.5:
                    results['bottleneck'] = 'MEMORY-BOUND'
                elif wait_ratio < 0.2:
                    results['bottleneck'] = 'COMPUTE-BOUND'
                else:
                    results['bottleneck'] = 'BALANCED'
        
        return results
    
    def analyze_stats(self) -> Dict:
        """Analyze kernel statistics."""
        if not self.stats:
            return {}
        
        # Sort by time
        sorted_stats = sorted(self.stats, key=lambda x: x['total_ns'], reverse=True)
        
        total_time = sum(s['total_ns'] for s in sorted_stats)
        
        results = {
            'total_kernels': len(sorted_stats),
            'total_time_ms': total_time / 1e6,
            'top_kernels': []
        }
        
        for s in sorted_stats[:5]:
            results['top_kernels'].append({
                'name': s['name'],
                'time_ms': s['total_ns'] / 1e6,
                'calls': s['calls'],
                'percentage': s['total_ns'] / total_time * 100 if total_time > 0 else 0
            })
        
        return results
    
    def generate_report(self) -> str:
        """Generate analysis report."""
        report = []
        report.append("=" * 60)
        report.append("         ROCm Profile Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Stats analysis
        stats_results = self.analyze_stats()
        if stats_results:
            report.append("KERNEL STATISTICS")
            report.append("-" * 40)
            report.append(f"Total kernels: {stats_results['total_kernels']}")
            report.append(f"Total GPU time: {stats_results['total_time_ms']:.2f} ms")
            report.append("")
            report.append("Top 5 kernels by time:")
            for i, k in enumerate(stats_results['top_kernels'], 1):
                report.append(f"  {i}. {k['name']}")
                report.append(f"     Time: {k['time_ms']:.3f} ms ({k['percentage']:.1f}%)")
                report.append(f"     Calls: {k['calls']}")
            report.append("")
        
        # Memory analysis
        mem_results = self.analyze_memory()
        if mem_results:
            report.append("MEMORY ANALYSIS")
            report.append("-" * 40)
            if 'total_fetch_gb' in mem_results:
                report.append(f"Data read: {mem_results['total_fetch_gb']:.3f} GB")
            if 'total_write_gb' in mem_results:
                report.append(f"Data written: {mem_results['total_write_gb']:.3f} GB")
            if 'bandwidth_gbps' in mem_results:
                report.append(f"Effective bandwidth: {mem_results['bandwidth_gbps']:.2f} GB/s")
            if 'l2_cache_hit_rate' in mem_results:
                report.append(f"L2 cache hit rate: {mem_results['l2_cache_hit_rate']:.1f}%")
            report.append("")
        
        # Compute analysis
        compute_results = self.analyze_compute()
        if compute_results:
            report.append("COMPUTE ANALYSIS")
            report.append("-" * 40)
            if 'total_waves' in compute_results:
                report.append(f"Total wavefronts: {compute_results['total_waves']:,}")
            if 'valu_insts' in compute_results:
                report.append(f"VALU instructions: {compute_results['valu_insts']:,}")
            if 'valu_per_wave' in compute_results:
                report.append(f"VALU/wave: {compute_results['valu_per_wave']:.1f}")
            if 'bottleneck' in compute_results:
                report.append(f"Bottleneck: {compute_results['bottleneck']}")
                report.append(f"  (memory stall ratio: {compute_results['memory_stall_ratio']:.2f})")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Analyze rocprof results')
    parser.add_argument('--memory', type=str, help='Memory counters CSV')
    parser.add_argument('--compute', type=str, help='Compute counters CSV')
    parser.add_argument('--stats', type=str, help='Stats output file')
    parser.add_argument('--output', type=str, help='Output report file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    analyzer = ProfileAnalyzer()
    
    if args.memory:
        analyzer.load_memory_counters(args.memory)
    
    if args.compute:
        analyzer.load_compute_counters(args.compute)
    
    if args.stats:
        analyzer.load_stats(args.stats)
    
    if args.json:
        results = {
            'memory': analyzer.analyze_memory(),
            'compute': analyzer.analyze_compute(),
            'stats': analyzer.analyze_stats()
        }
        print(json.dumps(results, indent=2))
    else:
        report = analyzer.generate_report()
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

if __name__ == '__main__':
    main()
