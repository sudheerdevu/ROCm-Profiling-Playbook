"""
Kernel Performance Analyzer

Specialized analyzer for HIP kernel performance optimization.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OccupancyIssue(Enum):
    """Types of occupancy issues"""
    REGISTER_LIMITED = "register_limited"
    LDS_LIMITED = "lds_limited"
    BLOCK_SIZE_LIMITED = "block_size_limited"
    WAVE_LIMITED = "wave_limited"
    OPTIMAL = "optimal"


@dataclass
class KernelProfile:
    """Detailed kernel performance profile"""
    name: str
    
    # Execution configuration
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    
    # Resource usage
    vgpr_count: int
    sgpr_count: int
    lds_bytes: int
    spill_bytes: int
    
    # Performance metrics
    achieved_occupancy: float
    theoretical_occupancy: float
    ipc: float  # Instructions per cycle
    
    # Memory metrics
    l1_hit_rate: float
    l2_hit_rate: float
    vram_read_throughput_gbps: float
    vram_write_throughput_gbps: float
    
    # Compute metrics
    valu_utilization: float
    salu_utilization: float
    vmem_utilization: float
    
    # Bottleneck analysis
    bottleneck: str
    recommendations: List[str]


class KernelAnalyzer:
    """
    Analyzes kernel performance and provides optimization recommendations.
    """
    
    # GFX architecture specifications
    ARCH_SPECS = {
        'gfx900': {
            'max_waves_per_cu': 40,
            'vgpr_per_simd': 256,
            'sgpr_per_cu': 800,
            'lds_per_cu': 65536,
            'wave_size': 64,
            'cus': 60,
        },
        'gfx906': {
            'max_waves_per_cu': 40,
            'vgpr_per_simd': 256,
            'sgpr_per_cu': 800,
            'lds_per_cu': 65536,
            'wave_size': 64,
            'cus': 60,
        },
        'gfx908': {
            'max_waves_per_cu': 40,
            'vgpr_per_simd': 256,
            'sgpr_per_cu': 800,
            'lds_per_cu': 65536,
            'wave_size': 64,
            'cus': 120,
        },
        'gfx90a': {
            'max_waves_per_cu': 40,
            'vgpr_per_simd': 512,
            'sgpr_per_cu': 800,
            'lds_per_cu': 65536,
            'wave_size': 64,
            'cus': 110,
        },
        'gfx1030': {
            'max_waves_per_cu': 32,
            'vgpr_per_simd': 512,
            'sgpr_per_cu': 800,
            'lds_per_cu': 131072,
            'wave_size': 32,
            'cus': 72,
        },
        'gfx1100': {
            'max_waves_per_cu': 32,
            'vgpr_per_simd': 512,
            'sgpr_per_cu': 800,
            'lds_per_cu': 131072,
            'wave_size': 32,
            'cus': 96,
        },
    }
    
    def __init__(self, arch: str = 'gfx908'):
        """
        Initialize analyzer for specific GPU architecture.
        
        Args:
            arch: GPU architecture (e.g., 'gfx908', 'gfx1100')
        """
        self.arch = arch
        self.specs = self.ARCH_SPECS.get(arch, self.ARCH_SPECS['gfx908'])
    
    def calculate_occupancy(self,
                            block_size: int,
                            vgprs: int,
                            lds_bytes: int) -> Dict[str, Any]:
        """
        Calculate theoretical occupancy for a kernel.
        
        Args:
            block_size: Threads per block
            vgprs: VGPRs used per thread
            lds_bytes: LDS bytes per block
            
        Returns:
            Dictionary with occupancy analysis
        """
        specs = self.specs
        wave_size = specs['wave_size']
        
        # Waves per block
        waves_per_block = (block_size + wave_size - 1) // wave_size
        
        # VGPR limited waves
        if vgprs > 0:
            vgpr_limited = specs['vgpr_per_simd'] // vgprs
        else:
            vgpr_limited = specs['max_waves_per_cu']
        
        # LDS limited waves
        if lds_bytes > 0:
            blocks_by_lds = specs['lds_per_cu'] // lds_bytes
            lds_limited = blocks_by_lds * waves_per_block
        else:
            lds_limited = specs['max_waves_per_cu']
        
        # Block size limited
        block_limited = specs['max_waves_per_cu']
        
        # Take minimum
        waves_per_cu = min(vgpr_limited, lds_limited, block_limited, specs['max_waves_per_cu'])
        
        theoretical_occupancy = (waves_per_cu / specs['max_waves_per_cu']) * 100
        
        # Determine limiting factor
        if waves_per_cu == vgpr_limited:
            limiting = OccupancyIssue.REGISTER_LIMITED
        elif waves_per_cu == lds_limited:
            limiting = OccupancyIssue.LDS_LIMITED
        elif waves_per_cu == block_limited:
            limiting = OccupancyIssue.BLOCK_SIZE_LIMITED
        else:
            limiting = OccupancyIssue.OPTIMAL
        
        return {
            'theoretical_occupancy': theoretical_occupancy,
            'waves_per_cu': waves_per_cu,
            'max_waves_per_cu': specs['max_waves_per_cu'],
            'limiting_factor': limiting,
            'vgpr_limited_waves': vgpr_limited,
            'lds_limited_waves': lds_limited,
        }
    
    def analyze_bottleneck(self,
                           valu_util: float,
                           vmem_util: float,
                           lds_util: float,
                           l2_hit_rate: float) -> Tuple[str, List[str]]:
        """
        Analyze kernel bottleneck from utilization metrics.
        
        Returns:
            Tuple of (bottleneck_type, recommendations)
        """
        recommendations = []
        
        # Determine primary bottleneck
        if vmem_util > 0.8 and valu_util < 0.5:
            bottleneck = "MEMORY_BOUND"
            recommendations.extend([
                "Consider using LDS to cache frequently accessed data",
                "Coalesce global memory accesses",
                "Increase arithmetic intensity",
                "Use vector loads (float4, etc.)",
            ])
            
            if l2_hit_rate < 0.5:
                recommendations.append("Poor L2 cache hit rate - improve memory access patterns")
        
        elif valu_util > 0.8:
            bottleneck = "COMPUTE_BOUND"
            recommendations.extend([
                "Kernel is compute-bound (good!)",
                "Consider FP16 or INT8 if precision allows",
                "Use packed math operations where possible",
            ])
        
        elif lds_util > 0.8:
            bottleneck = "LDS_BOUND"
            recommendations.extend([
                "LDS bandwidth limited",
                "Reduce LDS bank conflicts",
                "Consider padding LDS arrays",
            ])
        
        else:
            bottleneck = "LATENCY_BOUND"
            recommendations.extend([
                "Kernel may be latency bound",
                "Increase occupancy to hide latency",
                "Ensure sufficient parallelism",
                "Check for synchronization bottlenecks",
            ])
        
        return bottleneck, recommendations
    
    def suggest_block_size(self,
                           vgprs: int,
                           lds_bytes: int) -> List[Tuple[int, float]]:
        """
        Suggest optimal block sizes for a kernel.
        
        Args:
            vgprs: VGPRs per thread
            lds_bytes: LDS per block
            
        Returns:
            List of (block_size, occupancy) tuples sorted by occupancy
        """
        wave_size = self.specs['wave_size']
        
        # Try common block sizes
        candidates = [64, 128, 256, 512, 1024]
        
        results = []
        for size in candidates:
            # Adjust LDS for different block sizes
            adjusted_lds = lds_bytes * (size / 256) if lds_bytes > 0 else 0
            
            analysis = self.calculate_occupancy(size, vgprs, int(adjusted_lds))
            results.append((size, analysis['theoretical_occupancy']))
        
        # Sort by occupancy descending
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def format_recommendations(self, profile: KernelProfile) -> str:
        """
        Format detailed recommendations for a kernel.
        
        Args:
            profile: KernelProfile with analysis data
            
        Returns:
            Formatted recommendation string
        """
        lines = [
            f"=== Analysis for {profile.name} ===",
            "",
            "Configuration:",
            f"  Grid: {profile.grid_size}",
            f"  Block: {profile.block_size}",
            f"  VGPRs: {profile.vgpr_count}",
            f"  LDS: {profile.lds_bytes} bytes",
            "",
            f"Occupancy: {profile.achieved_occupancy:.1f}% (theoretical: {profile.theoretical_occupancy:.1f}%)",
            f"Bottleneck: {profile.bottleneck}",
            "",
            "Recommendations:",
        ]
        
        for i, rec in enumerate(profile.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)


def quick_occupancy_check(block_size: int,
                          vgprs: int,
                          lds_bytes: int = 0,
                          arch: str = 'gfx908') -> Dict[str, Any]:
    """
    Quick occupancy check utility function.
    
    Args:
        block_size: Threads per block
        vgprs: VGPRs per thread
        lds_bytes: LDS per block (optional)
        arch: GPU architecture
        
    Returns:
        Occupancy analysis dictionary
    """
    analyzer = KernelAnalyzer(arch)
    return analyzer.calculate_occupancy(block_size, vgprs, lds_bytes)
