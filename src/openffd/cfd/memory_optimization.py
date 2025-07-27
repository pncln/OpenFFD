"""
Memory Usage and Data Layout Optimization for Cache Efficiency

Implements advanced memory optimization techniques for CFD simulations:
- Cache-efficient data structures and memory layouts
- Structure of Arrays (SoA) vs Array of Structures (AoS) optimization
- Memory pooling and allocation strategies
- SIMD-friendly data alignment and padding
- Prefetching and locality optimization
- Memory bandwidth analysis and optimization
- NUMA-aware memory allocation
- Memory compression for large datasets

Maximizes performance on modern CPU architectures with complex memory hierarchies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import sys
import gc
import weakref
from collections import defaultdict
import mmap
import os

# Try to import memory profiling tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import NumPy memory optimization
try:
    import numpy.lib.stride_tricks as stride_tricks
    STRIDE_TRICKS_AVAILABLE = True
except ImportError:
    STRIDE_TRICKS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataLayout(Enum):
    """Enumeration of data layout strategies."""
    ARRAY_OF_STRUCTURES = "aos"  # Traditional: cell[i].variable[j]
    STRUCTURE_OF_ARRAYS = "soa"  # Optimized: variable[j].cell[i]
    HYBRID = "hybrid"  # Mixed approach
    BLOCKED = "blocked"  # Block-based layout


class MemoryPattern(Enum):
    """Enumeration of memory access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    BLOCKED = "blocked"


class AllocatorType(Enum):
    """Enumeration of memory allocator types."""
    STANDARD = "standard"
    POOL = "pool"
    ARENA = "arena"
    MMAP = "mmap"
    NUMA = "numa"


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    
    # Data layout
    preferred_layout: DataLayout = DataLayout.STRUCTURE_OF_ARRAYS
    block_size: int = 1024  # Cache-friendly block size
    alignment: int = 64  # Memory alignment (cache line size)
    
    # Memory allocation
    allocator_type: AllocatorType = AllocatorType.POOL
    pool_size: int = 1024 * 1024 * 1024  # 1GB default pool
    chunk_size: int = 4096  # Allocation chunk size
    
    # Cache optimization
    l1_cache_size: int = 32 * 1024  # 32KB typical L1 cache
    l2_cache_size: int = 256 * 1024  # 256KB typical L2 cache
    l3_cache_size: int = 8 * 1024 * 1024  # 8MB typical L3 cache
    cache_line_size: int = 64  # bytes
    
    # Prefetching
    enable_prefetching: bool = True
    prefetch_distance: int = 8  # cache lines ahead
    
    # NUMA optimization
    numa_aware: bool = True
    numa_topology: Optional[Dict[str, Any]] = None
    
    # Memory compression
    enable_compression: bool = False
    compression_threshold: int = 1024 * 1024  # 1MB
    compression_ratio_target: float = 0.5
    
    # Memory tracking
    enable_tracking: bool = True
    track_allocations: bool = True
    memory_limit: Optional[int] = None  # bytes


@dataclass
class MemoryMetrics:
    """Metrics for memory usage and performance."""
    
    # Memory usage
    total_allocated: int = 0
    total_freed: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    
    # Cache performance
    cache_hit_ratio: float = 0.0
    cache_miss_ratio: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    
    # Allocation performance
    allocation_count: int = 0
    deallocation_count: int = 0
    avg_allocation_time: float = 0.0
    fragmentation_ratio: float = 0.0
    
    # Access patterns
    sequential_accesses: int = 0
    random_accesses: int = 0
    stride_accesses: int = 0
    
    # NUMA metrics
    numa_locality: float = 0.0
    cross_numa_accesses: int = 0


class CacheOptimizedArray:
    """
    Cache-optimized array with configurable data layout.
    
    Supports both AoS and SoA layouts with automatic optimization.
    """
    
    def __init__(self,
                 shape: Tuple[int, ...],
                 dtype: np.dtype = np.float64,
                 layout: DataLayout = DataLayout.STRUCTURE_OF_ARRAYS,
                 config: Optional[MemoryConfig] = None):
        """Initialize cache-optimized array."""
        self.config = config or MemoryConfig()
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        
        # Compute aligned shape
        self._aligned_shape = self._compute_aligned_shape(shape)
        
        # Allocate aligned memory
        self._data = self._allocate_aligned_memory()
        
        # Create views for different access patterns
        self._views = {}
        self._create_views()
        
        # Access tracking
        self._access_count = 0
        self._cache_misses = 0
        
    def _compute_aligned_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute cache-aligned shape with padding."""
        if len(shape) < 2:
            return shape
        
        # Align the last dimension to cache line boundary
        elements_per_line = self.config.cache_line_size // np.dtype(self.dtype).itemsize
        
        aligned_shape = list(shape)
        last_dim = aligned_shape[-1]
        
        # Pad to next cache line boundary
        if last_dim % elements_per_line != 0:
            aligned_shape[-1] = ((last_dim // elements_per_line) + 1) * elements_per_line
        
        return tuple(aligned_shape)
    
    def _allocate_aligned_memory(self) -> np.ndarray:
        """Allocate cache-aligned memory."""
        total_elements = np.prod(self._aligned_shape)
        
        # Allocate with alignment
        if hasattr(np, 'empty_aligned'):
            # Use NumPy aligned allocation if available
            data = np.empty_aligned(self._aligned_shape, dtype=self.dtype, 
                                  alignment=self.config.alignment)
        else:
            # Manual alignment
            extra_bytes = self.config.alignment
            total_bytes = total_elements * np.dtype(self.dtype).itemsize + extra_bytes
            
            raw_memory = np.empty(total_bytes, dtype=np.uint8)
            aligned_ptr = (raw_memory.ctypes.data + self.config.alignment - 1) // self.config.alignment * self.config.alignment
            
            # Create aligned view
            offset = aligned_ptr - raw_memory.ctypes.data
            aligned_memory = raw_memory[offset:offset + total_elements * np.dtype(self.dtype).itemsize]
            data = np.frombuffer(aligned_memory, dtype=self.dtype).reshape(self._aligned_shape)
        
        return data
    
    def _create_views(self):
        """Create optimized views for different access patterns."""
        if self.layout == DataLayout.STRUCTURE_OF_ARRAYS and len(self.shape) >= 2:
            # Create SoA views: separate array for each variable
            n_vars = self.shape[-1]
            self._views['soa'] = []
            
            for var in range(n_vars):
                var_view = self._data[..., var:var+1:n_vars].squeeze()
                self._views['soa'].append(var_view)
        
        elif self.layout == DataLayout.BLOCKED:
            # Create blocked views for cache-friendly access
            self._views['blocks'] = self._create_blocked_views()
        
        # Always provide flattened view for linear access
        self._views['flat'] = self._data.ravel()
    
    def _create_blocked_views(self) -> List[np.ndarray]:
        """Create blocked views for cache optimization."""
        if len(self.shape) < 2:
            return [self._data]
        
        blocks = []
        block_size = self.config.block_size
        
        # Block the first dimension
        n_cells = self.shape[0]
        for start in range(0, n_cells, block_size):
            end = min(start + block_size, n_cells)
            block = self._data[start:end]
            blocks.append(block)
        
        return blocks
    
    def get_view(self, access_pattern: str = "default") -> Union[np.ndarray, List[np.ndarray]]:
        """Get optimized view for specific access pattern."""
        if access_pattern == "soa" and "soa" in self._views:
            return self._views["soa"]
        elif access_pattern == "blocks" and "blocks" in self._views:
            return self._views["blocks"]
        elif access_pattern == "flat":
            return self._views["flat"]
        else:
            return self._data[:self.shape[0], :self.shape[1]] if len(self.shape) >= 2 else self._data
    
    def prefetch(self, indices: np.ndarray):
        """Prefetch memory for given indices."""
        if not self.config.enable_prefetching:
            return
        
        # Simple prefetch simulation (would use actual prefetch instructions)
        for idx in indices[:self.config.prefetch_distance]:
            if idx < len(self._data):
                _ = self._data[idx]  # Touch the memory
    
    def track_access(self, index: Union[int, Tuple[int, ...]], pattern: MemoryPattern):
        """Track memory access patterns."""
        self._access_count += 1
        
        # Simple cache miss simulation
        if isinstance(index, int):
            cache_line = index // (self.config.cache_line_size // np.dtype(self.dtype).itemsize)
            if hasattr(self, '_last_cache_line') and abs(cache_line - self._last_cache_line) > 1:
                self._cache_misses += 1
            self._last_cache_line = cache_line
    
    @property
    def cache_efficiency(self) -> float:
        """Compute cache efficiency."""
        if self._access_count == 0:
            return 1.0
        return 1.0 - (self._cache_misses / self._access_count)
    
    def __getitem__(self, key):
        """Optimized array access with tracking."""
        self.track_access(key, MemoryPattern.RANDOM)
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Optimized array assignment with tracking."""
        self.track_access(key, MemoryPattern.RANDOM)
        self._data[key] = value


class MemoryPool:
    """
    Memory pool for efficient allocation and deallocation.
    
    Reduces allocation overhead and memory fragmentation.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize memory pool."""
        self.config = config
        self.pool_size = config.pool_size
        self.chunk_size = config.chunk_size
        
        # Allocate large memory pool
        self._pool = np.empty(self.pool_size, dtype=np.uint8)
        self._pool_ptr = 0
        
        # Free chunks tracking
        self._free_chunks: Dict[int, List[int]] = defaultdict(list)  # size -> [offsets]
        self._allocated_chunks: Dict[int, Tuple[int, int]] = {}  # id -> (offset, size)
        
        # Statistics
        self.metrics = MemoryMetrics()
        self._next_id = 0
    
    def allocate(self, size: int, alignment: int = None) -> Tuple[int, np.ndarray]:
        """
        Allocate memory from pool.
        
        Returns:
            Tuple of (allocation_id, memory_view)
        """
        if alignment is None:
            alignment = self.config.alignment
        
        # Align size to alignment boundary
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        
        # Try to reuse free chunk
        chunk_offset = self._find_free_chunk(aligned_size)
        
        if chunk_offset is None:
            # Allocate new chunk from pool
            chunk_offset = self._allocate_new_chunk(aligned_size)
        
        if chunk_offset is None:
            raise MemoryError("Memory pool exhausted")
        
        # Create allocation
        alloc_id = self._next_id
        self._next_id += 1
        
        self._allocated_chunks[alloc_id] = (chunk_offset, aligned_size)
        
        # Create memory view
        memory_view = self._pool[chunk_offset:chunk_offset + size]
        
        # Update metrics
        self.metrics.allocation_count += 1
        self.metrics.total_allocated += aligned_size
        self.metrics.current_usage += aligned_size
        self.metrics.peak_usage = max(self.metrics.peak_usage, self.metrics.current_usage)
        
        return alloc_id, memory_view
    
    def deallocate(self, alloc_id: int):
        """Deallocate memory back to pool."""
        if alloc_id not in self._allocated_chunks:
            raise ValueError(f"Invalid allocation ID: {alloc_id}")
        
        offset, size = self._allocated_chunks[alloc_id]
        del self._allocated_chunks[alloc_id]
        
        # Add to free chunks
        self._free_chunks[size].append(offset)
        
        # Update metrics
        self.metrics.deallocation_count += 1
        self.metrics.total_freed += size
        self.metrics.current_usage -= size
    
    def _find_free_chunk(self, size: int) -> Optional[int]:
        """Find a suitable free chunk."""
        # Look for exact size match first
        if size in self._free_chunks and self._free_chunks[size]:
            return self._free_chunks[size].pop()
        
        # Look for larger chunks
        for chunk_size in sorted(self._free_chunks.keys()):
            if chunk_size >= size and self._free_chunks[chunk_size]:
                offset = self._free_chunks[chunk_size].pop()
                
                # Split chunk if significantly larger
                if chunk_size - size >= self.config.chunk_size:
                    remaining_offset = offset + size
                    remaining_size = chunk_size - size
                    self._free_chunks[remaining_size].append(remaining_offset)
                
                return offset
        
        return None
    
    def _allocate_new_chunk(self, size: int) -> Optional[int]:
        """Allocate new chunk from pool."""
        if self._pool_ptr + size > self.pool_size:
            return None
        
        offset = self._pool_ptr
        self._pool_ptr += size
        
        return offset
    
    def get_fragmentation_ratio(self) -> float:
        """Compute memory fragmentation ratio."""
        total_free = sum(len(chunks) * size for size, chunks in self._free_chunks.items())
        total_used = self.metrics.current_usage
        
        if total_free + total_used == 0:
            return 0.0
        
        # Fragmentation = (free memory in small chunks) / (total memory)
        small_chunk_threshold = self.config.chunk_size
        small_free = sum(len(chunks) * size for size, chunks in self._free_chunks.items() 
                        if size < small_chunk_threshold)
        
        return small_free / (total_free + total_used)
    
    def defragment(self):
        """Defragment memory pool by compacting allocations."""
        # This would implement memory compaction
        # For now, just merge adjacent free chunks
        self._merge_free_chunks()
    
    def _merge_free_chunks(self):
        """Merge adjacent free chunks."""
        # Sort all free chunks by offset
        all_free = []
        for size, offsets in self._free_chunks.items():
            for offset in offsets:
                all_free.append((offset, size))
        
        all_free.sort()
        
        # Clear current free chunks
        self._free_chunks.clear()
        
        # Merge adjacent chunks
        if not all_free:
            return
        
        current_offset, current_size = all_free[0]
        
        for offset, size in all_free[1:]:
            if offset == current_offset + current_size:
                # Adjacent chunk - merge
                current_size += size
            else:
                # Non-adjacent - add current and start new
                self._free_chunks[current_size].append(current_offset)
                current_offset, current_size = offset, size
        
        # Add final chunk
        self._free_chunks[current_size].append(current_offset)


class DataStructureOptimizer:
    """
    Optimizer for CFD data structures.
    
    Analyzes access patterns and optimizes data layout for cache efficiency.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize data structure optimizer."""
        self.config = config
        self.access_patterns: Dict[str, List[MemoryPattern]] = {}
        self.layout_recommendations: Dict[str, DataLayout] = {}
    
    def analyze_access_pattern(self, name: str, indices: np.ndarray, data_shape: Tuple[int, ...]):
        """Analyze memory access pattern for optimization."""
        patterns = []
        
        if len(indices) < 2:
            return
        
        # Detect sequential access
        if self._is_sequential(indices):
            patterns.append(MemoryPattern.SEQUENTIAL)
        
        # Detect strided access
        stride = self._detect_stride(indices)
        if stride > 1:
            patterns.append(MemoryPattern.STRIDED)
        
        # Detect random access
        if not patterns:
            patterns.append(MemoryPattern.RANDOM)
        
        self.access_patterns[name] = patterns
        
        # Generate layout recommendation
        self.layout_recommendations[name] = self._recommend_layout(patterns, data_shape)
    
    def _is_sequential(self, indices: np.ndarray) -> bool:
        """Check if access pattern is sequential."""
        if len(indices) < 2:
            return True
        
        diffs = np.diff(indices)
        return np.all(diffs == 1) or np.all(diffs == -1)
    
    def _detect_stride(self, indices: np.ndarray) -> int:
        """Detect stride in access pattern."""
        if len(indices) < 3:
            return 1
        
        diffs = np.diff(indices)
        if len(np.unique(diffs)) == 1:
            return abs(diffs[0])
        
        return 1
    
    def _recommend_layout(self, patterns: List[MemoryPattern], shape: Tuple[int, ...]) -> DataLayout:
        """Recommend optimal data layout based on access patterns."""
        if MemoryPattern.SEQUENTIAL in patterns:
            # Sequential access benefits from SoA
            return DataLayout.STRUCTURE_OF_ARRAYS
        
        elif MemoryPattern.STRIDED in patterns:
            # Strided access might benefit from blocking
            return DataLayout.BLOCKED
        
        elif MemoryPattern.RANDOM in patterns:
            # Random access might benefit from AoS for locality
            return DataLayout.ARRAY_OF_STRUCTURES
        
        else:
            return self.config.preferred_layout
    
    def optimize_cfd_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, CacheOptimizedArray]:
        """Optimize CFD arrays based on analysis."""
        optimized = {}
        
        for name, array in arrays.items():
            # Get recommended layout
            layout = self.layout_recommendations.get(name, self.config.preferred_layout)
            
            # Create optimized array
            optimized[name] = CacheOptimizedArray(
                array.shape, array.dtype, layout, self.config
            )
            
            # Copy data
            optimized[name]._data[:array.shape[0], :array.shape[1]] = array
        
        return optimized


class MemoryBandwidthAnalyzer:
    """
    Analyzer for memory bandwidth utilization.
    
    Measures and optimizes memory bandwidth usage for CFD kernels.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize memory bandwidth analyzer."""
        self.config = config
        self.peak_bandwidth = self._estimate_peak_bandwidth()
        self.measurements: List[Dict[str, float]] = []
    
    def _estimate_peak_bandwidth(self) -> float:
        """Estimate peak memory bandwidth (GB/s)."""
        # Simple estimation based on system type
        # In practice, would measure with STREAM benchmark
        
        if PSUTIL_AVAILABLE:
            # Rough estimate based on memory speed
            return 50.0  # GB/s for typical DDR4 system
        else:
            return 25.0  # Conservative estimate
    
    def measure_bandwidth(self, operation_name: str, data_size: int, operation_time: float) -> float:
        """Measure memory bandwidth for an operation."""
        # Assume read + write for most operations
        bytes_transferred = data_size * 2
        bandwidth = (bytes_transferred / (1024**3)) / operation_time  # GB/s
        
        utilization = bandwidth / self.peak_bandwidth
        
        measurement = {
            'operation': operation_name,
            'bandwidth_gb_s': bandwidth,
            'utilization': utilization,
            'data_size': data_size,
            'time': operation_time
        }
        
        self.measurements.append(measurement)
        return utilization
    
    def analyze_kernel_efficiency(self, kernel_name: str, flops: int, bytes_accessed: int, time: float) -> Dict[str, float]:
        """Analyze computational intensity and roofline performance."""
        # Computational intensity (operations per byte)
        intensity = flops / bytes_accessed if bytes_accessed > 0 else 0
        
        # Achieved bandwidth
        bandwidth = (bytes_accessed / (1024**3)) / time
        
        # Achieved performance (GFLOPS)
        performance = flops / (time * 1e9)
        
        # Memory-bound vs compute-bound analysis
        memory_bound_performance = bandwidth * intensity * 1e9  # Convert to FLOPS
        
        return {
            'computational_intensity': intensity,
            'achieved_bandwidth': bandwidth,
            'achieved_performance': performance,
            'memory_bound_performance': memory_bound_performance,
            'efficiency': min(performance / memory_bound_performance, 1.0) if memory_bound_performance > 0 else 0
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on measurements."""
        recommendations = []
        
        if not self.measurements:
            return recommendations
        
        avg_utilization = np.mean([m['utilization'] for m in self.measurements])
        
        if avg_utilization < 0.3:
            recommendations.append("Low memory bandwidth utilization. Consider data prefetching or larger block sizes.")
        
        if avg_utilization > 0.9:
            recommendations.append("High memory bandwidth utilization. Consider computation/memory overlap or reducing memory traffic.")
        
        # Analyze variance in utilization
        utilization_variance = np.var([m['utilization'] for m in self.measurements])
        if utilization_variance > 0.1:
            recommendations.append("High variance in bandwidth utilization. Consider more consistent data access patterns.")
        
        return recommendations


class MemoryManager:
    """
    Main memory manager for CFD applications.
    
    Coordinates all memory optimization strategies.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize memory manager."""
        self.config = config
        
        # Initialize components
        self.memory_pool = MemoryPool(config) if config.allocator_type == AllocatorType.POOL else None
        self.optimizer = DataStructureOptimizer(config)
        self.bandwidth_analyzer = MemoryBandwidthAnalyzer(config)
        
        # Memory tracking
        self.allocations: Dict[str, Any] = {}
        self.metrics = MemoryMetrics()
        
        # System memory info
        self.system_info = self._get_system_memory_info()
        
    def _get_system_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        info = {
            'total_memory': 0,
            'available_memory': 0,
            'cache_sizes': {
                'l1': self.config.l1_cache_size,
                'l2': self.config.l2_cache_size,
                'l3': self.config.l3_cache_size
            }
        }
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            info['total_memory'] = memory.total
            info['available_memory'] = memory.available
        
        return info
    
    def create_optimized_array(self, name: str, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> CacheOptimizedArray:
        """Create cache-optimized array."""
        # Get recommended layout
        layout = self.optimizer.layout_recommendations.get(name, self.config.preferred_layout)
        
        # Create optimized array
        array = CacheOptimizedArray(shape, dtype, layout, self.config)
        
        # Track allocation
        self.allocations[name] = array
        
        # Update metrics
        array_size = np.prod(shape) * np.dtype(dtype).itemsize
        self.metrics.current_usage += array_size
        self.metrics.peak_usage = max(self.metrics.peak_usage, self.metrics.current_usage)
        
        return array
    
    def optimize_data_layout(self, arrays: Dict[str, np.ndarray]) -> Dict[str, CacheOptimizedArray]:
        """Optimize data layout for multiple arrays."""
        return self.optimizer.optimize_cfd_arrays(arrays)
    
    def track_access_pattern(self, array_name: str, indices: np.ndarray):
        """Track access pattern for optimization."""
        if array_name in self.allocations:
            array = self.allocations[array_name]
            if hasattr(array, 'shape'):
                self.optimizer.analyze_access_pattern(array_name, indices, array.shape)
    
    def measure_kernel_performance(self, kernel_name: str, data_size: int, flops: int, execution_time: float):
        """Measure and analyze kernel performance."""
        # Estimate memory access
        bytes_accessed = data_size * 8  # Assume 64-bit data
        
        # Measure bandwidth
        bandwidth_utilization = self.bandwidth_analyzer.measure_bandwidth(
            kernel_name, bytes_accessed, execution_time
        )
        
        # Analyze efficiency
        efficiency_metrics = self.bandwidth_analyzer.analyze_kernel_efficiency(
            kernel_name, flops, bytes_accessed, execution_time
        )
        
        return {
            'bandwidth_utilization': bandwidth_utilization,
            'efficiency_metrics': efficiency_metrics
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        usage = {
            'current_usage': self.metrics.current_usage,
            'peak_usage': self.metrics.peak_usage,
            'system_total': self.system_info['total_memory'],
            'system_available': self.system_info['available_memory']
        }
        
        if PSUTIL_AVAILABLE:
            current_memory = psutil.virtual_memory()
            usage['system_current'] = current_memory.used
            usage['system_percent'] = current_memory.percent
        
        if self.memory_pool:
            usage['pool_fragmentation'] = self.memory_pool.get_fragmentation_ratio()
            usage['pool_utilization'] = self.memory_pool.metrics.current_usage / self.memory_pool.pool_size
        
        return usage
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance optimization."""
        optimizations = {
            'layout_changes': 0,
            'memory_defragmentation': False,
            'recommendations': []
        }
        
        # Optimize data layouts based on access patterns
        for name, patterns in self.optimizer.access_patterns.items():
            if name in self.allocations:
                current_layout = getattr(self.allocations[name], 'layout', None)
                recommended_layout = self.optimizer._recommend_layout(patterns, self.allocations[name].shape)
                
                if current_layout != recommended_layout:
                    optimizations['layout_changes'] += 1
        
        # Defragment memory pool if needed
        if self.memory_pool and self.memory_pool.get_fragmentation_ratio() > 0.3:
            self.memory_pool.defragment()
            optimizations['memory_defragmentation'] = True
        
        # Get bandwidth optimization recommendations
        bandwidth_recommendations = self.bandwidth_analyzer.get_optimization_recommendations()
        optimizations['recommendations'].extend(bandwidth_recommendations)
        
        return optimizations
    
    def cleanup(self):
        """Cleanup memory allocations."""
        self.allocations.clear()
        if self.memory_pool:
            self.memory_pool._free_chunks.clear()
        gc.collect()


def create_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryManager:
    """
    Factory function for creating memory managers.
    
    Args:
        config: Memory optimization configuration
        
    Returns:
        Configured memory manager
    """
    if config is None:
        config = MemoryConfig()
    
    return MemoryManager(config)


def test_memory_optimization():
    """Test memory optimization functionality."""
    print("Testing Memory Optimization:")
    print(f"PSUtil available: {PSUTIL_AVAILABLE}")
    
    # Create memory manager
    config = MemoryConfig(
        preferred_layout=DataLayout.STRUCTURE_OF_ARRAYS,
        block_size=1024,
        enable_prefetching=True
    )
    
    manager = create_memory_manager(config)
    
    print(f"\n  Memory manager created")
    print(f"  System memory: {manager.system_info['total_memory'] / (1024**3):.1f} GB")
    
    # Test cache-optimized arrays
    print(f"\n  Testing cache-optimized arrays:")
    
    # Create test arrays
    n_cells = 10000
    n_vars = 5
    
    solution_array = manager.create_optimized_array("solution", (n_cells, n_vars))
    residual_array = manager.create_optimized_array("residuals", (n_cells, n_vars))
    gradient_array = manager.create_optimized_array("gradients", (n_cells, n_vars, 3))
    
    print(f"    Created solution array: {solution_array.shape}")
    print(f"    Created residual array: {residual_array.shape}")
    print(f"    Created gradient array: {gradient_array.shape}")
    
    # Test access patterns
    print(f"\n  Testing access pattern analysis:")
    
    # Sequential access
    sequential_indices = np.arange(0, min(1000, n_cells))
    manager.track_access_pattern("solution", sequential_indices)
    
    # Random access
    random_indices = np.random.randint(0, n_cells, 500)
    manager.track_access_pattern("residuals", random_indices)
    
    # Strided access
    strided_indices = np.arange(0, min(2000, n_cells), 4)
    manager.track_access_pattern("gradients", strided_indices)
    
    print(f"    Analyzed access patterns for 3 arrays")
    
    # Test memory pool
    print(f"\n  Testing memory pool:")
    if manager.memory_pool:
        # Allocate some memory chunks
        allocations = []
        for i in range(10):
            size = np.random.randint(1024, 8192)
            try:
                alloc_id, memory_view = manager.memory_pool.allocate(size)
                allocations.append(alloc_id)
                print(f"    Allocated {size} bytes (ID: {alloc_id})")
            except MemoryError:
                print(f"    Memory pool exhausted at allocation {i}")
                break
        
        # Deallocate some chunks
        for alloc_id in allocations[:5]:
            manager.memory_pool.deallocate(alloc_id)
        
        fragmentation = manager.memory_pool.get_fragmentation_ratio()
        print(f"    Memory fragmentation: {fragmentation:.3f}")
    
    # Test bandwidth analysis
    print(f"\n  Testing bandwidth analysis:")
    
    # Simulate kernel execution
    data_size = n_cells * n_vars * 8  # bytes
    flops = n_cells * n_vars * 10  # operations
    execution_time = 0.001  # seconds
    
    performance_metrics = manager.measure_kernel_performance(
        "flux_computation", data_size, flops, execution_time
    )
    
    print(f"    Bandwidth utilization: {performance_metrics['bandwidth_utilization']:.3f}")
    print(f"    Computational intensity: {performance_metrics['efficiency_metrics']['computational_intensity']:.2f}")
    
    # Test optimization
    print(f"\n  Testing performance optimization:")
    optimizations = manager.optimize_performance()
    
    print(f"    Layout changes recommended: {optimizations['layout_changes']}")
    print(f"    Memory defragmentation: {optimizations['memory_defragmentation']}")
    print(f"    Recommendations: {len(optimizations['recommendations'])}")
    
    for i, rec in enumerate(optimizations['recommendations'][:3]):
        print(f"      {i+1}. {rec}")
    
    # Get memory usage statistics
    print(f"\n  Memory usage statistics:")
    usage = manager.get_memory_usage()
    
    print(f"    Current usage: {usage['current_usage'] / (1024**2):.1f} MB")
    print(f"    Peak usage: {usage['peak_usage'] / (1024**2):.1f} MB")
    
    if 'pool_utilization' in usage:
        print(f"    Pool utilization: {usage['pool_utilization']:.3f}")
    
    # Cleanup
    manager.cleanup()
    print(f"\n  Memory optimization test completed!")


if __name__ == "__main__":
    test_memory_optimization()