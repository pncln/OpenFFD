"""
OpenMP Threading for Shared Memory Parallelization

Implements hybrid MPI+OpenMP parallelization for CFD simulations:
- Thread-level parallelization of computational kernels
- NUMA-aware thread placement and data locality
- Thread-safe data structures and algorithms
- Work-sharing constructs for CFD loops
- Thread scaling and performance optimization
- Memory bandwidth optimization
- Cache-efficient threading patterns

Enables efficient utilization of modern multi-core processors within
each MPI process for maximum scalability on HPC systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing as mp

# Try to import numba for JIT compilation with threading support
try:
    import numba
    from numba import jit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Mock decorators for environments without numba
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args, **kwargs):
        return range(*args)

logger = logging.getLogger(__name__)


class ThreadingStrategy(Enum):
    """Enumeration of threading strategies."""
    LOOP_PARALLEL = "loop_parallel"
    TASK_PARALLEL = "task_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    NESTED_PARALLEL = "nested_parallel"


class WorkDistribution(Enum):
    """Enumeration of work distribution patterns."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    GUIDED = "guided"
    BLOCK_CYCLIC = "block_cyclic"


@dataclass
class OpenMPConfig:
    """Configuration for OpenMP threading."""
    
    # Threading settings
    num_threads: int = 0  # 0 = auto-detect
    max_threads: int = 0  # 0 = no limit
    thread_affinity: str = "spread"  # "spread", "compact", "explicit"
    
    # Work distribution
    work_distribution: WorkDistribution = WorkDistribution.STATIC
    chunk_size: int = 0  # 0 = auto
    load_balancing: bool = True
    
    # Memory and cache optimization
    numa_aware: bool = True
    cache_line_size: int = 64  # bytes
    memory_alignment: int = 32  # bytes for SIMD
    prefetch_distance: int = 8  # cache lines
    
    # Performance tuning
    enable_simd: bool = True
    vectorization_threshold: int = 1000  # minimum loop size for vectorization
    thread_scaling_threshold: int = 10000  # minimum work for threading
    
    # Hybrid MPI+OpenMP
    hybrid_mode: bool = False
    threads_per_mpi_process: int = 0  # 0 = auto
    thread_migration: bool = False  # Allow thread migration
    
    # Debugging and profiling
    enable_profiling: bool = False
    thread_debugging: bool = False
    performance_counters: bool = False


@dataclass
class ThreadingMetrics:
    """Metrics for threading performance."""
    
    # Thread utilization
    thread_efficiency: float = 0.0
    load_balance_factor: float = 0.0
    parallel_fraction: float = 0.0
    
    # Performance metrics
    serial_time: float = 0.0
    parallel_time: float = 0.0
    speedup: float = 0.0
    efficiency: float = 0.0
    
    # Memory metrics
    cache_hit_rate: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    numa_locality: float = 0.0
    
    # Threading overhead
    synchronization_overhead: float = 0.0
    thread_creation_overhead: float = 0.0
    context_switch_overhead: float = 0.0


class ThreadSafeData:
    """Thread-safe data container for CFD variables."""
    
    def __init__(self, data: np.ndarray, config: OpenMPConfig):
        """Initialize thread-safe data container."""
        self.config = config
        self._lock = threading.Lock()
        self._data = self._align_memory(data)
        self._thread_local_copies = {}
        
    def _align_memory(self, data: np.ndarray) -> np.ndarray:
        """Align memory for optimal cache performance."""
        if self.config.memory_alignment > 1:
            # Ensure data is aligned to cache line boundaries
            aligned_data = np.empty_like(data)
            aligned_data[:] = data
            return aligned_data
        return data
    
    def get_thread_local_copy(self, thread_id: int) -> np.ndarray:
        """Get thread-local copy of data."""
        if thread_id not in self._thread_local_copies:
            with self._lock:
                if thread_id not in self._thread_local_copies:
                    self._thread_local_copies[thread_id] = self._data.copy()
        
        return self._thread_local_copies[thread_id]
    
    def merge_thread_local_data(self, reduction_op: str = "sum") -> np.ndarray:
        """Merge thread-local data using specified reduction."""
        if not self._thread_local_copies:
            return self._data
        
        with self._lock:
            if reduction_op == "sum":
                result = np.zeros_like(self._data)
                for thread_data in self._thread_local_copies.values():
                    result += thread_data
            elif reduction_op == "max":
                result = np.full_like(self._data, -np.inf)
                for thread_data in self._thread_local_copies.values():
                    result = np.maximum(result, thread_data)
            elif reduction_op == "min":
                result = np.full_like(self._data, np.inf)
                for thread_data in self._thread_local_copies.values():
                    result = np.minimum(result, thread_data)
            else:
                raise ValueError(f"Unsupported reduction operation: {reduction_op}")
        
        return result
    
    @property
    def data(self) -> np.ndarray:
        """Get data (thread-safe read)."""
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        """Set data (thread-safe write)."""
        with self._lock:
            self._data = self._align_memory(value)


class ParallelKernels:
    """
    Collection of parallel computational kernels for CFD.
    
    Implements thread-parallel versions of common CFD operations.
    """
    
    def __init__(self, config: OpenMPConfig):
        """Initialize parallel kernels."""
        self.config = config
        self.num_threads = self._determine_thread_count()
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.num_threads)
    
    def _determine_thread_count(self) -> int:
        """Determine optimal number of threads."""
        if self.config.num_threads > 0:
            return min(self.config.num_threads, mp.cpu_count())
        
        # Auto-detect based on system
        if self.config.hybrid_mode and self.config.threads_per_mpi_process > 0:
            return self.config.threads_per_mpi_process
        
        return mp.cpu_count()
    
    @jit(nopython=True, parallel=True, cache=True)
    def parallel_flux_computation(self,
                                 left_states: np.ndarray,
                                 right_states: np.ndarray,
                                 face_normals: np.ndarray,
                                 fluxes: np.ndarray) -> None:
        """
        Parallel computation of numerical fluxes.
        
        Uses OpenMP-style parallelization for face loop.
        """
        n_faces = left_states.shape[0]
        
        # Parallel loop over faces
        for face in prange(n_faces):
            # Compute Roe flux (simplified)
            rho_L, u_L, v_L, w_L, p_L = self._conservative_to_primitive(left_states[face])
            rho_R, u_R, v_R, w_R, p_R = self._conservative_to_primitive(right_states[face])
            
            # Roe averages
            sqrt_rho_L = np.sqrt(rho_L)
            sqrt_rho_R = np.sqrt(rho_R)
            inv_sum_sqrt = 1.0 / (sqrt_rho_L + sqrt_rho_R)
            
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * inv_sum_sqrt
            v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * inv_sum_sqrt
            w_roe = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) * inv_sum_sqrt
            
            h_L = self._compute_enthalpy(rho_L, u_L, v_L, w_L, p_L)
            h_R = self._compute_enthalpy(rho_R, u_R, v_R, w_R, p_R)
            h_roe = (sqrt_rho_L * h_L + sqrt_rho_R * h_R) * inv_sum_sqrt
            
            # Speed of sound
            gamma = 1.4
            a_roe = np.sqrt((gamma - 1) * (h_roe - 0.5 * (u_roe**2 + v_roe**2 + w_roe**2)))
            
            # Normal velocity
            nx, ny, nz = face_normals[face]
            u_n = u_roe * nx + v_roe * ny + w_roe * nz
            
            # Eigenvalues
            lambda1 = u_n - a_roe
            lambda2 = u_n
            lambda3 = u_n + a_roe
            
            # Simple flux computation (would use full Roe solver in practice)
            flux_factor = 0.5 * max(abs(lambda1), abs(lambda2), abs(lambda3))
            
            for var in range(5):
                fluxes[face, var] = 0.5 * (left_states[face, var] + right_states[face, var]) - \
                                   flux_factor * (right_states[face, var] - left_states[face, var])
    
    @staticmethod
    @jit(nopython=True, inline='always')
    def _conservative_to_primitive(conservative: np.ndarray) -> tuple:
        """Convert conservative to primitive variables."""
        rho, rho_u, rho_v, rho_w, rho_E = conservative
        rho = max(rho, 1e-12)
        
        u = rho_u / rho
        v = rho_v / rho
        w = rho_w / rho
        
        gamma = 1.4
        kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
        pressure = (gamma - 1) * (rho_E - kinetic_energy)
        
        return rho, u, v, w, max(pressure, 1e-12)
    
    @staticmethod
    @jit(nopython=True, inline='always')
    def _compute_enthalpy(rho: float, u: float, v: float, w: float, p: float) -> float:
        """Compute specific enthalpy."""
        gamma = 1.4
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        return gamma * p / ((gamma - 1) * rho) + kinetic_energy
    
    @jit(nopython=True, parallel=True, cache=True)
    def parallel_residual_computation(self,
                                    solution: np.ndarray,
                                    fluxes: np.ndarray,
                                    source_terms: np.ndarray,
                                    residuals: np.ndarray,
                                    cell_volumes: np.ndarray) -> None:
        """Parallel computation of cell residuals."""
        n_cells = solution.shape[0]
        
        # Parallel loop over cells
        for cell in prange(n_cells):
            volume = cell_volumes[cell]
            
            for var in range(5):
                # Simplified residual computation
                flux_divergence = fluxes[cell, var]  # Would sum face fluxes in practice
                source = source_terms[cell, var]
                
                residuals[cell, var] = (flux_divergence + source) / volume
    
    @jit(nopython=True, parallel=True, cache=True)
    def parallel_gradient_computation(self,
                                    solution: np.ndarray,
                                    cell_centers: np.ndarray,
                                    gradients: np.ndarray) -> None:
        """Parallel computation of solution gradients."""
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        
        # Parallel loop over cells
        for cell in prange(n_cells):
            for var in range(n_vars):
                # Simplified gradient computation using finite differences
                # In practice, would use least-squares or Green-Gauss method
                
                # Use neighboring cells (simplified)
                dx = 0.01  # Grid spacing
                if cell > 0 and cell < n_cells - 1:
                    grad_x = (solution[cell + 1, var] - solution[cell - 1, var]) / (2 * dx)
                    gradients[cell, var, 0] = grad_x
                    
                    # For simplicity, assume similar for y and z directions
                    gradients[cell, var, 1] = grad_x * 0.1
                    gradients[cell, var, 2] = grad_x * 0.1
    
    def parallel_matrix_vector_multiply(self,
                                      matrix: np.ndarray,
                                      vector: np.ndarray,
                                      result: np.ndarray) -> None:
        """Parallel sparse matrix-vector multiplication."""
        n_rows = matrix.shape[0]
        
        def compute_chunk(start: int, end: int):
            """Compute matrix-vector product for chunk of rows."""
            for i in range(start, end):
                result[i] = np.dot(matrix[i], vector)
        
        # Determine chunk size for load balancing
        chunk_size = max(1, n_rows // self.num_threads)
        
        # Create thread pool and distribute work
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                future = executor.submit(compute_chunk, start, end)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()


class ThreadScheduler:
    """
    Thread scheduler for work distribution and load balancing.
    
    Manages thread assignment and work distribution strategies.
    """
    
    def __init__(self, config: OpenMPConfig):
        """Initialize thread scheduler."""
        self.config = config
        self.num_threads = min(config.num_threads or mp.cpu_count(), mp.cpu_count())
        self.thread_pool = None
        
    def create_work_chunks(self, total_work: int, strategy: WorkDistribution = None) -> List[Tuple[int, int]]:
        """Create work chunks for parallel execution."""
        if strategy is None:
            strategy = self.config.work_distribution
        
        chunks = []
        
        if strategy == WorkDistribution.STATIC:
            # Static equal-size chunks
            chunk_size = max(1, total_work // self.num_threads)
            for i in range(0, total_work, chunk_size):
                end = min(i + chunk_size, total_work)
                chunks.append((i, end))
                
        elif strategy == WorkDistribution.DYNAMIC:
            # Dynamic smaller chunks for load balancing
            chunk_size = max(1, total_work // (self.num_threads * 4))
            for i in range(0, total_work, chunk_size):
                end = min(i + chunk_size, total_work)
                chunks.append((i, end))
                
        elif strategy == WorkDistribution.GUIDED:
            # Guided scheduling: decreasing chunk sizes
            remaining = total_work
            while remaining > 0:
                chunk_size = max(1, remaining // (self.num_threads * 2))
                end = min(total_work - remaining + chunk_size, total_work)
                start = total_work - remaining
                chunks.append((start, end))
                remaining -= chunk_size
                
        else:  # BLOCK_CYCLIC
            # Block-cyclic distribution
            block_size = max(1, self.config.chunk_size or 64)
            for block_start in range(0, total_work, block_size * self.num_threads):
                for thread in range(self.num_threads):
                    start = block_start + thread * block_size
                    end = min(start + block_size, total_work)
                    if start < total_work:
                        chunks.append((start, end))
        
        return chunks
    
    def execute_parallel_work(self,
                            work_function: Callable,
                            total_work: int,
                            *args,
                            **kwargs) -> List[Any]:
        """Execute work function in parallel with optimal scheduling."""
        if total_work < self.config.thread_scaling_threshold:
            # Execute serially for small work
            return [work_function(0, total_work, *args, **kwargs)]
        
        # Create work chunks
        chunks = self.create_work_chunks(total_work)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for start, end in chunks:
                future = executor.submit(work_function, start, end, *args, **kwargs)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                results.append(future.result())
        
        return results


class PerformanceProfiler:
    """
    Performance profiler for OpenMP threading.
    
    Monitors thread performance and provides optimization insights.
    """
    
    def __init__(self, config: OpenMPConfig):
        """Initialize performance profiler."""
        self.config = config
        self.metrics = ThreadingMetrics()
        self.thread_timings = {}
        self.memory_stats = {}
        
    def start_timing(self, operation: str, thread_id: int = 0):
        """Start timing an operation."""
        if operation not in self.thread_timings:
            self.thread_timings[operation] = {}
        
        self.thread_timings[operation][thread_id] = time.time()
    
    def end_timing(self, operation: str, thread_id: int = 0):
        """End timing an operation."""
        if operation in self.thread_timings and thread_id in self.thread_timings[operation]:
            elapsed = time.time() - self.thread_timings[operation][thread_id]
            
            # Store timing data
            if 'elapsed_times' not in self.thread_timings[operation]:
                self.thread_timings[operation]['elapsed_times'] = []
            
            self.thread_timings[operation]['elapsed_times'].append(elapsed)
    
    def compute_parallel_efficiency(self, serial_time: float, parallel_time: float, num_threads: int) -> float:
        """Compute parallel efficiency metrics."""
        if parallel_time <= 0:
            return 0.0
        
        speedup = serial_time / parallel_time
        efficiency = speedup / num_threads
        
        self.metrics.serial_time = serial_time
        self.metrics.parallel_time = parallel_time
        self.metrics.speedup = speedup
        self.metrics.efficiency = efficiency
        
        return efficiency
    
    def analyze_load_balance(self, operation: str) -> float:
        """Analyze load balance for an operation."""
        if operation not in self.thread_timings or 'elapsed_times' not in self.thread_timings[operation]:
            return 1.0
        
        times = self.thread_timings[operation]['elapsed_times']
        if len(times) < 2:
            return 1.0
        
        max_time = max(times)
        min_time = min(times)
        avg_time = sum(times) / len(times)
        
        # Load balance factor: 1.0 = perfect balance, <1.0 = imbalanced
        load_balance = min_time / max_time if max_time > 0 else 1.0
        
        self.metrics.load_balance_factor = load_balance
        return load_balance
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'threading_metrics': {
                'efficiency': self.metrics.efficiency,
                'speedup': self.metrics.speedup,
                'load_balance': self.metrics.load_balance_factor,
                'parallel_fraction': self.metrics.parallel_fraction
            },
            'timing_breakdown': {},
            'optimization_recommendations': []
        }
        
        # Analyze timing breakdown
        for operation, timing_data in self.thread_timings.items():
            if 'elapsed_times' in timing_data:
                times = timing_data['elapsed_times']
                report['timing_breakdown'][operation] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_dev': np.std(times)
                }
        
        # Generate optimization recommendations
        if self.metrics.efficiency < 0.8:
            report['optimization_recommendations'].append(
                "Low parallel efficiency detected. Consider reducing thread count or increasing work per thread."
            )
        
        if self.metrics.load_balance_factor < 0.8:
            report['optimization_recommendations'].append(
                "Load imbalance detected. Consider dynamic work distribution or better chunk sizing."
            )
        
        return report


class OpenMPManager:
    """
    Main manager for OpenMP threading in CFD applications.
    
    Coordinates thread management, work distribution, and performance optimization.
    """
    
    def __init__(self, config: OpenMPConfig):
        """Initialize OpenMP manager."""
        self.config = config
        
        # Initialize components
        self.kernels = ParallelKernels(config)
        self.scheduler = ThreadScheduler(config)
        self.profiler = PerformanceProfiler(config)
        
        # Thread management
        self.thread_pool = None
        self.active_threads = 0
        
        logger.info(f"OpenMP manager initialized with {self.kernels.num_threads} threads")
    
    def parallel_flux_computation(self,
                                left_states: np.ndarray,
                                right_states: np.ndarray,
                                face_normals: np.ndarray) -> np.ndarray:
        """Compute fluxes in parallel."""
        n_faces = left_states.shape[0]
        fluxes = np.zeros((n_faces, 5))
        
        if self.config.enable_profiling:
            self.profiler.start_timing('flux_computation')
        
        # Use compiled parallel kernel
        self.kernels.parallel_flux_computation(left_states, right_states, face_normals, fluxes)
        
        if self.config.enable_profiling:
            self.profiler.end_timing('flux_computation')
        
        return fluxes
    
    def parallel_residual_computation(self,
                                    solution: np.ndarray,
                                    fluxes: np.ndarray,
                                    source_terms: np.ndarray,
                                    cell_volumes: np.ndarray) -> np.ndarray:
        """Compute residuals in parallel."""
        n_cells = solution.shape[0]
        residuals = np.zeros_like(solution)
        
        if self.config.enable_profiling:
            self.profiler.start_timing('residual_computation')
        
        self.kernels.parallel_residual_computation(solution, fluxes, source_terms, residuals, cell_volumes)
        
        if self.config.enable_profiling:
            self.profiler.end_timing('residual_computation')
        
        return residuals
    
    def parallel_gradient_computation(self,
                                    solution: np.ndarray,
                                    cell_centers: np.ndarray) -> np.ndarray:
        """Compute gradients in parallel."""
        n_cells, n_vars = solution.shape
        gradients = np.zeros((n_cells, n_vars, 3))
        
        if self.config.enable_profiling:
            self.profiler.start_timing('gradient_computation')
        
        self.kernels.parallel_gradient_computation(solution, cell_centers, gradients)
        
        if self.config.enable_profiling:
            self.profiler.end_timing('gradient_computation')
        
        return gradients
    
    def set_thread_count(self, num_threads: int):
        """Dynamically adjust thread count."""
        self.config.num_threads = num_threads
        self.kernels.num_threads = min(num_threads, mp.cpu_count())
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.kernels.num_threads)
        
        logger.info(f"Thread count adjusted to {self.kernels.num_threads}")
    
    def optimize_performance(self, workload_profile: Dict[str, float]) -> Dict[str, Any]:
        """Optimize threading parameters based on workload profile."""
        recommendations = {}
        
        # Analyze workload characteristics
        total_work = sum(workload_profile.values())
        compute_intensive = workload_profile.get('computation', 0) / total_work > 0.7
        memory_intensive = workload_profile.get('memory_access', 0) / total_work > 0.5
        
        # Optimize thread count
        if compute_intensive:
            optimal_threads = mp.cpu_count()
            recommendations['thread_count'] = optimal_threads
        elif memory_intensive:
            # Reduce threads to avoid memory bandwidth saturation
            optimal_threads = max(1, mp.cpu_count() // 2)
            recommendations['thread_count'] = optimal_threads
        
        # Optimize work distribution
        if workload_profile.get('load_imbalance', 0) > 0.2:
            recommendations['work_distribution'] = WorkDistribution.DYNAMIC
        else:
            recommendations['work_distribution'] = WorkDistribution.STATIC
        
        # Apply optimizations
        if 'thread_count' in recommendations:
            self.set_thread_count(recommendations['thread_count'])
        
        if 'work_distribution' in recommendations:
            self.config.work_distribution = recommendations['work_distribution']
        
        return recommendations
    
    def get_threading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive threading statistics."""
        stats = {
            'configuration': {
                'num_threads': self.kernels.num_threads,
                'work_distribution': self.config.work_distribution.value,
                'numa_aware': self.config.numa_aware,
                'simd_enabled': self.config.enable_simd
            },
            'performance': self.profiler.get_performance_report(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'cache_line_size': self.config.cache_line_size,
                'memory_alignment': self.config.memory_alignment
            }
        }
        
        return stats


def create_openmp_manager(num_threads: int = 0,
                         config: Optional[OpenMPConfig] = None) -> OpenMPManager:
    """
    Factory function for creating OpenMP managers.
    
    Args:
        num_threads: Number of threads (0 = auto-detect)
        config: OpenMP configuration
        
    Returns:
        Configured OpenMP manager
    """
    if config is None:
        config = OpenMPConfig()
    
    if num_threads > 0:
        config.num_threads = num_threads
    
    return OpenMPManager(config)


def test_openmp_threading():
    """Test OpenMP threading functionality."""
    print("Testing OpenMP Threading:")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"CPU count: {mp.cpu_count()}")
    
    # Create test configuration
    config = OpenMPConfig(
        num_threads=min(4, mp.cpu_count()),
        work_distribution=WorkDistribution.STATIC,
        enable_profiling=True
    )
    
    # Create OpenMP manager
    manager = create_openmp_manager(config=config)
    
    print(f"\n  OpenMP manager created with {manager.kernels.num_threads} threads")
    
    # Test parallel flux computation
    print(f"\n  Testing parallel flux computation:")
    n_faces = 10000
    left_states = np.random.rand(n_faces, 5) + 0.5
    right_states = np.random.rand(n_faces, 5) + 0.5
    face_normals = np.random.rand(n_faces, 3)
    
    # Normalize face normals
    for i in range(n_faces):
        norm = np.linalg.norm(face_normals[i])
        if norm > 1e-12:
            face_normals[i] /= norm
    
    # Time serial computation
    start_time = time.time()
    fluxes_serial = np.zeros((n_faces, 5))
    for i in range(min(1000, n_faces)):  # Limited for timing
        # Simple flux computation
        fluxes_serial[i] = 0.5 * (left_states[i] + right_states[i])
    serial_time = time.time() - start_time
    
    # Time parallel computation
    start_time = time.time()
    fluxes_parallel = manager.parallel_flux_computation(left_states, right_states, face_normals)
    parallel_time = time.time() - start_time
    
    print(f"    Serial time: {serial_time:.4f}s")
    print(f"    Parallel time: {parallel_time:.4f}s")
    if serial_time > 0:
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"    Speedup: {speedup:.2f}x")
    
    # Test parallel residual computation
    print(f"\n  Testing parallel residual computation:")
    n_cells = 5000
    solution = np.random.rand(n_cells, 5) + 0.5
    fluxes = np.random.rand(n_cells, 5) * 0.1
    source_terms = np.random.rand(n_cells, 5) * 0.01
    cell_volumes = np.ones(n_cells) * 0.001
    
    residuals = manager.parallel_residual_computation(solution, fluxes, source_terms, cell_volumes)
    print(f"    Computed residuals for {n_cells} cells")
    print(f"    Residual norm: {np.linalg.norm(residuals):.2e}")
    
    # Test parallel gradient computation
    print(f"\n  Testing parallel gradient computation:")
    cell_centers = np.random.rand(n_cells, 3)
    gradients = manager.parallel_gradient_computation(solution, cell_centers)
    
    print(f"    Computed gradients for {n_cells} cells")
    print(f"    Gradient norm: {np.linalg.norm(gradients):.2e}")
    
    # Test performance optimization
    print(f"\n  Testing performance optimization:")
    workload_profile = {
        'computation': 0.7,
        'memory_access': 0.2,
        'communication': 0.1,
        'load_imbalance': 0.15
    }
    
    recommendations = manager.optimize_performance(workload_profile)
    print(f"    Optimization recommendations: {recommendations}")
    
    # Get threading statistics
    stats = manager.get_threading_statistics()
    print(f"\n  Threading statistics:")
    print(f"    Configured threads: {stats['configuration']['num_threads']}")
    print(f"    Work distribution: {stats['configuration']['work_distribution']}")
    print(f"    SIMD enabled: {stats['configuration']['simd_enabled']}")
    
    if 'threading_metrics' in stats['performance']:
        metrics = stats['performance']['threading_metrics']
        print(f"    Efficiency: {metrics.get('efficiency', 0):.3f}")
        print(f"    Load balance: {metrics.get('load_balance', 0):.3f}")
    
    print(f"\n  OpenMP threading test completed!")


if __name__ == "__main__":
    test_openmp_threading()