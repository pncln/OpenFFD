"""
Benchmark Hierarchical Free Form Deformation (HFFD) performance.
"""

import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import multiprocessing as mp

# Import OpenFFD components
from openffd.core.hierarchical import create_hierarchical_ffd, HierarchicalFFD
from openffd.utils.parallel import ParallelConfig
from .data_generator import MeshGenerator

logger = logging.getLogger(__name__)

@dataclass
class HFFDResult:
    """Result of a single HFFD benchmark run."""
    # Input parameters
    mesh_size: int
    geometry_type: str
    base_dims: Tuple[int, int, int]
    max_depth: int
    subdivision_factor: int
    parallel_enabled: bool
    max_workers: Optional[int]
    
    # Performance metrics
    execution_time: float
    memory_peak_mb: float
    success: bool
    error_message: Optional[str] = None
    
    # Output metrics
    total_levels: Optional[int] = None
    total_control_points: Optional[int] = None
    hierarchy_complexity: Optional[float] = None  # Total control points / base level control points
    
    # Detailed level information
    level_creation_times: Optional[List[float]] = None
    level_control_counts: Optional[List[int]] = None
    
    # System information
    cpu_count: int = mp.cpu_count()
    timestamp: float = time.time()

class HFFDBenchmark:
    """Benchmark suite for HFFD algorithm performance."""
    
    def __init__(self, config):
        """Initialize HFFD benchmark.
        
        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        self.results: List[HFFDResult] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            logger.warning("psutil not available, memory measurement disabled")
            return 0.0
    
    def _analyze_hffd_structure(self, hffd: HierarchicalFFD) -> Tuple[int, int, float, List[int]]:
        """Analyze HFFD hierarchy structure.
        
        Args:
            hffd: HierarchicalFFD instance
            
        Returns:
            Tuple of (total_levels, total_control_points, complexity, level_counts)
        """
        if not hffd or not hffd.levels:
            return 0, 0, 0.0, []
        
        total_levels = len(hffd.levels)
        level_counts = []
        total_control_points = 0
        
        for level_id, level in hffd.levels.items():
            if level.control_points is not None:
                count = len(level.control_points)
                level_counts.append(count)
                total_control_points += count
            else:
                level_counts.append(0)
        
        # Calculate complexity as ratio to base level
        base_level_count = level_counts[0] if level_counts else 1
        complexity = total_control_points / max(base_level_count, 1)
        
        return total_levels, total_control_points, complexity, level_counts
    
    def _run_single_hffd(self, mesh_points: np.ndarray, base_dims: Tuple[int, int, int],
                        max_depth: int, subdivision_factor: int, parallel_config: ParallelConfig,
                        geometry_type: str) -> HFFDResult:
        """Run a single HFFD benchmark.
        
        Args:
            mesh_points: Input mesh points
            base_dims: Base level control dimensions
            max_depth: Maximum hierarchy depth
            subdivision_factor: Subdivision factor between levels
            parallel_config: Parallel processing configuration
            geometry_type: Type of geometry being processed
            
        Returns:
            HFFDResult with benchmark metrics
        """
        memory_before = self._measure_memory_usage()
        level_times = []
        
        try:
            start_time = time.perf_counter()
            
            # Run HFFD creation
            hffd = create_hierarchical_ffd(
                mesh_points=mesh_points,
                base_dims=base_dims,
                max_depth=max_depth,
                subdivision_factor=subdivision_factor,
                margin=0.1,
                parallel_config=parallel_config
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            memory_after = self._measure_memory_usage()
            memory_peak = max(memory_before, memory_after)
            
            # Analyze HFFD structure
            total_levels, total_control_points, complexity, level_counts = self._analyze_hffd_structure(hffd)
            
            # Estimate per-level creation times (approximation)
            if total_levels > 0:
                avg_level_time = execution_time / total_levels
                level_times = [avg_level_time] * total_levels
            
            return HFFDResult(
                mesh_size=len(mesh_points),
                geometry_type=geometry_type,
                base_dims=base_dims,
                max_depth=max_depth,
                subdivision_factor=subdivision_factor,
                parallel_enabled=parallel_config.enabled,
                max_workers=parallel_config.max_workers,
                execution_time=execution_time,
                memory_peak_mb=memory_peak,
                success=True,
                total_levels=total_levels,
                total_control_points=total_control_points,
                hierarchy_complexity=complexity,
                level_creation_times=level_times,
                level_control_counts=level_counts
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"HFFD benchmark failed: {error_msg}")
            logger.debug(traceback.format_exc())
            
            return HFFDResult(
                mesh_size=len(mesh_points),
                geometry_type=geometry_type,
                base_dims=base_dims,
                max_depth=max_depth,
                subdivision_factor=subdivision_factor,
                parallel_enabled=parallel_config.enabled,
                max_workers=parallel_config.max_workers,
                execution_time=0.0,
                memory_peak_mb=memory_before,
                success=False,
                error_message=error_msg
            )
    
    def benchmark_mesh_size_scaling(self, geometry_types: List[str] = None) -> List[HFFDResult]:
        """Benchmark HFFD performance across different mesh sizes.
        
        Args:
            geometry_types: List of geometry types to test
            
        Returns:
            List of HFFDResult objects
        """
        if geometry_types is None:
            geometry_types = ["random", "sphere", "wing"]
        
        results = []
        base_dims = (4, 4, 4)  # Standard base dimensions
        max_depth = 3  # Standard depth
        subdivision_factor = 2  # Standard subdivision
        
        for geometry_type in geometry_types:
            logger.info(f"Benchmarking HFFD mesh size scaling for {geometry_type} geometry")
            
            for mesh_size in self.config.mesh_sizes:
                logger.info(f"  Testing mesh size: {mesh_size:,} points")
                
                # Generate test mesh
                mesh_points = MeshGenerator.generate_complex_geometry(
                    mesh_size, geometry_type, seed=42
                )
                
                # Test serial execution
                parallel_config = ParallelConfig(enabled=False)
                
                for rep in range(self.config.repetitions):
                    result = self._run_single_hffd(
                        mesh_points, base_dims, max_depth, subdivision_factor,
                        parallel_config, geometry_type
                    )
                    results.append(result)
                    
                    if not result.success:
                        logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                    else:
                        logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s, "
                                  f"{result.total_levels} levels, "
                                  f"{result.total_control_points} total control points")
        
        self.results.extend(results)
        return results
    
    def benchmark_hierarchy_complexity(self, mesh_size: int = 100_000,
                                     geometry_type: str = "random") -> List[HFFDResult]:
        """Benchmark HFFD performance across different hierarchy complexities.
        
        Args:
            mesh_size: Number of mesh points to use
            geometry_type: Type of geometry to test
            
        Returns:
            List of HFFDResult objects
        """
        logger.info(f"Benchmarking HFFD hierarchy complexity with {mesh_size:,} points")
        
        results = []
        
        # Generate test mesh
        mesh_points = MeshGenerator.generate_complex_geometry(
            mesh_size, geometry_type, seed=42
        )
        
        # Test different base dimensions
        for base_dims in self.config.hffd_base_dimensions:
            logger.info(f"  Testing base dimensions: {base_dims}")
            
            # Test different depths
            for max_depth in self.config.hffd_max_depths:
                logger.info(f"    Testing max depth: {max_depth}")
                
                # Test different subdivision factors
                for subdivision_factor in self.config.hffd_subdivision_factors:
                    logger.info(f"      Testing subdivision factor: {subdivision_factor}")
                    
                    # Test serial execution
                    parallel_config = ParallelConfig(enabled=False)
                    
                    for rep in range(self.config.repetitions):
                        result = self._run_single_hffd(
                            mesh_points, base_dims, max_depth, subdivision_factor,
                            parallel_config, geometry_type
                        )
                        results.append(result)
                        
                        if not result.success:
                            logger.warning(f"        Rep {rep+1}: FAILED - {result.error_message}")
                        else:
                            logger.info(f"        Rep {rep+1}: {result.execution_time:.3f}s, "
                                      f"complexity {result.hierarchy_complexity:.1f}x")
        
        self.results.extend(results)
        return results
    
    def benchmark_parallelization(self, mesh_size: int = 1_000_000,
                                geometry_type: str = "random") -> List[HFFDResult]:
        """Benchmark HFFD parallelization performance.
        
        Args:
            mesh_size: Number of mesh points to use
            geometry_type: Type of geometry to test
            
        Returns:
            List of HFFDResult objects
        """
        logger.info(f"Benchmarking HFFD parallelization with {mesh_size:,} points")
        
        results = []
        base_dims = (4, 4, 4)  # Standard base dimensions
        max_depth = 3  # Standard depth
        subdivision_factor = 2  # Standard subdivision
        
        # Generate test mesh
        mesh_points = MeshGenerator.generate_complex_geometry(
            mesh_size, geometry_type, seed=42
        )
        
        for worker_count in self.config.worker_counts:
            logger.info(f"  Testing {worker_count} workers")
            
            # Configure parallelization
            parallel_config = ParallelConfig(
                enabled=worker_count > 1,
                method="process",
                max_workers=worker_count if worker_count > 1 else None,
                threshold=min(50_000, mesh_size // 2)  # Adaptive threshold based on mesh size
            )
            
            for rep in range(self.config.repetitions):
                result = self._run_single_hffd(
                    mesh_points, base_dims, max_depth, subdivision_factor,
                    parallel_config, geometry_type
                )
                results.append(result)
                
                if not result.success:
                    logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                else:
                    logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s")
        
        self.results.extend(results)
        return results
    
    def benchmark_depth_scaling(self, mesh_size: int = 500_000,
                              geometry_type: str = "random") -> List[HFFDResult]:
        """Benchmark HFFD performance scaling with hierarchy depth.
        
        Args:
            mesh_size: Number of mesh points to use
            geometry_type: Type of geometry to test
            
        Returns:
            List of HFFDResult objects
        """
        logger.info(f"Benchmarking HFFD depth scaling with {mesh_size:,} points")
        
        results = []
        base_dims = (4, 4, 4)  # Standard base dimensions
        subdivision_factor = 2  # Standard subdivision
        
        # Generate test mesh
        mesh_points = MeshGenerator.generate_complex_geometry(
            mesh_size, geometry_type, seed=42
        )
        
        # Test increasing depths
        max_depths = range(2, min(7, max(self.config.hffd_max_depths) + 2))
        
        for max_depth in max_depths:
            logger.info(f"  Testing max depth: {max_depth}")
            
            # Test serial execution
            parallel_config = ParallelConfig(enabled=False)
            
            for rep in range(self.config.repetitions):
                result = self._run_single_hffd(
                    mesh_points, base_dims, max_depth, subdivision_factor,
                    parallel_config, geometry_type
                )
                results.append(result)
                
                if not result.success:
                    logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                else:
                    logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s, "
                              f"{result.total_control_points} total control points")
        
        self.results.extend(results)
        return results
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive HFFD benchmark suite.
        
        Returns:
            DataFrame with all benchmark results
        """
        logger.info("Starting comprehensive HFFD benchmark suite")
        
        # 1. Mesh size scaling
        logger.info("=== Phase 1: Mesh Size Scaling ===")
        self.benchmark_mesh_size_scaling()
        
        # 2. Hierarchy complexity scaling
        logger.info("=== Phase 2: Hierarchy Complexity Scaling ===")
        self.benchmark_hierarchy_complexity()
        
        # 3. Depth scaling
        logger.info("=== Phase 3: Depth Scaling ===")
        self.benchmark_depth_scaling()
        
        # 4. Parallelization performance (only for large meshes)
        if max(self.config.mesh_sizes) >= 100_000:  # Reduced threshold for more testing
            logger.info("=== Phase 4: Parallelization Performance ===")
            # Use the largest available mesh size for parallelization testing
            test_mesh_size = max(self.config.mesh_sizes)
            self.benchmark_parallelization(mesh_size=test_mesh_size)
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Save results
        results_file = self.config.results_dir / "hffd_benchmark_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        return df
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of benchmark results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Filter successful runs
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            return {"error": "No successful benchmark runs"}
        
        summary = {
            "total_runs": len(df),
            "successful_runs": len(successful_df),
            "success_rate": len(successful_df) / len(df) * 100,
            "avg_execution_time": successful_df['execution_time'].mean(),
            "median_execution_time": successful_df['execution_time'].median(),
            "min_execution_time": successful_df['execution_time'].min(),
            "max_execution_time": successful_df['execution_time'].max(),
            "avg_memory_usage": successful_df['memory_peak_mb'].mean(),
            "avg_hierarchy_complexity": successful_df['hierarchy_complexity'].mean(),
            "geometry_types_tested": df['geometry_type'].unique().tolist(),
            "mesh_sizes_tested": sorted(df['mesh_size'].unique().tolist()),
            "base_dims_tested": df['base_dims'].unique().tolist(),
            "max_depths_tested": sorted(df['max_depth'].unique().tolist()),
            "subdivision_factors_tested": sorted(df['subdivision_factor'].unique().tolist())
        }
        
        return summary