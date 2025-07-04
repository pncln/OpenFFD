"""
Benchmark Free Form Deformation (FFD) performance.
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
from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import ParallelConfig
from .data_generator import MeshGenerator

logger = logging.getLogger(__name__)

@dataclass
class FFDResult:
    """Result of a single FFD benchmark run."""
    # Input parameters
    mesh_size: int
    geometry_type: str
    control_dim: Tuple[int, int, int]
    parallel_enabled: bool
    max_workers: Optional[int]
    
    # Performance metrics
    execution_time: float
    memory_peak_mb: float
    success: bool
    error_message: Optional[str] = None
    
    # Output metrics
    n_control_points: Optional[int] = None
    control_box_volume: Optional[float] = None
    
    # System information
    cpu_count: int = mp.cpu_count()
    timestamp: float = time.time()

class FFDBenchmark:
    """Benchmark suite for FFD algorithm performance."""
    
    def __init__(self, config):
        """Initialize FFD benchmark.
        
        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        self.results: List[FFDResult] = []
        
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
    
    def _run_single_ffd(self, mesh_points: np.ndarray, control_dim: Tuple[int, int, int],
                       parallel_config: ParallelConfig, geometry_type: str) -> FFDResult:
        """Run a single FFD benchmark.
        
        Args:
            mesh_points: Input mesh points
            control_dim: Control box dimensions
            parallel_config: Parallel processing configuration
            geometry_type: Type of geometry being processed
            
        Returns:
            FFDResult with benchmark metrics
        """
        memory_before = self._measure_memory_usage()
        
        try:
            start_time = time.perf_counter()
            
            # Run FFD creation
            control_points, bbox = create_ffd_box(
                mesh_points=mesh_points,
                control_dim=control_dim,
                margin=0.1,
                custom_dims=None,
                parallel_config=parallel_config
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            memory_after = self._measure_memory_usage()
            memory_peak = max(memory_before, memory_after)
            
            # Calculate output metrics
            n_control_points = control_points.shape[0] if control_points is not None else 0
            
            # Calculate control box volume
            if bbox is not None:
                min_coords, max_coords = bbox
                volume = np.prod(max_coords - min_coords)
            else:
                volume = 0.0
            
            return FFDResult(
                mesh_size=len(mesh_points),
                geometry_type=geometry_type,
                control_dim=control_dim,
                parallel_enabled=parallel_config.enabled,
                max_workers=parallel_config.max_workers,
                execution_time=execution_time,
                memory_peak_mb=memory_peak,
                success=True,
                n_control_points=n_control_points,
                control_box_volume=volume
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"FFD benchmark failed: {error_msg}")
            logger.debug(traceback.format_exc())
            
            return FFDResult(
                mesh_size=len(mesh_points),
                geometry_type=geometry_type,
                control_dim=control_dim,
                parallel_enabled=parallel_config.enabled,
                max_workers=parallel_config.max_workers,
                execution_time=0.0,
                memory_peak_mb=memory_before,
                success=False,
                error_message=error_msg
            )
    
    def benchmark_mesh_size_scaling(self, geometry_types: List[str] = None) -> List[FFDResult]:
        """Benchmark FFD performance across different mesh sizes.
        
        Args:
            geometry_types: List of geometry types to test
            
        Returns:
            List of FFDResult objects
        """
        if geometry_types is None:
            geometry_types = ["random", "sphere", "wing"]
        
        results = []
        control_dim = (8, 8, 8)  # Standard control dimension
        
        for geometry_type in geometry_types:
            logger.info(f"Benchmarking FFD mesh size scaling for {geometry_type} geometry")
            
            for mesh_size in self.config.mesh_sizes:
                logger.info(f"  Testing mesh size: {mesh_size:,} points")
                
                # Generate test mesh
                mesh_points = MeshGenerator.generate_complex_geometry(
                    mesh_size, geometry_type, seed=42
                )
                
                # Test serial execution
                parallel_config = ParallelConfig(enabled=False)
                
                for rep in range(self.config.repetitions):
                    result = self._run_single_ffd(
                        mesh_points, control_dim, parallel_config, geometry_type
                    )
                    results.append(result)
                    
                    if not result.success:
                        logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                    else:
                        logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s")
        
        self.results.extend(results)
        return results
    
    def benchmark_control_complexity(self, mesh_size: int = 100_000, 
                                   geometry_type: str = "random") -> List[FFDResult]:
        """Benchmark FFD performance across different control box complexities.
        
        Args:
            mesh_size: Number of mesh points to use
            geometry_type: Type of geometry to test
            
        Returns:
            List of FFDResult objects
        """
        logger.info(f"Benchmarking FFD control complexity with {mesh_size:,} points")
        
        results = []
        
        # Generate test mesh
        mesh_points = MeshGenerator.generate_complex_geometry(
            mesh_size, geometry_type, seed=42
        )
        
        for control_dim in self.config.ffd_dimensions:
            logger.info(f"  Testing control dimensions: {control_dim}")
            
            # Test serial execution
            parallel_config = ParallelConfig(enabled=False)
            
            for rep in range(self.config.repetitions):
                result = self._run_single_ffd(
                    mesh_points, control_dim, parallel_config, geometry_type
                )
                results.append(result)
                
                if not result.success:
                    logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                else:
                    logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s, "
                              f"{result.n_control_points} control points")
        
        self.results.extend(results)
        return results
    
    def benchmark_parallelization(self, mesh_size: int = 1_000_000,
                                geometry_type: str = "random") -> List[FFDResult]:
        """Benchmark FFD parallelization performance.
        
        Args:
            mesh_size: Number of mesh points to use
            geometry_type: Type of geometry to test
            
        Returns:
            List of FFDResult objects
        """
        logger.info(f"Benchmarking FFD parallelization with {mesh_size:,} points")
        
        results = []
        control_dim = (8, 8, 8)  # Standard control dimension
        
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
                result = self._run_single_ffd(
                    mesh_points, control_dim, parallel_config, geometry_type
                )
                results.append(result)
                
                if not result.success:
                    logger.warning(f"    Rep {rep+1}: FAILED - {result.error_message}")
                else:
                    logger.info(f"    Rep {rep+1}: {result.execution_time:.3f}s")
        
        self.results.extend(results)
        return results
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive FFD benchmark suite.
        
        Returns:
            DataFrame with all benchmark results
        """
        logger.info("Starting comprehensive FFD benchmark suite")
        
        # 1. Mesh size scaling
        logger.info("=== Phase 1: Mesh Size Scaling ===")
        self.benchmark_mesh_size_scaling()
        
        # 2. Control complexity scaling
        logger.info("=== Phase 2: Control Complexity Scaling ===")
        self.benchmark_control_complexity()
        
        # 3. Parallelization performance (only for large meshes)
        if max(self.config.mesh_sizes) >= 100_000:  # Reduced threshold for more testing
            logger.info("=== Phase 3: Parallelization Performance ===")
            # Use the largest available mesh size for parallelization testing
            test_mesh_size = max(self.config.mesh_sizes)
            self.benchmark_parallelization(mesh_size=test_mesh_size)
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Save results
        results_file = self.config.results_dir / "ffd_benchmark_results.csv"
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
            "geometry_types_tested": df['geometry_type'].unique().tolist(),
            "mesh_sizes_tested": sorted(df['mesh_size'].unique().tolist()),
            "control_dims_tested": df['control_dim'].unique().tolist()
        }
        
        return summary