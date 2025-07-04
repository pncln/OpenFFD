"""
Configuration settings for benchmarking FFD and HFFD algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters."""
    
    # Mesh size parameters (number of points)
    mesh_sizes: List[int] = None
    
    # FFD control box dimensions
    ffd_dimensions: List[Tuple[int, int, int]] = None
    
    # HFFD parameters
    hffd_base_dimensions: List[Tuple[int, int, int]] = None
    hffd_max_depths: List[int] = None
    hffd_subdivision_factors: List[int] = None
    
    # Performance parameters
    repetitions: int = 5
    timeout_seconds: int = 300
    
    # Parallelization testing
    worker_counts: List[int] = None
    
    # Output settings
    results_dir: Path = Path("benchmarks/results")
    figures_dir: Path = Path("benchmarks/figures")
    data_dir: Path = Path("benchmarks/data")
    
    # Figure quality settings
    figure_dpi: int = 300
    figure_format: str = "pdf"
    figure_style: str = "seaborn-v0_8-paper"
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.mesh_sizes is None:
            self.mesh_sizes = [
                1_000, 5_000, 10_000, 25_000, 50_000, 
                100_000, 250_000, 500_000, 1_000_000, 2_000_000
            ]
        
        if self.ffd_dimensions is None:
            self.ffd_dimensions = [
                (4, 4, 4), (6, 6, 6), (8, 8, 8), 
                (10, 10, 10), (12, 12, 12), (16, 16, 16)
            ]
        
        if self.hffd_base_dimensions is None:
            self.hffd_base_dimensions = [
                (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)
            ]
        
        if self.hffd_max_depths is None:
            self.hffd_max_depths = [2, 3, 4, 5]
        
        if self.hffd_subdivision_factors is None:
            self.hffd_subdivision_factors = [2, 3]
        
        if self.worker_counts is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            self.worker_counts = [1, 2, 4, min(8, cpu_count), min(16, cpu_count)]
        
        # Ensure directories exist
        for directory in [self.results_dir, self.figures_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()

# Academic paper configuration (fewer points for faster execution)
PAPER_CONFIG = BenchmarkConfig(
    mesh_sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    ffd_dimensions=[(4, 4, 4), (8, 8, 8), (12, 12, 12)],
    hffd_base_dimensions=[(3, 3, 3), (4, 4, 4), (5, 5, 5)],
    hffd_max_depths=[2, 3, 4],
    repetitions=3,
    timeout_seconds=180
)

# Quick test configuration
QUICK_CONFIG = BenchmarkConfig(
    mesh_sizes=[1_000, 10_000, 100_000, 500_000],  # Added 500K for parallelization testing
    ffd_dimensions=[(4, 4, 4), (8, 8, 8)],
    hffd_base_dimensions=[(3, 3, 3), (4, 4, 4)],
    hffd_max_depths=[2, 3],
    worker_counts=[1, 2, 4],  # Reduced worker counts for faster testing
    repetitions=2,
    timeout_seconds=120  # Increased timeout for larger mesh
)