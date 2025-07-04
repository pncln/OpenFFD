#!/usr/bin/env python3
"""
Main script to run comprehensive FFD and HFFD benchmarks.

This script runs all benchmarks and generates publication-ready figures
suitable for academic research papers.

Usage:
    python benchmarks/run_benchmarks.py [--config {quick,paper,full}] [--output-dir DIR]
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import benchmark modules
from benchmarks.benchmark_config import DEFAULT_CONFIG, PAPER_CONFIG, QUICK_CONFIG
from benchmarks.data_generator import generate_benchmark_datasets
from benchmarks.ffd_benchmark import FFDBenchmark
from benchmarks.hffd_benchmark import HFFDBenchmark
from benchmarks.visualization import create_publication_figures

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log')
        ]
    )

def print_banner():
    """Print benchmark suite banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                  OpenFFD Benchmark Suite                     ║
    ║                                                              ║
    ║  Comprehensive performance evaluation of Free Form          ║
    ║  Deformation (FFD) and Hierarchical FFD (HFFD) algorithms   ║
    ║                                                              ║
    ║  Generates publication-ready figures for academic papers     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_comprehensive_benchmark(config, force_regenerate: bool = False) -> Dict[str, Any]:
    """Run comprehensive benchmark suite.
    
    Args:
        config: BenchmarkConfig instance
        force_regenerate: Whether to regenerate existing datasets
        
    Returns:
        Dictionary with benchmark results and metadata
    """
    logger = logging.getLogger(__name__)
    
    # Phase 1: Generate benchmark datasets
    logger.info("="*60)
    logger.info("PHASE 1: GENERATING BENCHMARK DATASETS")
    logger.info("="*60)
    
    start_time = time.time()
    datasets = generate_benchmark_datasets(config, force_regenerate)
    dataset_time = time.time() - start_time
    
    logger.info(f"Generated {len(datasets)} datasets in {dataset_time:.2f} seconds")
    
    # Phase 2: Run FFD benchmarks
    logger.info("="*60)
    logger.info("PHASE 2: RUNNING FFD BENCHMARKS")
    logger.info("="*60)
    
    ffd_benchmark = FFDBenchmark(config)
    ffd_start = time.time()
    ffd_results = ffd_benchmark.run_comprehensive_benchmark()
    ffd_time = time.time() - ffd_start
    
    logger.info(f"FFD benchmarks completed in {ffd_time:.2f} seconds")
    logger.info(f"FFD results: {len(ffd_results)} runs")
    
    # Phase 3: Run HFFD benchmarks
    logger.info("="*60)
    logger.info("PHASE 3: RUNNING HFFD BENCHMARKS")
    logger.info("="*60)
    
    hffd_benchmark = HFFDBenchmark(config)
    hffd_start = time.time()
    hffd_results = hffd_benchmark.run_comprehensive_benchmark()
    hffd_time = time.time() - hffd_start
    
    logger.info(f"HFFD benchmarks completed in {hffd_time:.2f} seconds")
    logger.info(f"HFFD results: {len(hffd_results)} runs")
    
    # Phase 4: Generate publication figures
    logger.info("="*60)
    logger.info("PHASE 4: GENERATING PUBLICATION FIGURES")
    logger.info("="*60)
    
    figures_start = time.time()
    saved_figures = create_publication_figures(ffd_results, hffd_results, config)
    figures_time = time.time() - figures_start
    
    logger.info(f"Publication figures generated in {figures_time:.2f} seconds")
    logger.info(f"Saved {len(saved_figures)} figure sets")
    
    # Generate summary statistics
    ffd_summary = ffd_benchmark.get_summary_statistics()
    hffd_summary = hffd_benchmark.get_summary_statistics()
    
    total_time = time.time() - start_time
    
    return {
        'config': config,
        'datasets': datasets,
        'ffd_results': ffd_results,
        'hffd_results': hffd_results,
        'ffd_summary': ffd_summary,
        'hffd_summary': hffd_summary,
        'saved_figures': saved_figures,
        'timing': {
            'total_time': total_time,
            'dataset_generation': dataset_time,
            'ffd_benchmarks': ffd_time,
            'hffd_benchmarks': hffd_time,
            'figure_generation': figures_time
        }
    }

def print_benchmark_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive benchmark summary."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Timing summary
    timing = results['timing']
    print(f"\n📊 TIMING SUMMARY:")
    print(f"   Total execution time: {timing['total_time']:.1f} seconds ({timing['total_time']/60:.1f} minutes)")
    print(f"   Dataset generation:   {timing['dataset_generation']:.1f} seconds")
    print(f"   FFD benchmarks:       {timing['ffd_benchmarks']:.1f} seconds")
    print(f"   HFFD benchmarks:      {timing['hffd_benchmarks']:.1f} seconds")
    print(f"   Figure generation:    {timing['figure_generation']:.1f} seconds")
    
    # FFD summary
    ffd_summary = results['ffd_summary']
    if 'error' not in ffd_summary:
        print(f"\n🔷 FFD BENCHMARK RESULTS:")
        print(f"   Total runs:           {ffd_summary['total_runs']}")
        print(f"   Successful runs:      {ffd_summary['successful_runs']}")
        print(f"   Success rate:         {ffd_summary['success_rate']:.1f}%")
        print(f"   Average time:         {ffd_summary['avg_execution_time']:.3f} seconds")
        print(f"   Median time:          {ffd_summary['median_execution_time']:.3f} seconds")
        print(f"   Time range:           {ffd_summary['min_execution_time']:.3f} - {ffd_summary['max_execution_time']:.3f} seconds")
        print(f"   Average memory:       {ffd_summary['avg_memory_usage']:.1f} MB")
        print(f"   Mesh sizes tested:    {len(ffd_summary['mesh_sizes_tested'])} different sizes")
        print(f"   Geometry types:       {', '.join(ffd_summary['geometry_types_tested'])}")
    else:
        print(f"\n❌ FFD BENCHMARKS: {ffd_summary['error']}")
    
    # HFFD summary
    hffd_summary = results['hffd_summary']
    if 'error' not in hffd_summary:
        print(f"\n🔶 HFFD BENCHMARK RESULTS:")
        print(f"   Total runs:           {hffd_summary['total_runs']}")
        print(f"   Successful runs:      {hffd_summary['successful_runs']}")
        print(f"   Success rate:         {hffd_summary['success_rate']:.1f}%")
        print(f"   Average time:         {hffd_summary['avg_execution_time']:.3f} seconds")
        print(f"   Median time:          {hffd_summary['median_execution_time']:.3f} seconds")
        print(f"   Time range:           {hffd_summary['min_execution_time']:.3f} - {hffd_summary['max_execution_time']:.3f} seconds")
        print(f"   Average memory:       {hffd_summary['avg_memory_usage']:.1f} MB")
        print(f"   Avg hierarchy complexity: {hffd_summary['avg_hierarchy_complexity']:.1f}x")
        print(f"   Mesh sizes tested:    {len(hffd_summary['mesh_sizes_tested'])} different sizes")
        print(f"   Depths tested:        {hffd_summary['max_depths_tested']}")
    else:
        print(f"\n❌ HFFD BENCHMARKS: {hffd_summary['error']}")
    
    # Figure summary
    saved_figures = results['saved_figures']
    print(f"\n📈 GENERATED FIGURES:")
    for figure_name, file_paths in saved_figures.items():
        print(f"   {figure_name}:")
        for path in file_paths:
            print(f"     {path}")
    
    # Output directories
    config = results['config']
    print(f"\n📁 OUTPUT DIRECTORIES:")
    print(f"   Results:  {config.results_dir}")
    print(f"   Figures:  {config.figures_dir}")
    print(f"   Data:     {config.data_dir}")
    
    print("\n✅ Benchmark suite completed successfully!")
    print("📊 Publication-ready figures are available in the figures directory.")
    print("📋 Detailed results are saved in CSV format in the results directory.")
    print("="*80)

def save_benchmark_metadata(results: Dict[str, Any], output_file: Path) -> None:
    """Save benchmark metadata to JSON file."""
    import json
    
    # Prepare metadata (exclude large DataFrames)
    metadata = {
        'timestamp': time.time(),
        'config': {
            'mesh_sizes': results['config'].mesh_sizes,
            'ffd_dimensions': results['config'].ffd_dimensions,
            'hffd_base_dimensions': results['config'].hffd_base_dimensions,
            'hffd_max_depths': results['config'].hffd_max_depths,
            'repetitions': results['config'].repetitions,
            'worker_counts': results['config'].worker_counts
        },
        'summary': {
            'ffd_summary': results['ffd_summary'],
            'hffd_summary': results['hffd_summary'],
            'timing': results['timing']
        },
        'datasets': {str(k): str(v) for k, v in results['datasets'].items()},
        'figures': {k: [str(p) for p in paths] for k, paths in results['saved_figures'].items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Benchmark metadata saved to: {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive FFD and HFFD benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration options:
  quick  - Fast benchmark with minimal test cases (2-5 minutes)
  paper  - Medium benchmark suitable for academic papers (10-30 minutes)  
  full   - Comprehensive benchmark with all test cases (1-3 hours)

Example usage:
  python benchmarks/run_benchmarks.py --config paper
  python benchmarks/run_benchmarks.py --config quick --force-regenerate
        """
    )
    
    parser.add_argument(
        '--config', 
        choices=['quick', 'paper', 'full'], 
        default='paper',
        help='Benchmark configuration to use (default: paper)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory (default: benchmarks/)'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of existing datasets'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--save-metadata',
        action='store_true',
        help='Save benchmark metadata to JSON file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print banner
    print_banner()
    
    # Select configuration
    config_map = {
        'quick': QUICK_CONFIG,
        'paper': PAPER_CONFIG,
        'full': DEFAULT_CONFIG
    }
    
    config = config_map[args.config]
    
    # Override output directory if specified
    if args.output_dir:
        base_dir = args.output_dir
        config.results_dir = base_dir / "results"
        config.figures_dir = base_dir / "figures"
        config.data_dir = base_dir / "data"
        
        # Ensure directories exist
        for directory in [config.results_dir, config.figures_dir, config.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting benchmark with '{args.config}' configuration")
    logger.info(f"Output directories: {config.results_dir.parent}")
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark(config, args.force_regenerate)
        
        # Print summary
        print_benchmark_summary(results)
        
        # Save metadata if requested
        if args.save_metadata:
            metadata_file = config.results_dir / "benchmark_metadata.json"
            save_benchmark_metadata(results, metadata_file)
        
        # Return success
        return 0
        
    except KeyboardInterrupt:
        logger.error("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug("Exception details:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())