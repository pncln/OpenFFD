# OpenFFD Benchmark Suite

This directory contains a comprehensive benchmark suite for OpenFFD, designed for academic performance analysis and publication-ready results. The benchmarks evaluate OpenFFD's performance across multiple dimensions including parallelization scalability, mesh complexity handling, hierarchical FFD capabilities, and visualization performance.

## Overview

The benchmark suite consists of four main benchmark modules and supporting utilities:

### Core Benchmark Modules

1. **Parallelization Scalability** (`parallelization_scalability.py`)
   - Evaluates parallel processing performance across varying mesh sizes and worker counts
   - Analyzes strong and weak scaling efficiency
   - Measures speedup, efficiency, and throughput metrics

2. **Mesh Complexity Scaling** (`mesh_complexity_scaling.py`)
   - Tests performance across different mesh sizes (10³ to 10⁷ points)
   - Analyzes various geometry types and aspect ratios
   - Evaluates control grid resolution effects
   - Determines computational complexity classes

3. **Hierarchical FFD Performance** (`hierarchical_ffd_benchmark.py`)
   - Benchmarks hierarchical FFD creation and manipulation
   - Tests different hierarchy depths and subdivision factors
   - Analyzes memory usage and control point scaling
   - Evaluates deformation performance

4. **Visualization Performance** (`visualization_performance.py`)
   - Tests visualization subsystem performance
   - Evaluates different rendering modes and quality levels
   - Analyzes parallel visualization processing
   - Measures memory efficiency for large datasets

### Supporting Utilities

- **Academic Plotting Utils** (`academic_plotting_utils.py`)
  - Publication-quality plotting functions
  - Academic styling and formatting
  - Statistical analysis visualizations
  - Multi-metric comparison tools

- **Benchmark Runner** (`benchmark_runner.py`)
  - Orchestrates all benchmark modules
  - Generates comprehensive reports
  - Creates publication-ready figures
  - Provides comparative analysis across benchmarks

## Quick Start

### Running Individual Benchmarks

Each benchmark can be run independently:

```bash
# Parallelization scalability
python parallelization_scalability.py --output-dir results/parallel --quick

# Mesh complexity scaling  
python mesh_complexity_scaling.py --output-dir results/mesh --quick

# Hierarchical FFD performance
python hierarchical_ffd_benchmark.py --output-dir results/hierarchical --quick

# Visualization performance
python visualization_performance.py --output-dir results/visualization --quick
```

### Running Complete Benchmark Suite

For comprehensive analysis, use the benchmark runner:

```bash
# Full benchmark suite (recommended for publication)
python benchmark_runner.py --output-dir comprehensive_results

# Quick benchmark for testing
python benchmark_runner.py --output-dir test_results --quick

# Limited benchmark for development
python benchmark_runner.py --output-dir dev_results --limited

# Run specific benchmarks only
python benchmark_runner.py --benchmarks parallelization mesh_complexity
```

## Configuration Options

### Performance Levels

- **Full Benchmark**: Complete analysis with all test cases (recommended for publication)
- **Quick Benchmark**: Reduced test cases for faster execution (~2-4 hours)
- **Limited Benchmark**: Minimal test cases for development/testing (~30 minutes)

### Command Line Options

```bash
# Benchmark Runner Options
--output-dir DIR          # Output directory for results
--benchmarks LIST         # Specific benchmarks to run
--quick                   # Run quick benchmark
--limited                 # Run limited benchmark for testing
--sequential              # Run benchmarks sequentially (not parallel)
--no-figures              # Skip publication figure generation

# Individual Benchmark Options
--quick                   # Reduced test cases
--max-workers N           # Maximum number of parallel workers
--max-mesh-size N         # Maximum mesh size to test
--max-depth N             # Maximum hierarchy depth (hierarchical FFD)
```

## Output Structure

Each benchmark generates organized output:

```
benchmark_results/
├── raw_results.csv           # Raw benchmark data
├── statistics.csv            # Statistical summaries
├── analysis.json            # Computational analysis
├── report.md                # Comprehensive report
├── *.png/*.pdf              # Publication figures
└── benchmark.log            # Execution log
```

Comprehensive results include:

```
comprehensive_benchmark_results/
├── parallelization/         # Parallelization benchmark results
├── mesh_complexity/         # Mesh complexity results
├── hierarchical_ffd/        # Hierarchical FFD results
├── visualization/           # Visualization results
├── publication_figures/     # Combined publication figures
├── comprehensive_analysis.json
├── comprehensive_benchmark_report.md
└── benchmark_suite.log
```

## Publication-Ready Outputs

The benchmark suite generates publication-ready materials:

### Figures
- High-resolution PNG, PDF, and EPS formats
- Academic typography (Times Roman, proper axis labels)
- Consistent color schemes and styling
- Statistical error bars and confidence intervals

### Reports
- Comprehensive Markdown reports
- Statistical significance analysis
- Performance recommendations
- System requirements analysis

### Data
- Raw CSV data for further analysis
- Statistical summaries
- JSON metadata with analysis parameters

## Academic Usage

This benchmark suite is designed for academic research and publication:

### For AIAA Journal Submission
- All figures follow AIAA formatting guidelines
- Professional typography and styling
- Publication-quality resolution (300 DPI)
- Comprehensive statistical analysis

### Performance Metrics
- **Execution Time**: Wall-clock time for operations
- **Speedup**: Parallel performance relative to sequential
- **Efficiency**: Speedup divided by number of workers
- **Throughput**: Operations per second
- **Memory Usage**: Peak memory consumption
- **Computational Complexity**: Scaling exponent analysis

### Statistical Analysis
- Multiple repetitions for statistical significance
- Mean, standard deviation, and confidence intervals
- Correlation analysis and R² values
- Complexity classification (linear, quadratic, etc.)

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8 GB RAM
- 4 CPU cores
- 10 GB free disk space

### Recommended for Full Benchmarks
- Python 3.9+
- 32 GB RAM
- 16+ CPU cores
- 100 GB free disk space
- Dedicated GPU for visualization benchmarks

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy
pip install pyvista  # For enhanced visualization
pip install psutil   # For memory monitoring
```

## Customization

### Adding Custom Benchmarks

1. Create a new benchmark class following the pattern:
```python
class CustomBenchmark:
    def __init__(self, output_dir):
        # Initialize benchmark
        
    def run_full_benchmark(self):
        # Main benchmark execution
        
    def generate_report(self):
        # Generate analysis report
```

2. Add to benchmark runner:
```python
self.benchmarks['custom'] = CustomBenchmark(output_dir)
```

### Custom Test Cases

Modify benchmark parameters in individual modules:
```python
# Example: Custom mesh sizes
benchmark.mesh_sizes = [1000, 10000, 100000, 1000000]

# Example: Custom worker counts  
benchmark.worker_counts = [1, 2, 4, 8, 16, 32]
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce mesh sizes or enable disk caching
2. **Import Errors**: Ensure all dependencies are installed
3. **Long Execution**: Use `--quick` or `--limited` options
4. **Permission Errors**: Check write permissions in output directory

### Performance Tips

1. **For Large Benchmarks**: Use SSD storage and close other applications
2. **For Parallel Tests**: Ensure sufficient CPU cores and memory
3. **For Visualization**: Install PyVista and ensure graphics drivers are updated

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To contribute new benchmarks or improvements:

1. Follow the existing code structure and style
2. Include comprehensive documentation
3. Add appropriate test cases
4. Ensure publication-quality output formatting

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{openffd_benchmarks,
  author    = {OpenFFD Development Team},
  title     = {OpenFFD Comprehensive Performance Benchmark Suite},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/pncln/openffd}
}
```

## License

This benchmark suite is distributed under the same license as OpenFFD (MIT License).