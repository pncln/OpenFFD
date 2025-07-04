# OpenFFD Benchmark Suite

A comprehensive benchmarking framework for evaluating the performance of Free Form Deformation (FFD) and Hierarchical FFD (HFFD) algorithms. This suite generates publication-ready figures suitable for academic research papers.

## Features

- **Comprehensive Performance Evaluation**: Tests FFD and HFFD algorithms across multiple dimensions
- **Publication-Ready Figures**: Generates high-quality figures suitable for academic papers
- **Multiple Geometry Types**: Tests on various geometric shapes (random, sphere, cylinder, wing profiles)
- **Parallelization Analysis**: Evaluates parallel processing performance and efficiency
- **Statistical Analysis**: Includes error bars, trend fitting, and statistical measures
- **Configurable Test Suites**: Quick, paper, and comprehensive benchmark configurations

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install numpy pandas matplotlib seaborn scipy psutil
```

### Running Benchmarks

```bash
# Quick benchmark (2-5 minutes)
python benchmarks/run_benchmarks.py --config quick

# Paper-quality benchmark (10-30 minutes)
python benchmarks/run_benchmarks.py --config paper

# Comprehensive benchmark (1-3 hours)
python benchmarks/run_benchmarks.py --config full
```

## Configuration Options

### Benchmark Configurations

1. **Quick** (`--config quick`):
   - 3 mesh sizes: 1K, 10K, 100K points
   - 2 FFD dimensions, 2 HFFD configurations
   - 2 repetitions per test
   - Execution time: 2-5 minutes

2. **Paper** (`--config paper`):
   - 5 mesh sizes: 10K to 1M points
   - 3 FFD dimensions, 3 HFFD configurations
   - 3 repetitions per test
   - Execution time: 10-30 minutes

3. **Full** (`--config full`):
   - 10 mesh sizes: 1K to 2M points
   - 6 FFD dimensions, 4 HFFD configurations
   - 5 repetitions per test
   - Execution time: 1-3 hours

### Command Line Options

```bash
python benchmarks/run_benchmarks.py [OPTIONS]

Options:
  --config {quick,paper,full}    Benchmark configuration (default: paper)
  --output-dir DIR               Override output directory
  --force-regenerate             Regenerate existing datasets
  --log-level {DEBUG,INFO,WARN,ERROR}  Logging level
  --save-metadata                Save benchmark metadata to JSON
  --help                         Show help message
```

## Output Structure

```
benchmarks/
├── data/                    # Generated mesh datasets
│   ├── random_10000_points.npz
│   ├── sphere_10000_points.npz
│   └── ...
├── results/                 # Benchmark results (CSV)
│   ├── ffd_benchmark_results.csv
│   ├── hffd_benchmark_results.csv
│   └── benchmark_metadata.json
└── figures/                 # Publication-ready figures
    ├── mesh_size_scaling.pdf
    ├── control_complexity_scaling.pdf
    ├── parallelization_performance.pdf
    ├── memory_usage.pdf
    └── hffd_hierarchy_analysis.pdf
```

## Generated Figures

All figures are created as separate plots without titles for maximum flexibility in academic publications. Titles should be provided in figure captions.

### 1. FFD Mesh Size Scaling
- **File**: `ffd_mesh_scaling.pdf`
- **Content**: FFD execution time vs mesh size for different geometry types
- **Analysis**: Power law fitting, scaling behavior analysis
- **Usage**: Demonstrates FFD computational complexity

### 2. HFFD Mesh Size Scaling
- **File**: `hffd_mesh_scaling.pdf`
- **Content**: HFFD execution time vs mesh size for different geometry types
- **Analysis**: Power law fitting, hierarchical scaling behavior
- **Usage**: Demonstrates HFFD computational complexity

### 3. FFD Control Complexity
- **File**: `ffd_control_complexity.pdf`
- **Content**: FFD performance vs control box complexity
- **Analysis**: Impact of control point density on execution time
- **Usage**: Shows FFD scaling with control complexity

### 4. HFFD Hierarchy Complexity
- **File**: `hffd_hierarchy_complexity.pdf`
- **Content**: HFFD performance vs hierarchy complexity
- **Analysis**: Impact of hierarchical structure on execution time
- **Usage**: Shows HFFD scaling with hierarchy complexity

### 5. Parallelization Speedup
- **File**: `parallelization_speedup.pdf`
- **Content**: Speedup analysis for FFD and HFFD algorithms
- **Analysis**: Speedup vs worker count with ideal speedup comparison
- **Usage**: Evaluates parallel processing effectiveness

### 6. Parallelization Efficiency
- **File**: `parallelization_efficiency.pdf`
- **Content**: Efficiency analysis for parallel processing
- **Analysis**: Efficiency percentage vs worker count
- **Usage**: Shows parallel processing overhead

### 7. Memory Usage
- **File**: `memory_usage.pdf`
- **Content**: Memory consumption vs mesh size for both algorithms
- **Analysis**: Peak memory usage comparison
- **Usage**: Memory scalability assessment

### 8. HFFD Depth Scaling
- **File**: `hffd_depth_scaling.pdf`
- **Content**: HFFD performance vs hierarchy depth
- **Analysis**: Impact of hierarchy depth on execution time
- **Usage**: Detailed HFFD depth analysis

## Benchmark Metrics

### Performance Metrics
- **Execution Time**: Wall-clock time for algorithm completion
- **Memory Usage**: Peak memory consumption during execution
- **Success Rate**: Percentage of successful benchmark runs
- **Speedup**: Parallel performance improvement factor
- **Efficiency**: Parallel efficiency percentage

### Algorithm Analysis
- **Power Law Fitting**: Computational complexity analysis (O(n^k))
- **Statistical Analysis**: Mean, median, standard deviation
- **Trend Analysis**: Performance scaling characteristics
- **Error Analysis**: Error bars and confidence intervals

## Academic Usage

### For Research Papers

The generated figures are designed for direct inclusion in academic papers:

- **High Resolution**: 300 DPI for print quality
- **Professional Formatting**: Consistent fonts, colors, and styling
- **Statistical Rigor**: Error bars, trend fitting, R² values
- **Colorblind-Friendly**: Accessible color schemes
- **Multiple Formats**: PDF (vector) and PNG (raster) outputs

### Citation Data

The benchmark results include detailed metadata suitable for academic citation:

```json
{
  "timestamp": "2024-XX-XX",
  "system_info": {...},
  "algorithm_versions": {...},
  "test_configurations": {...},
  "statistical_summary": {...}
}
```

### Reproducibility

All benchmarks are designed for reproducibility:
- Fixed random seeds for consistent results
- Detailed configuration logging
- Version tracking for algorithms and dependencies
- Complete parameter documentation

## Customization

### Adding New Geometries

```python
# In data_generator.py
def generate_custom_geometry(n_points: int, **params) -> np.ndarray:
    # Your custom geometry generation
    return points

# Register in MeshGenerator.generate_complex_geometry()
```

### Adding New Metrics

```python
# In ffd_benchmark.py or hffd_benchmark.py
@dataclass
class CustomResult:
    # Add your custom metrics
    custom_metric: float = 0.0
    
def _run_single_benchmark(self, ...):
    # Calculate your custom metrics
    custom_value = calculate_custom_metric(...)
    return CustomResult(..., custom_metric=custom_value)
```

### Custom Visualizations

```python
# In visualization.py
def plot_custom_analysis(self, df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    # Your custom plotting code
    return fig
```

## Performance Optimization

The benchmark suite includes several optimizations:

1. **Parallel Dataset Generation**: Multiple geometries generated concurrently
2. **Caching**: Reuse existing datasets unless forced regeneration
3. **Memory Monitoring**: Track and report memory usage
4. **Timeout Protection**: Prevent infinite execution on problematic inputs
5. **Error Recovery**: Graceful handling of failed benchmark runs

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce mesh sizes or use smaller configurations
2. **Timeout Errors**: Increase timeout values in configuration
3. **Import Errors**: Ensure OpenFFD is properly installed
4. **Permission Errors**: Check write permissions for output directories

### Debug Mode

```bash
# Enable debug logging
python benchmarks/run_benchmarks.py --config quick --log-level DEBUG
```

### Verbose Output

The benchmark suite provides detailed logging:
- Progress indicators for long-running operations
- Performance statistics for each test
- Error reporting with stack traces
- Memory usage monitoring

## Contributing

To contribute new benchmarks or improvements:

1. Follow the existing code structure
2. Add appropriate documentation
3. Include statistical analysis
4. Ensure publication-quality output
5. Test with all configuration levels

## Example Results

Typical benchmark results on a modern system:

- **FFD (100K points)**: ~0.05 seconds
- **HFFD (100K points, depth 3)**: ~0.15 seconds
- **Memory Usage**: 50-200 MB for typical workloads
- **Parallel Efficiency**: 60-80% for large datasets

Results vary significantly based on:
- Hardware specifications (CPU, memory)
- Problem size (mesh points, control complexity)
- Algorithm parameters (hierarchy depth, subdivision)
- System load and background processes