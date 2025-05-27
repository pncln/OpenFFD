# OpenFFD Parallel Processing Guide

OpenFFD now supports parallel processing for both FFD box creation and visualization, which can significantly improve performance when working with large meshes.

## Basic Usage

By default, parallel processing is disabled to maintain backward compatibility and ensure optimal performance for small meshes. To enable parallel processing:

```bash
python3 -m openffd /path/to/mesh.msh --parallel
```

## Command-line Options

OpenFFD provides several command-line options to control parallel processing:

### Core Parallel Processing Options

- `--parallel`: Enable parallel processing for large meshes and visualization (default: disabled)
- `--no-parallel`: Disable parallel processing completely
- `--parallel-method {process,thread}`: Method for parallelization (default: process)
  - `process`: Uses multiprocessing (faster but more memory usage)
  - `thread`: Uses threading (lower memory usage but slower for CPU-bound tasks)
- `--parallel-workers N`: Number of worker processes/threads (default: auto-detect based on CPU count)
- `--parallel-chunk-size N`: Size of data chunks for parallel processing (default: auto-calculate)
- `--parallel-threshold N`: Minimum data size to trigger parallelization (default: 100000 points)

### Visualization-specific Options

- `--parallel-viz`: Enable parallel processing for visualization only
- `--no-parallel-viz`: Disable parallel processing for visualization even if parallel is enabled

## Examples

### Basic FFD Generation with Parallel Processing

```bash
# Generate FFD box with parallel processing
python3 -m openffd 14.msh --dims 10 10 10 --parallel
```

### Visualization with Parallel Processing

```bash
# Generate FFD box and visualize with parallel processing
python3 -m openffd 14.msh --dims 10 10 10 --plot --parallel
```

### Fine-tuning Parallel Processing

```bash
# Specify number of worker processes and chunk size
python3 -m openffd 14.msh --dims 10 10 10 --parallel --parallel-workers 4 --parallel-chunk-size 50000
```

### Parallel Processing for Visualization Only

```bash
# Enable parallel processing only for visualization
python3 -m openffd 14.msh --dims 10 10 10 --plot --parallel-viz
```

### Original Command with Parallel Processing

Here's an example of using parallel processing with the same parameters as before:

```bash
# Without parallel processing:
python3 -m openffd /Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh --y-max -6 --show-original-mesh --dims 50 2 2 -o ffd_box.xyz --ffd-point-size 5.0 --ffd-color red --ffd-alpha 0.5 --detail-level high --view-axis x

# With parallel processing:
python3 -m openffd /Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh --y-max -6 --show-original-mesh --dims 50 2 2 -o ffd_box.xyz --ffd-point-size 5.0 --ffd-color red --ffd-alpha 0.5 --detail-level high --view-axis x --parallel
```

## Performance Considerations

- **Small Meshes**: For small meshes (less than 100,000 points), parallel processing may actually be slower due to the overhead of creating and managing worker processes. It's best to use parallel processing only for larger meshes.

- **Optimal Worker Count**: The default worker count is set to the number of available CPU cores minus one, which is typically optimal. Setting too many workers can lead to excessive context switching and memory usage, while too few workers may not fully utilize available CPU resources.

- **Memory Usage**: Process-based parallelization uses more memory than thread-based parallelization, as each worker process requires its own memory space. For very large meshes, consider using thread-based parallelization (`--parallel-method thread`) if memory usage is a concern.

- **Visualization**: Parallel processing for visualization can significantly improve performance when working with large meshes, but may not be beneficial for smaller meshes. Use the `--parallel-viz` and `--no-parallel-viz` options to control this behavior independently of other parallel processing operations.
