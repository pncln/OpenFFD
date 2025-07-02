# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenFFD is a comprehensive Free-Form Deformation (FFD) framework for computational design and optimization. It provides both standard FFD and hierarchical FFD (H-FFD) capabilities with support for large-scale meshes, parallel processing, and integration with CFD solvers like OpenFOAM.

## Core Architecture

### Source Organization (`src/openffd/`)
- **`core/`**: FFD algorithms, parallel processing, hierarchical FFD implementation
- **`cli/`**: Command-line interface with comprehensive argument parsing  
- **`gui/`**: PyQt6-based graphical interface with 3D visualization
- **`mesh/`**: Mesh I/O with native Fluent parser and zone detection
- **`solvers/`**: Solver integration (OpenFOAM, sonicFoam, adjoint)
- **`visualization/`**: 3D visualization components using PyVista
- **`utils/`**: Parallel configuration, benchmarking utilities

### Key Components

**FFD Control Box Creation**: Use `create_ffd_box()` from `core/control_box.py` with parameter `custom_dims=None` (not `bounds=None`)

**Hierarchical FFD**: Multi-resolution control lattices with adaptive refinement in `core/hierarchical.py`

**Parallel Processing**: Configuration-driven via `ParallelConfig` with threshold-based activation for large datasets

**Mesh Handling**: Native Fluent parser with comprehensive zone detection, fallback to meshio for additional formats

## Common Commands

### Running Applications
```bash
# CLI interface
python -m openffd --input mesh.cas --output result.3df --control-dim 8 8 8

# GUI interface  
python -m openffd.gui

# Zone extraction
python -m openffd.cli.zone_extractor --input mesh.cas --zone-id 3
```

### Development and Testing
```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_core.py::test_ffd_creation

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Benchmarking
```bash
# Comprehensive benchmark suite
python benchmarks/benchmark_runner.py --max-workers 8 --output-dir results

# Individual benchmarks
python benchmarks/parallelization_scalability.py --output-dir parallel_results
python benchmarks/mesh_complexity_scaling.py --quick --output-dir mesh_results
python benchmarks/hierarchical_ffd_benchmark.py --output-dir hierarchical_results
```

## Architecture Patterns

### Function Signatures
- **FFD Creation**: Always use `custom_dims=None` parameter, never `bounds=None`
- **Parallel Config**: Configure via `ParallelConfig(enabled=True, method='process', max_workers=8, threshold=50000)`
- **Mesh Reading**: Use `read_general_mesh()` for automatic format detection

### Error Handling
- Mesh parsing errors often indicate zone detection issues - check Fluent zone boundaries
- Threading errors on macOS require `matplotlib.use('Agg')` for benchmarks
- Missing visualization functions should be implemented in respective `*_viz.py` modules

### Benchmark System
- All benchmarks generate individual figures (no combined 2x2 layouts)
- Publication-ready outputs with no titles in PNG/PDF files
- Academic plotting utilities in `benchmarks/academic_plotting_utils.py` provide enhanced visibility for side-by-side paper placement
- Use `--max-workers` parameter to optimize parallel performance (typically 8-10 for 12-core machines)

### GUI Integration
- PyQt6-based interface with integrated 3D visualization
- Zone extraction and mesh manipulation through interactive panels
- FFD and H-FFD unified in single interface

### Solver Integration
- OpenFOAM integration through `solvers/openfoam/` with sonicFoam and adjoint support
- Export formats for optimization frameworks (DAFoam, SU2, PyOpt)
- Use `write_ffd_3df()` for standard FFD output format

## Key Dependencies

**Core**: NumPy, SciPy, Matplotlib
**Visualization**: PyVista, PyVistaQt  
**GUI**: PyQt6
**Optional**: meshio (additional mesh formats), vtk

## Testing Strategy

Run full test suite before major changes. GUI tests require display environment. Zone extraction tests validate Fluent mesh parsing capabilities. Visualization tests ensure proper 3D rendering functionality.

## Performance Considerations

- Enable parallel processing for meshes >100,000 points
- Use process-based parallelism for CPU-intensive operations
- H-FFD provides better scalability for complex geometries
- Benchmark suite validates performance characteristics across configurations

## Git
- Make commit after significant changes