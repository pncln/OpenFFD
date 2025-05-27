<div align="center">

# OpenFFD

**Modern Free-Form Deformation Control Box Generator for Computational Design**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/pncln/openffd/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>

## 📋 Overview

OpenFFD is a high-performance, open-source tool for generating Free-Form Deformation (FFD) control boxes for computational mesh files. This framework enables precise shape manipulation for aerodynamic optimization, structural analysis, and design workflows.

The software leverages parallel processing capabilities for handling large-scale meshes efficiently, making it suitable for industrial-grade CFD and FEA applications.

<div align="center">
  <img src="https://github.com/pncln/openffd/raw/main/docs/images/ffd_visualization.png" alt="FFD Visualization Example" width="700px">
</div>

## ✨ Key Features

- **Universal Mesh Support**: Processes Fluent mesh (.cas, .msh), VTK, STL, OBJ, Gmsh and more through a unified interface
- **Parallel Processing**: Utilizes multi-core processing for handling large-scale meshes with millions of points efficiently
- **Selective Targeting**: Targets specific mesh zones, patches, or cell sets for precise control
- **Multi-format Export**: Exports control points in .3df (standard) or .xyz (DAFoam-compatible) formats
- **Advanced Visualization**: Provides interactive 3D visualization with complete grid connectivity display
- **Plot3D Integration**: Supports direct export to Plot3D format for CFD post-processing
- **Modular Architecture**: Features a clean, extensible codebase using modern Python packaging standards
- **Optimization Integration**: Offers seamless compatibility with aerodynamic shape optimization frameworks

## 🚀 Installation

### From Source

```bash
git clone https://github.com/pncln/openffd.git
cd openffd
pip install -e .
```

With optional dependencies:

```bash
# For development tools (testing, linting)
pip install -e ".[dev]"

# For additional mesh format support
pip install -e ".[meshio]"

# For documentation building
pip install -e ".[docs]"

# For all optional dependencies
pip install -e ".[all]"
```

### Dependencies

- **Core**: NumPy, SciPy, Matplotlib, PyVista
- **Mesh Processing**: meshio (optional)
- **Visualization**: PyVista (>= 0.37.0)
- **Development**: pytest, black, isort, mypy, flake8, pre-commit

## 🏁 Quick Start

```bash
# Basic usage with default parameters
python -m openffd mesh_file.msh

# Specify control lattice dimensions and visualize
python -m openffd mesh_file.stl --dims 8 6 4 --plot

# Export in DAFoam compatible format with margins
python -m openffd airfoil.cas --output airfoil_ffd.xyz --margin 0.05 --export-xyz

# Enable parallel processing for large meshes
python -m openffd large_mesh.msh --dims 10 10 10 --parallel
```

## 📘 Usage Guide

### Command Line Interface

```
python -m openffd <mesh_file> [options]
```

#### Core Options

| Option | Description |
|--------|-------------|
| `<mesh_file>` | Path to mesh file |
| `-p, --patch PATCH` | Name of zone, cell set/patch or Gmsh physical group |
| `--dims N N N` | Control lattice dimensions: Nx Ny Nz (default: 4 4 4) |
| `--margin VALUE` | Margin padding around the mesh (default: 0.0) |
| `--output FILE` | Output filename (default: ffd_box.3df) |
| `--export-xyz` | Export in DAFoam-compatible .xyz format |

#### Boundary Options

| Option | Description |
|--------|-------------|
| `--x-min VALUE` | Minimum x-coordinate for custom bounds |
| `--x-max VALUE` | Maximum x-coordinate for custom bounds |
| `--y-min VALUE` | Minimum y-coordinate for custom bounds |
| `--y-max VALUE` | Maximum y-coordinate for custom bounds |
| `--z-min VALUE` | Minimum z-coordinate for custom bounds |
| `--z-max VALUE` | Maximum z-coordinate for custom bounds |

#### Visualization Options

| Option | Description |
|--------|-------------|
| `--plot` | Visualize FFD lattice |
| `--save-plot FILE` | Save visualization to specified file path |
| `--show-mesh` | Show mesh points in visualization |
| `--mesh-size VALUE` | Size of mesh points in visualization |
| `--ffd-point-size VALUE` | Size of FFD control points |
| `--ffd-color COLOR` | Color of FFD control points and grid |
| `--ffd-alpha VALUE` | Opacity of FFD control points and grid |
| `--view-axis {x,y,z,-x,-y,-z}` | View axis for visualization |
| `--detail-level {low,medium,high}` | Detail level for mesh visualization |

#### Parallel Processing Options

| Option | Description |
|--------|-------------|
| `--parallel` | Enable parallel processing for large meshes |
| `--no-parallel` | Disable parallel processing completely |
| `--parallel-method {process,thread}` | Method for parallelization |
| `--parallel-workers N` | Number of worker processes/threads |
| `--parallel-threshold N` | Minimum data size to trigger parallelization |
| `--parallel-viz` | Enable parallel processing for visualization only |

#### Advanced Options

| Option | Description |
|--------|-------------|
| `--debug` | Enable debug output |
| `--force-ascii` | Force ASCII reading for Fluent mesh |
| `--force-binary` | Force binary reading for Fluent mesh |


### Example Workflows

#### Basic Workflow

```bash
# Generate and visualize an FFD box around a mesh
python -m openffd wing.msh --dims 8 6 4 --plot
```

#### Aerodynamic Optimization

```bash
# Generate high-resolution control box for wing optimization with custom bounds
python -m openffd wing.msh --patch wing_surface --dims 12 8 6 --output wing_ffd.xyz --export-xyz \
    --x-min 0.0 --x-max 1.0 --y-min -0.5 --y-max 0.5
```

#### Large Mesh Processing with Parallel Execution

```bash
# Process a large mesh with parallel execution
python -m openffd large_mesh.msh --dims 75 50 2 --parallel --parallel-workers 8 --plot
```

#### Export to Plot3D Format

```bash
# Generate an FFD box and export to Plot3D format for CFD post-processing
python -m openffd aircraft.msh --dims 20 15 10 --output aircraft_ffd.p3d --plot
```

## 🎨 Visualization

OpenFFD provides advanced visualization capabilities for examining the generated FFD control boxes:

### Interactive 3D Visualization

The `--plot` flag enables a feature-rich interactive 3D visualization:

- **Complete Grid Visualization**: View all control points connected by lines in each dimension (x, y, z)
- **Original Mesh Display**: See the original mesh geometry with the FFD box overlaid
- **Surface Representation**: Utilize face connectivity data for accurate surface rendering
- **Zone Coloring**: Distinguish different mesh zones with automatic color assignment
- **Parallel Processing**: Leverage multi-core capabilities for large mesh visualization

### Interactive Controls

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Screenshot**: Press 'S' key
- **Reset View**: Press 'R' key

### Customization Options

```bash
# Customize visualization with point size, color, and transparency
python -m openffd mesh.msh --plot --ffd-point-size 8.0 --ffd-color blue --ffd-alpha 0.7

# View from a specific axis
python -m openffd mesh.msh --plot --view-axis z

# Use parallel processing for visualizing large meshes
python -m openffd large_mesh.msh --plot --parallel-viz
```

## 📄 Output Formats

OpenFFD supports multiple output formats for different downstream applications:

### .3df Format (Default)

Standard FFD control box format that includes dimensional information:

```
Nx Ny Nz
x1 y1 z1
x2 y2 z2
...
```

### .xyz Format (DAFoam-compatible)

A simplified format compatible with DAFoam and other optimization frameworks:

```
x1 y1 z1
x2 y2 z2
...
```

Enable with the `--export-xyz` flag.

### Plot3D Format

Structured grid format widely used in CFD applications, with proper connectivity information preserved:

```
Ni Nj Nk
x1 y1 z1
x2 y2 z2
...
```

This format maintains the structured nature of the control lattice, making it ideal for CFD post-processing.

## 🔄 Integration with Optimization

OpenFFD is designed to integrate seamlessly with various optimization frameworks:

### Aerodynamic Shape Optimization

- **DAFoam**: Direct integration for gradient-based aerodynamic shape optimization
- **SU2**: Compatible with the SU2 CFD suite's deformation tools
- **OpenMDAO**: Ready for inclusion in multidisciplinary optimization workflows

### General Purpose Optimization

- **PyOpt**: Integration with Python-based optimization frameworks
- **SWIG FFD**: Compatible with FFD libraries using SWIG bindings
- **Custom Frameworks**: Simple API allows easy integration with proprietary tools

### Example Integration

```python
from openffd.core.control_box import create_ffd_box
from openffd.mesh.general import read_general_mesh

# Read mesh
mesh_points = read_general_mesh("wing.msh")

# Create FFD control box
control_points, bbox = create_ffd_box(
    mesh_points, 
    control_dim=(10, 8, 6), 
    margin=0.05
)

# Use in optimization framework
# ...
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use OpenFFD in your research, please cite:

```bibtex
@software{openffd,
  author    = {Emil Mammadli},
  title     = {OpenFFD: A High-Performance Free-Form Deformation Tool for Computational Design},
  year      = {2025},
  month     = {5},
  publisher = {GitHub},
  url       = {https://github.com/pncln/openffd},
  version   = {1.1.0}
}
```

---

<div align="center">
  <sub>Built with ❤️ by Emil Mammadli and contributors.</sub>
</div>

## 🧩 Project Structure

OpenFFD follows a modern Python package structure:

```
openffd/
├── docs/                 # Documentation
├── src/
│   └── openffd/          # Main package
│       ├── cli/          # Command-line interface
│       ├── core/         # Core FFD functionality
│       ├── io/           # Input/output operations
│       ├── mesh/         # Mesh handling utilities
│       ├── utils/        # Utility functions
│       └── visualization/ # Visualization tools
├── tests/                # Test suite
├── pyproject.toml        # Package configuration
├── LICENSE               # MIT License
└── README.md            # This file
```

## 🔍 Advanced Features

### Parallel Processing

OpenFFD utilizes parallel processing for handling large meshes efficiently:

```bash
# Enable parallel processing with 8 worker processes
python -m openffd large_mesh.msh --parallel --parallel-workers 8

# Use thread-based parallelism for lower memory usage
python -m openffd large_mesh.msh --parallel --parallel-method thread

# Adjust parallelization threshold
python -m openffd mesh.msh --parallel --parallel-threshold 500000
```

For more details, see the [Parallel Processing Guide](docs/parallel_processing.md).

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Set up pre-commit hooks: `pre-commit install`
5. Make your changes (ensure tests pass and code is formatted)
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the computational design and CFD communities for valuable feedback
