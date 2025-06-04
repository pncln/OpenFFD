<div align="center">

# OpenFFD

**Advanced Free-Form Deformation Framework for Computational Design and Optimization**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/pncln/openffd/graphs/commit-activity)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>

## 📋 Overview

OpenFFD represents a high-performance open-source system which enables advanced Free-Form Deformation (FFD) operations in computational design and optimization. The framework provides both standard and hierarchical FFD capabilities which enable precise shape manipulation for applications in aerodynamic optimization and structural analysis as well as other engineering design workflows.

The system uses parallel processing to handle big mesh data sets efficiently which makes it appropriate for industrial CFD/FEA applications and allows integration with adjoint-based optimization frameworks like OpenFOAM.

<div align="center">
  <img src="https://github.com/pncln/openffd/raw/main/docs/images/hffd1.png" alt="FFD Visualization Example" width="700px">
</div>

## ✨ Key Features

Hierarchical FFD: The framework features multiple-level FFD with adjustable subdivision parameters for precise control over specific regions.
- **Universal Mesh Support**: The system accepts Fluent mesh (.cas,.msh) files as well as VTK, STL, OBJ and Gmsh and more through a unified interface.
Enhanced Zone Extraction: Advanced detection and extraction functionality of complex CFD zones combined with robust boundary handling capabilities.
OpenFOAM Integration: Direct integration with OpenFOAM and sonicFoamAdjoint for shape optimization of supersonic flows.
Parallel Processing: Optimized multi-core processing for handling large-scale meshes with millions of points efficiently
Custom Bounds Control: Precise control box dimensions in both standard and hierarchical FFD modes
Advanced Visualization: Interactive 3D visualization with level-based coloring for hierarchical FFD structures
- Unified GUI: Integrated interface for both standard and hierarchical FFD with seamless switching between modes

## 🚀 Installation

### From Source

```bash
git clone https://github.com/pncln/openffd.git
cd openffd
pip install -e.
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
- **Visualization**: PyVista (>= 0.37.0), PyVistaQt
- **GUI**: PyQt6 for the graphical user interface
OpenFOAM Integration: OpenFOAM (v2406+) for sonicFoamAdjoint integration.
Development: pytest, black, isort, mypy, flake8, pre-commit

## 🏁 Quick Start

```bash
# Basic usage with default parameters
python -m openffd mesh_file.msh

# Specify control lattice dimensions and visualize
python -m openffd mesh_file.stl --dims 8 6 4 --plot

# Create a hierarchical FFD with 3 levels and visualize
python -m openffd aircraft.msh --hierarchical --depth 3 --base-dims 4 4 4 --plot

# Extract specific zones and apply FFD
python -m openffd nozzle.cas --list-zones  # List available zones
python -m openffd nozzle.cas --extract-boundary wing --margin 0.05 --plot

# Generate FFD and export for OpenFOAM/sonicFoam optimization
python -m openffd wing.msh --output wing_ffd.3df --export-openfoam

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

#### Hierarchical FFD Options

| Option | Description |
|--------|-------------|
| `--hierarchical` | Enable hierarchical FFD mode |
| `--depth N` | Maximum depth of the hierarchical FFD (default: 3) |
| `--base-dims N N N` | Base lattice dimensions for hierarchical FFD (default: 4 4 4) |
| `--subdivision N` | Subdivision factor between hierarchy levels (default: 2) |

#### Zone Extraction Options

| Option | Description |
|--------|-------------|
| `--list-zones` | List available zones in the mesh file || `--extract-boundary NAME` | Extract specific boundary/zone for FFD application |
| `--save-boundary FILE` | Save extracted boundary mesh to a separate file |

#### OpenFOAM Integration

| Option | Description |
| `--export-openfoam` | Export in format compatible with OpenFOAM integration |
| `--adjoint-format` | Format FFD for sonicFoamAdjoint sensitivity mapping |
| `--export-xyz` | Export in DAFoam-compatible.xyz format |

#### Boundary Options

| Option | Description |
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

### Graphical User Interface

The graphical interface of OpenFFD launches after users initiate the application through this command:

```bash
python -m openffd
```

#### Unified FFD Panel

The graphical user interface contains an FFD panel that enables users to shift between standard and hierarchical FFD modes:

The user interface includes a standard FFD Mode that allows users to adjust control point dimensions alongside margins and custom bounds. The Hierarchical FFD Mode contains configuration options for base dimensions and hierarchy depth as well as subdivision factors and custom bounds.

#### Key GUI Features

The application accepts all mesh formats which automatically detect zones during loading.
A user interface lets users view and extract particular zones together with their boundaries.
The application features an interactive 3D viewer that enables users to rotate and pan while zooming between levels of hierarchical FFD through color coding.
The application enables users to specify exact control box dimensions in both Standard and Hierarchical FFD modes.
The application provides multiple export options that allow users to save FFD configurations for optimization frameworks.
The system interface lets users adjust parallel processing parameters when working with extensive mesh data.

#### OpenFOAM Integration

The software interface enables seamless integration with OpenFOAM functionality.

1. Users can export FFD for sonicFoamAdjoint solver applications.
The system creates mappings between FFD control points and adjoint sensitivity values.
The system enables FFD setup for supersonic flow-based shape optimization applications.

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

Users can generate and visualize FFD boxes around meshes using this command:

python -m openffd wing.msh --dims 8 6 4 --plot
```

#### Aerodynamic Optimization

This command produces detailed wing optimization control boxes from the wing surface with defined boundaries.

python -m openffd wing.msh --patch wing_surface --dims 12 8 6 --output wing_ffd.xyz --export-xyz \
    --x-min 0.0 --x-max 1.0 --y-min -0.5 --y-max 0.5

#### Large Mesh Processing with Parallel Execution

The code runs this command for large mesh processing through parallel execution:

python -m openffd large_mesh.msh --dims 75 50 2 --parallel --parallel-workers 8 --plot
```

#### Export to Plot3D Format

```bash
# Create an FFD box for aircraft meshing with 20x15x10 dimensions then export to Plot3D format for CFD analysis.
python -m openffd aircraft.msh --dims 20 15 10 --output aircraft_ffd.p3d --plot
```

## 🎨 Visualization

OpenFFD provides advanced visualization capabilities for examining the generated FFD control boxes:

### Interactive 3D Visualization

The `--plot` flag enables a feature-rich interactive 3D visualization:

- **Complete Grid Visualization**: All control points are displayed by lines in each dimension (x, y, z)
- **Original Mesh Display**: View the original mesh geometry with the FFD box overlaid.
- **Surface Representation**: Utilize face connectivity data for accurate surface rendering.
- **Zone Coloring**: Distinguish different mesh zones with automatic color assignment.
- **Parallel Processing**: Leverage multi-core capabilities for large mesh visualization.

### Interactive Controls

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Screenshot**: Press 'S' key
- **Reset View**: Press 'R' key.

### Customization Options

```bash
# Custom visualization by setting point size, color, and transparency for the FFD output
python -m openffd mesh.msh --plot --ffd-point-size 8.0 --ffd-color blue --ffd-alpha 0.7

# View from a specific axis
python -m openffd mesh.msh --plot --view-axis z

# Use parallel processing for visualizing large meshes
python -m openffd large_mesh.msh --plot --parallel-viz
```

## 📄 Output Formats

OpenFFD supports multiple output formats for different downstream applications:

###.3df Format (Default)

Standard FFD control box format that includes dimensional information:

```
Nx Ny Nz
x1 y1 z1
x2 y2 z2
...
```

###.xyz Format (DAFoam-compatible)

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
- **SU2**: Compatible with the SU2 CFD suite’s deformation tools
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
#...
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use OpenFFD in your research, please cite:

```bibtex
@software{openffd,
  author    = {Emil Mammadli},
  title     = {OpenFFD: Advanced Free-Form Deformation Framework for Computational Design and Optimization},
  year      = {2025},
  month     = {5},
  publisher = {GitHub},
  url       = {https://github.com/pncln/openffd},
  version   = {1.1.0}
}
```

---

<div align="center">
  <sub>Built with ❤️ by Emil Mammadli.</sub>
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

OpenFFD utilizes parallel processing for handling large meshes efficiently:```bash
Python runs the openffd tool on large_mesh.msh with parallel processing enabled by using 8 worker processes.

When you run the command with the thread-based parallelism method you can lower memory usage.
The command to use thread-based parallelism for parallel processing can be seen below.
The parallelization threshold can be adjusted by using python -m openffd mesh.msh --parallel --parallel-threshold 500000

The Parallel Processing Guide is available for further information in the docs/parallel_processing.md section.

## 🤝 Contributing

All help in the form of contributions is welcome. Here’s how you can help:

1. Fork the repository
2. Create your feature branch: git checkout -b feature/amazing-feature
3. Install development dependencies: pip install -e ".[dev]"
4. Set up pre-commit hooks: pre-commit install
5. Make your changes (ensure tests pass and code is formatted)
6. Commit your changes: git commit -m 'Add amazing feature'
7. Push to the branch: git push origin feature/amazing-feature
8. Open a Pull Request

## 🙏 Acknowledgments

Thanks to all the contributors who have contributed to this project
A special thanks to the computational design and CFD communities for their feedback
