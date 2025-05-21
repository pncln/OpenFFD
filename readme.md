# OpenFFD -- Open-Source FFD Control Box Generator

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.7+-blue.svg" />
</p>

A robust tool for generating Free-Form Deformation (FFD) control boxes for computational mesh files. This framework enables precise shape manipulation for aerodynamic optimization, structural analysis, and design workflows.

![FFD Visualization Example](/Users/pncln/Documents/tubitak/verynew/ffd_gen/ffd_view_20250521_163553.png)

## 🌟 Features

- **Universal Mesh Support**: Handles Fluent mesh (.cas, .msh), VTK, STL, OBJ, Gmsh and more
- **Selective Targeting**: Restrict FFD boxes to specific mesh zones, patches, or cell sets
- **Multi-format Export**: Generate control points in .3df or .xyz (DAFoam) formats
- **Advanced Visualization**: Interactive 3D plotting with complete grid visualization showing all control point connections
- **High-Density Support**: Efficiently handles high point density grids (e.g., 75×50×2 for rocket launch pad meshes)
- **Optimization Integration**: Direct compatibility with aerodynamic shape optimization frameworks

## 📦 Installation

### From PyPI

```bash
pip install ffd-generator
```

### From Source

```bash
git clone https://github.com/username/ffd_generator.git
cd ffd_generator
pip install -e .
```

For full support of additional mesh formats:

```bash
pip install -e ".[meshio]"
```

## 📖 Dependencies

- **Core**: NumPy, SciPy, Matplotlib, PyVista
- **Optional**: meshio, fluent_reader

## 🚀 Quick Start

```bash
# Basic usage with default parameters
python ffd_generator.py mesh_file.msh

# Specify control lattice dimensions and visualize
python ffd_generator.py mesh_file.stl -d 8 6 4 --plot

# Export in DAFoam compatible format with margins
python ffd_generator.py airfoil.cas -o airfoil_ffd.xyz -m 0.05
```

## 🛠️ Usage Guide

### Command Line Interface

```
python ffd_generator.py <mesh_file> [options]
```

| Option | Description |
|--------|-------------|
| `<mesh_file>` | Path to mesh file |
| `-p, --patch PATCH` | Name of zone, cell set/patch or Gmsh physical group |
| `-d, --dims N N N` | Control lattice dimensions: Nx Ny Nz (default: 4 4 4) |
| `-m, --margin VALUE` | Margin padding around the mesh (default: 0.0) |
| `-o, --output FILE` | Output filename: .3df or .xyz (default: ffd_box.3df) |
| `--plot` | Visualize FFD lattice |
| `--debug` | Enable debug output |
| `--force-ascii` | Force ASCII reading for Fluent mesh |
| `--force-binary` | Force binary reading for Fluent mesh |

### Example Workflows

#### Aerodynamic Optimization

```bash
# Generate high-resolution control box for wing optimization
python ffd_generator.py wing.msh -p wing_surface -d 12 8 6 -o wing_ffd.xyz
```

#### Rocket Launch Pad Mesh

```bash
# Generate high-density control grid (75×50×2) for launch pad
python ffd_generator.py launchpad.msh -d 75 50 2 --plot
```

## 📊 Visualization

The `--plot` flag enables an interactive 3D visualization featuring:

- Complete grid visualization with all control points connected by lines in each dimension (x, y, z)
- Original mesh geometry or bounding box representation
- Original face connectivity data for accurate surface representation
- Interactive controls:
  - **Rotate**: Click and drag
  - **Zoom**: Scroll wheel
  - **Pan**: Right-click and drag
  - **Screenshot**: Press 'S' key

## 📄 Output Formats

### .3df Format

Standard FFD control box format:
```
Nx Ny Nz
x1 y1 z1
x2 y2 z2
...
```

### .xyz Format

DAFoam-compatible format:
```
x1 y1 z1
x2 y2 z2
...
```

## 🔄 Integration with Optimization

The FFD control boxes generated with this tool can be seamlessly integrated with:

- **DAFoam**: Direct integration for aerodynamic shape optimization
- **SWIG FFD**: Compatible with FFD libraries using SWIG bindings
- **PyOpt**: Ready for inclusion in Python-based optimization workflows

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{ffd_generator,
  author    = {Your Name},
  title     = {FFD Control Box Generator},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/username/ffd_generator}
}
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the computational design community for valuable feedback
