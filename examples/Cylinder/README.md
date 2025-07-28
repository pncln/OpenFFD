# Cylinder Flow Optimization Test Case

This directory contains a complete shape optimization test case for flow around a cylinder using our discrete adjoint CFD framework with OpenFOAM mesh integration.

## Case Description

- **Geometry**: 2D cylinder in cross-flow
- **Mesh**: OpenFOAM polyMesh format (5000 points, 9850 faces, 2450 cells)
- **Flow**: Incompressible viscous flow around cylinder
- **Objective**: Minimize drag while maintaining flow characteristics
- **Method**: Free Form Deformation (FFD) with discrete adjoint gradients

## Files Structure

```
Cylinder/
├── polyMesh/              # OpenFOAM mesh files
│   ├── boundary           # Boundary patch definitions
│   ├── points.gz          # Vertex coordinates (compressed)
│   ├── faces.gz           # Face connectivity (compressed)
│   ├── owner.gz           # Cell-face ownership (compressed)
│   └── neighbour.gz       # Cell-face neighbors (compressed)
├── optimization_config.yaml    # Optimization configuration
├── run_cylinder_optimization.py # Main optimization script
├── analyze_results.py          # Post-processing and analysis
├── create_visualization.py     # Visualization generation
└── results/                    # Output directory (created during run)
    ├── optimization_history.json
    ├── deformed_meshes/
    ├── convergence_plots/
    └── final_report.txt
```

## Boundary Conditions

- **cylinder**: Wall boundary (no-slip)
- **inout**: Velocity inlet/pressure outlet
- **symmetry1**: Symmetry plane (top)
- **symmetry2**: Symmetry plane (bottom)

## Design Variables

- **Parameterization**: Free Form Deformation (FFD)
- **Control Points**: 4×3×2 lattice around cylinder
- **Design Variables**: 60 total (movement in x, y, z directions)
- **Constraints**: Volume preservation, thickness limits

## Usage

### Quick Start
```bash
cd examples/Cylinder
python run_cylinder_optimization.py
```

### Custom Configuration
```bash
python run_cylinder_optimization.py --config custom_config.yaml --max-iter 50
```

### Analysis Only
```bash
python analyze_results.py --results-dir results/
```

## Expected Results

- **Drag Reduction**: 5-15% improvement
- **Optimization Time**: ~10-30 minutes (depending on iterations)
- **Mesh Quality**: Maintained throughout optimization
- **Convergence**: Typically 20-50 iterations

## Validation

The case includes validation against:
- **Analytical Solutions**: Potential flow around cylinder
- **Experimental Data**: Drag coefficients at various Reynolds numbers
- **CFD Benchmarks**: Standard cylinder flow cases

## Advanced Usage

### Parallel Execution
```bash
mpirun -np 4 python run_cylinder_optimization.py --parallel
```

### Adjoint Verification
```bash
python run_cylinder_optimization.py --verify-gradients
```

### Custom Objectives
```bash
python run_cylinder_optimization.py --objective lift_to_drag_ratio
```

## References

1. Cylinder flow benchmarks: Schäfer & Turek (1996)
2. Adjoint methods in CFD: Giles & Pierce (2000)
3. Shape optimization: Jameson (1988)