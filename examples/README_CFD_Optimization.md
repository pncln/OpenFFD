# CFD Optimization Examples

This directory contains examples demonstrating the OpenFFD CFD optimization capabilities.

## New Streamlined Structure

The optimization functionality has been consolidated into the `openffd.cfd.optimization` module for better organization and reusability.

### Main Example

**`naca0012_optimization_clean.py`** - Complete NACA0012 airfoil optimization example
- Uses the consolidated `openffd.cfd` module
- Professional OpenFOAM case generation
- Automated optimization workflow
- Result analysis and visualization

### Key Features

1. **Airfoil Generation**
   ```python
   from openffd.cfd import AirfoilGenerator
   
   # Generate NACA0012
   x, y = AirfoilGenerator.generate_naca0012(n_points=100)
   
   # Generate any NACA 4-digit airfoil
   x, y = AirfoilGenerator.generate_naca_4digit("2412", n_points=100)
   
   # Create STL file
   AirfoilGenerator.create_stl_file(x, y, "airfoil.stl")
   ```

2. **OpenFOAM Case Creation**
   ```python
   from openffd.cfd import create_naca_airfoil_case
   
   success = create_naca_airfoil_case(
       case_dir="./my_case",
       naca_code="0012",
       config={
           "control": {"endTime": 1000, "magUInf": 30},
           "mesh": {"nx": 100, "ny": 80},
           "initial": {"U_inlet": [30, 0, 0]}
       }
   )
   ```

3. **Complete Optimization**
   ```python
   from openffd.cfd import optimize_naca0012_airfoil, OptimizationConfig
   
   config = OptimizationConfig(
       max_iterations=10,
       convergence_tolerance=1e-6
   )
   
   results = optimize_naca0012_airfoil("./optimization_case", config)
   ```

## Prerequisites

1. **OpenFOAM Installation**
   ```bash
   # macOS with Homebrew
   brew install openfoam
   
   # Or download OpenFOAM.app
   # https://github.com/gerlero/openfoam-app
   ```

2. **Environment Activation**
   ```bash
   openfoam  # Activate OpenFOAM environment
   ```

3. **OpenFFD Package**
   ```bash
   pip install -e .  # From project root
   ```

## Usage

```bash
# Activate OpenFOAM environment
openfoam

# Run the optimization example
python3 naca0012_optimization_clean.py
```

## Generated Files

After running the optimization:

- `naca0012_optimization_clean/` - Complete OpenFOAM case
- `optimization_results.txt` - Detailed results summary
- `convergence_history.png` - Optimization convergence plot
- `naca0012_optimization.log` - Execution log

## OpenFOAM Case Structure

The generated OpenFOAM cases include:

```
case_directory/
├── 0/                    # Initial conditions
│   ├── U                 # Velocity field
│   └── p                 # Pressure field
├── constant/             # Physical properties
│   ├── transportProperties
│   ├── turbulenceProperties
│   └── triSurface/
│       └── airfoil.stl   # Airfoil geometry
└── system/               # Discretization settings
    ├── controlDict       # Simulation control
    ├── blockMeshDict      # Background mesh
    ├── snappyHexMeshDict  # Body-fitted mesh
    ├── fvSchemes          # Numerical schemes
    └── fvSolution         # Linear solvers
```

## Manual OpenFOAM Execution

You can also run the OpenFOAM cases manually:

```bash
cd naca0012_optimization_clean
openfoam
blockMesh
snappyHexMesh -overwrite
simpleFoam
```

## Advanced Usage

### Custom Optimization Configuration

```python
from openffd.cfd import OptimizationConfig, OptimizationManager

config = OptimizationConfig(
    objective="minimize_drag",
    max_iterations=50,
    convergence_tolerance=1e-6,
    initial_step_size=0.001,
    max_step_size=0.01,
    parallel_evaluation=True,
    n_processes=4
)

optimizer = OptimizationManager(config)
results = optimizer.optimize_naca0012(Path("./my_optimization"))
```

### Custom Case Configuration

```python
from openffd.cfd import OpenFOAMCaseBuilder, AirfoilGenerator

# Generate custom airfoil
x, y = AirfoilGenerator.generate_naca_4digit("4412")

# Create custom case
builder = OpenFOAMCaseBuilder("./custom_case")
config = {
    "control": {
        "solver": "simpleFoam",
        "endTime": 2000,
        "magUInf": 50,
        "rhoInf": 1.225
    },
    "mesh": {
        "nx": 120, "ny": 100,
        "surface_max_level": 5,
        "boundary_layers": 8
    },
    "transport": {
        "nu": 1.5e-05
    }
}

success = builder.create_complete_case((x, y), config)
```

## Old Examples

Previous examples have been moved to `old_examples/` directory for reference but are no longer recommended for use.

## Troubleshooting

1. **OpenFOAM Not Found**
   - Ensure OpenFOAM is installed and the `openfoam` command works
   - Check environment variables: `echo $FOAM_APP`

2. **Import Errors**
   - Ensure OpenFFD is installed: `pip install -e .`
   - Check Python path includes OpenFFD

3. **Mesh Generation Failures**
   - Check STL file quality
   - Verify snappyHexMesh settings
   - Examine log files for detailed errors

4. **Simulation Failures**
   - Check mesh quality with `checkMesh`
   - Verify boundary conditions
   - Monitor residuals for convergence issues