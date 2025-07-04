# NACA 0012 Airfoil Optimization Example

This example demonstrates the universal CFD optimization framework with a NACA 0012 airfoil case.

## Prerequisites

1. **OpenFOAM Installation**: Ensure OpenFOAM is installed and properly sourced
2. **Python Dependencies**: Install required Python packages
3. **OpenFFD**: The OpenFFD framework must be installed

## Case Setup

This example includes:

- `case_config.yaml`: Universal optimization configuration
- `run_optimization.py`: Universal optimization script
- Standard OpenFOAM case structure:
  - `constant/`: Contains mesh and physical properties
  - `system/`: Contains solver settings
  - `0/`: Contains initial and boundary conditions

## Configuration

The `case_config.yaml` file defines:

### Case Type and Solver
```yaml
case_type: "airfoil"
solver: "simpleFoam"
physics: "incompressible_flow"
```

### Optimization Objectives
```yaml
objectives:
  - name: "drag_coefficient"
    weight: 1.0
    patches: ["airfoil"]
    direction: [1.0, 0.0, 0.0]
```

### FFD Configuration
```yaml
ffd_config:
  control_points: [8, 6, 2]
  domain: "auto"
  ffd_type: "ffd"
```

### Optimization Settings
```yaml
optimization:
  max_iterations: 50
  tolerance: 1e-6
  algorithm: "slsqp"
```

## Running the Optimization

### Method 1: Using the Universal Script
```bash
# In the case directory
python run_optimization.py
```

### Method 2: Custom Configuration
```bash
python run_optimization.py my_custom_config.yaml
```

### Method 3: Python API
```python
from openffd.cfd.optimization.optimizer import UniversalOptimizer

# Initialize optimizer
optimizer = UniversalOptimizer(".", "case_config.yaml")

# Run optimization
results = optimizer.optimize()
```

## What Happens During Optimization

1. **Case Detection**: Framework automatically detects this is an airfoil case
2. **Configuration Loading**: Reads optimization parameters from YAML
3. **Mesh Preparation**: Prepares OpenFOAM mesh for FFD deformation
4. **FFD Setup**: Creates Free-Form Deformation volume around airfoil
5. **Optimization Loop**:
   - Deform airfoil shape using FFD control points
   - Run CFD simulation with OpenFOAM
   - Extract objective values (drag coefficient)
   - Compute gradients (finite differences or adjoint)
   - Update design variables
6. **Results**: Saves optimization history and final optimized shape

## Output Files

- `optimization_history.json`: Complete optimization history
- `optimization_config_used.yaml`: Configuration actually used
- `postProcessing/`: OpenFOAM post-processing data
- `{solver}.log`: CFD solver log files

## Customization

### Different Objectives
Modify `case_config.yaml` to optimize for:
- Lift coefficient: `lift_coefficient`
- Moment coefficient: `moment_coefficient`
- Lift-to-drag ratio: Custom objective function

### Different Algorithms
Change the optimization algorithm:
- `slsqp`: Sequential Least Squares Programming
- `genetic`: Genetic Algorithm

### Parallel Execution
Enable parallel CFD runs:
```yaml
optimization:
  parallel: true
```

## Troubleshooting

### OpenFOAM Not Found
Ensure OpenFOAM is properly sourced:
```bash
source /opt/openfoam/OpenFOAM-v2112/etc/bashrc
```

### Configuration Errors
The framework will auto-detect and suggest fixes for common configuration issues.

### Convergence Issues
Adjust tolerance or increase maximum iterations:
```yaml
optimization:
  max_iterations: 100
  tolerance: 1e-5
```

## Advanced Usage

### Custom Case Types
The framework can handle any OpenFOAM case by:
1. Creating a custom case handler
2. Defining appropriate objectives
3. Configuring FFD domain

### Multi-Objective Optimization
Add multiple objectives with weights:
```yaml
objectives:
  - name: "drag_coefficient"
    weight: 0.8
  - name: "lift_coefficient"
    weight: 0.2
    target: 0.5
    constraint_type: "min"
```

This universal framework makes CFD optimization accessible and reusable across different case types!