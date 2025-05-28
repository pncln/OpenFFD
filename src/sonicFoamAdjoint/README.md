# sonicFoamAdjoint

An adjoint-enabled version of OpenFOAM's sonicFoam solver for sensitivity analysis in supersonic flows.

## Overview

This solver extends the standard OpenFOAM sonicFoam solver with discrete adjoint capabilities for:

- Computing sensitivities for aerodynamic shape optimization
- Supporting multiple objective functions (drag, lift, moment, pressure loss, uniformity)
- Exporting sensitivity data for use with external mesh deformation tools like OpenFFD

## Compilation

To compile the solver:

```bash
cd /path/to/sonicFoamAdjoint
wmake
```

This will create the executable in `$FOAM_APPBIN`.

## Usage

```bash
sonicFoamAdjoint -case <case_dir> -objective <objective_name>
```

Where `<objective_name>` is one of:
- drag
- lift
- moment
- pressure_loss
- uniformity

## Workflow

1. The solver first checks if a converged forward solution exists
2. If not, it runs the forward solution (equivalent to sonicFoam)
3. It then initializes adjoint variables and solves the adjoint equations
4. Sensitivities are calculated and exported in various formats

## Integration with OpenFFD

The solver exports sensitivities in formats compatible with OpenFFD. Look for the exported files in:

```
<case>/sensitivities/sensitivity_openffd.dat
```

This file contains surface points and their associated sensitivities and can be directly used with OpenFFD for mesh deformation.

## Mathematical Background

The adjoint method computes the gradient of an objective function J with respect to design variables α by solving the adjoint equations:

[A]^T [ψ] = [∂J/∂U]

where [A] is the Jacobian of the flow equations, [ψ] are the adjoint variables, and [∂J/∂U] are the derivatives of the objective function with respect to the flow variables.

The mesh sensitivities are then computed as:

dJ/dα = ∂J/∂α + ψ^T [∂R/∂α]

where R represents the residuals of the flow equations.
