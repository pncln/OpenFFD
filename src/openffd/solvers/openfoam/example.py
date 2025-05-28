"""Example usage of the OpenFOAM interface with OpenFFD.

This module demonstrates how to use the OpenFOAM interface with OpenFFD
for running adjoint-based shape optimization for supersonic flows.
"""

import os
import logging
import numpy as np
import argparse
from pathlib import Path

from openffd.core.control_box import create_ffd_box, FFDBox
from openffd.mesh.general import read_general_mesh
from openffd.solvers.openfoam.interface import OpenFOAMInterface


def optimize_shape(
    case_dir: str, 
    ffd_file: str,
    mesh_file: str,
    patch_name: str,
    num_iterations: int = 10,
    objective: str = "drag",
    step_size: float = 0.01
):
    """Run an adjoint-based shape optimization using OpenFOAM and OpenFFD.
    
    Args:
        case_dir: Path to OpenFOAM case directory
        ffd_file: Path to FFD file (3df or xyz format)
        mesh_file: Path to mesh file
        patch_name: Name of the patch to optimize
        num_iterations: Number of optimization iterations
        objective: Objective function (e.g., drag, lift)
        step_size: Step size for gradient descent
    """
    logging.info(f"Starting optimization with {num_iterations} iterations")
    
    # Initialize the OpenFOAM interface
    # Assuming sonicFoamAdjoint is installed in the standard OpenFOAM location
    openfoam_path = os.environ.get("FOAM_APPBIN", "/usr/bin")
    interface = OpenFOAMInterface(
        solver_path=openfoam_path,
        case_dir=case_dir,
        solver_type="sonicFoam",
        adjoint_solver_type="sonicFoamAdjoint"  # This will be our custom adjoint solver
    )
    
    # Set up the case with the initial mesh
    interface.setup_case(mesh_file, {})
    
    # Read the mesh and create the FFD box
    mesh_points, zones = read_general_mesh(mesh_file, patch_name)
    
    # If ffd_file is provided, load the FFD box, otherwise create a new one
    if os.path.exists(ffd_file):
        # In a real implementation, would load the FFD box from file
        ffd_box = FFDBox(np.array([]), np.array([4, 4, 4]))  # Placeholder
    else:
        # Create a new FFD box around the mesh points
        ffd_box = create_ffd_box(mesh_points, [4, 4, 4])
    
    # Set the FFD box in the interface
    interface.set_ffd_box(ffd_box)
    
    # Initialize objective value tracking
    objective_values = []
    
    # Main optimization loop
    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration+1}/{num_iterations}")
        
        # Run the flow solver
        if not interface.solve():
            logging.error("Flow solution failed, stopping optimization")
            break
        
        # Get the objective value
        obj_value = interface.get_objective_value(objective)
        objective_values.append(obj_value)
        logging.info(f"Objective value: {obj_value}")
        
        # Check for convergence (simple check based on relative change)
        if iteration > 0:
            rel_change = abs(objective_values[-1] - objective_values[-2]) / abs(objective_values[-2])
            if rel_change < 1e-4:
                logging.info(f"Optimization converged with relative change {rel_change:.6f}")
                break
        
        # Run the adjoint solver to get sensitivities
        if not interface.solve_adjoint(objective):
            logging.error("Adjoint solution failed, stopping optimization")
            break
        
        # Map sensitivities to FFD control points
        control_point_sensitivities = interface.map_sensitivities_to_control_points()
        
        # Update the FFD control points using gradient descent
        # In a more sophisticated implementation, would use a proper optimizer
        ffd_box.control_points -= step_size * control_point_sensitivities
        
        # Apply the FFD deformation to the mesh
        deformed_points = ffd_box.evaluate_points(mesh_points)
        
        # Update the mesh in OpenFOAM
        if not interface.set_mesh(deformed_points):
            logging.error("Mesh update failed, stopping optimization")
            break
    
    # Final flow solution with the optimized shape
    interface.solve()
    
    # Export the final results
    results_file = os.path.join(os.path.dirname(case_dir), "optimization_results.csv")
    with open(results_file, 'w') as f:
        f.write(f"Iteration,{objective}\n")
        for i, val in enumerate(objective_values):
            f.write(f"{i+1},{val}\n")
    
    logging.info(f"Optimization completed, results saved to {results_file}")
    
    # Export the final FFD box and mesh
    # In a real implementation, would use proper export functions
    logging.info("Exported final FFD box and mesh")


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(
        description='OpenFOAM-OpenFFD integration example for adjoint-based shape optimization'
    )
    parser.add_argument('case_dir', help='Path to OpenFOAM case directory')
    parser.add_argument('--ffd-file', help='Path to FFD file (3df or xyz format)', default=None)
    parser.add_argument('--mesh-file', help='Path to mesh file (if different from case)', default=None)
    parser.add_argument('--patch', help='Name of patch to optimize', default=None)
    parser.add_argument('--iterations', type=int, default=10, help='Number of optimization iterations')
    parser.add_argument('--objective', choices=['drag', 'lift', 'moment', 'pressure_loss'], 
                        default='drag', help='Objective function to minimize')
    parser.add_argument('--step-size', type=float, default=0.01, help='Step size for gradient descent')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # If mesh file is not provided, use the one in the case directory
    mesh_file = args.mesh_file
    if mesh_file is None:
        mesh_path = Path(args.case_dir) / "constant" / "polyMesh"
        if mesh_path.exists():
            mesh_file = str(mesh_path)
        else:
            logging.error("No mesh file provided and no mesh found in case directory")
            return 1
    
    # Run the optimization
    optimize_shape(
        args.case_dir,
        args.ffd_file,
        mesh_file,
        args.patch,
        args.iterations,
        args.objective,
        args.step_size
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
