"""Example of using sonicAdjointFoam with OpenFFD for shape optimization.

This example demonstrates how to set up and run an adjoint-based shape
optimization workflow for supersonic flow around an airfoil using sonicAdjointFoam
and OpenFFD's free-form deformation capabilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from openffd.core.control_box import FFDBox
from openffd.solvers.openfoam.sonic_adjoint import (
    SonicAdjointInterface,
    create_sonic_adjoint_interface
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_single_optimization_iteration(
    interface: SonicAdjointInterface,
    step_size: float = 0.01
) -> Tuple[float, float]:
    """Run a single optimization iteration.
    
    Args:
        interface: SonicAdjointInterface instance
        step_size: Step size for shape updates
        
    Returns:
        Tuple[float, float]: Initial and optimized objective values
    """
    # Get initial objective value
    logger.info("Running initial analysis...")
    initial_obj = interface.run_adjoint_analysis()
    logger.info(f"Initial objective value: {initial_obj}")
    
    # Get control point sensitivities
    gradient = interface.get_objective_gradient()
    logger.info(f"Gradient shape: {gradient.shape}")
    logger.info(f"Max sensitivity magnitude: {np.max(np.abs(gradient))}")
    
    # Apply updates to control points (negative gradient direction for minimization)
    interface.apply_control_point_update(-gradient, scale_factor=step_size)
    
    # Run analysis with updated shape
    logger.info("Running analysis with updated shape...")
    updated_obj = interface.run_solver()
    logger.info(f"Updated objective value: {updated_obj}")
    
    # Calculate improvement
    improvement = (initial_obj - updated_obj) / initial_obj * 100
    logger.info(f"Improvement: {improvement:.2f}%")
    
    return initial_obj, updated_obj


def run_multi_step_optimization(
    interface: SonicAdjointInterface,
    n_iterations: int = 5,
    step_size: float = 0.01
) -> List[float]:
    """Run a multi-step optimization process.
    
    Args:
        interface: SonicAdjointInterface instance
        n_iterations: Number of optimization iterations
        step_size: Step size for shape updates
        
    Returns:
        List[float]: History of objective values
    """
    obj_history = []
    
    # Initial analysis
    logger.info("Running initial analysis...")
    obj = interface.run_adjoint_analysis()
    obj_history.append(obj)
    logger.info(f"Initial objective value: {obj}")
    
    # Optimization loop
    for i in range(n_iterations):
        logger.info(f"\n{'='*50}\nIteration {i+1}/{n_iterations}\n{'='*50}")
        
        # Get control point sensitivities
        gradient = interface.get_objective_gradient()
        
        # Apply updates
        interface.apply_control_point_update(-gradient, scale_factor=step_size)
        
        # Run analysis with updated shape
        obj = interface.run_adjoint_analysis()
        obj_history.append(obj)
        
        # Calculate improvement
        improvement = (obj_history[0] - obj) / obj_history[0] * 100
        rel_improvement = (obj_history[-2] - obj) / obj_history[-2] * 100
        
        logger.info(f"Objective value: {obj}")
        logger.info(f"Total improvement: {improvement:.2f}%")
        logger.info(f"Relative improvement: {rel_improvement:.2f}%")
        
        # Export intermediate results
        output_dir = os.path.join(interface.case_dir, f"optimization_step_{i+1}")
        interface.export_results(output_dir)
        
        # Adjust step size based on progress
        if rel_improvement < 0.1 and i > 0:
            step_size *= 0.5
            logger.info(f"Reducing step size to {step_size}")
    
    return obj_history


def plot_optimization_results(
    obj_history: List[float],
    output_path: Optional[str] = None
) -> None:
    """Plot optimization history.
    
    Args:
        obj_history: List of objective values
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(obj_history, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.grid(True)
    
    # Calculate improvement
    improvement = (obj_history[0] - obj_history[-1]) / obj_history[0] * 100
    plt.annotate(
        f'Total improvement: {improvement:.2f}%',
        xy=(len(obj_history)-1, obj_history[-1]),
        xytext=(-100, 30),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->'),
        fontsize=12
    )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()


def main():
    """Run the complete optimization example."""
    # Parameters
    case_dir = "/path/to/your/openfoam/case"  # Replace with your case path
    ffd_file = "/path/to/your/ffd_box.3df"    # Replace with your FFD file
    
    # Objective function settings
    settings = {
        "objective_type": "drag",
        "step_size": 0.01,
        "smooth_sensitivity": True,
        "target_patches": ["airfoil"],       # Replace with your patch names
        "export_vtk": True
    }
    
    # Create the interface
    interface = create_sonic_adjoint_interface(
        case_dir=case_dir,
        ffd_file=ffd_file,
        settings=settings
    )
    
    # Set up the optimization problem
    interface.setup_optimization(
        objective_type="drag",
        target_patches=["airfoil"],           # Replace with your patch names
        direction=[1.0, 0.0, 0.0],            # x-direction for drag
        reference_values={
            "rhoRef": 1.0,
            "URef": 340.0,                    # ~Mach 1
            "LRef": 1.0,
            "pRef": 101325.0                  # Standard atmospheric pressure
        }
    )
    
    # Run the optimization
    obj_history = run_multi_step_optimization(
        interface=interface,
        n_iterations=5,
        step_size=0.01
    )
    
    # Plot results
    plot_optimization_results(
        obj_history=obj_history,
        output_path=os.path.join(case_dir, "optimization_history.png")
    )
    
    # Export final results
    interface.export_results(os.path.join(case_dir, "optimization_final"))
    
    logger.info("Optimization completed successfully!")


if __name__ == "__main__":
    main()
