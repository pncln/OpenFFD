#!/usr/bin/env python3
"""Example script demonstrating the use of SonicFoamInterface with OpenFFD.

This example shows how to set up and run the sonicFoam solver with OpenFFD's
shape optimization capabilities for sonic/supersonic flow applications.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import Dict, Any

from openffd.solvers.openfoam.sonic_foam_interface import create_sonic_foam_interface

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sonicFoam with OpenFFD shape optimization"
    )
    parser.add_argument("--case", required=True, help="Path to OpenFOAM case directory")
    parser.add_argument("--ffd", required=True, help="Path to FFD box file")
    parser.add_argument("--steps", type=int, default=5, help="Number of optimization steps")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for optimization")
    parser.add_argument("--objective", default="drag", 
                       choices=["drag", "lift", "force", "pressure_uniformity"],
                       help="Objective function to minimize")
    parser.add_argument("--surfaces", nargs="+", default=["wall"],
                       help="Surface names for objective function calculation")
    return parser.parse_args()


def run_optimization(args):
    """Run the shape optimization process."""
    # Prepare settings
    settings = {
        "objective_type": args.objective,
        "surface_ids": args.surfaces,
        "step_size": args.step_size,
        "write_interval": 1
    }
    
    # Create interface
    logger.info(f"Creating SonicFoamInterface with case: {args.case}, FFD: {args.ffd}")
    interface = create_sonic_foam_interface(args.case, args.ffd, settings)
    
    # Run initial analysis
    logger.info("Running initial flow analysis")
    initial_obj = interface.run_flow_analysis()
    logger.info(f"Initial objective value: {initial_obj}")
    
    # Run optimization steps
    best_obj = initial_obj
    best_points = interface.ffd_box.get_control_points().copy()
    
    for i in range(args.steps):
        logger.info(f"Starting optimization step {i+1}/{args.steps}")
        
        # Perform optimization step
        prev_obj, current_obj = interface.perform_optimization_step()
        
        logger.info(f"Step {i+1} complete: {prev_obj} -> {current_obj}")
        
        # Store best result
        if current_obj < best_obj:
            best_obj = current_obj
            best_points = interface.ffd_box.get_control_points().copy()
            logger.info(f"New best objective: {best_obj}")
    
    # Restore best shape
    interface.ffd_box.set_control_points(best_points)
    new_points = interface.ffd_box.deform_mesh(interface.mesh_adapter.get_original_mesh_points())
    interface.mesh_adapter.update_mesh(new_points)
    
    # Final analysis
    logger.info("Running final analysis with best shape")
    final_obj = interface.run_flow_analysis()
    
    # Report results
    improvement = (initial_obj - final_obj) / initial_obj * 100
    logger.info(f"Optimization complete:")
    logger.info(f"  Initial objective: {initial_obj}")
    logger.info(f"  Final objective: {final_obj}")
    logger.info(f"  Improvement: {improvement:.2f}%")
    
    # Export optimized FFD box
    ffd_dir = os.path.dirname(args.ffd)
    ffd_base = os.path.basename(args.ffd)
    ffd_name, ffd_ext = os.path.splitext(ffd_base)
    optimized_ffd = os.path.join(ffd_dir, f"{ffd_name}_optimized{ffd_ext}")
    interface.ffd_box.to_file(optimized_ffd)
    logger.info(f"Optimized FFD box saved to: {optimized_ffd}")


if __name__ == "__main__":
    args = parse_args()
    run_optimization(args)
