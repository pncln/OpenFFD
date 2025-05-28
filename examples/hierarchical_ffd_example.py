#!/usr/bin/env python3
"""
Example demonstrating the use of Hierarchical FFD for mesh deformation.

This example shows how to:
1. Create a hierarchical FFD structure
2. Visualize the hierarchy of control lattices
3. Deform a mesh using hierarchical FFD
4. Analyze the influence of different levels
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import OpenFFD modules
from openffd.core.hierarchical import create_hierarchical_ffd
from openffd.mesh.general import read_general_mesh
from openffd.utils.parallel import ParallelConfig
from openffd.visualization.hierarchical_viz import (
    visualize_hierarchical_ffd_pyvista,
    visualize_influence_distribution,
    visualize_hierarchical_deformation
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run the hierarchical FFD example."""
    # Path to mesh file (use a sample mesh or your own)
    mesh_file = "../14.msh"  # Update with your mesh file path
    output_dir = "./output_hierarchical"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create parallel configuration
    parallel_config = ParallelConfig(enabled=True, n_workers=4, chunk_size=10000)
    
    # Load mesh
    print(f"Loading mesh from {mesh_file}")
    mesh_data = read_general_mesh(mesh_file)
    mesh_points = mesh_data.points
    
    # Create hierarchical FFD
    print("Creating hierarchical FFD structure...")
    hffd = create_hierarchical_ffd(
        mesh_points=mesh_points,
        base_dims=(6, 6, 6),           # Base level dimensions
        max_depth=3,                   # Number of hierarchical levels
        subdivision_factor=2,          # Factor for each subdivision
        parallel_config=parallel_config
    )
    
    # Print information about the hierarchical levels
    print("\nHierarchical FFD Structure:")
    for level_info in hffd.get_level_info():
        print(
            f"Level {level_info['level_id']} (depth {level_info['depth']}): "
            f"{level_info['dims']} dims, {level_info['num_control_points']} control points, "
            f"weight: {level_info['weight_factor']:.2f}"
        )
    
    # Visualize the hierarchical structure
    print("\nVisualizing hierarchical FFD structure...")
    visualize_hierarchical_ffd_pyvista(
        hffd=hffd,
        title="Hierarchical FFD Control Lattice",
        save_path=os.path.join(output_dir, "hierarchical_ffd_structure.png"),
        mesh_points=mesh_points,
        show_mesh=True,
        mesh_size=3.0,
        mesh_alpha=0.2,
        point_size=10.0,
        color_by_level=True,
        show_influence=True,
        off_screen=True,
        parallel_config=parallel_config
    )
    
    # Visualize the influence distribution
    print("\nVisualizing influence distribution...")
    visualize_influence_distribution(
        hffd=hffd,
        mesh_points=mesh_points,
        save_path=os.path.join(output_dir, "influence_distribution.png")
    )
    
    # Create a sample deformation by moving control points
    print("\nCreating sample deformation...")
    deformed_control_points = {}
    
    # Deform level 0 (coarse global deformation)
    level_0_points = hffd.levels[0].control_points.copy()
    # Move some control points upward
    center = np.mean(level_0_points, axis=0)
    for i in range(len(level_0_points)):
        dist = np.linalg.norm(level_0_points[i] - center)
        # Apply a sinusoidal deformation based on distance from center
        level_0_points[i, 1] += 5.0 * np.sin(dist / 20.0)
    deformed_control_points[0] = level_0_points
    
    # Deform level 1 (medium-scale deformation)
    if 1 in hffd.levels:
        level_1_points = hffd.levels[1].control_points.copy()
        # Add some local bumps
        bbox_min, bbox_max = hffd.levels[1].bbox
        bbox_center = (bbox_min + bbox_max) / 2
        for i in range(len(level_1_points)):
            # Calculate normalized position within bounding box
            normalized_pos = (level_1_points[i] - bbox_min) / (bbox_max - bbox_min)
            # Add some noise to create local variations
            if 0.4 < normalized_pos[0] < 0.6 and 0.4 < normalized_pos[2] < 0.6:
                level_1_points[i, 1] += 2.0 * np.exp(-((normalized_pos[0]-0.5)**2 + (normalized_pos[2]-0.5)**2) / 0.01)
        deformed_control_points[1] = level_1_points
    
    # Deform level 2 (fine-scale details) if it exists
    if 2 in hffd.levels:
        level_2_points = hffd.levels[2].control_points.copy()
        # Add fine-scale noise
        for i in range(len(level_2_points)):
            # Only affect a specific region
            bbox_min, bbox_max = hffd.levels[2].bbox
            normalized_pos = (level_2_points[i] - bbox_min) / (bbox_max - bbox_min)
            if 0.3 < normalized_pos[0] < 0.7 and 0.3 < normalized_pos[2] < 0.7:
                # Add small ripples
                level_2_points[i, 1] += 0.5 * np.sin(normalized_pos[0] * 20) * np.sin(normalized_pos[2] * 20)
        deformed_control_points[2] = level_2_points
    
    # Perform the hierarchical deformation
    print("\nPerforming hierarchical deformation...")
    deformed_points = hffd.deform_mesh(deformed_control_points, mesh_points)
    
    # Save the deformed points
    output_file = os.path.join(output_dir, "deformed_mesh_points.txt")
    np.savetxt(output_file, deformed_points, delimiter=",", header="x,y,z")
    print(f"Saved deformed points to {output_file}")
    
    # Visualize the deformation
    print("\nVisualizing deformation...")
    visualize_hierarchical_deformation(
        hffd=hffd,
        deformed_control_points=deformed_control_points,
        mesh_points=mesh_points,
        title="Hierarchical FFD Deformation",
        save_path=os.path.join(output_dir, "hierarchical_deformation.png"),
        show_original=True,
        original_alpha=0.3,
        off_screen=True
    )
    
    # Demonstrate individual level deformations to show multi-resolution effect
    print("\nDemonstrating individual level deformations...")
    
    # Apply only level 0 (coarse)
    level0_only = {0: deformed_control_points[0]}
    deformed_points_level0 = hffd.deform_mesh(level0_only, mesh_points)
    
    # Apply only level 1 (medium) if it exists
    if 1 in deformed_control_points:
        level1_only = {1: deformed_control_points[1]}
        deformed_points_level1 = hffd.deform_mesh(level1_only, mesh_points)
    
    # Apply only level 2 (fine) if it exists
    if 2 in deformed_control_points:
        level2_only = {2: deformed_control_points[2]}
        deformed_points_level2 = hffd.deform_mesh(level2_only, mesh_points)
    
    # Save comparison metrics
    if 1 in deformed_control_points and 2 in deformed_control_points:
        # Calculate displacement magnitudes
        full_displacement = np.linalg.norm(deformed_points - mesh_points, axis=1)
        level0_displacement = np.linalg.norm(deformed_points_level0 - mesh_points, axis=1)
        level1_displacement = np.linalg.norm(deformed_points_level1 - mesh_points, axis=1)
        level2_displacement = np.linalg.norm(deformed_points_level2 - mesh_points, axis=1)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        plt.hist(level0_displacement, bins=50, alpha=0.7, label="Level 0 (Coarse)")
        plt.hist(level1_displacement, bins=50, alpha=0.7, label="Level 1 (Medium)")
        plt.hist(level2_displacement, bins=50, alpha=0.7, label="Level 2 (Fine)")
        plt.hist(full_displacement, bins=50, alpha=0.7, label="Full Hierarchical")
        plt.xlabel("Displacement Magnitude")
        plt.ylabel("Count")
        plt.title("Displacement Distribution by Hierarchical Level")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "displacement_comparison.png"), dpi=300, bbox_inches='tight')
        
        print(f"Saved displacement comparison to {os.path.join(output_dir, 'displacement_comparison.png')}")
    
    print("\nHierarchical FFD example completed. Results saved to", output_dir)


if __name__ == "__main__":
    main()
