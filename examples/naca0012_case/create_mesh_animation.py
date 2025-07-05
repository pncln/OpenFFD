#!/usr/bin/env python3
"""
Create Animation of Mesh Deformation During Optimization

This script creates an animated visualization showing how the airfoil mesh
changes throughout the optimization iterations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import json
import re
import sys
from pathlib import Path

# Add the source path for imports
sys.path.insert(0, '/Users/pncln/Documents/tubitak/verynew/ffd_gen/src')

def load_mesh_points(mesh_dir):
    """Load mesh points from OpenFOAM points file."""
    points_file = Path(mesh_dir) / "points"
    
    if not points_file.exists():
        print(f"Points file not found: {points_file}")
        return None
    
    with open(points_file, 'r') as f:
        content = f.read()
    
    # Parse OpenFOAM points file
    lines = content.split('\n')
    start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit():
            start_idx = i + 2  # Skip the '(' line
            break
    
    # Extract coordinate data
    points = []
    for i, line in enumerate(lines[start_idx:], start_idx):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            coords_str = line[1:-1]  # Remove parentheses
            try:
                coords = [float(x) for x in coords_str.split()]
                if len(coords) >= 3:
                    points.append(coords[:3])
            except ValueError:
                continue
        elif line == ')':
            break
    
    return np.array(points)

def get_airfoil_boundary_points(mesh_dir):
    """Get airfoil boundary point indices."""
    boundary_file = Path(mesh_dir) / "boundary"
    
    with open(boundary_file, 'r') as f:
        content = f.read()
    
    # Find walls boundary
    walls_pattern = r'walls\s*\{[^}]*\}'
    walls_match = re.search(walls_pattern, content, re.DOTALL)
    
    if not walls_match:
        print("Could not find 'walls' boundary")
        return None
    
    walls_section = walls_match.group(0)
    
    # Extract nFaces and startFace
    nfaces_match = re.search(r'nFaces\s+(\d+);', walls_section)
    startface_match = re.search(r'startFace\s+(\d+);', walls_section)
    
    if not nfaces_match or not startface_match:
        print("Could not find nFaces or startFace")
        return None
    
    n_faces = int(nfaces_match.group(1))
    start_face = int(startface_match.group(1))
    
    # Load faces to get point indices
    faces_file = Path(mesh_dir) / "faces"
    with open(faces_file, 'r') as f:
        faces_content = f.read()
    
    # Parse faces file
    lines = faces_content.split('\n')
    faces_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit():
            faces_start_idx = i + 2
            break
    
    # Extract airfoil point indices from faces
    airfoil_point_set = set()
    for i in range(n_faces):
        face_idx = start_face + i
        line_idx = faces_start_idx + face_idx
        if line_idx < len(lines):
            line = lines[line_idx].strip()
            if '(' in line and ')' in line:
                paren_start = line.find('(')
                paren_end = line.find(')')
                if paren_start != -1 and paren_end != -1:
                    indices_str = line[paren_start+1:paren_end]
                    try:
                        indices = [int(x) for x in indices_str.split()]
                        airfoil_point_set.update(indices)
                    except ValueError:
                        continue
    
    return sorted(list(airfoil_point_set))

def load_optimization_history():
    """Load optimization history to get iteration data."""
    history_file = Path("optimization_history.json")
    
    if not history_file.exists():
        print("Optimization history file not found")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def collect_iteration_meshes():
    """Collect mesh data from all optimization iterations."""
    print("Collecting mesh data from optimization iterations...")
    
    # Load optimization history
    history = load_optimization_history()
    if not history:
        return None
    
    # Load original mesh
    original_mesh_dir = Path("constant/polyMesh_original")
    if not original_mesh_dir.exists():
        original_mesh_dir = Path("constant/polyMesh")
    
    original_points = load_mesh_points(original_mesh_dir)
    airfoil_indices = get_airfoil_boundary_points(original_mesh_dir)
    
    if original_points is None or airfoil_indices is None:
        print("Failed to load original mesh data")
        return None
    
    print(f"Found {len(airfoil_indices)} airfoil boundary points")
    
    # Collect iteration data
    iteration_data = []
    
    # Add baseline (original) mesh
    original_airfoil = original_points[airfoil_indices]
    iteration_data.append({
        'iteration': 0,
        'design_vars': [0.0, 0.0, 0.0, 0.0],
        'airfoil_points': original_airfoil,
        'objective': 0.013,  # Approximate baseline
        'label': 'Baseline'
    })
    
    # Add iteration meshes
    optimization_meshes_dir = Path("optimization_meshes")
    if optimization_meshes_dir.exists():
        for iteration_info in history.get('history', []):
            iteration = iteration_info['iteration']
            design_vars = iteration_info['design_vars']
            objective = iteration_info['objective_value']
            
            mesh_dir = optimization_meshes_dir / f"iteration_{iteration:03d}" / "polyMesh"
            if mesh_dir.exists():
                points = load_mesh_points(mesh_dir)
                if points is not None:
                    airfoil_points = points[airfoil_indices]
                    iteration_data.append({
                        'iteration': iteration,
                        'design_vars': design_vars,
                        'airfoil_points': airfoil_points,
                        'objective': objective,
                        'label': f'Iteration {iteration}'
                    })
                    print(f"Loaded iteration {iteration}: Cd = {objective:.6f}")
    
    print(f"Collected {len(iteration_data)} iterations")
    return iteration_data

def create_mesh_animation():
    """Create animated visualization of mesh deformation."""
    print("Creating mesh deformation animation...")
    
    # Collect iteration data
    iteration_data = collect_iteration_meshes()
    if not iteration_data:
        print("No iteration data found")
        return
    
    # Set up the figure and animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Setup airfoil plot
    ax1.set_title('Airfoil Shape Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X/c')
    ax1.set_ylabel('Y/c')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Setup objective plot
    ax2.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Drag Coefficient (Cd)')
    ax2.grid(True, alpha=0.3)
    
    # Prepare data for animation
    all_airfoil_points = [data['airfoil_points'] for data in iteration_data]
    objectives = [data['objective'] for data in iteration_data]
    iterations = [data['iteration'] for data in iteration_data]
    
    # Normalize coordinates by chord length
    chord_length = 35.0492  # From the mesh analysis
    
    # Initialize plots
    airfoil_line, = ax1.plot([], [], 'b-', linewidth=2, label='Current')
    baseline_line, = ax1.plot([], [], 'k--', linewidth=1, alpha=0.5, label='Baseline')
    
    obj_line, = ax2.plot([], [], 'r-', linewidth=2, marker='o', markersize=4)
    current_point, = ax2.plot([], [], 'ro', markersize=8)
    
    # Set axis limits
    all_points = np.vstack(all_airfoil_points)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    # Normalize by chord and add margins
    x_range = (x_max - x_min) / chord_length
    y_range = (y_max - y_min) / chord_length
    x_center = (x_max + x_min) / (2 * chord_length)
    y_center = (y_max + y_min) / (2 * chord_length)
    
    margin = 0.1
    ax1.set_xlim(x_center - x_range/2 - margin, x_center + x_range/2 + margin)
    ax1.set_ylim(y_center - y_range/2 - margin, y_center + y_range/2 + margin)
    
    if len(objectives) > 1:
        obj_range = max(objectives) - min(objectives)
        obj_margin = max(0.001, obj_range * 0.1)
        ax2.set_xlim(-0.5, max(iterations) + 0.5)
        ax2.set_ylim(min(objectives) - obj_margin, max(objectives) + obj_margin)
    
    # Add legend and text
    ax1.legend()
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot baseline
    baseline_points = all_airfoil_points[0]
    baseline_x = baseline_points[:, 0] / chord_length
    baseline_y = baseline_points[:, 1] / chord_length
    
    # Sort points for proper line plotting
    sorted_indices = np.argsort(baseline_x)
    baseline_line.set_data(baseline_x[sorted_indices], baseline_y[sorted_indices])
    
    def animate(frame):
        """Animation function."""
        data = iteration_data[frame]
        airfoil_points = data['airfoil_points']
        
        # Normalize airfoil coordinates
        x_norm = airfoil_points[:, 0] / chord_length
        y_norm = airfoil_points[:, 1] / chord_length
        
        # Sort points for proper line plotting (by x-coordinate)
        sorted_indices = np.argsort(x_norm)
        airfoil_line.set_data(x_norm[sorted_indices], y_norm[sorted_indices])
        
        # Update objective plot
        current_objectives = objectives[:frame+1]
        current_iterations = iterations[:frame+1]
        obj_line.set_data(current_iterations, current_objectives)
        current_point.set_data([iterations[frame]], [objectives[frame]])
        
        # Update info text
        info_str = f"{data['label']}\n"
        info_str += f"Design Variables:\n"
        info_str += f"  [{data['design_vars'][0]:+.3f}, {data['design_vars'][1]:+.3f},\n"
        info_str += f"   {data['design_vars'][2]:+.3f}, {data['design_vars'][3]:+.3f}]\n"
        info_str += f"Drag Coefficient: {data['objective']:.6f}"
        info_text.set_text(info_str)
        
        return airfoil_line, obj_line, current_point, info_text
    
    # Create animation
    print(f"Creating animation with {len(iteration_data)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(iteration_data),
                                 interval=1000, blit=False, repeat=True)
    
    # Save animation
    output_file = "mesh_deformation_animation.gif"
    print(f"Saving animation to {output_file}...")
    
    try:
        anim.save(output_file, writer='pillow', fps=1, dpi=100)
        print(f"Animation saved successfully: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Displaying animation instead...")
        plt.tight_layout()
        plt.show()
    
    return anim

def create_comparison_plot():
    """Create a static comparison plot showing multiple iterations."""
    print("Creating static comparison plot...")
    
    iteration_data = collect_iteration_meshes()
    if not iteration_data:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot airfoil evolution
    chord_length = 35.0492
    colors = plt.cm.viridis(np.linspace(0, 1, len(iteration_data)))
    
    for i, (data, color) in enumerate(zip(iteration_data, colors)):
        airfoil_points = data['airfoil_points']
        x_norm = airfoil_points[:, 0] / chord_length
        y_norm = airfoil_points[:, 1] / chord_length
        
        # Sort points for proper plotting
        sorted_indices = np.argsort(x_norm)
        
        alpha = 1.0 if i == 0 or i == len(iteration_data)-1 else 0.6
        linewidth = 2 if i == 0 or i == len(iteration_data)-1 else 1
        
        label = data['label'] if i == 0 or i == len(iteration_data)-1 else None
        ax1.plot(x_norm[sorted_indices], y_norm[sorted_indices], 
                color=color, alpha=alpha, linewidth=linewidth, label=label)
    
    ax1.set_title('Airfoil Shape Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X/c')
    ax1.set_ylabel('Y/c')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Plot optimization convergence
    objectives = [data['objective'] for data in iteration_data]
    iterations = [data['iteration'] for data in iteration_data]
    
    ax2.plot(iterations, objectives, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Drag Coefficient (Cd)')
    ax2.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(objectives) > 1:
        improvement = (objectives[0] - objectives[-1]) / objectives[0] * 100
        ax2.text(0.02, 0.98, f'Improvement: {improvement:.1f}%', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('airfoil_optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("Static comparison plot saved: airfoil_optimization_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("=== Mesh Deformation Animation Creator ===")
    print()
    
    # Check if we're in the right directory
    if not Path("optimization_history.json").exists():
        print("Error: optimization_history.json not found")
        print("Please run this script from the case directory")
        sys.exit(1)
    
    try:
        # Create animated visualization
        anim = create_mesh_animation()
        
        # Create static comparison
        create_comparison_plot()
        
        print("\nAnimation creation completed!")
        print("Files created:")
        print("  - mesh_deformation_animation.gif")
        print("  - airfoil_optimization_comparison.png")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()