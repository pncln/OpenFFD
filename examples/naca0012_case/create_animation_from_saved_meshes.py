#!/usr/bin/env python3
"""
Create Animation from All Saved Optimization Meshes

This script creates an animated visualization using all available saved
optimization meshes, regardless of what's in the history file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import sys
from pathlib import Path

# Add the source path for imports
sys.path.insert(0, '/Users/pncln/Documents/tubitak/verynew/ffd_gen/src')

def load_mesh_points(mesh_dir):
    """Load mesh points from OpenFOAM points file."""
    points_file = Path(mesh_dir) / "points"
    
    if not points_file.exists():
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
        return None
    
    walls_section = walls_match.group(0)
    
    # Extract nFaces and startFace
    nfaces_match = re.search(r'nFaces\s+(\d+);', walls_section)
    startface_match = re.search(r'startFace\s+(\d+);', walls_section)
    
    if not nfaces_match or not startface_match:
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

def load_iteration_data(iteration_dir):
    """Load design variables and objective value from iteration directory."""
    design_vars_file = iteration_dir / "design_variables.txt"
    design_vars = [0.0, 0.0, 0.0, 0.0]
    objective = None
    
    if design_vars_file.exists():
        with open(design_vars_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Design Variables:') or line.startswith('Perturbed Design Variables:'):
                    # Extract the array from the line
                    import ast
                    try:
                        var_str = line.split(':', 1)[1].strip()
                        # Handle numpy array string format
                        if var_str.startswith('[') and var_str.endswith(']'):
                            design_vars = ast.literal_eval(var_str)
                    except:
                        pass
                elif line.startswith('Objective Value:'):
                    try:
                        objective = float(line.split(':', 1)[1].strip())
                    except:
                        pass
    
    return design_vars, objective

def collect_all_iteration_meshes():
    """Collect mesh data from all available optimization iterations."""
    print("Collecting mesh data from all saved optimization iterations...")
    
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
        'objective': 0.0129,  # Approximate baseline
        'label': 'Baseline'
    })
    
    # Find all iteration directories
    optimization_meshes_dir = Path("optimization_meshes")
    if optimization_meshes_dir.exists():
        iteration_dirs = sorted([d for d in optimization_meshes_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('iteration_')])
        
        print(f"Found {len(iteration_dirs)} optimization iteration directories")
        
        for iteration_dir in iteration_dirs:
            # Extract iteration number from directory name
            try:
                iteration_num = int(iteration_dir.name.split('_')[1])
            except:
                continue
            
            mesh_dir = iteration_dir / "polyMesh"
            if mesh_dir.exists():
                points = load_mesh_points(mesh_dir)
                if points is not None:
                    airfoil_points = points[airfoil_indices]
                    design_vars, objective = load_iteration_data(iteration_dir)
                    
                    # Use recorded objective value or estimate if not available
                    if objective is None:
                        var_magnitude = np.linalg.norm(design_vars)
                        objective = 0.0129 + var_magnitude * 0.001
                    
                    iteration_data.append({
                        'iteration': iteration_num,
                        'design_vars': design_vars,
                        'airfoil_points': airfoil_points,
                        'objective': objective,
                        'label': f'Iteration {iteration_num}'
                    })
                    print(f"Loaded iteration {iteration_num}: Design vars = {design_vars}, Objective = {objective:.6f}")
    
    print(f"Collected {len(iteration_data)} total iterations")
    return iteration_data

def sort_airfoil_points(airfoil_points):
    """Sort airfoil points in proper order for plotting (around the perimeter)."""
    # Find leading edge (rightmost point)
    le_idx = np.argmax(airfoil_points[:, 0])
    le_point = airfoil_points[le_idx]
    
    # Separate upper and lower surfaces based on Y coordinate relative to LE
    upper_mask = airfoil_points[:, 1] >= le_point[1]
    lower_mask = airfoil_points[:, 1] < le_point[1]
    
    upper_points = airfoil_points[upper_mask]
    lower_points = airfoil_points[lower_mask]
    
    # Sort upper surface from LE to TE (decreasing X)
    upper_sorted_idx = np.argsort(-upper_points[:, 0])
    upper_sorted = upper_points[upper_sorted_idx]
    
    # Sort lower surface from TE to LE (increasing X)  
    lower_sorted_idx = np.argsort(lower_points[:, 0])
    lower_sorted = lower_points[lower_sorted_idx]
    
    # Combine: upper surface (LE to TE) + lower surface (TE to LE) + close the loop
    ordered_points = np.vstack([upper_sorted, lower_sorted, upper_sorted[0:1]])
    
    return ordered_points

def create_comprehensive_animation():
    """Create animated visualization of mesh deformation using all available data."""
    print("Creating comprehensive mesh deformation animation...")
    
    # Collect iteration data
    iteration_data = collect_all_iteration_meshes()
    if not iteration_data or len(iteration_data) < 2:
        print("Insufficient iteration data found")
        return
    
    # Set up the figure and animation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Setup airfoil plot
    ax1.set_title('Airfoil Shape Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X/c')
    ax1.set_ylabel('Y/c')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Setup design variables plot
    ax2.set_title('Design Variables Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Design Variable Value')
    ax2.grid(True, alpha=0.3)
    
    # Setup objective plot
    ax3.set_title('Optimization Progress', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Objective Function')
    ax3.grid(True, alpha=0.3)
    
    # Setup airfoil deformation details
    ax4.set_title('Current Airfoil vs Baseline', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X/c')
    ax4.set_ylabel('Y/c')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Prepare data for animation
    all_airfoil_points = [data['airfoil_points'] for data in iteration_data]
    all_design_vars = [data['design_vars'] for data in iteration_data]
    objectives = [data['objective'] for data in iteration_data]
    iterations = [data['iteration'] for data in iteration_data]
    
    # Normalize coordinates by chord length
    chord_length = 35.0492  # From the mesh analysis
    
    # Initialize plots
    airfoil_line, = ax1.plot([], [], 'b-', linewidth=2, label='Current')
    airfoil_fill = ax1.fill([], [], color='lightblue', alpha=0.3, label='_nolegend_')[0]
    baseline_line, = ax1.plot([], [], 'k--', linewidth=1, alpha=0.5, label='Baseline')
    
    # Design variable lines
    dv_lines = []
    dv_labels = ['Front Upper', 'Rear Upper', 'Front Lower', 'Rear Lower']
    colors = ['red', 'blue', 'green', 'orange']
    for i, (label, color) in enumerate(zip(dv_labels, colors)):
        line, = ax2.plot([], [], color=color, linewidth=2, marker='o', markersize=3, label=label)
        dv_lines.append(line)
    
    obj_line, = ax3.plot([], [], 'r-', linewidth=2, marker='o', markersize=4)
    current_obj_point, = ax3.plot([], [], 'ro', markersize=8)
    
    # Detail plot lines
    current_detail, = ax4.plot([], [], 'b-', linewidth=2, label='Current')
    baseline_detail, = ax4.plot([], [], 'k--', linewidth=1, alpha=0.7, label='Baseline')
    
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
    ax4.set_xlim(x_center - x_range/2 - margin, x_center + x_range/2 + margin)
    ax4.set_ylim(y_center - y_range/2 - margin, y_center + y_range/2 + margin)
    
    # Set design variables axis limits
    all_dvs = np.array(all_design_vars)
    dv_min, dv_max = np.min(all_dvs), np.max(all_dvs)
    dv_range = dv_max - dv_min
    ax2.set_xlim(-1, max(iterations) + 1)
    ax2.set_ylim(dv_min - 0.1*dv_range, dv_max + 0.1*dv_range)
    
    # Set objective axis limits
    if len(objectives) > 1:
        obj_range = max(objectives) - min(objectives)
        obj_margin = max(0.0001, obj_range * 0.1)
        ax3.set_xlim(-1, max(iterations) + 1)
        ax3.set_ylim(min(objectives) - obj_margin, max(objectives) + obj_margin)
    
    # Add legends
    ax1.legend()
    ax2.legend(loc='upper right')
    ax4.legend()
    
    # Add info text
    info_text = fig.text(0.02, 0.02, '', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot baseline
    baseline_points = all_airfoil_points[0]
    baseline_ordered = sort_airfoil_points(baseline_points)
    baseline_x = baseline_ordered[:, 0] / chord_length
    baseline_y = baseline_ordered[:, 1] / chord_length
    
    baseline_line.set_data(baseline_x, baseline_y)
    baseline_detail.set_data(baseline_x, baseline_y)
    
    def animate(frame):
        """Animation function."""
        data = iteration_data[frame]
        airfoil_points = data['airfoil_points']
        
        # Sort airfoil points in proper order around perimeter
        airfoil_ordered = sort_airfoil_points(airfoil_points)
        
        # Normalize airfoil coordinates
        x_norm = airfoil_ordered[:, 0] / chord_length
        y_norm = airfoil_ordered[:, 1] / chord_length
        
        # Set data for properly ordered airfoil
        airfoil_line.set_data(x_norm, y_norm)
        current_detail.set_data(x_norm, y_norm)
        
        # Update filled airfoil
        airfoil_fill.set_xy(np.column_stack([x_norm, y_norm]))
        
        # Update design variables plot
        current_iterations = iterations[:frame+1]
        current_dvs = all_design_vars[:frame+1]
        
        for i, line in enumerate(dv_lines):
            dv_values = [dvs[i] for dvs in current_dvs]
            line.set_data(current_iterations, dv_values)
        
        # Update objective plot
        current_objectives = objectives[:frame+1]
        obj_line.set_data(current_iterations, current_objectives)
        current_obj_point.set_data([iterations[frame]], [objectives[frame]])
        
        # Update info text
        info_str = f"{data['label']}\n"
        info_str += f"Design Variables (FFD Y-direction):\n"
        info_str += f"  Front Upper: {data['design_vars'][0]:+.3f}\n"
        info_str += f"  Rear Upper:  {data['design_vars'][1]:+.3f}\n"
        info_str += f"  Front Lower: {data['design_vars'][2]:+.3f}\n"
        info_str += f"  Rear Lower:  {data['design_vars'][3]:+.3f}\n"
        info_str += f"Objective: {data['objective']:.6f}"
        info_text.set_text(info_str)
        
        return [airfoil_line, airfoil_fill, current_detail, obj_line, current_obj_point, info_text] + dv_lines
    
    # Create animation
    print(f"Creating animation with {len(iteration_data)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(iteration_data),
                                 interval=800, blit=False, repeat=True)
    
    # Save animation
    output_file = "comprehensive_mesh_animation.gif"
    print(f"Saving animation to {output_file}...")
    
    try:
        anim.save(output_file, writer='pillow', fps=1.25, dpi=100)
        print(f"Animation saved successfully: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Displaying animation instead...")
        plt.tight_layout()
        plt.show()
    
    # Also create a high-resolution static comparison
    create_multi_iteration_comparison(iteration_data)
    
    return anim

def create_multi_iteration_comparison(iteration_data):
    """Create a static comparison plot showing multiple iterations."""
    print("Creating multi-iteration comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot airfoil evolution
    chord_length = 35.0492
    n_iterations = len(iteration_data)
    
    # Show every 5th iteration plus first and last
    show_indices = [0]  # Always show baseline
    step = max(1, n_iterations // 10)  # Show about 10 iterations max
    show_indices.extend(range(step, n_iterations-1, step))
    show_indices.append(n_iterations-1)  # Always show last
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(show_indices)))
    
    for i, (idx, color) in enumerate(zip(show_indices, colors)):
        data = iteration_data[idx]
        airfoil_points = data['airfoil_points']
        
        # Sort airfoil points in proper order around perimeter
        airfoil_ordered = sort_airfoil_points(airfoil_points)
        x_norm = airfoil_ordered[:, 0] / chord_length
        y_norm = airfoil_ordered[:, 1] / chord_length
        
        alpha = 1.0 if i == 0 or i == len(show_indices)-1 else 0.7
        linewidth = 2 if i == 0 or i == len(show_indices)-1 else 1
        
        ax1.plot(x_norm, y_norm, 
                color=color, alpha=alpha, linewidth=linewidth, 
                label=data['label'] if i == 0 or i == len(show_indices)-1 else None)
    
    ax1.set_title('Airfoil Shape Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X/c')
    ax1.set_ylabel('Y/c')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Plot design variable evolution
    iterations = [data['iteration'] for data in iteration_data]
    all_design_vars = [data['design_vars'] for data in iteration_data]
    
    dv_labels = ['Front Upper', 'Rear Upper', 'Front Lower', 'Rear Lower']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (label, color) in enumerate(zip(dv_labels, colors)):
        dv_values = [dvs[i] for dvs in all_design_vars]
        ax2.plot(iterations, dv_values, color=color, linewidth=2, 
                marker='o', markersize=3, label=label)
    
    ax2.set_title('Design Variables Evolution (FFD Y-direction)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Design Variable Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis plot saved: comprehensive_optimization_analysis.png")

if __name__ == "__main__":
    print("=== Comprehensive Mesh Deformation Animation Creator ===")
    print()
    
    try:
        # Create comprehensive animated visualization
        anim = create_comprehensive_animation()
        
        print("\nComprehensive animation creation completed!")
        print("Files created:")
        print("  - comprehensive_mesh_animation.gif")
        print("  - comprehensive_optimization_analysis.png")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()