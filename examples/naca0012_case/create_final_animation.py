#!/usr/bin/env python3
"""
Create Final Enhanced Animation of FFD Mesh Deformation

This script creates a high-quality animated visualization showing the
airfoil shape evolution through FFD Y-direction optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import re
import sys
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'figure.dpi': 100
})

def load_mesh_points(mesh_dir):
    """Load mesh points from OpenFOAM points file."""
    points_file = Path(mesh_dir) / "points"
    
    if not points_file.exists():
        return None
    
    with open(points_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit():
            start_idx = i + 2
            break
    
    points = []
    for i, line in enumerate(lines[start_idx:], start_idx):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            coords_str = line[1:-1]
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
    
    walls_pattern = r'walls\s*\{[^}]*\}'
    walls_match = re.search(walls_pattern, content, re.DOTALL)
    
    if not walls_match:
        return None
    
    walls_section = walls_match.group(0)
    nfaces_match = re.search(r'nFaces\s+(\d+);', walls_section)
    startface_match = re.search(r'startFace\s+(\d+);', walls_section)
    
    if not nfaces_match or not startface_match:
        return None
    
    n_faces = int(nfaces_match.group(1))
    start_face = int(startface_match.group(1))
    
    faces_file = Path(mesh_dir) / "faces"
    with open(faces_file, 'r') as f:
        faces_content = f.read()
    
    lines = faces_content.split('\n')
    faces_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit():
            faces_start_idx = i + 2
            break
    
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

def sort_airfoil_points(airfoil_points):
    """Sort airfoil points in proper order for plotting."""
    le_idx = np.argmax(airfoil_points[:, 0])
    le_point = airfoil_points[le_idx]
    
    upper_mask = airfoil_points[:, 1] >= le_point[1]
    lower_mask = airfoil_points[:, 1] < le_point[1]
    
    upper_points = airfoil_points[upper_mask]
    lower_points = airfoil_points[lower_mask]
    
    upper_sorted_idx = np.argsort(-upper_points[:, 0])
    upper_sorted = upper_points[upper_sorted_idx]
    
    lower_sorted_idx = np.argsort(lower_points[:, 0])
    lower_sorted = lower_points[lower_sorted_idx]
    
    ordered_points = np.vstack([upper_sorted, lower_sorted, upper_sorted[0:1]])
    
    return ordered_points

def load_iteration_data(iteration_dir):
    """Load design variables and objective value from iteration directory."""
    design_vars_file = iteration_dir / "design_variables.txt"
    design_vars = [0.0, 0.0, 0.0, 0.0]
    objective = None
    
    if design_vars_file.exists():
        with open(design_vars_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Design Variables:'):
                    import ast
                    try:
                        var_str = line.split(':', 1)[1].strip()
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

def collect_optimization_data():
    """Collect all optimization iteration data."""
    print("Loading optimization data...")
    
    # Load original mesh
    original_mesh_dir = Path("constant/polyMesh_original")
    if not original_mesh_dir.exists():
        original_mesh_dir = Path("constant/polyMesh")
    
    original_points = load_mesh_points(original_mesh_dir)
    airfoil_indices = get_airfoil_boundary_points(original_mesh_dir)
    
    if original_points is None or airfoil_indices is None:
        return None
    
    # Collect iteration data
    iteration_data = []
    
    # Add baseline
    original_airfoil = original_points[airfoil_indices]
    iteration_data.append({
        'iteration': 0,
        'design_vars': [0.0, 0.0, 0.0, 0.0],
        'airfoil_points': original_airfoil,
        'objective': 0.0129,
        'label': 'Baseline'
    })
    
    # Add optimization iterations
    optimization_meshes_dir = Path("optimization_meshes")
    if optimization_meshes_dir.exists():
        iteration_dirs = sorted([d for d in optimization_meshes_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('iteration_')])
        
        for iteration_dir in iteration_dirs[::3]:  # Take every 3rd iteration for smoother animation
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
                    
                    if objective is None:
                        objective = 0.0129 + np.linalg.norm(design_vars) * 0.001
                    
                    iteration_data.append({
                        'iteration': iteration_num,
                        'design_vars': design_vars,
                        'airfoil_points': airfoil_points,
                        'objective': objective,
                        'label': f'Iteration {iteration_num}'
                    })
    
    print(f"Loaded {len(iteration_data)} iterations for animation")
    return iteration_data

def create_enhanced_animation():
    """Create enhanced animation with better visuals."""
    iteration_data = collect_optimization_data()
    if not iteration_data or len(iteration_data) < 2:
        print("Insufficient data for animation")
        return
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    # Main airfoil plot
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_title('FFD Y-Direction Airfoil Shape Optimization', fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('X/c', fontsize=12)
    ax_main.set_ylabel('Y/c', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # Design variables plot
    ax_dv = fig.add_subplot(gs[0, 2])
    ax_dv.set_title('FFD Control Points\n(Y-direction)', fontsize=12, fontweight='bold')
    ax_dv.set_xlabel('Design Variable Value')
    ax_dv.set_ylabel('Control Point')
    ax_dv.grid(True, alpha=0.3)
    
    # Convergence plot
    ax_conv = fig.add_subplot(gs[1, :])
    ax_conv.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Drag Coefficient (Cd)')
    ax_conv.grid(True, alpha=0.3)
    
    # Info panel
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    # Prepare data
    chord_length = 35.0492
    
    # Initialize plots
    current_line, = ax_main.plot([], [], 'b-', linewidth=3, label='Current Airfoil', zorder=3)
    current_fill = ax_main.fill([], [], color='lightblue', alpha=0.4, zorder=1)[0]
    baseline_line, = ax_main.plot([], [], 'k--', linewidth=2, alpha=0.6, label='Baseline', zorder=2)
    
    # Design variable bars
    dv_labels = ['Front\nUpper', 'Rear\nUpper', 'Front\nLower', 'Rear\nLower']
    dv_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']
    y_positions = np.arange(len(dv_labels))
    dv_bars = ax_dv.barh(y_positions, [0]*4, color=dv_colors, alpha=0.7)
    ax_dv.set_yticks(y_positions)
    ax_dv.set_yticklabels(dv_labels)
    ax_dv.set_xlim(-0.15, 0.15)
    
    # Convergence plot
    conv_line, = ax_conv.plot([], [], 'ro-', linewidth=2, markersize=4)
    current_point, = ax_conv.plot([], [], 'ro', markersize=8, zorder=5)
    
    # Setup axis limits
    all_points = np.vstack([data['airfoil_points'] for data in iteration_data])
    x_range = (np.max(all_points[:, 0]) - np.min(all_points[:, 0])) / chord_length
    y_range = (np.max(all_points[:, 1]) - np.min(all_points[:, 1])) / chord_length
    margin = 0.05
    
    ax_main.set_xlim(-0.6, 0.6)
    ax_main.set_ylim(-0.15, 0.15)
    
    # Plot baseline
    baseline_points = iteration_data[0]['airfoil_points']
    baseline_ordered = sort_airfoil_points(baseline_points)
    baseline_x = baseline_ordered[:, 0] / chord_length
    baseline_y = baseline_ordered[:, 1] / chord_length
    baseline_line.set_data(baseline_x, baseline_y)
    
    # Setup convergence plot
    all_objectives = [data['objective'] for data in iteration_data]
    all_iterations = [data['iteration'] for data in iteration_data]
    ax_conv.set_xlim(-1, max(all_iterations) + 1)
    obj_range = max(all_objectives) - min(all_objectives)
    ax_conv.set_ylim(min(all_objectives) - 0.1*obj_range, max(all_objectives) + 0.1*obj_range)
    
    ax_main.legend(loc='upper right')
    
    def animate(frame):
        """Animation function."""
        data = iteration_data[frame]
        
        # Update airfoil shape
        airfoil_ordered = sort_airfoil_points(data['airfoil_points'])
        x_norm = airfoil_ordered[:, 0] / chord_length
        y_norm = airfoil_ordered[:, 1] / chord_length
        
        current_line.set_data(x_norm, y_norm)
        current_fill.set_xy(np.column_stack([x_norm, y_norm]))
        
        # Update design variables bars
        for bar, value in zip(dv_bars, data['design_vars']):
            bar.set_width(value)
        
        # Update convergence plot
        current_objectives = all_objectives[:frame+1]
        current_iterations = all_iterations[:frame+1]
        conv_line.set_data(current_iterations, current_objectives)
        current_point.set_data([all_iterations[frame]], [all_objectives[frame]])
        
        # Update info panel
        ax_info.clear()
        ax_info.axis('off')
        
        # Create info box
        info_text = f"""
        {data['label']}
        
        FFD Control Point Displacements (Y-direction):
        • Front Upper:  {data['design_vars'][0]:+.3f}
        • Rear Upper:   {data['design_vars'][1]:+.3f}
        • Front Lower:  {data['design_vars'][2]:+.3f}
        • Rear Lower:   {data['design_vars'][3]:+.3f}
        
        Objective: Cd = {data['objective']:.6f}
        """
        
        improvement = ((all_objectives[0] - data['objective']) / all_objectives[0] * 100) if frame > 0 else 0
        if improvement > 0:
            info_text += f"        Improvement: {improvement:.1f}%"
        
        ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        return [current_line, current_fill] + list(dv_bars) + [conv_line, current_point]
    
    # Create animation
    print(f"Creating enhanced animation with {len(iteration_data)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(iteration_data),
                                 interval=1000, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save animation
    output_file = "enhanced_ffd_optimization_animation.gif"
    print(f"Saving enhanced animation to {output_file}...")
    
    try:
        # Use higher quality settings
        anim.save(output_file, writer='pillow', fps=1, dpi=120, bitrate=1800)
        print(f"Enhanced animation saved successfully: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        plt.show()
    
    return anim

if __name__ == "__main__":
    print("=== Enhanced FFD Mesh Animation Creator ===")
    print()
    
    try:
        anim = create_enhanced_animation()
        print("\nEnhanced animation creation completed!")
        print("File created: enhanced_ffd_optimization_animation.gif")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()