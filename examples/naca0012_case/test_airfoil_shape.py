#!/usr/bin/env python3
"""
Test script to verify airfoil shape plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

def load_mesh_points(mesh_dir):
    """Load mesh points from OpenFOAM points file."""
    points_file = Path(mesh_dir) / "points"
    
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
    # Find leading edge (rightmost point)
    le_idx = np.argmax(airfoil_points[:, 0])
    le_point = airfoil_points[le_idx]
    
    # Separate upper and lower surfaces
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
    
    # Combine: upper surface + lower surface + close the loop
    ordered_points = np.vstack([upper_sorted, lower_sorted, upper_sorted[0:1]])
    
    return ordered_points

# Test with baseline and one optimization iteration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Load baseline
original_mesh_dir = Path("constant/polyMesh_original")
original_points = load_mesh_points(original_mesh_dir)
airfoil_indices = get_airfoil_boundary_points(original_mesh_dir)

if original_points is not None and airfoil_indices is not None:
    baseline_airfoil = original_points[airfoil_indices]
    baseline_ordered = sort_airfoil_points(baseline_airfoil)
    
    chord_length = 35.0492
    baseline_x = baseline_ordered[:, 0] / chord_length
    baseline_y = baseline_ordered[:, 1] / chord_length
    
    # Plot baseline
    ax1.plot(baseline_x, baseline_y, 'k-', linewidth=2, label='Baseline (Ordered)')
    ax1.fill(baseline_x, baseline_y, color='lightgray', alpha=0.3)
    ax1.set_title('Baseline Airfoil Shape')
    ax1.set_xlabel('X/c')
    ax1.set_ylabel('Y/c')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Show the difference between ordered and unordered
    unordered_x = baseline_airfoil[:, 0] / chord_length
    unordered_y = baseline_airfoil[:, 1] / chord_length
    ax2.plot(unordered_x, unordered_y, 'r-', linewidth=1, alpha=0.7, label='Unordered (Zig-zag)')
    ax2.plot(baseline_x, baseline_y, 'b-', linewidth=2, label='Ordered (Smooth)')
    ax2.set_title('Comparison: Ordered vs Unordered Points')
    ax2.set_xlabel('X/c')
    ax2.set_ylabel('Y/c')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend()
    
    print(f"Airfoil points: {len(airfoil_indices)}")
    print(f"Baseline airfoil extent: X=[{np.min(baseline_x):.3f}, {np.max(baseline_x):.3f}], Y=[{np.min(baseline_y):.3f}, {np.max(baseline_y):.3f}]")

plt.tight_layout()
plt.savefig('airfoil_shape_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("Airfoil shape test completed!")
print("File created: airfoil_shape_test.png")