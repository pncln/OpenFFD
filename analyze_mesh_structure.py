#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/pncln/Documents/tubitak/verynew/ffd_gen/src')

import meshio
import numpy as np

def analyze_mesh_structure():
    """Analyze the mesh file structure in detail to understand zone mapping"""
    print("=== ANALYZING MESH STRUCTURE ===")
    
    # Read mesh with meshio
    mesh = meshio.read('/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh')
    
    print(f"Points: {len(mesh.points)}")
    print(f"Cell blocks: {len(mesh.cells)}")
    print(f"Point data keys: {list(mesh.point_data.keys()) if mesh.point_data else 'None'}")
    print(f"Cell data keys: {list(mesh.cell_data.keys()) if mesh.cell_data else 'None'}")
    print(f"Field data: {mesh.field_data if mesh.field_data else 'None'}")
    
    print("\n=== CELL BLOCKS ANALYSIS ===")
    for i, cell_block in enumerate(mesh.cells):
        print(f"\nBlock {i}: {cell_block.type}")
        print(f"  Elements: {len(cell_block.data)}")
        
        # Get spatial bounds of this block
        if len(cell_block.data) > 0:
            # Sample up to 100 cells for spatial analysis
            sample_size = min(100, len(cell_block.data))
            sample_cells = cell_block.data[:sample_size]
            
            # Get all points used by sampled cells
            unique_points = set()
            for cell in sample_cells:
                unique_points.update(cell)
            
            if unique_points:
                points_array = np.array([mesh.points[p] for p in unique_points])
                bounds = {
                    'X': [points_array[:, 0].min(), points_array[:, 0].max()],
                    'Y': [points_array[:, 1].min(), points_array[:, 1].max()],
                    'Z': [points_array[:, 2].min(), points_array[:, 2].max()]
                }
                
                print(f"  Spatial bounds (sampled):")
                for axis, (min_val, max_val) in bounds.items():
                    span = max_val - min_val
                    print(f"    {axis}: [{min_val:.3f}, {max_val:.3f}] (span: {span:.3f})")
    
    # Check if there's cell data that might indicate zones
    print("\n=== CELL DATA ANALYSIS ===")
    if mesh.cell_data:
        for key, data_list in mesh.cell_data.items():
            print(f"\nCell data key: '{key}'")
            for i, data in enumerate(data_list):
                if len(data) > 0:
                    unique_values = np.unique(data)
                    print(f"  Block {i}: {len(data)} values, unique: {unique_values}")
    
    # Check field data for zone information
    print("\n=== FIELD DATA ANALYSIS ===")
    if mesh.field_data:
        for name, (tag, dim) in mesh.field_data.items():
            print(f"Field: '{name}' -> tag={tag}, dim={dim}")
    
    print("\n=== EXPECTED ZONES ===")
    expected_zones = [
        'launchpad', 'deflector', 'rocket', 'symmetry', 
        'outlet', 'inlet', 'wedge_pos', 'wedge_neg'
    ]
    
    for zone in expected_zones:
        print(f"Expected zone: {zone}")
    
    # Try to correlate cell blocks with expected zones based on size
    print("\n=== SIZE-BASED CORRELATION ATTEMPT ===")
    cell_sizes = [(i, len(block.data), block.type) for i, block in enumerate(mesh.cells)]
    cell_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for i, (block_idx, size, cell_type) in enumerate(cell_sizes):
        print(f"Block {block_idx}: {size:,} {cell_type} elements")
        
        # Try to guess which zone this might be
        if cell_type in ['triangle', 'quad']:  # Boundary elements
            if size > 1000000:
                guess = "launchpad (largest boundary)"
            elif size > 500000:
                guess = "symmetry or wedge_neg (large boundary)"
            elif size > 1000:
                guess = "rocket, deflector, or other wall"
            else:
                guess = "small boundary (inlet, outlet, etc.)"
            print(f"  → Likely: {guess}")

if __name__ == "__main__":
    analyze_mesh_structure()
