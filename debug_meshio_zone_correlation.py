#!/usr/bin/env python3
"""
Debug script to find the actual correlation between meshio cell blocks and Fluent zones.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import meshio
import numpy as np

def debug_meshio_zone_correlation():
    """Try to find zone information preserved in meshio data."""
    
    mesh_file = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/14.msh"
    
    print("=== MESHIO ZONE CORRELATION ANALYSIS ===")
    
    # Load with meshio
    try:
        mesh = meshio.read(mesh_file)
        
        print(f"Points: {len(mesh.points)}")
        print(f"Cell blocks: {len(mesh.cells)}")
        
        # Detailed cell block analysis
        print(f"\nDetailed cell blocks:")
        for i, cell_block in enumerate(mesh.cells):
            print(f"  Block {i}: type='{cell_block.type}', count={len(cell_block.data)}")
            
            # Sample some cell data to see if there are patterns
            if len(cell_block.data) > 0:
                sample_cells = cell_block.data[:3]  # First 3 cells
                print(f"    Sample cells: {sample_cells}")
                
                # Check point indices used by this block
                all_indices = set()
                for cell in cell_block.data[:100]:  # Sample 100 cells
                    all_indices.update(cell)
                
                if all_indices:
                    indices_array = np.array(list(all_indices))
                    # Get spatial bounds of points used by this cell block
                    block_points = mesh.points[indices_array]
                    bounds = {
                        'x': (block_points[:, 0].min(), block_points[:, 0].max()),
                        'y': (block_points[:, 1].min(), block_points[:, 1].max()),
                        'z': (block_points[:, 2].min(), block_points[:, 2].max())
                    }
                    print(f"    Spatial bounds: X=[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}], "
                          f"Y=[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}], "
                          f"Z=[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
                    
                    # Check if this might be a specific zone based on bounds
                    x_span = bounds['x'][1] - bounds['x'][0]
                    y_span = bounds['y'][1] - bounds['y'][0]
                    z_span = bounds['z'][1] - bounds['z'][0]
                    
                    print(f"    Spans: X={x_span:.3f}, Y={y_span:.3f}, Z={z_span:.3f}")
                    
                    # Try to identify zone type based on geometry
                    if z_span < 0.001:  # Very thin in Z
                        if x_span < 1.0 and y_span < 1.0:
                            print(f"    -> Likely: Small boundary (wedge_pos?)")
                        elif x_span > 15.0 and y_span > 15.0:
                            print(f"    -> Likely: Large boundary (launchpad?)")
                        elif bounds['z'][0] == bounds['z'][1] == 0.0:
                            print(f"    -> Likely: Symmetry plane (Z=0)")
                        else:
                            print(f"    -> Likely: Medium boundary")
                    else:
                        print(f"    -> Likely: Volume or thick boundary")
        
        # Check cell data for zone tags
        print(f"\nCell data analysis:")
        print(f"Cell data keys: {list(mesh.cell_data.keys())}")
        
        if mesh.cell_data:
            for key, data_list in mesh.cell_data.items():
                print(f"  Key '{key}':")
                for i, data in enumerate(data_list):
                    if len(data) > 0:
                        unique_vals = np.unique(data)
                        print(f"    Block {i}: {len(data)} values, unique: {unique_vals[:10]}...")
        
        # Check point data
        print(f"\nPoint data keys: {list(mesh.point_data.keys())}")
        
        # Check field data (zone names might be here)
        print(f"\nField data: {mesh.field_data}")
        
        # Check info
        print(f"\nMesh info: {mesh.info}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_meshio_zone_correlation()
