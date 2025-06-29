#!/usr/bin/env python3
"""
Test script for the new FFD generation modes.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from openffd.core.control_box import create_ffd_box

def test_ffd_modes():
    """Test all FFD generation modes with a simple wing-like shape."""
    print("Testing new FFD generation modes...")
    
    # Create a simple wing-like point cloud for testing
    np.random.seed(42)
    
    # Wing points - simulate a wing shape
    x = np.linspace(0, 10, 50)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(0, 1, 10)
    
    points = []
    for xi in x:
        for yi in y:
            for zi in z:
                # Create a wing-like shape
                wing_thickness = 0.2 * (1 - xi/10) * np.exp(-yi**2/2)
                if zi <= wing_thickness:
                    points.append([xi, yi, zi])
    
    mesh_points = np.array(points)
    print(f"Created test mesh with {len(mesh_points)} points")
    
    control_dim = (5, 4, 3)
    margin = 0.1
    
    # Test all modes
    modes = ["box", "convex", "surface"]
    
    for mode in modes:
        print(f"\n--- Testing {mode.upper()} mode ---")
        try:
            control_points, bounding_box = create_ffd_box(
                mesh_points,
                control_dim=control_dim,
                margin=margin,
                generation_mode=mode
            )
            
            print(f"✓ {mode} mode: Generated {len(control_points)} control points")
            print(f"  Bounding box: {bounding_box[0]} to {bounding_box[1]}")
            
            # Basic validation
            expected_points = control_dim[0] * control_dim[1] * control_dim[2]
            if len(control_points) == expected_points:
                print(f"  ✓ Correct number of control points ({expected_points})")
            else:
                print(f"  ✗ Wrong number of control points (expected {expected_points}, got {len(control_points)})")
                
        except Exception as e:
            print(f"✗ {mode} mode failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nFFD mode testing completed!")

if __name__ == "__main__":
    test_ffd_modes()
