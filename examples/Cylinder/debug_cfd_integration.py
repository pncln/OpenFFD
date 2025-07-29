#!/usr/bin/env python3
"""
Debug CFD Integration

Step-by-step debugging of CFD solver integration.
"""

import sys
import numpy as np
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

def test_navier_stokes_solver_direct():
    """Test the Navier-Stokes solver directly."""
    print("Testing Navier-Stokes solver directly...")
    
    try:
        from openffd.cfd.navier_stokes_solver import NavierStokesSolver, FlowProperties, BoundaryCondition, BoundaryType
        
        # Create simple mesh data
        mesh_data = {
            'vertices': [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0]
            ],
            'cells': [[0], [1], [2], [3]],
            'faces': [],
            'boundary_patches': {}
        }
        
        # Create flow properties
        flow_properties = FlowProperties(
            reynolds_number=100.0,
            mach_number=0.1,
            reference_velocity=1.0,
            reference_density=1.0,
            reference_pressure=101325.0,
            reference_temperature=288.15,
            reference_length=1.0
        )
        
        print(f"  Flow properties: Re={flow_properties.reynolds_number}, U∞={flow_properties.reference_velocity}")
        
        # Create solver
        solver = NavierStokesSolver(mesh_data, flow_properties)
        print(f"  Solver created successfully")
        
        # Set boundary conditions
        wall_bc = BoundaryCondition(boundary_type=BoundaryType.WALL, velocity=np.array([0.0, 0.0, 0.0]))
        solver.set_boundary_condition('wall', wall_bc)
        print(f"  Boundary conditions set")
        
        # Solve
        result = solver.solve_steady_state(max_iterations=5, convergence_tolerance=1e-3)
        print(f"  Solution completed: converged={result['converged']}, iterations={result['iterations']}")
        
        # Compute forces
        forces = solver.compute_forces()
        print(f"  Forces: Cd={forces['drag_coefficient']:.6f}, Cl={forces['lift_coefficient']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_loading():
    """Test mesh loading from optimization runner."""
    print("\nTesting mesh loading...")
    
    try:
        from run_cylinder_optimization import CylinderOptimizationRunner
        
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        runner.load_mesh()
        
        print(f"  Mesh loaded: {len(runner.mesh_data['vertices'])} vertices")
        print(f"  Mesh keys: {list(runner.mesh_data.keys())}")
        
        # Check mesh structure
        vertices = runner.mesh_data['vertices']
        print(f"  Vertex sample: {vertices[0]} (first vertex)")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_boundary_conditions():
    """Test boundary condition setup."""
    print("\nTesting boundary condition setup...")
    
    try:
        from run_cylinder_optimization import CylinderOptimizationRunner
        
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        runner.load_mesh()
        
        # Test boundary condition parsing
        bc_data = runner._get_openfoam_boundary_conditions()
        print(f"  Boundary conditions parsed:")
        for name, data in bc_data.items():
            print(f"    {name}: {data}")
            
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfd_solver_creation():
    """Test CFD solver creation with actual mesh."""
    print("\nTesting CFD solver creation with actual mesh...")
    
    try:
        from run_cylinder_optimization import CylinderOptimizationRunner
        from openffd.cfd.navier_stokes_solver import NavierStokesSolver, FlowProperties
        
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        runner.load_mesh()
        
        # Setup flow properties
        flow_properties = FlowProperties(
            reynolds_number=100.0,
            mach_number=0.1,
            reference_velocity=1.0,
            reference_density=1.0,
            reference_pressure=101325.0,
            reference_temperature=288.15,
            reference_length=1.0
        )
        
        print(f"  Creating Navier-Stokes solver...")
        solver = NavierStokesSolver(runner.mesh_data, flow_properties)
        print(f"  ✓ Solver created successfully")
        
        # Check geometry
        print(f"    Geometry: {len(solver.geometry.cell_centers)} cells")
        print(f"    Cell volume range: [{np.min(solver.geometry.cell_volumes):.2e}, {np.max(solver.geometry.cell_volumes):.2e}]")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debugging tests."""
    print("=" * 60)
    print("CFD INTEGRATION DEBUGGING")
    print("=" * 60)
    
    tests = [
        ("Direct Navier-Stokes solver test", test_navier_stokes_solver_direct),
        ("Mesh loading test", test_mesh_loading),
        ("Boundary condition test", test_boundary_conditions),
        ("CFD solver creation test", test_cfd_solver_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED")
        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            
    print(f"\n" + "=" * 60)
    print(f"DEBUGGING SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All debugging tests passed!")
        return 0
    else:
        print("🔧 Some issues found - need further investigation")
        return 1

if __name__ == "__main__":
    sys.exit(main())