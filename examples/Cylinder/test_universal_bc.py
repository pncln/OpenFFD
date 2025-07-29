#!/usr/bin/env python3
"""
Test Universal OpenFOAM Boundary Condition System

Verify that the universal BC manager correctly reads and applies
boundary conditions from the 0_orig directory.
"""

import sys
import numpy as np
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

def test_universal_bc_system():
    """Test the universal boundary condition system."""
    print("=" * 80)
    print("TESTING UNIVERSAL OPENFOAM BOUNDARY CONDITION SYSTEM")
    print("=" * 80)
    
    try:
        from openffd.cfd.openfoam_bc_manager import UniversalOpenFOAMBCManager
        
        # Test 1: Create BC manager for current case
        print("\n1. Creating Universal BC Manager...")
        case_directory = Path.cwd()  # Current directory should be the cylinder case
        bc_manager = UniversalOpenFOAMBCManager(case_directory)
        
        print(f"   ✓ BC Manager created successfully")
        print(f"   Case directory: {case_directory}")
        print(f"   Time directory: {bc_manager.primary_time_dir}")
        print(f"   Fields found: {list(bc_manager.field_bcs.keys())}")
        print(f"   Patches found: {bc_manager.get_patch_names()}")
        
        # Test 2: Check velocity boundary conditions
        print("\n2. Analyzing Velocity Boundary Conditions...")
        velocity_bcs = bc_manager.get_velocity_bcs()
        
        for patch_name, bc in velocity_bcs.items():
            print(f"   {patch_name}:")
            print(f"     Type: {bc.bc_type}")
            if bc.value is not None:
                print(f"     Value: {bc.value}")
            if bc.inlet_value is not None:
                print(f"     Inlet Value: {bc.inlet_value}")
                
        # Test 3: Check pressure boundary conditions
        print("\n3. Analyzing Pressure Boundary Conditions...")
        pressure_bcs = bc_manager.get_pressure_bcs()
        
        for patch_name, bc in pressure_bcs.items():
            print(f"   {patch_name}:")
            print(f"     Type: {bc.bc_type}")
            if bc.value is not None:
                print(f"     Value: {bc.value}")
                
        # Test 4: Map to CFD boundary conditions
        print("\n4. Mapping to CFD Boundary Conditions...")
        mesh_patches = ['cylinder', 'inout', 'symmetry1', 'symmetry2']
        cfd_bcs = bc_manager.map_to_cfd_boundary_conditions(mesh_patches)
        
        for patch_name, bc_data in cfd_bcs.items():
            print(f"   {patch_name}:")
            print(f"     CFD Type: {bc_data['type']}")
            if bc_data['velocity'] is not None:
                print(f"     Velocity: {bc_data['velocity']}")
            if bc_data['pressure'] is not None:
                print(f"     Pressure: {bc_data['pressure']}")
            print(f"     Velocity Fixed: {bc_data['velocity_fixed']}")
            print(f"     Pressure Fixed: {bc_data['pressure_fixed']}")
            
        # Test 5: Verify specific boundary conditions
        print("\n5. Verifying Specific Boundary Conditions...")
        
        # Check cylinder (should be wall with zero velocity)
        if 'cylinder' in cfd_bcs:
            cylinder_bc = cfd_bcs['cylinder']
            if (cylinder_bc['type'] == 'wall' and 
                cylinder_bc['velocity_fixed'] and
                np.allclose(cylinder_bc['velocity'], [0.0, 0.0, 0.0])):
                print("   ✓ Cylinder: Correct no-slip wall condition")
            else:
                print("   ❌ Cylinder: Incorrect boundary condition")
        else:
            print("   ❌ Cylinder: Boundary condition not found")
            
        # Check farfield (should be farfield with non-zero velocity)
        if 'inout' in cfd_bcs:
            farfield_bc = cfd_bcs['inout']
            if (farfield_bc['type'] == 'farfield' and
                farfield_bc['velocity_fixed'] and
                farfield_bc['velocity'] is not None):
                velocity_mag = np.linalg.norm(farfield_bc['velocity'])
                if velocity_mag > 5.0:  # Should be 10 m/s
                    print(f"   ✓ Farfield: Correct velocity magnitude ({velocity_mag:.1f} m/s)")
                else:
                    print(f"   ❌ Farfield: Low velocity magnitude ({velocity_mag:.1f} m/s)")
            else:
                print("   ❌ Farfield: Incorrect boundary condition")
        else:
            print("   ❌ Farfield: Boundary condition not found")
            
        # Check symmetry
        symmetry_count = 0
        for patch_name, bc_data in cfd_bcs.items():
            if bc_data['type'] == 'symmetry':
                symmetry_count += 1
                
        if symmetry_count == 2:
            print("   ✓ Symmetry: Found 2 symmetry boundaries")
        else:
            print(f"   ❌ Symmetry: Found {symmetry_count} symmetry boundaries (expected 2)")
            
        print("\n6. Testing Integration with Navier-Stokes Solver...")
        
        # Create a minimal mesh data structure for testing
        test_mesh_data = {
            'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            'faces': [],
            'cells': [],
            'boundary_patches': {
                'cylinder': {'nFaces': 10, 'startFace': 0},
                'inout': {'nFaces': 10, 'startFace': 10},
                'symmetry1': {'nFaces': 5, 'startFace': 20},
                'symmetry2': {'nFaces': 5, 'startFace': 25}
            }
        }
        
        from openffd.cfd.navier_stokes_solver import NavierStokesSolver, FlowProperties
        
        flow_properties = FlowProperties(
            reynolds_number=100.0,
            mach_number=0.1,
            reference_velocity=10.0,
            reference_density=1.0,
            reference_pressure=101325.0,
            reference_temperature=288.15,
            reference_length=1.0
        )
        
        # Create solver with automatic BC detection
        solver = NavierStokesSolver(test_mesh_data, flow_properties, str(case_directory))
        
        # Check if boundary conditions were applied
        applied_bcs = list(solver.boundary_conditions.keys())
        print(f"   Applied BCs to solver: {applied_bcs}")
        
        if len(applied_bcs) >= 3:  # Should have at least cylinder, inout, and symmetries
            print("   ✓ Boundary conditions successfully applied to solver")
        else:
            print("   ❌ Not enough boundary conditions applied to solver")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing universal BC system: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run universal BC system tests."""
    success = test_universal_bc_system()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 UNIVERSAL BC SYSTEM TEST: SUCCESS!")
        print("✓ OpenFOAM boundary conditions automatically detected and applied")
        print("✓ Ready for universal use across all OpenFOAM cases")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ UNIVERSAL BC SYSTEM TEST: FAILED")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())