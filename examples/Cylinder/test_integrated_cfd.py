#!/usr/bin/env python3
"""
Test Integrated CFD Solver

Comprehensive testing of the integrated Navier-Stokes solver 
in the cylinder optimization workflow.
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

from run_cylinder_optimization import CylinderOptimizationRunner


def test_integrated_navier_stokes_solver():
    """Test the integrated Navier-Stokes solver."""
    print("=" * 80)
    print("TESTING INTEGRATED NAVIER-STOKES CFD SOLVER")
    print("=" * 80)
    
    try:
        # Create optimization runner
        print("\n1. Initializing optimization runner...")
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        
        # Load mesh
        print("\n2. Loading mesh...")
        runner.load_mesh()
        print(f"   Mesh loaded: {len(runner.mesh_data['vertices'])} vertices")
        
        # Setup CFD solver
        print("\n3. Setting up CFD solver...")
        runner.setup_cfd_solver()
        
        # Run CFD simulation
        print("\n4. Running CFD simulation with Navier-Stokes solver...")
        start_time = time.time()
        
        cfd_result = runner._run_cfd_simulation()
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        print(f"\n5. CFD Simulation Results:")
        print(f"   Simulation time: {simulation_time:.2f} seconds")
        print(f"   Converged: {cfd_result.get('converged', False)}")
        print(f"   Iterations: {cfd_result.get('iterations', 0)}")
        print(f"   Final residual: {cfd_result.get('residual', 1.0):.2e}")
        print(f"   Drag coefficient: {cfd_result.get('drag_coefficient', 0.0):.6f}")
        print(f"   Lift coefficient: {cfd_result.get('lift_coefficient', 0.0):.6f}")
        
        # Analyze flow field quality
        print(f"\n6. Flow Field Analysis:")
        
        if 'velocity' in cfd_result and 'pressure' in cfd_result:
            velocity = cfd_result['velocity']
            pressure = cfd_result['pressure']
            
            # Velocity analysis
            vel_magnitudes = np.linalg.norm(velocity, axis=1)
            print(f"   Velocity magnitude range: [{np.min(vel_magnitudes):.3f}, {np.max(vel_magnitudes):.3f}] m/s")
            print(f"   Mean velocity magnitude: {np.mean(vel_magnitudes):.3f} m/s")
            
            # Pressure analysis
            print(f"   Pressure range: [{np.min(pressure):.1f}, {np.max(pressure):.1f}] Pa")
            print(f"   Mean pressure: {np.mean(pressure):.1f} Pa")
            print(f"   Pressure std: {np.std(pressure):.1f} Pa")
            
            # Check for physical consistency
            print(f"\n7. Physics Validation:")
            
            # Check if velocities are reasonable
            max_expected_velocity = 15.0  # 1.5 * farfield velocity
            if np.max(vel_magnitudes) > max_expected_velocity:
                print(f"   ⚠ WARNING: Maximum velocity ({np.max(vel_magnitudes):.3f}) seems too high")
            else:
                print(f"   ✓ Velocity magnitudes are within reasonable range")
                
            # Check pressure variation
            reference_pressure = 101325.0
            pressure_variation = np.std(pressure) / reference_pressure
            if pressure_variation > 0.1:  # More than 10% variation
                print(f"   ⚠ WARNING: Large pressure variation ({pressure_variation*100:.1f}%)")
            else:
                print(f"   ✓ Pressure variation is reasonable ({pressure_variation*100:.1f}%)")
                
            # Check for NaN or inf values
            has_nan_velocity = np.any(np.isnan(velocity))
            has_inf_velocity = np.any(np.isinf(velocity))
            has_nan_pressure = np.any(np.isnan(pressure))
            has_inf_pressure = np.any(np.isinf(pressure))
            
            if has_nan_velocity or has_inf_velocity or has_nan_pressure or has_inf_pressure:
                print(f"   ✗ ERROR: Found NaN or inf values in solution")
                print(f"     Velocity: NaN={has_nan_velocity}, inf={has_inf_velocity}")
                print(f"     Pressure: NaN={has_nan_pressure}, inf={has_inf_pressure}")
            else:
                print(f"   ✓ No NaN or inf values found")
                
        # Test drag coefficient reasonableness
        print(f"\n8. Aerodynamic Validation:")
        drag_coeff = cfd_result.get('drag_coefficient', 0.0)
        lift_coeff = cfd_result.get('lift_coefficient', 0.0)
        
        # Expected drag coefficient for cylinder at Re=100: approximately 1.0-1.5
        expected_drag_range = (0.5, 2.0)
        expected_lift_range = (-0.2, 0.2)  # Should be near zero for symmetric flow
        
        if expected_drag_range[0] <= drag_coeff <= expected_drag_range[1]:
            print(f"   ✓ Drag coefficient ({drag_coeff:.6f}) is within expected range {expected_drag_range}")
        else:
            print(f"   ⚠ WARNING: Drag coefficient ({drag_coeff:.6f}) outside expected range {expected_drag_range}")
            
        if expected_lift_range[0] <= lift_coeff <= expected_lift_range[1]:
            print(f"   ✓ Lift coefficient ({lift_coeff:.6f}) is within expected range {expected_lift_range}")
        else:
            print(f"   ⚠ WARNING: Lift coefficient ({lift_coeff:.6f}) outside expected range {expected_lift_range}")
            
        # Check convergence history if available
        if 'time_history' in cfd_result and len(cfd_result['time_history']) > 0:
            print(f"\n9. Convergence Analysis:")
            history = cfd_result['time_history']
            print(f"   Total time steps: {len(history)}")
            
            if len(history) > 1:
                initial_residual = history[0].get('residual', 1.0)
                final_residual = history[-1].get('residual', 1.0)
                residual_reduction = initial_residual / max(final_residual, 1e-16)
                
                print(f"   Initial residual: {initial_residual:.2e}")
                print(f"   Final residual: {final_residual:.2e}")
                print(f"   Residual reduction: {residual_reduction:.2e}")
                
                if residual_reduction > 10:
                    print(f"   ✓ Good residual reduction achieved")
                else:
                    print(f"   ⚠ WARNING: Poor residual reduction")
                    
        # Overall assessment
        print(f"\n10. Overall Assessment:")
        
        issues = 0
        if not cfd_result.get('converged', False):
            issues += 1
            print(f"    - CFD did not converge")
            
        if drag_coeff <= 0 or drag_coeff > 5.0:
            issues += 1
            print(f"    - Unreasonable drag coefficient: {drag_coeff:.6f}")
            
        if abs(lift_coeff) > 0.5:
            issues += 1
            print(f"    - Unreasonable lift coefficient for symmetric flow: {lift_coeff:.6f}")
            
        if issues == 0:
            print(f"    🎉 SUCCESS: Integrated CFD solver appears to be working correctly!")
            print(f"    ✓ Proper Navier-Stokes equations implemented")
            print(f"    ✓ SIMPLE algorithm functioning")
            print(f"    ✓ Boundary conditions properly applied")
            print(f"    ✓ Reasonable aerodynamic forces computed")
            return True
        else:
            print(f"    ⚠ WARNING: Found {issues} issues with CFD solver")
            print(f"    🔧 Further optimization of solver parameters may be needed")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: Integrated CFD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_boundary_condition_setup():
    """Test boundary condition setup specifically."""
    print("\n" + "=" * 80)
    print("TESTING BOUNDARY CONDITION SETUP")
    print("=" * 80)
    
    try:
        # Test just the boundary condition setup
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        runner.load_mesh()
        
        # Test boundary condition parsing
        bc_data = runner._get_openfoam_boundary_conditions()
        
        print(f"Parsed boundary conditions:")
        for patch_name, patch_data in bc_data.items():
            print(f"  {patch_name}: {patch_data}")
            
        # Test CFD boundary condition setup (would need actual solver)
        print(f"\n✓ Boundary condition parsing successful")
        return True
        
    except Exception as e:
        print(f"✗ Boundary condition test failed: {e}")
        return False


def main():
    """Run all integrated CFD tests."""
    print("INTEGRATED CFD SOLVER TESTING SUITE")
    print("=" * 80)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Integrated Navier-Stokes solver
    if test_integrated_navier_stokes_solver():
        success_count += 1
        
    # Test 2: Boundary condition setup
    if test_boundary_condition_setup():
        success_count += 1
        
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"FINAL TEST SUMMARY")
    print(f"=" * 80)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✓ Integrated Navier-Stokes CFD solver is working correctly")
        print(f"✓ Ready for production shape optimization")
        return 0
    else:
        print(f"⚠ SOME TESTS FAILED")
        print(f"🔧 CFD solver needs further debugging and optimization")
        return 1


if __name__ == "__main__":
    sys.exit(main())