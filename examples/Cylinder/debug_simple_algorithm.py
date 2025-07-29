#!/usr/bin/env python3
"""
Advanced SIMPLE Algorithm Debugging

Deep diagnostic analysis of the SIMPLE algorithm implementation
to identify why it converges immediately with zero drag coefficient.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_simple_algorithm():
    """Comprehensive debugging of SIMPLE algorithm."""
    print("=" * 80)
    print("ADVANCED SIMPLE ALGORITHM DEBUGGING")
    print("=" * 80)
    
    try:
        # Import the actual cylinder optimization runner
        from run_cylinder_optimization import CylinderOptimizationRunner
        
        # Load the actual mesh
        runner = CylinderOptimizationRunner('optimization_config.yaml')
        runner.load_mesh()
        
        print(f"\n1. MESH ANALYSIS:")
        print(f"   Vertices: {len(runner.mesh_data['vertices'])}")
        print(f"   Cells: {len(runner.mesh_data.get('cells', []))}")
        print(f"   Faces: {len(runner.mesh_data.get('faces', []))}")
        print(f"   Boundary patches: {list(runner.mesh_data.get('boundary_patches', {}).keys())}")
        
        # Now create the Navier-Stokes solver directly
        from openffd.cfd.navier_stokes_solver import NavierStokesSolver, FlowProperties, BoundaryCondition, BoundaryType
        
        # Setup flow properties
        flow_properties = FlowProperties(
            reynolds_number=100.0,
            mach_number=0.1,
            reference_velocity=10.0,  # Use actual velocity
            reference_density=1.0,
            reference_pressure=101325.0,
            reference_temperature=288.15,
            reference_length=1.0
        )
        
        print(f"\n2. FLOW PROPERTIES:")
        print(f"   Reynolds number: {flow_properties.reynolds_number}")
        print(f"   Reference velocity: {flow_properties.reference_velocity} m/s")
        print(f"   Dynamic viscosity: {flow_properties.dynamic_viscosity:.2e} Pa·s")
        print(f"   Kinematic viscosity: {flow_properties.kinematic_viscosity:.2e} m²/s")
        
        # Create solver
        solver = NavierStokesSolver(runner.mesh_data, flow_properties)
        
        print(f"\n3. MESH GEOMETRY ANALYSIS:")
        print(f"   Cell centers shape: {solver.geometry.cell_centers.shape}")
        print(f"   Cell volumes range: [{np.min(solver.geometry.cell_volumes):.2e}, {np.max(solver.geometry.cell_volumes):.2e}]")
        print(f"   Face centers shape: {solver.geometry.face_centers.shape}")
        print(f"   Face areas range: [{np.min(solver.geometry.face_areas):.2e}, {np.max(solver.geometry.face_areas):.2e}]")
        print(f"   Face owner shape: {solver.geometry.face_owner.shape}")
        print(f"   Face neighbor shape: {solver.geometry.face_neighbor.shape}")
        print(f"   Boundary faces: {list(solver.geometry.boundary_faces.keys())}")
        
        # Check boundary face connectivity
        total_boundary_faces = sum(len(faces) for faces in solver.geometry.boundary_faces.values())
        print(f"   Total boundary faces: {total_boundary_faces}")
        
        print(f"\n4. BOUNDARY CONDITION ANALYSIS:")
        
        # Setup boundary conditions like in the real code
        wall_bc = BoundaryCondition(boundary_type=BoundaryType.WALL, velocity=np.array([0.0, 0.0, 0.0]))
        solver.set_boundary_condition('cylinder', wall_bc)
        
        farfield_bc = BoundaryCondition(
            boundary_type=BoundaryType.FARFIELD, 
            velocity=np.array([10.0, 0.0, 0.0]),
            pressure=101325.0
        )
        solver.set_boundary_condition('farfield', farfield_bc)
        
        print(f"   Boundary conditions set: {list(solver.boundary_conditions.keys())}")
        
        print(f"\n5. INITIAL FLOW FIELD:")
        print(f"   Velocity shape: {solver.flow_field.velocity.shape}")
        print(f"   Velocity range: [{np.min(solver.flow_field.velocity):.3f}, {np.max(solver.flow_field.velocity):.3f}]")
        print(f"   Pressure range: [{np.min(solver.flow_field.pressure):.1f}, {np.max(solver.flow_field.pressure):.1f}]")
        
        # Sample some cells to check initial conditions
        n_sample = min(5, len(solver.flow_field.velocity))
        print(f"   Sample velocities (first {n_sample}):")
        for i in range(n_sample):
            print(f"     Cell {i}: velocity = {solver.flow_field.velocity[i]}, pressure = {solver.flow_field.pressure[i]:.1f}")
        
        print(f"\n6. DETAILED SIMPLE ITERATION ANALYSIS:")
        
        # Let's manually step through one SIMPLE iteration
        print("   Starting manual SIMPLE iteration...")
        
        # Store initial state
        u_initial = solver.flow_field.velocity.copy()
        p_initial = solver.flow_field.pressure.copy()
        
        print(f"   Initial velocity L2 norm: {np.linalg.norm(u_initial):.6f}")
        print(f"   Initial pressure L2 norm: {np.linalg.norm(p_initial):.6f}")
        
        # Step 1: Solve momentum equations
        print("   \nStep 1: Momentum equation analysis...")
        momentum_residuals = []
        
        for component in range(3):
            try:
                A, b = solver._build_momentum_system(component)
                print(f"     Component {component}:")
                print(f"       Matrix shape: {A.shape}")
                print(f"       Matrix nnz: {A.nnz}")
                print(f"       RHS norm: {np.linalg.norm(b):.2e}")
                print(f"       Matrix condition number: {np.linalg.cond(A.toarray()) if A.shape[0] < 1000 else 'too large'}")
                
                # Check if matrix is singular
                diagonal = A.diagonal()
                zero_diag = np.sum(np.abs(diagonal) < 1e-15)
                print(f"       Zero diagonal entries: {zero_diag}/{len(diagonal)}")
                print(f"       Diagonal range: [{np.min(np.abs(diagonal[diagonal != 0])):.2e}, {np.max(np.abs(diagonal)):.2e}]")
                
                momentum_residuals.append(np.linalg.norm(b))
                
            except Exception as e:
                print(f"       Error building momentum system: {e}")
                momentum_residuals.append(1e6)
        
        # Step 2: Pressure correction analysis
        print("   \nStep 2: Pressure correction analysis...")
        try:
            # Let's examine the pressure correction system
            n_cells = len(solver.geometry.cell_centers)
            
            # Check velocity divergence
            divergences = []
            for i in range(min(10, n_cells)):
                div = solver._compute_velocity_divergence(i)
                divergences.append(div)
                
            print(f"     Sample velocity divergences: {divergences[:5]}")
            print(f"     Max divergence: {np.max(np.abs(divergences)):.2e}")
            
            # Try to build pressure correction system manually
            print("     Building pressure correction matrix...")
            A_pressure = solver._build_pressure_correction_matrix()
            if A_pressure is not None:
                print(f"     Pressure matrix shape: {A_pressure.shape}")
                print(f"     Pressure matrix nnz: {A_pressure.nnz}")
                
        except Exception as e:
            print(f"     Error in pressure correction: {e}")
        
        print(f"\n7. BOUNDARY CONDITION APPLICATION TEST:")
        
        # Apply boundary conditions manually and see what happens
        solver._apply_boundary_conditions()
        
        u_after_bc = solver.flow_field.velocity.copy()
        p_after_bc = solver.flow_field.pressure.copy()
        
        velocity_change = np.linalg.norm(u_after_bc - u_initial)
        pressure_change = np.linalg.norm(p_after_bc - p_initial)
        
        print(f"   Velocity change after BC application: {velocity_change:.2e}")
        print(f"   Pressure change after BC application: {pressure_change:.2e}")
        
        print(f"\n8. FACE CONNECTIVITY VALIDATION:")
        
        # Check face connectivity
        n_faces = len(solver.geometry.face_owner)
        internal_faces = np.sum(solver.geometry.face_neighbor >= 0)
        boundary_faces = n_faces - internal_faces
        
        print(f"   Total faces: {n_faces}")
        print(f"   Internal faces: {internal_faces}")
        print(f"   Boundary faces: {boundary_faces}")
        
        # Check if face normals are reasonable
        face_normal_mags = np.linalg.norm(solver.geometry.face_normals, axis=1)
        print(f"   Face normal magnitudes: [{np.min(face_normal_mags):.3f}, {np.max(face_normal_mags):.3f}]")
        print(f"   Zero normals: {np.sum(face_normal_mags < 1e-12)}")
        
        print(f"\n9. RUNNING ACTUAL SOLVE WITH DIAGNOSTICS:")
        
        # Reset to initial conditions
        solver._initialize_flow_field()
        
        # Run with very detailed monitoring
        result = debug_simple_solve(solver, max_iterations=5)
        
        print(f"\n10. FINAL DIAGNOSIS:")
        print(f"    Converged: {result['converged']}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Final residual: {result['final_residual']:.2e}")
        
        # Check final forces
        final_forces = solver.compute_forces()
        print(f"    Final drag coefficient: {final_forces['drag_coefficient']:.6f}")
        print(f"    Final lift coefficient: {final_forces['lift_coefficient']:.6f}")
        
        return solver, result
        
    except Exception as e:
        print(f"Critical error in debugging: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_simple_solve(solver, max_iterations=5):
    """Debug version of SIMPLE solve with detailed diagnostics."""
    print("   Starting diagnostic SIMPLE solve...")
    
    history = []
    
    for iteration in range(max_iterations):
        print(f"\n   === ITERATION {iteration} ===")
        
        # Store previous solution
        u_old = solver.flow_field.velocity.copy()
        p_old = solver.flow_field.pressure.copy()
        
        print(f"   Pre-iteration velocity L2: {np.linalg.norm(u_old):.6f}")
        print(f"   Pre-iteration pressure L2: {np.linalg.norm(p_old):.6f}")
        
        # Step 1: Momentum equations
        print("   Solving momentum equations...")
        try:
            momentum_residuals = solver._solve_momentum_equations()
            print(f"     Momentum residuals: {momentum_residuals}")
        except Exception as e:
            print(f"     Momentum solve failed: {e}")
            momentum_residuals = [1e6, 1e6, 1e6]
        
        u_after_momentum = solver.flow_field.velocity.copy()
        momentum_change = np.linalg.norm(u_after_momentum - u_old)
        print(f"     Velocity change from momentum: {momentum_change:.2e}")
        
        # Step 2: Pressure correction
        print("   Solving pressure correction...")
        try:
            pressure_residual = solver._solve_pressure_correction()
            print(f"     Pressure residual: {pressure_residual:.2e}")
        except Exception as e:
            print(f"     Pressure solve failed: {e}")
            pressure_residual = 1e6
        
        p_after_pressure = solver.flow_field.pressure.copy()
        pressure_change = np.linalg.norm(p_after_pressure - p_old)
        print(f"     Pressure change: {pressure_change:.2e}")
        
        # Step 3: Velocity correction
        print("   Correcting velocities...")
        solver._correct_velocity_and_pressure()
        
        u_after_correction = solver.flow_field.velocity.copy()
        correction_change = np.linalg.norm(u_after_correction - u_after_momentum)
        print(f"     Velocity change from correction: {correction_change:.2e}")
        
        # Step 4: Boundary conditions
        print("   Applying boundary conditions...")
        solver._apply_boundary_conditions()
        
        u_final = solver.flow_field.velocity.copy()
        p_final = solver.flow_field.pressure.copy()
        
        bc_velocity_change = np.linalg.norm(u_final - u_after_correction)
        bc_pressure_change = np.linalg.norm(p_final - p_after_pressure)
        print(f"     BC velocity change: {bc_velocity_change:.2e}")
        print(f"     BC pressure change: {bc_pressure_change:.2e}")
        
        # Compute total residuals
        velocity_residual = np.max(np.linalg.norm(u_final - u_old, axis=1))
        pressure_residual_total = np.max(np.abs(p_final - p_old))
        total_residual = max(velocity_residual, pressure_residual_total)
        
        print(f"   Total residuals:")
        print(f"     Velocity: {velocity_residual:.2e}")
        print(f"     Pressure: {pressure_residual_total:.2e}")
        print(f"     Combined: {total_residual:.2e}")
        
        # Store iteration data
        iteration_data = {
            'iteration': iteration,
            'velocity_residual': float(velocity_residual),
            'pressure_residual': float(pressure_residual_total),
            'total_residual': float(total_residual),
            'momentum_residuals': [float(r) for r in momentum_residuals],
            'converged': total_residual < 1e-6
        }
        history.append(iteration_data)
        
        # Check convergence
        if iteration_data['converged']:
            print(f"   CONVERGED after {iteration + 1} iterations")
            break
        elif total_residual == 0.0:
            print(f"   WARNING: Zero residual detected - likely algorithmic issue")
            break
            
        # Check for divergence
        if total_residual > 1e6:
            print(f"   DIVERGED: Residual too large")
            break
    
    return {
        'converged': iteration_data['converged'] if history else False,
        'iterations': len(history),
        'final_residual': float(total_residual) if history else 1.0,
        'history': history
    }

def main():
    """Run comprehensive SIMPLE debugging."""
    solver, result = debug_simple_algorithm()
    
    if solver and result:
        print("\n" + "=" * 80)
        print("DEBUGGING COMPLETE - ISSUES IDENTIFIED:")
        
        if result['final_residual'] == 0.0:
            print("❌ CRITICAL: Zero residual indicates no actual solving")
            print("   The SIMPLE algorithm is not properly implemented")
            
        if result['iterations'] <= 1:
            print("❌ CRITICAL: Immediate convergence indicates algorithmic failure")
            
        print("\nNext steps: Fix the identified issues in the SIMPLE implementation")
        print("=" * 80)
        
        return 0
    else:
        print("❌ DEBUGGING FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())