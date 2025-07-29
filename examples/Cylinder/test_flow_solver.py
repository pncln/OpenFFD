#!/usr/bin/env python3
"""
Comprehensive Flow Solver Testing and Debugging

Tests the CFD flow solver implementation for cylinder flow to identify
and fix all issues with physics, boundary conditions, and convergence.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

from run_cylinder_optimization import CylinderOptimizationRunner


class FlowSolverTester:
    """Comprehensive flow solver testing suite."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        self.issues = []
        self.test_dir = Path("flow_solver_tests")
        self.test_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self):
        """Run comprehensive flow solver tests."""
        print("=" * 80)
        print("COMPREHENSIVE FLOW SOLVER TESTING AND DEBUGGING")
        print("=" * 80)
        
        # Test 1: Basic CFD simulation
        print("\n1. Testing basic CFD simulation...")
        self.test_basic_simulation()
        
        # Test 2: Boundary condition validation
        print("\n2. Testing boundary condition implementation...")
        self.test_boundary_conditions()
        
        # Test 3: Flow physics validation
        print("\n3. Testing flow physics...")
        self.test_flow_physics()
        
        # Test 4: Convergence behavior
        print("\n4. Testing convergence behavior...")
        self.test_convergence()
        
        # Test 5: Compare with analytical solutions
        print("\n5. Comparing with analytical solutions...")
        self.test_analytical_comparison()
        
        # Test 6: Mesh dependency
        print("\n6. Testing mesh dependency...")
        self.test_mesh_dependency()
        
        # Generate comprehensive report
        self.generate_test_report()
        
    def test_basic_simulation(self):
        """Test basic CFD simulation functionality."""
        try:
            # Create optimization runner
            runner = CylinderOptimizationRunner('optimization_config.yaml')
            runner.load_mesh()
            runner.setup_cfd_solver()
            
            # Run a short CFD simulation
            print("    Running CFD simulation...")
            cfd_result = runner._run_cfd_simulation()
            
            # Analyze results
            self.results['basic_simulation'] = {
                'success': True,
                'drag_coefficient': cfd_result.get('drag_coefficient', 0.0),
                'lift_coefficient': cfd_result.get('lift_coefficient', 0.0),
                'converged': cfd_result.get('converged', False),
                'iterations': cfd_result.get('iterations', 0),
                'residual': cfd_result.get('residual', 1.0),
                'has_velocity_field': 'velocity' in cfd_result and len(cfd_result['velocity']) > 0,
                'has_pressure_field': 'pressure' in cfd_result and len(cfd_result['pressure']) > 0,
                'has_temperature_field': 'temperature' in cfd_result and len(cfd_result['temperature']) > 0
            }
            
            # Check for basic issues
            if cfd_result.get('drag_coefficient', 0.0) <= 0:
                self.issues.append("CRITICAL: Drag coefficient is non-positive")
            
            if not cfd_result.get('converged', False):
                self.issues.append("WARNING: CFD simulation did not converge")
                
            if cfd_result.get('iterations', 0) == 0:
                self.issues.append("CRITICAL: No CFD iterations performed")
                
            print(f"    ✓ Basic simulation completed")
            print(f"      - Drag coefficient: {cfd_result.get('drag_coefficient', 0.0):.6f}")
            print(f"      - Converged: {cfd_result.get('converged', False)}")
            print(f"      - Iterations: {cfd_result.get('iterations', 0)}")
            
            # Save flow field data for analysis
            if 'velocity' in cfd_result:
                velocity = cfd_result['velocity']
                pressure = cfd_result['pressure']
                vertices = runner.mesh_data['vertices']
                
                # Save raw data
                np.save(self.test_dir / 'velocity_field.npy', velocity)
                np.save(self.test_dir / 'pressure_field.npy', pressure)
                np.save(self.test_dir / 'vertices.npy', vertices)
                
                # Analyze flow field statistics
                vel_mag = np.linalg.norm(velocity, axis=1)
                self.results['flow_statistics'] = {
                    'velocity_magnitude_min': float(np.min(vel_mag)),
                    'velocity_magnitude_max': float(np.max(vel_mag)),
                    'velocity_magnitude_mean': float(np.mean(vel_mag)),
                    'pressure_min': float(np.min(pressure)),
                    'pressure_max': float(np.max(pressure)),
                    'pressure_mean': float(np.mean(pressure)),
                    'n_vertices': len(vertices)
                }
                
                print(f"      - Velocity range: [{np.min(vel_mag):.3f}, {np.max(vel_mag):.3f}] m/s")
                print(f"      - Pressure range: [{np.min(pressure):.1f}, {np.max(pressure):.1f}] Pa")
                
        except Exception as e:
            self.results['basic_simulation'] = {'success': False, 'error': str(e)}
            self.issues.append(f"CRITICAL: Basic simulation failed: {e}")
            print(f"    ✗ Basic simulation failed: {e}")
            
    def test_boundary_conditions(self):
        """Test boundary condition implementation."""
        try:
            runner = CylinderOptimizationRunner('optimization_config.yaml')
            runner.load_mesh()
            runner.setup_cfd_solver()
            
            # Test boundary condition parsing
            bc_data = runner._get_openfoam_boundary_conditions()
            
            print("    Testing boundary condition parsing...")
            print(f"      - Cylinder velocity BC: {bc_data['cylinder']['velocity']}")
            print(f"      - Farfield velocity BC: {bc_data['inout']['velocity']}")
            print(f"      - Farfield pressure BC: {bc_data['inout']['pressure']}")
            
            # Validate boundary conditions
            cylinder_vel = bc_data['cylinder']['velocity']
            farfield_vel = bc_data['inout']['velocity']
            
            # Check cylinder no-slip condition
            if not np.allclose(cylinder_vel, [0.0, 0.0, 0.0], atol=1e-10):
                self.issues.append(f"ERROR: Cylinder should have no-slip BC, got {cylinder_vel}")
            else:
                print("      ✓ Cylinder no-slip boundary condition correct")
                
            # Check farfield velocity magnitude
            farfield_speed = np.linalg.norm(farfield_vel)
            if farfield_speed < 1e-6:
                self.issues.append("ERROR: Farfield velocity is essentially zero")
            else:
                print(f"      ✓ Farfield velocity magnitude: {farfield_speed:.3f} m/s")
                
            # Test boundary condition application to flow field
            print("    Testing boundary condition application...")
            vertices = runner.mesh_data['vertices']
            n_vertices = len(vertices)
            
            pressure = np.zeros(n_vertices)
            velocity = np.zeros((n_vertices, 3))
            temperature = np.full(n_vertices, 288.15)
            
            # Apply boundary conditions
            pressure, velocity, temperature = runner._apply_boundary_conditions_to_flow_field(
                vertices, bc_data, 1.0, pressure, velocity, temperature
            )
            
            # Check cylinder surface points (inside cylinder radius)
            cylinder_indices = []
            farfield_indices = []
            
            for i, vertex in enumerate(vertices):
                x, y, z = vertex
                r = np.sqrt(x**2 + y**2)
                if r < 0.5:  # Inside cylinder
                    cylinder_indices.append(i)
                else:
                    farfield_indices.append(i)
                    
            print(f"      - Found {len(cylinder_indices)} cylinder surface points")
            print(f"      - Found {len(farfield_indices)} farfield points")
            
            if len(cylinder_indices) > 0:
                # Check cylinder surface velocities
                cylinder_velocities = velocity[cylinder_indices]
                cylinder_vel_mags = np.linalg.norm(cylinder_velocities, axis=1)
                max_cylinder_vel = np.max(cylinder_vel_mags)
                
                if max_cylinder_vel > 1e-6:
                    self.issues.append(f"ERROR: Cylinder surface has non-zero velocity: max = {max_cylinder_vel:.6f}")
                else:
                    print("      ✓ Cylinder surface velocities are zero (no-slip)")
                    
            if len(farfield_indices) > 0:
                # Check farfield velocities
                farfield_velocities = velocity[farfield_indices]
                farfield_vel_mags = np.linalg.norm(farfield_velocities, axis=1)
                mean_farfield_vel = np.mean(farfield_vel_mags)
                
                print(f"      - Mean farfield velocity magnitude: {mean_farfield_vel:.3f} m/s")
                
                if mean_farfield_vel < 0.1:
                    self.issues.append(f"WARNING: Farfield velocities seem too low: mean = {mean_farfield_vel:.6f}")
                    
            self.results['boundary_conditions'] = {
                'cylinder_velocity_bc': cylinder_vel,
                'farfield_velocity_bc': farfield_vel,
                'n_cylinder_points': len(cylinder_indices),
                'n_farfield_points': len(farfield_indices),
                'max_cylinder_velocity': float(max_cylinder_vel) if len(cylinder_indices) > 0 else 0.0,
                'mean_farfield_velocity': float(mean_farfield_vel) if len(farfield_indices) > 0 else 0.0
            }
                    
        except Exception as e:
            self.issues.append(f"ERROR: Boundary condition test failed: {e}")
            print(f"    ✗ Boundary condition test failed: {e}")
            
    def test_flow_physics(self):
        """Test flow physics implementation."""
        try:
            print("    Analyzing flow physics...")
            
            # Load saved flow field data
            velocity = np.load(self.test_dir / 'velocity_field.npy')
            pressure = np.load(self.test_dir / 'pressure_field.npy')
            vertices = np.load(self.test_dir / 'vertices.npy')
            
            # Check continuity equation (∇·v = 0 for incompressible flow)
            print("      Checking continuity equation...")
            
            # Analyze velocity components
            u_velocities = velocity[:, 0]  # x-velocity
            v_velocities = velocity[:, 1]  # y-velocity
            w_velocities = velocity[:, 2]  # z-velocity
            
            print(f"        - U velocity range: [{np.min(u_velocities):.3f}, {np.max(u_velocities):.3f}]")
            print(f"        - V velocity range: [{np.min(v_velocities):.3f}, {np.max(v_velocities):.3f}]")
            print(f"        - W velocity range: [{np.min(w_velocities):.3f}, {np.max(w_velocities):.3f}]")
            
            # Check if flow is predominantly in x-direction (expected for cylinder flow)
            u_mean = np.mean(np.abs(u_velocities))
            v_mean = np.mean(np.abs(v_velocities))
            w_mean = np.mean(np.abs(w_velocities))
            
            print(f"        - Mean |U|: {u_mean:.3f}, Mean |V|: {v_mean:.3f}, Mean |W|: {w_mean:.3f}")
            
            if u_mean < v_mean:
                self.issues.append("WARNING: Flow is not predominantly in x-direction as expected")
                
            # Check Bernoulli's equation (p + 0.5*ρ*v² = constant for inviscid flow)
            print("      Checking Bernoulli's equation...")
            rho = 1.0  # Reference density
            velocity_squared = np.sum(velocity**2, axis=1)
            total_pressure = pressure + 0.5 * rho * velocity_squared
            
            total_pressure_std = np.std(total_pressure)
            total_pressure_mean = np.mean(total_pressure)
            
            print(f"        - Total pressure mean: {total_pressure_mean:.1f} Pa")
            print(f"        - Total pressure std: {total_pressure_std:.1f} Pa")
            print(f"        - Total pressure variation: {total_pressure_std/total_pressure_mean*100:.2f}%")
            
            # For potential flow, total pressure should be nearly constant
            if total_pressure_std/total_pressure_mean > 0.1:  # More than 10% variation
                self.issues.append(f"WARNING: Large total pressure variation: {total_pressure_std/total_pressure_mean*100:.1f}%")
                
            # Check stagnation point behavior
            print("      Checking stagnation point...")
            vel_magnitudes = np.linalg.norm(velocity, axis=1)
            min_vel_idx = np.argmin(vel_magnitudes)
            stagnation_point = vertices[min_vel_idx]
            stagnation_velocity = vel_magnitudes[min_vel_idx]
            stagnation_pressure = pressure[min_vel_idx]
            
            print(f"        - Stagnation point: ({stagnation_point[0]:.3f}, {stagnation_point[1]:.3f}, {stagnation_point[2]:.3f})")
            print(f"        - Stagnation velocity: {stagnation_velocity:.6f} m/s")
            print(f"        - Stagnation pressure: {stagnation_pressure:.1f} Pa")
            
            # Stagnation point should be at upstream cylinder surface (x ≈ -0.5, y ≈ 0)
            expected_stag_x = -0.5
            if abs(stagnation_point[0] - expected_stag_x) > 0.2:
                self.issues.append(f"WARNING: Stagnation point x-location seems wrong: {stagnation_point[0]:.3f}, expected ≈ {expected_stag_x}")
                
            self.results['flow_physics'] = {
                'u_velocity_range': [float(np.min(u_velocities)), float(np.max(u_velocities))],
                'v_velocity_range': [float(np.min(v_velocities)), float(np.max(v_velocities))],
                'w_velocity_range': [float(np.min(w_velocities)), float(np.max(w_velocities))],
                'total_pressure_variation': float(total_pressure_std/total_pressure_mean),
                'stagnation_point': stagnation_point.tolist(),
                'stagnation_velocity': float(stagnation_velocity),
                'stagnation_pressure': float(stagnation_pressure)
            }
            
        except Exception as e:
            self.issues.append(f"ERROR: Flow physics test failed: {e}")
            print(f"    ✗ Flow physics test failed: {e}")
            
    def test_convergence(self):
        """Test CFD convergence behavior."""
        try:
            print("    Testing convergence behavior...")
            
            runner = CylinderOptimizationRunner('optimization_config.yaml')
            runner.load_mesh()
            runner.setup_cfd_solver()
            
            # Run time integration and capture history
            history = runner._run_time_integration_with_history(100, 1e-6)
            
            print(f"      - Time steps completed: {len(history)}")
            
            if len(history) == 0:
                self.issues.append("CRITICAL: No time integration history captured")
                return
                
            # Analyze residual evolution
            residuals = [step['residual'] for step in history]
            converged_flags = [step['converged'] for step in history]
            
            final_converged = converged_flags[-1] if converged_flags else False
            final_residual = residuals[-1] if residuals else 1.0
            
            print(f"      - Final converged: {final_converged}")
            print(f"      - Final residual: {final_residual:.2e}")
            
            # Check residual reduction
            if len(residuals) > 1:
                initial_residual = residuals[0] 
                residual_reduction = initial_residual / final_residual
                print(f"      - Residual reduction: {residual_reduction:.2e}")
                
                if residual_reduction < 10:
                    self.issues.append(f"WARNING: Poor residual reduction: {residual_reduction:.2e}")
                    
            # Check for monotonic residual decrease
            non_monotonic_steps = 0
            for i in range(1, len(residuals)):
                if residuals[i] > residuals[i-1]:
                    non_monotonic_steps += 1
                    
            if non_monotonic_steps > len(residuals) * 0.3:  # More than 30% non-monotonic
                self.issues.append(f"WARNING: Non-monotonic residual behavior: {non_monotonic_steps}/{len(residuals)} steps")
                
            # Plot convergence history
            if len(residuals) > 1:
                plt.figure(figsize=(10, 6))
                plt.semilogy(residuals, 'b-', linewidth=2)
                plt.xlabel('Time Step')
                plt.ylabel('Residual')
                plt.title('CFD Convergence History')
                plt.grid(True)
                plt.savefig(self.test_dir / 'convergence_history.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"      ✓ Convergence plot saved to {self.test_dir / 'convergence_history.png'}")
                
            self.results['convergence'] = {
                'n_time_steps': len(history),
                'final_converged': final_converged,
                'final_residual': float(final_residual),
                'residual_reduction': float(residual_reduction) if len(residuals) > 1 else 1.0,
                'non_monotonic_fraction': float(non_monotonic_steps / len(residuals)) if residuals else 0.0
            }
            
        except Exception as e:
            self.issues.append(f"ERROR: Convergence test failed: {e}")
            print(f"    ✗ Convergence test failed: {e}")
            
    def test_analytical_comparison(self):
        """Compare with analytical solutions."""
        try:
            print("    Comparing with analytical potential flow solution...")
            
            # Load flow field data
            velocity = np.load(self.test_dir / 'velocity_field.npy')
            pressure = np.load(self.test_dir / 'pressure_field.npy')
            vertices = np.load(self.test_dir / 'vertices.npy')
            
            # Analytical potential flow around cylinder
            # u = U∞(1 - a²/r²)(1 + cos(2θ))
            # v = -U∞(a²/r²)sin(2θ)
            # p = p∞ - 0.5ρU∞²(1 - 4sin²θ/r²)
            
            U_inf = 10.0  # Farfield velocity from BC
            a = 0.5      # Cylinder radius
            rho = 1.0    # Density
            p_inf = 101325.0  # Reference pressure
            
            analytical_u = np.zeros(len(vertices))
            analytical_v = np.zeros(len(vertices))
            analytical_p = np.zeros(len(vertices))
            
            for i, vertex in enumerate(vertices):
                x, y, z = vertex
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                
                if r > a + 1e-6:  # Outside cylinder
                    # Potential flow solution
                    analytical_u[i] = U_inf * (1 - a**2/r**2) * (1 + np.cos(2*theta))
                    analytical_v[i] = -U_inf * (a**2/r**2) * np.sin(2*theta)
                    analytical_p[i] = p_inf - 0.5*rho*U_inf**2*(1 - 4*np.sin(theta)**2/r**2)
                else:
                    # Inside cylinder (should be zero velocity)
                    analytical_u[i] = 0.0
                    analytical_v[i] = 0.0
                    analytical_p[i] = p_inf + 0.5*rho*U_inf**2  # Stagnation pressure
                    
            # Compare velocities
            u_computed = velocity[:, 0]
            v_computed = velocity[:, 1]
            
            # Only compare points outside cylinder
            outside_mask = np.array([np.sqrt(v[0]**2 + v[1]**2) > a + 1e-6 for v in vertices])
            
            if np.sum(outside_mask) > 0:
                u_error = np.mean(np.abs(u_computed[outside_mask] - analytical_u[outside_mask]))
                v_error = np.mean(np.abs(v_computed[outside_mask] - analytical_v[outside_mask]))
                p_error = np.mean(np.abs(pressure[outside_mask] - analytical_p[outside_mask]))
                
                u_relative_error = u_error / U_inf
                v_relative_error = v_error / U_inf
                p_relative_error = p_error / (0.5*rho*U_inf**2)
                
                print(f"      - U velocity MAE: {u_error:.3f} m/s ({u_relative_error*100:.1f}%)")
                print(f"      - V velocity MAE: {v_error:.3f} m/s ({v_relative_error*100:.1f}%)")
                print(f"      - Pressure MAE: {p_error:.1f} Pa ({p_relative_error*100:.1f}%)")
                
                # Check if errors are reasonable
                if u_relative_error > 0.5:  # More than 50% error
                    self.issues.append(f"ERROR: Large U velocity error vs analytical: {u_relative_error*100:.1f}%")
                if v_relative_error > 0.5:
                    self.issues.append(f"ERROR: Large V velocity error vs analytical: {v_relative_error*100:.1f}%")
                if p_relative_error > 0.5:
                    self.issues.append(f"ERROR: Large pressure error vs analytical: {p_relative_error*100:.1f}%")
                    
                # Calculate drag coefficient from pressure integration
                # For potential flow around cylinder, theoretical drag = 0 (d'Alembert's paradox)
                print("      - Theoretical drag coefficient (potential flow): 0.0")
                
                self.results['analytical_comparison'] = {
                    'u_velocity_mae': float(u_error),
                    'v_velocity_mae': float(v_error),
                    'pressure_mae': float(p_error),
                    'u_relative_error': float(u_relative_error),
                    'v_relative_error': float(v_relative_error),
                    'p_relative_error': float(p_relative_error),
                    'comparison_points': int(np.sum(outside_mask))
                }
                
            else:
                self.issues.append("ERROR: No points found outside cylinder for analytical comparison")
                
        except Exception as e:
            self.issues.append(f"ERROR: Analytical comparison failed: {e}")
            print(f"    ✗ Analytical comparison failed: {e}")
            
    def test_mesh_dependency(self):
        """Test mesh dependency and consistency."""
        try:
            print("    Testing mesh consistency...")
            
            vertices = np.load(self.test_dir / 'vertices.npy')
            
            # Check mesh bounds
            x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
            
            print(f"      - X bounds: [{x_min:.3f}, {x_max:.3f}]")
            print(f"      - Y bounds: [{y_min:.3f}, {y_max:.3f}]") 
            print(f"      - Z bounds: [{z_min:.3f}, {z_max:.3f}]")
            
            # Check if cylinder is centered
            cylinder_points = []
            for vertex in vertices:
                x, y, z = vertex
                r = np.sqrt(x**2 + y**2)
                if r <= 0.51:  # Near cylinder surface
                    cylinder_points.append(vertex)
                    
            if len(cylinder_points) > 0:
                cylinder_points = np.array(cylinder_points)
                cylinder_center = np.mean(cylinder_points, axis=0)
                print(f"      - Cylinder center: ({cylinder_center[0]:.3f}, {cylinder_center[1]:.3f}, {cylinder_center[2]:.3f})")
                
                if abs(cylinder_center[0]) > 0.1 or abs(cylinder_center[1]) > 0.1:
                    self.issues.append(f"WARNING: Cylinder not centered: center = ({cylinder_center[0]:.3f}, {cylinder_center[1]:.3f})")
                    
            # Check domain size relative to cylinder
            domain_width = x_max - x_min
            domain_height = y_max - y_min
            cylinder_diameter = 1.0
            
            print(f"      - Domain width: {domain_width:.1f} (cylinder diameters: {domain_width/cylinder_diameter:.1f})")
            print(f"      - Domain height: {domain_height:.1f} (cylinder diameters: {domain_height/cylinder_diameter:.1f})")
            
            if domain_width < 10 * cylinder_diameter or domain_height < 10 * cylinder_diameter:
                self.issues.append("WARNING: Domain may be too small for accurate far-field conditions")
                
            self.results['mesh_dependency'] = {
                'domain_bounds': {
                    'x': [float(x_min), float(x_max)],
                    'y': [float(y_min), float(y_max)],
                    'z': [float(z_min), float(z_max)]
                },
                'cylinder_center': cylinder_center.tolist() if len(cylinder_points) > 0 else [0.0, 0.0, 0.0],
                'domain_width_diameters': float(domain_width/cylinder_diameter),
                'domain_height_diameters': float(domain_height/cylinder_diameter),
                'n_cylinder_points': len(cylinder_points)
            }
            
        except Exception as e:
            self.issues.append(f"ERROR: Mesh dependency test failed: {e}")
            print(f"    ✗ Mesh dependency test failed: {e}")
            
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("FLOW SOLVER TEST REPORT")
        print("=" * 80)
        
        # Summary
        total_issues = len(self.issues)
        critical_issues = len([issue for issue in self.issues if issue.startswith("CRITICAL")])
        error_issues = len([issue for issue in self.issues if issue.startswith("ERROR")])
        warning_issues = len([issue for issue in self.issues if issue.startswith("WARNING")])
        
        print(f"\nSUMMARY:")
        print(f"  Total Issues Found: {total_issues}")
        print(f"  - Critical: {critical_issues}")
        print(f"  - Errors: {error_issues}")
        print(f"  - Warnings: {warning_issues}")
        
        # Issue details
        if self.issues:
            print(f"\nISSUES FOUND:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i:2d}. {issue}")
                
        # Test results summary
        print(f"\nTEST RESULTS:")
        
        if 'basic_simulation' in self.results:
            basic = self.results['basic_simulation']
            if basic.get('success', False):
                print(f"  ✓ Basic Simulation: SUCCESS")
                print(f"    - Drag coefficient: {basic.get('drag_coefficient', 0.0):.6f}")
                print(f"    - Converged: {basic.get('converged', False)}")
            else:
                print(f"  ✗ Basic Simulation: FAILED")
                
        if 'flow_physics' in self.results:
            physics = self.results['flow_physics']
            print(f"  ✓ Flow Physics Analysis: COMPLETED")
            print(f"    - Total pressure variation: {physics.get('total_pressure_variation', 0.0)*100:.1f}%")
            
        if 'convergence' in self.results:
            conv = self.results['convergence']
            print(f"  ✓ Convergence Analysis: COMPLETED")
            print(f"    - Final residual: {conv.get('final_residual', 1.0):.2e}")
            print(f"    - Residual reduction: {conv.get('residual_reduction', 1.0):.2e}")
            
        if 'analytical_comparison' in self.results:
            analytical = self.results['analytical_comparison']
            print(f"  ✓ Analytical Comparison: COMPLETED")
            print(f"    - U velocity error: {analytical.get('u_relative_error', 0.0)*100:.1f}%")
            print(f"    - V velocity error: {analytical.get('v_relative_error', 0.0)*100:.1f}%")
            print(f"    - Pressure error: {analytical.get('p_relative_error', 0.0)*100:.1f}%")
            
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if critical_issues > 0:
            print("  🚨 CRITICAL ISSUES MUST BE FIXED IMMEDIATELY")
        if error_issues > 0:
            print("  ⚠️  ERROR ISSUES SHOULD BE ADDRESSED")
        if warning_issues > 0:
            print("  ℹ️  WARNING ISSUES SHOULD BE REVIEWED")
            
        if total_issues == 0:
            print("  🎉 All tests passed! Flow solver appears to be working correctly.")
        else:
            print("  🔧 Flow solver needs debugging and fixes.")
            
        # Save detailed results
        with open(self.test_dir / 'test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_issues': total_issues,
                    'critical_issues': critical_issues,
                    'error_issues': error_issues,
                    'warning_issues': warning_issues
                },
                'issues': self.issues,
                'results': self.results
            }, f, indent=2)
            
        print(f"\n📄 Detailed results saved to {self.test_dir / 'test_results.json'}")
        print(f"📊 Test artifacts saved to {self.test_dir}/")
        
        return total_issues == 0


def main():
    """Run comprehensive flow solver testing."""
    tester = FlowSolverTester()
    success = tester.run_all_tests()
    
    if not success:
        print(f"\n{'='*80}")
        print("NEXT STEPS:")
        print("1. Review all issues listed above")
        print("2. Fix critical and error issues first")
        print("3. Implement proper Navier-Stokes equations")
        print("4. Validate boundary condition implementation")
        print("5. Re-run tests until all issues are resolved")
        print("="*80)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())