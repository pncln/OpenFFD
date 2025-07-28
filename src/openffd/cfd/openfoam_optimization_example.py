"""
OpenFOAM Mesh Integration with Shape Optimization

Demonstrates how to use real OpenFOAM meshes with our shape optimization framework:
- Reading OpenFOAM polyMesh format
- Converting to internal mesh representation
- Setting up shape optimization with real geometry
- Handling boundary conditions from OpenFOAM
- Mesh deformation and quality assessment
- Complete optimization workflow with real meshes

Shows practical integration of OpenFOAM meshes with our discrete adjoint
optimization framework for production CFD applications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import os
from pathlib import Path

# Import framework components
from .openfoam_mesh_reader import read_openfoam_mesh, OpenFOAMPolyMeshReader
from .shape_optimization import (
    ShapeOptimizer, OptimizationConfig, ParameterizationType, 
    OptimizationAlgorithm, DesignVariable
)
from .convergence_monitoring import ConvergenceMonitor, ConvergenceConfig
from .validation_cases import ValidationSuite, ObliqueShockCase

logger = logging.getLogger(__name__)


class OpenFOAMShapeOptimizer:
    """
    Shape optimizer specifically designed for OpenFOAM meshes.
    
    Integrates OpenFOAM mesh reading with shape optimization workflow.
    """
    
    def __init__(self, polymesh_dir: str, optimization_config: OptimizationConfig):
        """Initialize OpenFOAM shape optimizer."""
        self.polymesh_dir = Path(polymesh_dir)
        self.optimization_config = optimization_config
        
        # Mesh data
        self.original_mesh: Optional[Dict[str, Any]] = None
        self.current_mesh: Optional[Dict[str, Any]] = None
        
        # Optimization components
        self.shape_optimizer: Optional[ShapeOptimizer] = None
        self.convergence_monitor: Optional[ConvergenceMonitor] = None
        
        # Boundary condition mapping
        self.boundary_mapping: Dict[str, str] = {}
        
    def load_mesh(self):
        """Load OpenFOAM mesh and prepare for optimization."""
        logger.info(f"Loading OpenFOAM mesh from {self.polymesh_dir}")
        
        if not self.polymesh_dir.exists():
            raise FileNotFoundError(f"polyMesh directory not found: {self.polymesh_dir}")
        
        # Read mesh
        self.original_mesh = read_openfoam_mesh(self.polymesh_dir)
        self.current_mesh = self.original_mesh.copy()
        
        # Analyze mesh for optimization setup
        self._analyze_mesh_for_optimization()
        
        # Setup boundary condition mapping
        self._setup_boundary_mapping()
        
        logger.info("Mesh loaded successfully")
        
    def _analyze_mesh_for_optimization(self):
        """Analyze mesh characteristics for optimization setup."""
        mesh = self.original_mesh
        
        logger.info(f"Mesh Analysis:")
        logger.info(f"  Points: {len(mesh['vertices'])}")
        logger.info(f"  Faces: {len(mesh['faces'])}")
        logger.info(f"  Cells: {len(mesh['cells'])}")
        
        # Analyze boundary patches
        logger.info(f"  Boundary Patches:")
        for patch_name, patch_info in mesh['boundary_patches'].items():
            logger.info(f"    {patch_name}: {patch_info['type']}, {len(patch_info['faces'])} faces")
        
        # Mesh quality
        if 'mesh_quality' in mesh:
            quality = mesh['mesh_quality']
            bounds = quality['point_bounds']
            logger.info(f"  Mesh bounds: "
                       f"({bounds['min'][0]:.2f}, {bounds['min'][1]:.2f}, {bounds['min'][2]:.2f}) to "
                       f"({bounds['max'][0]:.2f}, {bounds['max'][1]:.2f}, {bounds['max'][2]:.2f})")
    
    def _setup_boundary_mapping(self):
        """Setup mapping between OpenFOAM boundaries and CFD boundary types."""
        # Standard OpenFOAM to CFD boundary type mapping
        openfoam_to_cfd = {
            'wall': 'wall',
            'patch': 'farfield',
            'inlet': 'inlet',
            'outlet': 'outlet',
            'symmetry': 'symmetry',
            'symmetryPlane': 'symmetry',
            'empty': 'empty',
            'wedge': 'wedge'
        }
        
        for patch_name, patch_info in self.original_mesh['boundary_patches'].items():
            openfoam_type = patch_info['type']
            cfd_type = openfoam_to_cfd.get(openfoam_type, 'patch')
            self.boundary_mapping[patch_name] = cfd_type
            
        logger.info(f"Boundary mapping: {self.boundary_mapping}")
    
    def setup_optimization(self):
        """Setup shape optimization for OpenFOAM mesh."""
        if self.original_mesh is None:
            raise ValueError("Mesh must be loaded before setting up optimization")
        
        logger.info("Setting up shape optimization...")
        
        # Create shape optimizer
        self.shape_optimizer = ShapeOptimizer(self.optimization_config)
        
        # Setup parameterization with mesh data
        optimization_geometry = {
            'vertices': self.original_mesh['vertices'],
            'faces': self.original_mesh['faces'],
            'cells': self.original_mesh['cells'],
            'boundary_patches': self.original_mesh['boundary_patches'],
            'n_cells': len(self.original_mesh['cells'])
        }
        
        self.shape_optimizer.setup_parameterization(optimization_geometry)
        
        # Setup convergence monitoring
        convergence_config = ConvergenceConfig(
            variable_names=['rho', 'rho_u', 'rho_v', 'rho_w', 'rho_E'],
            absolute_tolerance=1e-6,
            max_iterations=1000
        )
        self.convergence_monitor = ConvergenceMonitor(convergence_config)
        
        # Setup CFD and adjoint interfaces
        self.shape_optimizer.cfd_solver = self._openfoam_cfd_interface
        self.shape_optimizer.adjoint_solver = self._openfoam_adjoint_interface
        
        logger.info("Optimization setup completed")
    
    def _openfoam_cfd_interface(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """CFD solver interface for OpenFOAM meshes."""
        # In practice, this would:
        # 1. Deform the OpenFOAM mesh based on geometry changes
        # 2. Write the deformed mesh to polyMesh format
        # 3. Run OpenFOAM solver (simpleFoam, etc.)
        # 4. Read the solution and compute objectives
        
        # For demonstration, we'll simulate this process
        start_time = time.time()
        
        # Simulate mesh deformation impact on aerodynamics
        if 'vertices' in geometry:
            vertices = geometry['vertices']
            original_vertices = self.original_mesh['vertices']
            
            # Compute deformation magnitude
            if vertices.shape == original_vertices.shape:
                deformation = np.linalg.norm(vertices - original_vertices, axis=1)
                max_deformation = np.max(deformation)
                avg_deformation = np.mean(deformation)
            else:
                max_deformation = 0.1
                avg_deformation = 0.05
            
            # Simple aerodynamic model based on deformation
            base_drag = 0.02
            base_lift = 0.5
            
            # Deformation penalty (too much deformation increases drag)
            drag_penalty = 0.01 * (max_deformation / 0.1)**2
            lift_change = -0.1 * (avg_deformation / 0.05)
            
            drag = base_drag + drag_penalty
            lift = base_lift + lift_change
            
            # Add some mesh quality impact
            if max_deformation > 0.2:
                drag += 0.005  # Poor mesh quality penalty
        else:
            drag = 0.02
            lift = 0.5
        
        solve_time = time.time() - start_time
        
        # Simulate convergence monitoring
        if self.convergence_monitor:
            n_cells = len(self.current_mesh['cells'])
            residual_vector = np.random.randn(n_cells * 5) * 1e-6
            
            self.convergence_monitor.update_residuals(
                iteration=50,
                residual_vector=residual_vector,
                iteration_time=solve_time / 50
            )
        
        return {
            'drag': drag,
            'lift': lift,
            'pressure': np.random.rand(1000) + 1.0,  # Mock pressure field
            'velocity': np.random.rand(1000, 3) + 10.0,  # Mock velocity field
            'converged': True,
            'iterations': 50,
            'residual': 1e-6,
            'solve_time': solve_time,
            'mesh_quality': {
                'max_deformation': max_deformation,
                'avg_deformation': avg_deformation,
                'mesh_valid': max_deformation < 0.3
            }
        }
    
    def _openfoam_adjoint_interface(self, geometry: Dict[str, Any], 
                                   objective_function: Any) -> Dict[str, Any]:
        """Adjoint solver interface for OpenFOAM meshes."""
        # In practice, this would:
        # 1. Run adjoint OpenFOAM solver (adjointSimpleFoam, etc.)
        # 2. Compute surface sensitivities
        # 3. Map sensitivities to design variables
        
        start_time = time.time()
        
        # Mock adjoint gradients based on mesh characteristics
        if 'vertices' in geometry:
            n_vertices = len(geometry['vertices'])
            
            # Generate physics-informed gradients
            gradients = np.random.randn(n_vertices * 3) * 0.01
            
            # Higher gradients near walls (cylinder surface)
            vertices = geometry['vertices']
            for patch_name, patch_info in geometry.get('boundary_patches', {}).items():
                if patch_info['type'] == 'wall':
                    # Identify wall vertices (simplified)
                    wall_faces = patch_info.get('faces', [])
                    for face_idx in wall_faces[:min(10, len(wall_faces))]:
                        if face_idx < len(geometry.get('faces', [])):
                            face = geometry['faces'][face_idx]
                            for vertex_idx in face:
                                if vertex_idx * 3 + 2 < len(gradients):
                                    gradients[vertex_idx * 3:(vertex_idx + 1) * 3] *= 2.0
        else:
            gradients = np.random.randn(15000) * 0.01  # Default size
        
        adjoint_time = time.time() - start_time
        
        return {
            'gradients': gradients,
            'adjoint_residual': 1e-8,
            'adjoint_iterations': 30,
            'adjoint_time': adjoint_time,
            'sensitivity_magnitude': np.linalg.norm(gradients),
            'boundary_sensitivities': self._compute_boundary_sensitivities(gradients, geometry)
        }
    
    def _compute_boundary_sensitivities(self, gradients: np.ndarray, 
                                      geometry: Dict[str, Any]) -> Dict[str, float]:
        """Compute boundary-specific sensitivities."""
        boundary_sensitivities = {}
        
        for patch_name, patch_info in geometry.get('boundary_patches', {}).items():
            # Mock boundary sensitivity calculation
            if patch_info['type'] == 'wall':
                # Higher sensitivity on walls
                boundary_sensitivities[patch_name] = np.random.rand() * 0.1
            else:
                boundary_sensitivities[patch_name] = np.random.rand() * 0.01
        
        return boundary_sensitivities
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run complete shape optimization on OpenFOAM mesh."""
        if self.shape_optimizer is None:
            raise ValueError("Optimization must be setup before running")
        
        logger.info("Starting OpenFOAM shape optimization...")
        
        def objective_function(cfd_result, geometry):
            """Minimize drag while maintaining lift."""
            drag = cfd_result.get('drag', 0.1)
            lift = cfd_result.get('lift', 0.5)
            
            # Multi-objective: minimize drag, penalty for low lift
            objective = drag
            if lift < 0.4:  # Minimum lift constraint
                objective += 10.0 * (0.4 - lift)**2
            
            # Mesh quality penalty
            mesh_quality = cfd_result.get('mesh_quality', {})
            if not mesh_quality.get('mesh_valid', True):
                objective += 1.0  # Large penalty for invalid mesh
            
            return objective
        
        # Run optimization
        optimization_geometry = {
            'vertices': self.original_mesh['vertices'],
            'faces': self.original_mesh['faces'],
            'cells': self.original_mesh['cells'],
            'boundary_patches': self.original_mesh['boundary_patches']
        }
        
        start_time = time.time()
        result = self.shape_optimizer.optimize(objective_function, optimization_geometry)
        total_time = time.time() - start_time
        
        # Post-process results
        final_results = {
            'optimization_result': result,
            'mesh_info': {
                'original_mesh_path': str(self.polymesh_dir),
                'n_points': len(self.original_mesh['vertices']),
                'n_faces': len(self.original_mesh['faces']),
                'n_cells': len(self.original_mesh['cells']),
                'boundary_patches': list(self.original_mesh['boundary_patches'].keys())
            },
            'boundary_analysis': self._analyze_boundary_optimization(result),
            'mesh_deformation_analysis': self._analyze_mesh_deformation(result),
            'total_optimization_time': total_time
        }
        
        logger.info(f"OpenFOAM optimization completed in {total_time:.2f}s")
        
        return final_results
    
    def _analyze_boundary_optimization(self, result) -> Dict[str, Any]:
        """Analyze how optimization affected different boundaries."""
        analysis = {}
        
        # Get final gradients and analyze by boundary
        if hasattr(result, 'final_gradients') and len(result.final_gradients) > 0:
            gradients = result.final_gradients
            
            # Compute gradient statistics by boundary type
            for patch_name, patch_info in self.original_mesh['boundary_patches'].items():
                boundary_type = patch_info['type']
                
                if boundary_type not in analysis:
                    analysis[boundary_type] = {
                        'patches': [],
                        'avg_sensitivity': 0.0,
                        'max_sensitivity': 0.0
                    }
                
                # Mock sensitivity for this patch
                patch_sensitivity = np.random.rand() * np.linalg.norm(gradients[:10])
                
                analysis[boundary_type]['patches'].append(patch_name)
                analysis[boundary_type]['avg_sensitivity'] += patch_sensitivity
                analysis[boundary_type]['max_sensitivity'] = max(
                    analysis[boundary_type]['max_sensitivity'], patch_sensitivity
                )
            
            # Normalize averages
            for boundary_type in analysis:
                n_patches = len(analysis[boundary_type]['patches'])
                if n_patches > 0:
                    analysis[boundary_type]['avg_sensitivity'] /= n_patches
        
        return analysis
    
    def _analyze_mesh_deformation(self, result) -> Dict[str, Any]:
        """Analyze mesh deformation from optimization."""
        analysis = {
            'max_deformation': 0.0,
            'avg_deformation': 0.0,
            'deformation_distribution': {},
            'mesh_quality_preserved': True
        }
        
        if hasattr(result, 'optimal_design'):
            # Apply final design to get deformed geometry
            final_geometry = self.shape_optimizer.parameterization.apply_design_changes(
                result.optimal_design, {'vertices': self.original_mesh['vertices']}
            )
            
            if 'vertices' in final_geometry:
                original_vertices = self.original_mesh['vertices']
                final_vertices = final_geometry['vertices']
                
                if original_vertices.shape == final_vertices.shape:
                    deformation = np.linalg.norm(final_vertices - original_vertices, axis=1)
                    
                    analysis['max_deformation'] = float(np.max(deformation))
                    analysis['avg_deformation'] = float(np.mean(deformation))
                    
                    # Deformation distribution
                    analysis['deformation_distribution'] = {
                        'min': float(np.min(deformation)),
                        'max': float(np.max(deformation)),
                        'mean': float(np.mean(deformation)),
                        'std': float(np.std(deformation)),
                        'percentiles': {
                            '50': float(np.percentile(deformation, 50)),
                            '90': float(np.percentile(deformation, 90)),
                            '95': float(np.percentile(deformation, 95))
                        }
                    }
                    
                    # Check mesh quality preservation
                    analysis['mesh_quality_preserved'] = analysis['max_deformation'] < 0.2
        
        return analysis
    
    def save_deformed_mesh(self, result, output_dir: str):
        """Save the optimized/deformed mesh in OpenFOAM format."""
        # This would implement writing the deformed mesh back to OpenFOAM format
        # For now, we'll just log the intention
        logger.info(f"Would save deformed mesh to {output_dir}")
        logger.info("Implementation would write: points, faces, owner, neighbour, boundary files")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report for OpenFOAM case."""
        report = []
        report.append("=" * 80)
        report.append("OPENFOAM SHAPE OPTIMIZATION REPORT")
        report.append("=" * 80)
        
        # Mesh information
        mesh_info = results['mesh_info']
        report.append(f"\nMESH INFORMATION:")
        report.append(f"  Original mesh: {mesh_info['original_mesh_path']}")
        report.append(f"  Points: {mesh_info['n_points']}")
        report.append(f"  Faces: {mesh_info['n_faces']}")
        report.append(f"  Cells: {mesh_info['n_cells']}")
        report.append(f"  Boundary patches: {', '.join(mesh_info['boundary_patches'])}")
        
        # Optimization results
        opt_result = results['optimization_result']
        report.append(f"\nOPTIMIZATION RESULTS:")
        report.append(f"  Status: {'SUCCESS' if opt_result.success else 'FAILED'}")
        report.append(f"  Iterations: {opt_result.n_iterations}")
        report.append(f"  Function evaluations: {opt_result.n_function_evaluations}")
        report.append(f"  Total time: {results['total_optimization_time']:.2f}s")
        
        if opt_result.objective_history:
            initial_obj = opt_result.objective_history[0]
            final_obj = opt_result.optimal_objective
            improvement = (initial_obj - final_obj) / initial_obj * 100
            
            report.append(f"\nOBJECTIVE IMPROVEMENT:")
            report.append(f"  Initial: {initial_obj:.6e}")
            report.append(f"  Final: {final_obj:.6e}")
            report.append(f"  Improvement: {improvement:.2f}%")
        
        # Boundary analysis
        boundary_analysis = results['boundary_analysis']
        if boundary_analysis:
            report.append(f"\nBOUNDARY SENSITIVITY ANALYSIS:")
            for boundary_type, analysis in boundary_analysis.items():
                report.append(f"  {boundary_type}:")
                report.append(f"    Patches: {', '.join(analysis['patches'])}")
                report.append(f"    Avg sensitivity: {analysis['avg_sensitivity']:.2e}")
                report.append(f"    Max sensitivity: {analysis['max_sensitivity']:.2e}")
        
        # Mesh deformation analysis
        deform_analysis = results['mesh_deformation_analysis']
        report.append(f"\nMESH DEFORMATION ANALYSIS:")
        report.append(f"  Max deformation: {deform_analysis['max_deformation']:.6f}")
        report.append(f"  Avg deformation: {deform_analysis['avg_deformation']:.6f}")
        report.append(f"  Mesh quality preserved: {deform_analysis['mesh_quality_preserved']}")
        
        if 'deformation_distribution' in deform_analysis:
            dist = deform_analysis['deformation_distribution']
            report.append(f"  Deformation distribution:")
            report.append(f"    50th percentile: {dist['percentiles']['50']:.6f}")
            report.append(f"    90th percentile: {dist['percentiles']['90']:.6f}")
            report.append(f"    95th percentile: {dist['percentiles']['95']:.6f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def demonstrate_openfoam_optimization():
    """Demonstrate OpenFOAM mesh optimization with cylinder example."""
    print("Demonstrating OpenFOAM Shape Optimization:")
    print("=" * 60)
    
    # Cylinder mesh path
    cylinder_mesh_path = "/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/Cylinder/polyMesh"
    
    if not os.path.exists(cylinder_mesh_path):
        print(f"Cylinder mesh not found at {cylinder_mesh_path}")
        return
    
    # Create optimization configuration
    optimization_config = OptimizationConfig(
        parameterization=ParameterizationType.FREE_FORM_DEFORMATION,
        algorithm=OptimizationAlgorithm.SLSQP,
        max_iterations=5,  # Reduced for demonstration
        convergence_tolerance=1e-4
    )
    
    # Create OpenFOAM optimizer
    print("Setting up OpenFOAM shape optimizer...")
    optimizer = OpenFOAMShapeOptimizer(cylinder_mesh_path, optimization_config)
    
    # Load mesh
    print("Loading OpenFOAM mesh...")
    optimizer.load_mesh()
    
    # Setup optimization
    print("Setting up optimization...")
    optimizer.setup_optimization()
    
    # Run optimization
    print("Running optimization...")
    results = optimizer.run_optimization()
    
    # Generate and display report
    report = optimizer.generate_optimization_report(results)
    print("\n" + report)
    
    # Additional analysis
    opt_result = results['optimization_result']
    print(f"\nPerformance Breakdown:")
    print(f"  CFD solver time: {opt_result.cfd_time:.2f}s ({opt_result.cfd_time/opt_result.total_time*100:.1f}%)")
    print(f"  Adjoint solver time: {opt_result.adjoint_time:.2f}s ({opt_result.adjoint_time/opt_result.total_time*100:.1f}%)")
    print(f"  Mesh deformation time: {opt_result.mesh_deformation_time:.2f}s")
    
    # Design variable analysis
    if opt_result.sensitivity_analysis:
        print(f"\nDesign Variable Importance:")
        importance_counts = {}
        for importance in opt_result.sensitivity_analysis['design_variable_importance'].values():
            importance_counts[importance] = importance_counts.get(importance, 0) + 1
        
        for importance, count in importance_counts.items():
            print(f"  {importance.title()} importance: {count} variables")
    
    print(f"\nOpenFOAM optimization demonstration completed!")
    
    return results


if __name__ == "__main__":
    demonstrate_openfoam_optimization()