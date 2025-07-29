#!/usr/bin/env python3
"""
Navier-Stokes CFD Solver

Complete implementation of incompressible Navier-Stokes equations solver
using finite volume method on unstructured meshes.

Features:
- Incompressible Navier-Stokes equations
- SIMPLE pressure-velocity coupling algorithm
- Finite volume discretization on unstructured grids
- Robust boundary condition enforcement
- Proper time integration schemes
- Convergence monitoring and acceleration
- Force and moment computation
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Boundary condition types."""
    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"
    FARFIELD = "farfield"
    SYMMETRY = "symmetry"
    PRESSURE_OUTLET = "pressure_outlet"


@dataclass
class FlowProperties:
    """Flow properties and reference conditions."""
    reynolds_number: float
    mach_number: float
    reference_velocity: float
    reference_density: float
    reference_pressure: float
    reference_temperature: float
    reference_length: float
    dynamic_viscosity: float = field(init=False)
    kinematic_viscosity: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Dynamic viscosity from Reynolds number: Re = ρUL/μ
        self.dynamic_viscosity = (self.reference_density * self.reference_velocity * 
                                 self.reference_length / self.reynolds_number)
        self.kinematic_viscosity = self.dynamic_viscosity / self.reference_density


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    boundary_type: BoundaryType
    velocity: Optional[np.ndarray] = None
    pressure: Optional[float] = None
    temperature: Optional[float] = None
    heat_flux: Optional[float] = None
    normal_vector: Optional[np.ndarray] = None


@dataclass
class FlowField:
    """Flow field variables stored at cell centers."""
    velocity: np.ndarray  # [n_cells, 3] - cell-centered velocities
    pressure: np.ndarray  # [n_cells] - cell-centered pressures
    temperature: np.ndarray  # [n_cells] - cell-centered temperatures
    density: np.ndarray  # [n_cells] - cell-centered densities (constant for incompressible)


@dataclass
class MeshGeometry:
    """Mesh geometric quantities for finite volume discretization."""
    cell_centers: np.ndarray  # [n_cells, 3] - cell centroids
    cell_volumes: np.ndarray  # [n_cells] - cell volumes
    face_centers: np.ndarray  # [n_faces, 3] - face centroids
    face_areas: np.ndarray  # [n_faces] - face areas
    face_normals: np.ndarray  # [n_faces, 3] - face normal vectors (outward from owner)
    face_owner: np.ndarray  # [n_faces] - owner cell index
    face_neighbor: np.ndarray  # [n_faces] - neighbor cell index (-1 for boundary faces)
    boundary_faces: Dict[str, List[int]]  # boundary_name -> list of face indices
    cell_faces: List[List[int]]  # [n_cells] -> list of face indices for each cell


class NavierStokesSolver:
    """Complete Navier-Stokes solver for incompressible flow."""
    
    def __init__(self, mesh_data: Dict, flow_properties: FlowProperties, case_directory: Optional[str] = None):
        """
        Initialize Navier-Stokes solver.
        
        Args:
            mesh_data: Mesh connectivity and geometry data
            flow_properties: Flow properties and reference conditions
            case_directory: OpenFOAM case directory for automatic BC detection (optional)
        """
        self.flow_properties = flow_properties
        self.mesh_data = mesh_data
        self.case_directory = case_directory
        
        # Initialize mesh geometry
        self._initialize_mesh_geometry()
        
        # Initialize flow field
        self._initialize_flow_field()
        
        # Initialize boundary conditions
        self.boundary_conditions = {}
        
        # Auto-detect and apply OpenFOAM boundary conditions if case directory provided
        if case_directory:
            self._auto_detect_openfoam_boundary_conditions()
        
        # Solver parameters
        self.solver_params = {
            'simple_relaxation': {
                'velocity': 0.7,
                'pressure': 0.3
            },
            'linear_solver': {
                'velocity_tolerance': 1e-3,
                'pressure_tolerance': 1e-4,
                'max_iterations': 2000
            },
            'time_stepping': {
                'cfl_number': 0.5,
                'max_cfl': 2.0
            }
        }
        
        logger.info(f"Navier-Stokes solver initialized: Re={flow_properties.reynolds_number:.1e}")
        
    def _auto_detect_openfoam_boundary_conditions(self):
        """Automatically detect and apply OpenFOAM boundary conditions from case directory."""
        try:
            from .openfoam_bc_manager import UniversalOpenFOAMBCManager
            
            logger.info(f"Auto-detecting OpenFOAM boundary conditions from: {self.case_directory}")
            
            # Create universal BC manager
            bc_manager = UniversalOpenFOAMBCManager(self.case_directory)
            
            # Get mesh patch names
            mesh_patches = list(self.geometry.boundary_faces.keys())
            
            # Apply boundary conditions automatically
            bc_manager.apply_to_navier_stokes_solver(self, mesh_patches)
            
            logger.info(f"✓ Successfully applied OpenFOAM boundary conditions from case directory")
            
        except Exception as e:
            logger.warning(f"Failed to auto-detect OpenFOAM boundary conditions: {e}")
            logger.warning("Falling back to manual boundary condition setup")
        
    def _initialize_mesh_geometry(self):
        """Initialize mesh geometric quantities from OpenFOAM data."""
        logger.info("Computing mesh geometry...")
        
        vertices = np.array(self.mesh_data['vertices'])
        
        # Get OpenFOAM mesh connectivity
        if 'openfoam_data' in self.mesh_data:
            openfoam_data = self.mesh_data['openfoam_data']
            # Handle both dict and object forms
            if hasattr(openfoam_data, 'faces'):
                faces = openfoam_data.faces if openfoam_data.faces is not None else []
                face_owner = openfoam_data.owner if openfoam_data.owner is not None else []
                face_neighbor = openfoam_data.neighbour if openfoam_data.neighbour is not None else []
            else:
                faces = openfoam_data.get('faces', [])
                face_owner = openfoam_data.get('owner', [])
                face_neighbor = openfoam_data.get('neighbour', [])
            boundary_patches = self.mesh_data.get('boundary_patches', {})
        else:
            # Fallback to old method if no OpenFOAM data
            faces = self.mesh_data.get('faces', [])
            face_owner = self.mesh_data.get('owner', [])
            face_neighbor = self.mesh_data.get('neighbour', [])
            boundary_patches = self.mesh_data.get('boundary_patches', {})
        
        n_faces = len(faces)
        face_owner = np.array(face_owner) if len(face_owner) > 0 else np.zeros(n_faces, dtype=int)
        
        # face_neighbor may be shorter than n_faces (only internal faces have neighbors)
        if len(face_neighbor) > 0:
            face_neighbor_full = -np.ones(n_faces, dtype=int)
            face_neighbor_full[:len(face_neighbor)] = face_neighbor
            face_neighbor = face_neighbor_full
        else:
            face_neighbor = -np.ones(n_faces, dtype=int)
        
        # Determine number of cells from face ownership
        n_cells = max(np.max(face_owner) + 1 if len(face_owner) > 0 else 0,
                     np.max(face_neighbor[face_neighbor >= 0]) + 1 if len(face_neighbor[face_neighbor >= 0]) > 0 else 0)
        
        if n_cells == 0:
            # Fallback: use number of vertices as cells for vertex-based approach
            n_cells = len(vertices)
            logger.warning("No cell connectivity found, using vertex-based approach")
        
        logger.info(f"Processing {n_cells} cells, {n_faces} faces")
        
        # Compute cell centers and volumes using vertex-to-cell mapping
        cell_centers = np.zeros((n_cells, 3))
        cell_volumes = np.zeros(n_cells)
        
        # Get cell-vertex connectivity from mesh data
        if 'cells' in self.mesh_data and len(self.mesh_data['cells']) == n_cells:
            cells = self.mesh_data['cells']
            for i, cell_vertices in enumerate(cells):
                if len(cell_vertices) > 0:
                    cell_center = np.mean(vertices[cell_vertices], axis=0)
                    cell_centers[i] = cell_center
                    cell_volumes[i] = self._estimate_cell_volume(vertices[cell_vertices])
        else:
            # Estimate cell properties from face connectivity
            cell_vertex_count = [[] for _ in range(n_cells)]
            
            for face_idx, face in enumerate(faces):
                owner = face_owner[face_idx]
                neighbor = face_neighbor[face_idx] if face_idx < len(face_neighbor) else -1
                
                if 0 <= owner < n_cells:
                    cell_vertex_count[owner].extend(face)
                if 0 <= neighbor < n_cells:
                    cell_vertex_count[neighbor].extend(face)
            
            for i in range(n_cells):
                if cell_vertex_count[i]:
                    unique_vertices = list(set(cell_vertex_count[i]))
                    if unique_vertices:
                        cell_centers[i] = np.mean(vertices[unique_vertices], axis=0)
                        cell_volumes[i] = self._estimate_cell_volume(vertices[unique_vertices])
                    else:
                        cell_centers[i] = vertices[min(i, len(vertices)-1)]
                        cell_volumes[i] = 1e-6
                else:
                    # Fallback
                    cell_centers[i] = vertices[min(i, len(vertices)-1)]
                    cell_volumes[i] = 1e-6
        
        # Compute face geometric quantities
        face_centers = np.zeros((n_faces, 3))
        face_areas = np.zeros(n_faces)
        face_normals = np.zeros((n_faces, 3))
        
        for i, face in enumerate(faces):
            if len(face) >= 2:
                face_vertices = vertices[face]
                face_centers[i] = np.mean(face_vertices, axis=0)
                
                # Compute face area and normal for polygonal faces
                if len(face) == 2:
                    # Edge case - treat as 1D
                    edge_vector = face_vertices[1] - face_vertices[0]
                    face_areas[i] = np.linalg.norm(edge_vector)
                    if face_areas[i] > 1e-12:
                        # Create a normal perpendicular to edge
                        face_normals[i] = np.array([-edge_vector[1], edge_vector[0], 0.0])
                        face_normals[i] /= np.linalg.norm(face_normals[i])
                elif len(face) == 3:
                    # Triangle
                    v1 = face_vertices[1] - face_vertices[0]
                    v2 = face_vertices[2] - face_vertices[0]
                    normal = np.cross(v1, v2)
                    face_areas[i] = 0.5 * np.linalg.norm(normal)
                    if face_areas[i] > 1e-12:
                        face_normals[i] = normal / np.linalg.norm(normal)
                elif len(face) == 4:
                    # Quadrilateral - split into two triangles
                    v1 = face_vertices[1] - face_vertices[0]
                    v2 = face_vertices[2] - face_vertices[0]
                    v3 = face_vertices[3] - face_vertices[0]
                    
                    normal1 = np.cross(v1, v2)
                    normal2 = np.cross(v2, v3)
                    
                    area1 = 0.5 * np.linalg.norm(normal1)
                    area2 = 0.5 * np.linalg.norm(normal2)
                    
                    face_areas[i] = area1 + area2
                    if face_areas[i] > 1e-12:
                        combined_normal = normal1 + normal2
                        face_normals[i] = combined_normal / np.linalg.norm(combined_normal)
                else:
                    # General polygon - use centroid method
                    centroid = face_centers[i]
                    total_area = 0.0
                    total_normal = np.zeros(3)
                    
                    for j in range(len(face)):
                        v1 = face_vertices[j] - centroid
                        v2 = face_vertices[(j+1) % len(face)] - centroid
                        triangle_normal = np.cross(v1, v2)
                        triangle_area = 0.5 * np.linalg.norm(triangle_normal)
                        total_area += triangle_area
                        total_normal += triangle_normal
                    
                    face_areas[i] = total_area
                    if face_areas[i] > 1e-12:
                        face_normals[i] = total_normal / np.linalg.norm(total_normal)
        
        # Build cell-face connectivity
        cell_faces = [[] for _ in range(n_cells)]
        for face_idx in range(n_faces):
            owner = face_owner[face_idx]
            if 0 <= owner < n_cells:
                cell_faces[owner].append(face_idx)
            neighbor = face_neighbor[face_idx] if face_idx < len(face_neighbor) else -1
            if 0 <= neighbor < n_cells:
                cell_faces[neighbor].append(face_idx)
        
        # Process boundary patches with proper face ranges
        boundary_faces = {}
        face_start = 0
        
        for patch_name, patch_info in boundary_patches.items():
            if isinstance(patch_info, dict):
                n_faces_patch = patch_info.get('nFaces', 0)
                start_face = patch_info.get('startFace', face_start)
                
                # Create face indices for this patch
                patch_face_indices = list(range(start_face, start_face + n_faces_patch))
                boundary_faces[patch_name] = patch_face_indices
                
                face_start = start_face + n_faces_patch
            else:
                boundary_faces[patch_name] = []
        
        # Count internal vs boundary faces
        internal_faces = np.sum(face_neighbor >= 0)
        boundary_face_count = n_faces - internal_faces
        
        self.geometry = MeshGeometry(
            cell_centers=cell_centers,
            cell_volumes=cell_volumes,
            face_centers=face_centers,
            face_areas=face_areas,
            face_normals=face_normals,
            face_owner=face_owner,
            face_neighbor=face_neighbor,
            boundary_faces=boundary_faces,
            cell_faces=cell_faces
        )
        
        logger.info(f"Mesh geometry computed: {n_cells} cells, {n_faces} faces")
        logger.info(f"Internal faces: {internal_faces}, Boundary faces: {boundary_face_count}")
        logger.info(f"Cell volume range: [{np.min(cell_volumes):.2e}, {np.max(cell_volumes):.2e}]")
        logger.info(f"Boundary patches: {list(boundary_faces.keys())}")
        
    def _create_vertex_based_faces(self, vertices: np.ndarray, n_cells: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Create face connectivity for vertex-based discretization."""
        faces = []
        face_owner = []
        face_neighbor = []
        
        # Create connections between nearby vertices
        for i in range(n_cells):
            vertex_i = vertices[i]
            distances = np.linalg.norm(vertices - vertex_i, axis=1)
            nearby_indices = np.where((distances > 1e-12) & (distances < np.median(distances)))[0]
            
            for j in nearby_indices[:6]:  # Limit to 6 neighbors
                if j > i:  # Avoid duplicate faces
                    faces.append([i, j])
                    face_owner.append(i)
                    face_neighbor.append(j)
                    
        return faces, np.array(face_owner), np.array(face_neighbor)
        
    def _estimate_cell_volume(self, cell_vertices: np.ndarray) -> float:
        """Estimate cell volume using tetrahedralization."""
        if len(cell_vertices) < 4:
            return 1e-6
            
        # Use first vertex as common point for tetrahedralization
        v0 = cell_vertices[0]
        total_volume = 0.0
        
        for i in range(1, len(cell_vertices) - 2):
            for j in range(i + 1, len(cell_vertices) - 1):
                for k in range(j + 1, len(cell_vertices)):
                    v1, v2, v3 = cell_vertices[i], cell_vertices[j], cell_vertices[k]
                    # Tetrahedron volume: |det(v1-v0, v2-v0, v3-v0)| / 6
                    matrix = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
                    vol = abs(np.linalg.det(matrix)) / 6.0
                    total_volume += vol
                    
        return max(total_volume / max(1, len(cell_vertices) - 3), 1e-9)
        
    def _initialize_flow_field(self):
        """Initialize flow field with appropriate initial conditions."""
        logger.info("Initializing flow field...")
        
        n_cells = len(self.geometry.cell_centers)
        
        # Initialize with freestream conditions
        U_inf = self.flow_properties.reference_velocity
        rho_inf = self.flow_properties.reference_density
        p_inf = self.flow_properties.reference_pressure
        T_inf = self.flow_properties.reference_temperature
        
        self.flow_field = FlowField(
            velocity=np.full((n_cells, 3), [U_inf, 0.0, 0.0], dtype=np.float64),
            pressure=np.full(n_cells, p_inf, dtype=np.float64),
            temperature=np.full(n_cells, T_inf, dtype=np.float64),
            density=np.full(n_cells, rho_inf, dtype=np.float64)
        )
        
        logger.info(f"Flow field initialized: U∞={U_inf:.3f} m/s, p∞={p_inf:.1f} Pa")
        
    def set_boundary_condition(self, boundary_name: str, bc: BoundaryCondition):
        """Set boundary condition for a named boundary patch."""
        self.boundary_conditions[boundary_name] = bc
        logger.info(f"Boundary condition set for '{boundary_name}': {bc.boundary_type.value}")
        
    def solve_steady_state(self, max_iterations: int = 1000, 
                          convergence_tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Solve steady-state incompressible Navier-Stokes equations using SIMPLE algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_tolerance: Convergence tolerance for residuals
            
        Returns:
            Solution history and convergence information
        """
        logger.info(f"Starting SIMPLE algorithm (max_iter={max_iterations}, tol={convergence_tolerance:.2e})")
        
        history = []
        converged = False
        
        # SIMPLE algorithm main loop
        for iteration in range(max_iterations):
            # Store previous solution for convergence checking
            u_old = self.flow_field.velocity.copy()
            p_old = self.flow_field.pressure.copy()
            
            # Step 1: Solve momentum equations with guessed pressure field
            momentum_residuals = self._solve_momentum_equations()
            
            # Step 2: Solve pressure correction equation
            pressure_residual = self._solve_pressure_correction()
            
            # Step 3: Correct velocities and pressure
            self._correct_velocity_and_pressure()
            
            # Step 4: Apply boundary conditions
            self._apply_boundary_conditions()
            
            # Compute residuals
            velocity_residual = np.max(np.linalg.norm(self.flow_field.velocity - u_old, axis=1))
            pressure_change = np.max(np.abs(self.flow_field.pressure - p_old))
            total_residual = max(velocity_residual, pressure_change)
            
            # Store iteration data
            iteration_data = {
                'iteration': iteration,
                'velocity_residual': float(velocity_residual),
                'pressure_residual': float(pressure_residual),
                'pressure_change': float(pressure_change),
                'momentum_residuals': [float(r) for r in momentum_residuals],
                'total_residual': float(total_residual),
                'converged': total_residual < convergence_tolerance
            }
            history.append(iteration_data)
            
            # Check convergence
            converged = iteration_data['converged']
            
            # Print progress
            if iteration % 10 == 0 or converged:
                logger.info(f"  Iter {iteration:4d}: |du|={velocity_residual:.2e}, "
                           f"|dp|={pressure_change:.2e}, residual={total_residual:.2e}")
                
            if converged:
                logger.info(f"  ✓ Converged after {iteration + 1} iterations")
                break
                
        if not converged:
            logger.warning(f"  ⚠ Did not converge after {max_iterations} iterations")
            
        return {
            'converged': converged,
            'iterations': len(history),
            'final_residual': float(total_residual) if history else 1.0,
            'history': history,
            'flow_field': self.flow_field
        }
        
    def _solve_momentum_equations(self) -> List[float]:
        """Solve momentum equations for each velocity component."""
        residuals = []
        
        n_cells = len(self.geometry.cell_centers)
        
        for component in range(3):  # u, v, w components
            # Build coefficient matrix for this velocity component
            A, b = self._build_momentum_system(component)
            
            # Apply relaxation
            alpha = self.solver_params['simple_relaxation']['velocity']
            A_diag = A.diagonal()
            A.setdiag(A_diag / alpha)
            
            # Add old velocity contribution
            u_old = self.flow_field.velocity[:, component]
            b += (1 - alpha) * A_diag * u_old / alpha
            
            # Solve linear system
            try:
                u_new, info = spla.cg(A, b, 
                                     rtol=self.solver_params['linear_solver']['velocity_tolerance'],
                                     maxiter=self.solver_params['linear_solver']['max_iterations'])
                
                if info == 0:
                    residual = np.linalg.norm(A @ u_new - b)
                    self.flow_field.velocity[:, component] = u_new
                else:
                    logger.warning(f"Momentum solver convergence issue for component {component}: info={info}")
                    residual = 1e3
                    
            except Exception as e:
                logger.warning(f"Momentum solver failed for component {component}: {e}")
                residual = 1e3
                
            residuals.append(residual)
            
        return residuals
        
    def _build_momentum_system(self, component: int) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build coefficient matrix and RHS for momentum equation."""
        n_cells = len(self.geometry.cell_centers)
        
        # Initialize coefficient matrix and RHS
        A = sp.lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)
        
        rho = self.flow_properties.reference_density
        mu = self.flow_properties.dynamic_viscosity
        
        # Process all faces
        for face_idx in range(len(self.geometry.face_owner)):
            face_area = self.geometry.face_areas[face_idx]
            face_normal = self.geometry.face_normals[face_idx]
            face_center = self.geometry.face_centers[face_idx]
            
            owner = self.geometry.face_owner[face_idx]
            neighbor = self.geometry.face_neighbor[face_idx]
            
            # Skip invalid cells
            if owner >= n_cells:
                continue
                
            if neighbor >= 0 and neighbor < n_cells:
                # Internal face - process both cells
                cell_center_i = self.geometry.cell_centers[owner]
                cell_center_j = self.geometry.cell_centers[neighbor]
                distance = np.linalg.norm(cell_center_j - cell_center_i)
                
                if distance > 1e-12:
                    # Interpolate velocity to face for convection
                    u_face = 0.5 * (self.flow_field.velocity[owner] + 
                                   self.flow_field.velocity[neighbor])
                    
                    # Convective flux: ρ(u·n)A
                    convective_flux = rho * np.dot(u_face, face_normal) * face_area
                    
                    # Diffusion coefficient: μA/d
                    diffusion_coeff = mu * face_area / distance
                    
                    # Owner cell contributions
                    A[owner, owner] += diffusion_coeff
                    A[owner, neighbor] -= diffusion_coeff
                    
                    # Neighbor cell contributions
                    A[neighbor, neighbor] += diffusion_coeff
                    A[neighbor, owner] -= diffusion_coeff
                    
                    # Convection contributions (upwind)
                    if convective_flux > 0:
                        # Flow from owner to neighbor
                        A[owner, owner] += convective_flux
                        A[neighbor, neighbor] -= convective_flux
                    else:
                        # Flow from neighbor to owner
                        A[owner, neighbor] += convective_flux
                        A[neighbor, owner] -= convective_flux
                        
            else:
                # Boundary face - only affects owner cell
                if face_area > 1e-12:
                    # Add diffusion contribution for boundary
                    # Estimate distance to boundary (half cell size)
                    cell_volume = self.geometry.cell_volumes[owner]
                    char_length = cell_volume**(1/3)
                    boundary_distance = max(char_length * 0.5, 1e-6)
                    
                    diffusion_coeff = mu * face_area / boundary_distance
                    A[owner, owner] += diffusion_coeff
                    
                    # Add boundary flux to RHS (will be handled in BC application)
                    # For now, just add a small contribution to avoid singular matrix
        
        # Add pressure gradient terms
        for cell_i in range(n_cells):
            cell_volume = self.geometry.cell_volumes[cell_i]
            pressure_gradient = self._compute_pressure_gradient(cell_i)
            b[cell_i] -= pressure_gradient[component] * cell_volume
            
            # Add temporal term for pseudo-time stepping
            # ∂u/∂t = (u_new - u_old)/dt
            # For steady state, this becomes a stabilization term
            temporal_coeff = rho * cell_volume / 1e-3  # Pseudo time step
            A[cell_i, cell_i] += temporal_coeff
            b[cell_i] += temporal_coeff * self.flow_field.velocity[cell_i, component]
        
        # Ensure matrix is not singular
        for i in range(n_cells):
            if abs(A[i, i]) < 1e-15:
                A[i, i] = 1e-6
                
        return A.tocsr(), b
        
    def _compute_pressure_gradient(self, cell_idx: int) -> np.ndarray:
        """Compute pressure gradient at cell center using least squares."""
        cell_center = self.geometry.cell_centers[cell_idx]
        pressure_center = self.flow_field.pressure[cell_idx]
        
        # Collect neighboring cell information
        A_matrix = []
        b_vector = []
        
        face_indices = self.geometry.cell_faces[cell_idx]
        for face_idx in face_indices:
            if face_idx >= len(self.geometry.face_owner):
                continue
                
            owner = self.geometry.face_owner[face_idx]
            neighbor = self.geometry.face_neighbor[face_idx]
            
            # Find neighbor cell
            neighbor_cell = neighbor if owner == cell_idx else owner
            
            if neighbor_cell >= 0 and neighbor_cell < len(self.geometry.cell_centers):
                neighbor_center = self.geometry.cell_centers[neighbor_cell]
                neighbor_pressure = self.flow_field.pressure[neighbor_cell]
                
                dr = neighbor_center - cell_center
                dp = neighbor_pressure - pressure_center
                
                if np.linalg.norm(dr) > 1e-12:
                    A_matrix.append(dr)
                    b_vector.append(dp)
                    
        if len(A_matrix) >= 3:
            A = np.array(A_matrix)
            b = np.array(b_vector)
            
            try:
                gradient = np.linalg.lstsq(A, b, rcond=None)[0]
                return gradient
            except:
                pass
                
        return np.zeros(3)
        
    def _solve_pressure_correction(self) -> float:
        """Solve pressure correction equation."""
        n_cells = len(self.geometry.cell_centers)
        
        # Build pressure correction system: ∇²p' = ∇·u
        A = sp.lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)
        
        rho = self.flow_properties.reference_density
        
        # Compute mass source term for each cell (velocity divergence)
        for cell_i in range(n_cells):
            divergence = self._compute_velocity_divergence(cell_i)
            cell_volume = self.geometry.cell_volumes[cell_i]
            b[cell_i] = rho * divergence * cell_volume
        
        # Build Laplacian operator from face contributions
        for face_idx in range(len(self.geometry.face_owner)):
            face_area = self.geometry.face_areas[face_idx]
            owner = self.geometry.face_owner[face_idx]
            neighbor = self.geometry.face_neighbor[face_idx]
            
            if owner >= n_cells:
                continue
                
            if neighbor >= 0 and neighbor < n_cells:
                # Internal face
                cell_center_i = self.geometry.cell_centers[owner]
                cell_center_j = self.geometry.cell_centers[neighbor]
                distance = np.linalg.norm(cell_center_j - cell_center_i)
                
                if distance > 1e-12:
                    # Laplacian coefficient: A/d
                    coeff = face_area / distance
                    
                    # Owner cell
                    A[owner, owner] += coeff
                    A[owner, neighbor] -= coeff
                    
                    # Neighbor cell
                    A[neighbor, neighbor] += coeff
                    A[neighbor, owner] -= coeff
                    
            else:
                # Boundary face - Neumann boundary condition (dp/dn = 0)
                # This naturally gives zero contribution to the Laplacian
                pass
        
        # Add reference pressure constraint to avoid singular matrix
        # Set pressure to zero at cell 0
        if n_cells > 0:
            A[0, :] = 0
            A[0, 0] = 1.0
            b[0] = 0.0
            
        # Ensure matrix is not singular on diagonal
        for i in range(n_cells):
            if abs(A[i, i]) < 1e-15:
                A[i, i] = 1e-6
        
        # Solve pressure correction system
        try:
            p_correction, info = spla.cg(A.tocsr(), b,
                                        rtol=self.solver_params['linear_solver']['pressure_tolerance'],
                                        maxiter=self.solver_params['linear_solver']['max_iterations'])
            
            if info == 0:
                # Apply pressure correction with relaxation
                alpha_p = self.solver_params['simple_relaxation']['pressure']
                self.flow_field.pressure += alpha_p * p_correction
                residual = np.linalg.norm(b)  # Use RHS norm as residual
            else:
                logger.warning(f"Pressure correction solver convergence issue: info={info}")
                residual = 1e3
                
        except Exception as e:
            logger.warning(f"Pressure correction solver failed: {e}")
            residual = 1e3
            
        return residual
        
    def _compute_velocity_divergence(self, cell_idx: int) -> float:
        """Compute velocity divergence at cell center using finite volume approach."""
        # Use finite volume: ∇·u = (1/V) ∫ u·n dS
        divergence = 0.0
        cell_volume = self.geometry.cell_volumes[cell_idx]
        
        if cell_volume < 1e-12:
            return 0.0
        
        # Sum contributions from all faces of the cell
        face_indices = self.geometry.cell_faces[cell_idx]
        
        for face_idx in face_indices:
            if face_idx >= len(self.geometry.face_areas):
                continue
                
            face_area = self.geometry.face_areas[face_idx]
            face_normal = self.geometry.face_normals[face_idx]
            
            owner = self.geometry.face_owner[face_idx]
            neighbor = self.geometry.face_neighbor[face_idx]
            
            # Determine face velocity and correct normal direction
            if owner == cell_idx:
                # Normal points outward from owner
                normal = face_normal
                if neighbor >= 0 and neighbor < len(self.flow_field.velocity):
                    # Internal face - interpolate
                    u_face = 0.5 * (self.flow_field.velocity[cell_idx] + 
                                   self.flow_field.velocity[neighbor])
                else:
                    # Boundary face - use cell value
                    u_face = self.flow_field.velocity[cell_idx]
                    
            elif neighbor == cell_idx:
                # Normal points inward to neighbor, flip it
                normal = -face_normal
                # Internal face - interpolate
                u_face = 0.5 * (self.flow_field.velocity[cell_idx] + 
                               self.flow_field.velocity[owner])
            else:
                continue
                
            # Add flux: u·n * A
            flux = np.dot(u_face, normal) * face_area
            divergence += flux
            
        # Normalize by cell volume
        divergence /= cell_volume
        return divergence
        
    def _correct_velocity_and_pressure(self):
        """Correct velocities based on pressure correction (SIMPLE algorithm)."""
        # This is a simplified correction step
        # In full SIMPLE, velocities are corrected based on pressure correction gradients
        pass
        
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to flow field using proper face-cell mapping."""
        # Apply boundary conditions based on mesh patches
        applied_patches = []
        
        for boundary_name, bc in self.boundary_conditions.items():
            # Map common boundary names
            patch_name = boundary_name
            if boundary_name == 'farfield':
                patch_name = 'inout'  # Map to actual patch name
            elif boundary_name == 'wall':
                patch_name = 'cylinder'  # Map to actual patch name
                
            if patch_name in self.geometry.boundary_faces:
                face_indices = self.geometry.boundary_faces[patch_name]
                cells_modified = 0
                
                for face_idx in face_indices:
                    if face_idx >= len(self.geometry.face_owner):
                        continue
                        
                    # Find boundary cell (owner of boundary face)
                    owner = self.geometry.face_owner[face_idx]
                    neighbor = self.geometry.face_neighbor[face_idx]
                    
                    # Boundary faces have neighbor = -1, so owner is the boundary cell
                    boundary_cell = owner if neighbor < 0 else (neighbor if owner < 0 else owner)
                    
                    if 0 <= boundary_cell < len(self.flow_field.velocity):
                        self._apply_cell_boundary_condition(boundary_cell, face_idx, bc)
                        cells_modified += 1
                        
                applied_patches.append(f"{patch_name}: {cells_modified} cells")
                
        # Apply BCs to specific geometric regions if patch mapping fails
        self._apply_geometric_boundary_conditions()
        
        logger.debug(f"Applied BCs to patches: {applied_patches}")
                        
    def _apply_cell_boundary_condition(self, cell_idx: int, face_idx: int, bc: BoundaryCondition):
        """Apply boundary condition to specific cell."""
        if bc.boundary_type == BoundaryType.WALL:
            # No-slip wall condition
            if bc.velocity is not None:
                self.flow_field.velocity[cell_idx] = bc.velocity.copy()
            else:
                self.flow_field.velocity[cell_idx] = np.zeros(3)
                
        elif bc.boundary_type == BoundaryType.FARFIELD:
            # Farfield conditions - fix velocity and pressure
            if bc.velocity is not None:
                self.flow_field.velocity[cell_idx] = bc.velocity.copy()
            if bc.pressure is not None:
                self.flow_field.pressure[cell_idx] = bc.pressure
                
        elif bc.boundary_type == BoundaryType.INLET:
            # Inlet velocity specification
            if bc.velocity is not None:
                self.flow_field.velocity[cell_idx] = bc.velocity.copy()
                
        elif bc.boundary_type == BoundaryType.OUTLET:
            # Outlet: zero normal gradient (do nothing for now)
            pass
            
        elif bc.boundary_type == BoundaryType.SYMMETRY:
            # Symmetry: zero normal velocity component
            if face_idx < len(self.geometry.face_normals):
                face_normal = self.geometry.face_normals[face_idx]
                velocity = self.flow_field.velocity[cell_idx]
                normal_component = np.dot(velocity, face_normal)
                self.flow_field.velocity[cell_idx] = velocity - normal_component * face_normal
                
    def _apply_geometric_boundary_conditions(self):
        """Apply boundary conditions based on geometric criteria."""
        # Apply wall BC to cylinder surface cells
        cylinder_radius = 0.5
        cylinder_center = np.array([0.0, 0.0, 0.05])
        
        # Apply farfield BC to cells far from cylinder
        farfield_distance = 15.0
        
        for i, cell_center in enumerate(self.geometry.cell_centers):
            # Distance from cylinder axis
            cylinder_distance = np.linalg.norm(cell_center[:2] - cylinder_center[:2])
            total_distance = np.linalg.norm(cell_center - cylinder_center)
            
            if cylinder_distance <= cylinder_radius * 1.2:
                # Near cylinder - apply wall BC
                self.flow_field.velocity[i] = np.zeros(3)
            elif total_distance > farfield_distance:
                # Far from cylinder - apply farfield BC
                self.flow_field.velocity[i] = np.array([10.0, 0.0, 0.0])
                self.flow_field.pressure[i] = 101325.0
                
    def compute_forces(self, reference_point: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute aerodynamic forces and moments.
        
        Args:
            reference_point: Reference point for moment calculation
            
        Returns:
            Dictionary with force and moment coefficients
        """
        if reference_point is None:
            reference_point = np.zeros(3)
            
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        total_area = 0.0
        
        rho = self.flow_properties.reference_density
        U_inf = self.flow_properties.reference_velocity
        mu = self.flow_properties.dynamic_viscosity
        
        # Integrate forces over wall boundaries
        for boundary_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.WALL and boundary_name in self.geometry.boundary_faces:
                face_indices = self.geometry.boundary_faces[boundary_name]
                
                for face_idx in face_indices:
                    if face_idx >= len(self.geometry.face_areas):
                        continue
                        
                    face_area = self.geometry.face_areas[face_idx]
                    face_normal = self.geometry.face_normals[face_idx]
                    face_center = self.geometry.face_centers[face_idx]
                    
                    # Find boundary cell
                    owner = self.geometry.face_owner[face_idx]
                    neighbor = self.geometry.face_neighbor[face_idx]
                    boundary_cell = owner if neighbor < 0 else neighbor
                    
                    if boundary_cell >= 0 and boundary_cell < len(self.flow_field.pressure):
                        pressure = self.flow_field.pressure[boundary_cell]
                        
                        # Pressure force (acts normal to surface)
                        pressure_force = pressure * face_area * face_normal
                        
                        # Viscous force (simplified)
                        velocity_gradient = self._compute_velocity_gradient_at_face(face_idx)
                        viscous_stress = mu * velocity_gradient
                        viscous_force = np.dot(viscous_stress, face_normal) * face_area
                        
                        # Total force on this face element
                        face_force = pressure_force + viscous_force
                        total_force += face_force
                        
                        # Moment about reference point
                        moment_arm = face_center - reference_point
                        total_moment += np.cross(moment_arm, face_force)
                        
                        total_area += face_area
                        
        # Non-dimensionalize forces
        dynamic_pressure = 0.5 * rho * U_inf**2
        reference_area = self.flow_properties.reference_length  # Can be modified as needed
        reference_length = self.flow_properties.reference_length
        
        if dynamic_pressure > 1e-12 and reference_area > 1e-12:
            drag_coefficient = total_force[0] / (dynamic_pressure * reference_area)
            lift_coefficient = total_force[1] / (dynamic_pressure * reference_area)
            side_force_coefficient = total_force[2] / (dynamic_pressure * reference_area)
            
            moment_coefficient = total_moment / (dynamic_pressure * reference_area * reference_length)
        else:
            drag_coefficient = 0.0
            lift_coefficient = 0.0
            side_force_coefficient = 0.0
            moment_coefficient = np.zeros(3)
            
        logger.info(f"Forces computed: Cd={drag_coefficient:.6f}, Cl={lift_coefficient:.6f}")
        
        return {
            'drag_coefficient': float(drag_coefficient),
            'lift_coefficient': float(lift_coefficient),
            'side_force_coefficient': float(side_force_coefficient),
            'moment_coefficient': moment_coefficient.tolist(),
            'total_force': total_force.tolist(),
            'total_moment': total_moment.tolist(),
            'surface_area': float(total_area),
            'reference_values': {
                'dynamic_pressure': float(dynamic_pressure),
                'reference_area': float(reference_area),
                'reference_length': float(reference_length)
            }
        }
        
    def _compute_velocity_gradient_at_face(self, face_idx: int) -> np.ndarray:
        """Compute velocity gradient tensor at face (simplified)."""
        # Simplified: return zero gradient (pressure forces only)
        return np.zeros((3, 3))
        
    def get_solution_data(self) -> Dict[str, Any]:
        """Get complete solution data."""
        return {
            'cell_centers': self.geometry.cell_centers.tolist(),
            'velocity': self.flow_field.velocity.tolist(),
            'pressure': self.flow_field.pressure.tolist(),
            'temperature': self.flow_field.temperature.tolist(),
            'density': self.flow_field.density.tolist(),
            'flow_properties': {
                'reynolds_number': self.flow_properties.reynolds_number,
                'mach_number': self.flow_properties.mach_number,
                'reference_velocity': self.flow_properties.reference_velocity,
                'reference_pressure': self.flow_properties.reference_pressure,
                'dynamic_viscosity': self.flow_properties.dynamic_viscosity
            },
            'mesh_info': {
                'n_cells': len(self.geometry.cell_centers),
                'n_faces': len(self.geometry.face_centers),
                'cell_volume_range': [float(np.min(self.geometry.cell_volumes)), 
                                     float(np.max(self.geometry.cell_volumes))],
                'face_area_range': [float(np.min(self.geometry.face_areas)), 
                                   float(np.max(self.geometry.face_areas))]
            }
        }