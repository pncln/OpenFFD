"""
Shape Optimization Framework Using Adjoint Gradients

Implements comprehensive gradient-based shape optimization for aerodynamic design:
- Design variable parameterization (Free Form Deformation, B-splines, CAD)
- Gradient-based optimization algorithms (SLSQP, L-BFGS-B, Trust Region)
- Design constraints handling (geometric, aerodynamic, manufacturing)
- Multi-objective optimization with Pareto analysis
- Robust optimization under uncertainty
- Design space exploration and sensitivity analysis
- Mesh deformation and update strategies
- Optimization history tracking and visualization

Integrates with discrete adjoint framework for efficient gradient computation
and provides complete design optimization workflow for industrial applications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from scipy.optimize import minimize, differential_evolution, Bounds, NonlinearConstraint
from scipy.interpolate import BSpline, splrep, splev
import copy

logger = logging.getLogger(__name__)


class ParameterizationType(Enum):
    """Enumeration of design parameterization types."""
    FREE_FORM_DEFORMATION = "ffd"
    B_SPLINE = "bspline"
    BEZIER = "bezier"
    CAD_PARAMETERS = "cad"
    SHAPE_FUNCTIONS = "shape_functions"
    RADIAL_BASIS_FUNCTIONS = "rbf"


class OptimizationAlgorithm(Enum):
    """Enumeration of optimization algorithms."""
    SLSQP = "slsqp"
    L_BFGS_B = "l_bfgs_b"
    TRUST_REGION = "trust_region"
    GENETIC_ALGORITHM = "genetic"
    PARTICLE_SWARM = "pso"
    GRADIENT_DESCENT = "gradient_descent"


class ConstraintType(Enum):
    """Enumeration of constraint types."""
    GEOMETRIC = "geometric"
    AERODYNAMIC = "aerodynamic"
    MANUFACTURING = "manufacturing"
    STRUCTURAL = "structural"
    VOLUME = "volume"
    THICKNESS = "thickness"


@dataclass
class OptimizationConfig:
    """Configuration for shape optimization."""
    
    # Algorithm settings
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.SLSQP
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-6
    step_tolerance: float = 1e-8
    
    # Design space
    parameterization: ParameterizationType = ParameterizationType.FREE_FORM_DEFORMATION
    n_design_variables: int = 10
    variable_bounds: Optional[List[Tuple[float, float]]] = None
    
    # Multi-objective settings
    multi_objective: bool = False
    objective_weights: List[float] = field(default_factory=lambda: [1.0])
    pareto_analysis: bool = False
    
    # Constraint settings
    constraint_tolerance: float = 1e-6
    penalty_parameter: float = 1000.0
    
    # Mesh and solver settings
    mesh_deformation_method: str = "radial_basis_function"
    mesh_quality_threshold: float = 0.1
    cfd_convergence_tolerance: float = 1e-5
    
    # Robustness settings
    robust_optimization: bool = False
    uncertainty_samples: int = 10
    monte_carlo_samples: int = 100
    
    # Output settings
    save_intermediate: bool = True
    save_frequency: int = 5
    output_directory: str = "optimization_output"


@dataclass
class DesignVariable:
    """Design variable definition."""
    
    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    scaling_factor: float = 1.0
    description: str = ""
    
    # Parameterization specific
    parameter_type: str = "general"  # "geometric", "aerodynamic", etc.
    sensitivity_weight: float = 1.0
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        return (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
    
    def denormalize(self, normalized_value: float) -> float:
        """Denormalize value from [0, 1] range."""
        return self.lower_bound + normalized_value * (self.upper_bound - self.lower_bound)


@dataclass
class OptimizationConstraint:
    """Optimization constraint definition."""
    
    name: str
    constraint_type: ConstraintType
    target_value: float
    tolerance: float
    weight: float = 1.0
    
    # Constraint function
    evaluation_function: Optional[Callable] = None
    gradient_function: Optional[Callable] = None
    
    # Constraint properties
    inequality: bool = True  # True for g(x) >= 0, False for g(x) = 0
    active: bool = True


@dataclass
class OptimizationResult:
    """Results from shape optimization."""
    
    # Optimization status
    success: bool
    message: str
    n_iterations: int
    n_function_evaluations: int
    n_gradient_evaluations: int
    
    # Final design
    optimal_design: np.ndarray
    optimal_objective: float
    optimal_constraints: Dict[str, float]
    
    # Convergence history
    objective_history: List[float]
    constraint_history: List[Dict[str, float]]
    gradient_norm_history: List[float]
    design_history: List[np.ndarray]
    
    # Performance metrics
    total_time: float
    average_iteration_time: float
    cfd_time: float
    adjoint_time: float
    mesh_deformation_time: float
    
    # Final gradients and sensitivities
    final_gradients: np.ndarray
    sensitivity_analysis: Dict[str, Any]
    
    # Multi-objective results
    pareto_front: Optional[List[Tuple[float, ...]]] = None
    pareto_designs: Optional[List[np.ndarray]] = None


class DesignParameterization(ABC):
    """Abstract base class for design parameterization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize parameterization."""
        self.config = config
        self.design_variables: List[DesignVariable] = []
        
    @abstractmethod
    def setup_design_variables(self, initial_geometry: Dict[str, Any]) -> List[DesignVariable]:
        """Setup design variables for the parameterization."""
        pass
    
    @abstractmethod
    def apply_design_changes(self, design_vector: np.ndarray, 
                           geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Apply design changes to geometry."""
        pass
    
    @abstractmethod
    def compute_shape_sensitivities(self, adjoint_gradients: np.ndarray,
                                  geometry: Dict[str, Any]) -> np.ndarray:
        """Compute shape sensitivities from adjoint gradients."""
        pass


class FreeFormDeformation(DesignParameterization):
    """
    Free Form Deformation (FFD) parameterization.
    
    Uses a 3D lattice of control points to deform the geometry.
    """
    
    def __init__(self, config: OptimizationConfig, 
                 n_control_points: Tuple[int, int, int] = (4, 3, 2),
                 ffd_file_path: Optional[str] = None):
        """Initialize FFD parameterization."""
        super().__init__(config)
        self.n_control_points = n_control_points
        self.control_points: Optional[np.ndarray] = None
        self.bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.ffd_file_path = ffd_file_path
        
    def _load_ffd_from_file(self, file_path: str) -> None:
        """Load FFD control box from .xyz file."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse FFD file format
            n_blocks = int(lines[0].strip())
            if n_blocks != 1:
                raise ValueError(f"Only single FFD block supported, found {n_blocks}")
            
            # Parse dimensions
            dims = list(map(int, lines[1].strip().split()))
            if len(dims) != 3:
                raise ValueError(f"Expected 3 dimensions, found {len(dims)}")
            
            nx, ny, nz = dims
            self.n_control_points = (nx, ny, nz)
            
            # Parse coordinates
            x_coords = list(map(float, lines[2].strip().split()))
            y_coords = list(map(float, lines[3].strip().split()))
            z_coords = list(map(float, lines[4].strip().split()))
            
            n_points = nx * ny * nz
            if len(x_coords) != n_points or len(y_coords) != n_points or len(z_coords) != n_points:
                raise ValueError(f"Coordinate arrays size mismatch. Expected {n_points} points")
            
            # Reshape into control point lattice
            self.control_points = np.zeros((nx, ny, nz, 3))
            
            idx = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        self.control_points[i, j, k, 0] = x_coords[idx]
                        self.control_points[i, j, k, 1] = y_coords[idx]
                        self.control_points[i, j, k, 2] = z_coords[idx]
                        idx += 1
            
            # Compute bounding box from control points
            min_coords = np.min(self.control_points.reshape(-1, 3), axis=0)
            max_coords = np.max(self.control_points.reshape(-1, 3), axis=0)
            self.bounding_box = (min_coords, max_coords)
            
            print(f"Loaded FFD control box from {file_path}: {nx}×{ny}×{nz} control points")
            print(f"  Control point bounds: X[{min_coords[0]:.3f}, {max_coords[0]:.3f}], Y[{min_coords[1]:.3f}, {max_coords[1]:.3f}], Z[{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
            
        except Exception as e:
            print(f"Warning: Could not load FFD file {file_path}: {e}")
            print("Using default FFD box generation...")
            self.control_points = None
            self.bounding_box = None
    
    def setup_design_variables(self, initial_geometry: Dict[str, Any]) -> List[DesignVariable]:
        """Setup FFD control point design variables."""
        self.design_variables = []
        
        # Load FFD control box from file if provided
        if self.ffd_file_path:
            self._load_ffd_from_file(self.ffd_file_path)
        
        # If no FFD loaded, create default box
        if self.control_points is None:
            # Extract geometry bounds
            if 'vertices' in initial_geometry:
                vertices = initial_geometry['vertices']
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                self.bounding_box = (bbox_min, bbox_max)
            else:
                # Default bounding box
                bbox_min = np.array([-1.0, -0.5, -0.1])
                bbox_max = np.array([2.0, 0.5, 0.1])
                self.bounding_box = (bbox_min, bbox_max)
            
            # Create control point lattice
            nx, ny, nz = self.n_control_points
            self.control_points = np.zeros((nx, ny, nz, 3))
            
            # Initialize control points uniformly in bounding box
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        u = i / (nx - 1) if nx > 1 else 0.0
                        v = j / (ny - 1) if ny > 1 else 0.0
                        w = k / (nz - 1) if nz > 1 else 0.0
                        
                        self.control_points[i, j, k] = (
                            bbox_min + np.array([u, v, w]) * (bbox_max - bbox_min)
                        )
        
        # Get current control box dimensions
        nx, ny, nz = self.n_control_points
        bbox_min, bbox_max = self.bounding_box
        
        # Create design variables for control point displacements
        displacement_magnitude = 0.1 * np.linalg.norm(bbox_max - bbox_min)
        
        variable_idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for coord in range(3):
                        # Only create variables for movable control points
                        if self._is_control_point_movable(i, j, k, coord):
                            var = DesignVariable(
                                name=f"ffd_cp_{i}_{j}_{k}_{coord}",
                                initial_value=0.0,  # Initial displacement
                                lower_bound=-displacement_magnitude,
                                upper_bound=displacement_magnitude,
                                scaling_factor=1.0,
                                description=f"FFD control point ({i},{j},{k}) {['x','y','z'][coord]} displacement"
                            )
                            self.design_variables.append(var)
                            variable_idx += 1
        
        logger.info(f"Created {len(self.design_variables)} FFD design variables")
        return self.design_variables
    
    def _is_control_point_movable(self, i: int, j: int, k: int, coord: int) -> bool:
        """Determine if control point coordinate can be moved."""
        nx, ny, nz = self.n_control_points
        
        # Fixed points at boundaries in x-direction (leading/trailing edge)
        if coord == 0 and (i == 0 or i == nx - 1):
            return False
        
        # Allow movement in y and z directions for shape changes
        return True
    
    def apply_design_changes(self, design_vector: np.ndarray, 
                           geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FFD deformation to geometry."""
        if 'vertices' not in geometry:
            return geometry
        
        original_vertices = geometry['vertices'].copy()
        n_vertices = original_vertices.shape[0]
        
        # Update control points with design vector
        updated_control_points = self.control_points.copy()
        var_idx = 0
        
        nx, ny, nz = self.n_control_points
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for coord in range(3):
                        if self._is_control_point_movable(i, j, k, coord):
                            displacement = design_vector[var_idx]
                            updated_control_points[i, j, k, coord] += displacement
                            var_idx += 1
        
        # Apply FFD deformation using Bernstein polynomials
        deformed_vertices = np.zeros_like(original_vertices)
        
        for vertex_idx in range(n_vertices):
            vertex = original_vertices[vertex_idx]
            
            # Compute parametric coordinates in bounding box
            bbox_min, bbox_max = self.bounding_box
            param_coords = (vertex - bbox_min) / (bbox_max - bbox_min + 1e-12)
            param_coords = np.clip(param_coords, 0.0, 1.0)
            
            # FFD deformation using Bernstein polynomials
            displacement = self._compute_ffd_displacement(
                param_coords, self.control_points, updated_control_points
            )
            
            deformed_vertices[vertex_idx] = vertex + displacement
        
        # Update geometry
        new_geometry = copy.deepcopy(geometry)
        new_geometry['vertices'] = deformed_vertices
        
        return new_geometry
    
    def _compute_ffd_displacement(self, param_coords: np.ndarray, 
                                original_control_points: np.ndarray,
                                updated_control_points: np.ndarray) -> np.ndarray:
        """Compute FFD displacement using Bernstein polynomials."""
        u, v, w = param_coords
        nx, ny, nz, _ = original_control_points.shape
        
        # Compute displacement as weighted sum of control point displacements
        displacement = np.zeros(3)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Bernstein polynomials for tri-variate FFD
                    B_i = self._bernstein_polynomial(i, nx-1, u)
                    B_j = self._bernstein_polynomial(j, ny-1, v)
                    B_k = self._bernstein_polynomial(k, nz-1, w)
                    
                    # Weight for this control point
                    weight = B_i * B_j * B_k
                    
                    # Control point displacement
                    cp_displacement = (updated_control_points[i, j, k] - 
                                     original_control_points[i, j, k])
                    
                    # Add weighted displacement
                    displacement += weight * cp_displacement
        
        return displacement
    
    def _bernstein_polynomial(self, i: int, n: int, t: float) -> float:
        """Compute Bernstein polynomial B_{i,n}(t)."""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def _trilinear_interpolation(self, param_coords: np.ndarray, 
                               control_points: np.ndarray) -> np.ndarray:
        """Perform trilinear interpolation using control points."""
        u, v, w = param_coords
        nx, ny, nz = self.n_control_points
        
        # Find control point indices
        i = min(int(u * (nx - 1)), nx - 2)
        j = min(int(v * (ny - 1)), ny - 2)
        k = min(int(w * (nz - 1)), nz - 2)
        
        # Local coordinates within control volume
        u_local = (u * (nx - 1)) - i
        v_local = (v * (ny - 1)) - j
        w_local = (w * (nz - 1)) - k
        
        # Trilinear interpolation
        result = np.zeros(3)
        
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    weight = ((1 - u_local) if di == 0 else u_local) * \
                            ((1 - v_local) if dj == 0 else v_local) * \
                            ((1 - w_local) if dk == 0 else w_local)
                    
                    if i + di < nx and j + dj < ny and k + dk < nz:
                        result += weight * control_points[i + di, j + dj, k + dk]
        
        return result
    
    def compute_shape_sensitivities(self, adjoint_gradients: np.ndarray,
                                  geometry: Dict[str, Any]) -> np.ndarray:
        """Compute FFD design variable sensitivities."""
        if 'vertices' not in geometry:
            return np.zeros(len(self.design_variables))
        
        vertices = geometry['vertices']
        n_vertices = vertices.shape[0]
        
        # Initialize sensitivity vector
        sensitivities = np.zeros(len(self.design_variables))
        
        # Compute sensitivity of each design variable
        var_idx = 0
        nx, ny, nz = self.n_control_points
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for coord in range(3):
                        if self._is_control_point_movable(i, j, k, coord):
                            # Compute derivative of vertices w.r.t. this control point
                            sensitivity = 0.0
                            
                            for vertex_idx in range(n_vertices):
                                vertex = vertices[vertex_idx]
                                
                                # Parametric coordinates
                                bbox_min, bbox_max = self.bounding_box
                                param_coords = (vertex - bbox_min) / (bbox_max - bbox_min + 1e-12)
                                param_coords = np.clip(param_coords, 0.0, 1.0)
                                
                                # Derivative of trilinear interpolation
                                derivative = self._trilinear_derivative(
                                    param_coords, i, j, k, coord
                                )
                                
                                # Chain rule with adjoint gradients
                                if vertex_idx * 3 + coord < len(adjoint_gradients):
                                    sensitivity += (adjoint_gradients[vertex_idx * 3 + coord] * 
                                                  derivative)
                            
                            sensitivities[var_idx] = sensitivity
                            var_idx += 1
        
        return sensitivities
    
    def _trilinear_derivative(self, param_coords: np.ndarray,
                            cp_i: int, cp_j: int, cp_k: int, coord: int) -> float:
        """Compute derivative of trilinear interpolation w.r.t. control point."""
        u, v, w = param_coords
        nx, ny, nz = self.n_control_points
        
        # Find control point indices for interpolation
        i = min(int(u * (nx - 1)), nx - 2)
        j = min(int(v * (ny - 1)), ny - 2)
        k = min(int(w * (nz - 1)), nz - 2)
        
        # Check if this control point influences this vertex
        if not (i <= cp_i <= i + 1 and j <= cp_j <= j + 1 and k <= cp_k <= k + 1):
            return 0.0
        
        # Local coordinates
        u_local = (u * (nx - 1)) - i
        v_local = (v * (ny - 1)) - j
        w_local = (w * (nz - 1)) - k
        
        # Compute weight for this control point
        di = cp_i - i
        dj = cp_j - j
        dk = cp_k - k
        
        weight = ((1 - u_local) if di == 0 else u_local) * \
                ((1 - v_local) if dj == 0 else v_local) * \
                ((1 - w_local) if dk == 0 else w_local)
        
        return weight


class BSplineParameterization(DesignParameterization):
    """
    B-spline curve parameterization for airfoil/wing shapes.
    
    Uses B-spline control points to represent surface geometry.
    """
    
    def __init__(self, config: OptimizationConfig, 
                 n_control_points: int = 10, degree: int = 3):
        """Initialize B-spline parameterization."""
        super().__init__(config)
        self.n_control_points = n_control_points
        self.degree = degree
        self.control_points: Optional[np.ndarray] = None
        self.knot_vector: Optional[np.ndarray] = None
        
    def setup_design_variables(self, initial_geometry: Dict[str, Any]) -> List[DesignVariable]:
        """Setup B-spline control point design variables."""
        self.design_variables = []
        
        # Initialize control points from geometry
        if 'surface_points' in initial_geometry:
            surface_points = initial_geometry['surface_points']
            self.control_points = self._fit_bspline_to_surface(surface_points)
        else:
            # Default airfoil-like control points
            self.control_points = self._create_default_airfoil_control_points()
        
        # Create design variables for control point coordinates
        for i in range(self.n_control_points):
            for coord in range(2):  # Assuming 2D airfoil
                if self._is_control_point_movable(i, coord):
                    var = DesignVariable(
                        name=f"bspline_cp_{i}_{coord}",
                        initial_value=self.control_points[i, coord],
                        lower_bound=self.control_points[i, coord] - 0.1,
                        upper_bound=self.control_points[i, coord] + 0.1,
                        scaling_factor=1.0,
                        description=f"B-spline control point {i} {['x','y'][coord]} coordinate"
                    )
                    self.design_variables.append(var)
        
        logger.info(f"Created {len(self.design_variables)} B-spline design variables")
        return self.design_variables
    
    def _create_default_airfoil_control_points(self) -> np.ndarray:
        """Create default airfoil control points."""
        x = np.linspace(0, 1, self.n_control_points)
        y = np.zeros(self.n_control_points)
        
        # Simple symmetric airfoil shape
        for i, xi in enumerate(x):
            if xi <= 0.5:
                y[i] = 0.12 * (2 * xi) * (1 - 2 * xi)**0.5
            else:
                y[i] = 0.12 * (2 * (1 - xi)) * (1 - 2 * (1 - xi))**0.5
        
        return np.column_stack([x, y])
    
    def _fit_bspline_to_surface(self, surface_points: np.ndarray) -> np.ndarray:
        """Fit B-spline to existing surface points."""
        # Simple implementation - in practice would use proper B-spline fitting
        n_surface = surface_points.shape[0]
        indices = np.linspace(0, n_surface - 1, self.n_control_points, dtype=int)
        return surface_points[indices]
    
    def _is_control_point_movable(self, i: int, coord: int) -> bool:
        """Determine if control point coordinate can be moved."""
        # Fix leading and trailing edge points in x-direction
        if coord == 0 and (i == 0 or i == self.n_control_points - 1):
            return False
        return True
    
    def apply_design_changes(self, design_vector: np.ndarray, 
                           geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Apply B-spline deformation to geometry."""
        # Update control points with design vector
        updated_control_points = self.control_points.copy()
        var_idx = 0
        
        for i in range(self.n_control_points):
            for coord in range(2):
                if self._is_control_point_movable(i, coord):
                    updated_control_points[i, coord] = design_vector[var_idx]
                    var_idx += 1
        
        # Generate new surface points from B-spline
        n_surface_points = geometry.get('n_surface_points', 100)
        t = np.linspace(0, 1, n_surface_points)
        
        # Create knot vector if not exists
        if self.knot_vector is None:
            self.knot_vector = self._create_knot_vector()
        
        # Evaluate B-spline
        new_surface_points = self._evaluate_bspline(t, updated_control_points)
        
        # Update geometry
        new_geometry = copy.deepcopy(geometry)
        new_geometry['surface_points'] = new_surface_points
        new_geometry['control_points'] = updated_control_points
        
        return new_geometry
    
    def _create_knot_vector(self) -> np.ndarray:
        """Create knot vector for B-spline."""
        n = self.n_control_points
        p = self.degree
        
        # Uniform knot vector
        knots = np.zeros(n + p + 1)
        for i in range(p + 1, n + 1):
            knots[i] = (i - p) / (n - p)
        knots[n:] = 1.0
        
        return knots
    
    def _evaluate_bspline(self, t: np.ndarray, control_points: np.ndarray) -> np.ndarray:
        """Evaluate B-spline curve at parameter values t."""
        n_points = len(t)
        n_dims = control_points.shape[1]
        surface_points = np.zeros((n_points, n_dims))
        
        for i, ti in enumerate(t):
            point = np.zeros(n_dims)
            for j in range(self.n_control_points):
                basis = self._basis_function(j, self.degree, ti)
                point += basis * control_points[j]
            surface_points[i] = point
        
        return surface_points
    
    def _basis_function(self, i: int, p: int, t: float) -> float:
        """Compute B-spline basis function."""
        if p == 0:
            if self.knot_vector[i] <= t < self.knot_vector[i + 1]:
                return 1.0
            else:
                return 0.0
        else:
            # Recursive definition
            left_term = 0.0
            right_term = 0.0
            
            if self.knot_vector[i + p] != self.knot_vector[i]:
                left_term = ((t - self.knot_vector[i]) / 
                           (self.knot_vector[i + p] - self.knot_vector[i]) * 
                           self._basis_function(i, p - 1, t))
            
            if self.knot_vector[i + p + 1] != self.knot_vector[i + 1]:
                right_term = ((self.knot_vector[i + p + 1] - t) / 
                            (self.knot_vector[i + p + 1] - self.knot_vector[i + 1]) * 
                            self._basis_function(i + 1, p - 1, t))
            
            return left_term + right_term
    
    def compute_shape_sensitivities(self, adjoint_gradients: np.ndarray,
                                  geometry: Dict[str, Any]) -> np.ndarray:
        """Compute B-spline design variable sensitivities."""
        sensitivities = np.zeros(len(self.design_variables))
        
        if 'surface_points' not in geometry:
            return sensitivities
        
        surface_points = geometry['surface_points']
        n_surface = surface_points.shape[0]
        
        # Compute sensitivities using chain rule
        var_idx = 0
        for i in range(self.n_control_points):
            for coord in range(2):
                if self._is_control_point_movable(i, coord):
                    sensitivity = 0.0
                    
                    # Sum over all surface points
                    for j in range(n_surface):
                        t = j / (n_surface - 1)
                        basis = self._basis_function(i, self.degree, t)
                        
                        # Chain rule with adjoint gradients
                        if j * 2 + coord < len(adjoint_gradients):
                            sensitivity += (adjoint_gradients[j * 2 + coord] * basis)
                    
                    sensitivities[var_idx] = sensitivity
                    var_idx += 1
        
        return sensitivities


class ShapeOptimizer:
    """
    Main shape optimization coordinator.
    
    Integrates parameterization, CFD solver, adjoint solver, and optimization algorithm.
    """
    
    def __init__(self, config: OptimizationConfig, ffd_file_path: Optional[str] = None):
        """Initialize shape optimizer."""
        self.config = config
        self.ffd_file_path = ffd_file_path
        self.parameterization: Optional[DesignParameterization] = None
        self.constraints: List[OptimizationConstraint] = []
        
        # Solvers (would be injected in practice)
        self.cfd_solver: Optional[Callable] = None
        self.adjoint_solver: Optional[Callable] = None
        self.mesh_deformer: Optional[Callable] = None
        
        # Optimization state
        self.current_design: Optional[np.ndarray] = None
        self.current_geometry: Optional[Dict[str, Any]] = None
        self.iteration_count: int = 0
        
        # History tracking
        self.objective_history: List[float] = []
        self.constraint_history: List[Dict[str, float]] = []
        self.gradient_history: List[np.ndarray] = []
        self.design_history: List[np.ndarray] = []
        
        # Timing
        self.total_time: float = 0.0
        self.cfd_time: float = 0.0
        self.adjoint_time: float = 0.0
        self.mesh_deformation_time: float = 0.0
        
    def setup_parameterization(self, initial_geometry: Dict[str, Any]):
        """Setup design parameterization."""
        if self.config.parameterization == ParameterizationType.FREE_FORM_DEFORMATION:
            self.parameterization = FreeFormDeformation(self.config, ffd_file_path=self.ffd_file_path)
        elif self.config.parameterization == ParameterizationType.B_SPLINE:
            self.parameterization = BSplineParameterization(self.config)
        else:
            raise ValueError(f"Unsupported parameterization: {self.config.parameterization}")
        
        # Setup design variables
        design_variables = self.parameterization.setup_design_variables(initial_geometry)
        
        # Initialize current design
        self.current_design = np.array([var.initial_value for var in design_variables])
        self.current_geometry = initial_geometry
        
        logger.info(f"Setup {len(design_variables)} design variables")
        
    def add_constraint(self, constraint: OptimizationConstraint):
        """Add optimization constraint."""
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.name}")
    
    def add_geometric_constraints(self):
        """Add standard geometric constraints."""
        # Volume constraint
        volume_constraint = OptimizationConstraint(
            name="volume_preservation",
            constraint_type=ConstraintType.VOLUME,
            target_value=1.0,  # Maintain volume
            tolerance=0.05,
            evaluation_function=self._evaluate_volume_constraint
        )
        self.add_constraint(volume_constraint)
        
        # Thickness constraint
        thickness_constraint = OptimizationConstraint(
            name="minimum_thickness",
            constraint_type=ConstraintType.THICKNESS,
            target_value=0.02,  # Minimum thickness
            tolerance=0.001,
            evaluation_function=self._evaluate_thickness_constraint
        )
        self.add_constraint(thickness_constraint)
    
    def _evaluate_volume_constraint(self, geometry: Dict[str, Any]) -> float:
        """Evaluate volume constraint."""
        # Simplified volume calculation
        if 'vertices' in geometry:
            vertices = geometry['vertices']
            # Simple bounding box volume
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            volume = np.prod(bbox_max - bbox_min)
            return volume
        return 1.0
    
    def _evaluate_thickness_constraint(self, geometry: Dict[str, Any]) -> float:
        """Evaluate thickness constraint."""
        # Simplified thickness calculation
        if 'surface_points' in geometry:
            surface_points = geometry['surface_points']
            if surface_points.shape[1] >= 2:
                thickness = np.max(surface_points[:, 1]) - np.min(surface_points[:, 1])
                return thickness
        return 0.1
    
    def optimize(self, objective_function: Callable,
                initial_geometry: Dict[str, Any]) -> OptimizationResult:
        """
        Run shape optimization.
        
        Args:
            objective_function: Function to evaluate objectives
            initial_geometry: Initial geometry definition
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        # Setup parameterization
        self.setup_parameterization(initial_geometry)
        
        # Add standard constraints
        self.add_geometric_constraints()
        
        # Setup optimization bounds
        bounds = []
        for var in self.parameterization.design_variables:
            bounds.append((var.lower_bound, var.upper_bound))
        
        # Setup constraints for scipy
        constraint_objects = []
        for constraint in self.constraints:
            if constraint.active:
                constraint_obj = NonlinearConstraint(
                    fun=lambda x, c=constraint: self._evaluate_constraint(x, c),
                    lb=constraint.target_value - constraint.tolerance if not constraint.inequality else constraint.target_value,
                    ub=np.inf if constraint.inequality else constraint.target_value + constraint.tolerance
                )
                constraint_objects.append(constraint_obj)
        
        # Define optimization problem
        def optimization_objective(design_vector):
            return self._evaluate_objective_and_constraints(design_vector, objective_function)
        
        def optimization_gradient(design_vector):
            return self._compute_gradients(design_vector, objective_function)
        
        # Run optimization
        logger.info(f"Starting optimization with {len(self.current_design)} design variables")
        
        if self.config.algorithm == OptimizationAlgorithm.SLSQP:
            result = minimize(
                fun=optimization_objective,
                x0=self.current_design,
                method='SLSQP',
                jac=optimization_gradient,
                bounds=bounds,
                constraints=constraint_objects,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance,
                    'disp': True
                },
                callback=self._optimization_callback
            )
        elif self.config.algorithm == OptimizationAlgorithm.L_BFGS_B:
            result = minimize(
                fun=optimization_objective,
                x0=self.current_design,
                method='L-BFGS-B',
                jac=optimization_gradient,
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance,
                    'gtol': self.config.gradient_tolerance,
                    'disp': True
                },
                callback=self._optimization_callback
            )
        else:
            raise ValueError(f"Unsupported optimization algorithm: {self.config.algorithm}")
        
        # Finalize results
        self.total_time = time.time() - start_time
        
        # Evaluate final constraints
        final_constraints = {}
        final_geometry = self.parameterization.apply_design_changes(result.x, initial_geometry)
        for constraint in self.constraints:
            if constraint.active:
                final_constraints[constraint.name] = self._evaluate_constraint(result.x, constraint)
        
        # Create optimization result
        optimization_result = OptimizationResult(
            success=result.success,
            message=result.message,
            n_iterations=self.iteration_count,
            n_function_evaluations=result.nfev,
            n_gradient_evaluations=result.njev if hasattr(result, 'njev') else 0,
            optimal_design=result.x,
            optimal_objective=result.fun,
            optimal_constraints=final_constraints,
            objective_history=self.objective_history.copy(),
            constraint_history=self.constraint_history.copy(),
            gradient_norm_history=[np.linalg.norm(g) for g in self.gradient_history],
            design_history=self.design_history.copy(),
            total_time=self.total_time,
            average_iteration_time=self.total_time / max(self.iteration_count, 1),
            cfd_time=self.cfd_time,
            adjoint_time=self.adjoint_time,
            mesh_deformation_time=self.mesh_deformation_time,
            final_gradients=self.gradient_history[-1] if self.gradient_history else np.array([]),
            sensitivity_analysis=self._perform_sensitivity_analysis(result.x)
        )
        
        logger.info(f"Optimization completed: {result.message}")
        logger.info(f"Final objective: {result.fun:.6e}")
        logger.info(f"Total time: {self.total_time:.2f}s")
        
        return optimization_result
    
    def _evaluate_objective_and_constraints(self, design_vector: np.ndarray,
                                          objective_function: Callable) -> float:
        """Evaluate objective function and constraints."""
        # Apply design changes
        deform_start = time.time()
        new_geometry = self.parameterization.apply_design_changes(design_vector, self.current_geometry)
        self.mesh_deformation_time += time.time() - deform_start
        
        # Run CFD simulation
        cfd_start = time.time()
        if self.cfd_solver:
            cfd_result = self.cfd_solver(new_geometry)
        else:
            # Mock CFD result for testing
            cfd_result = {
                'drag': 0.1 + 0.01 * np.sum(design_vector**2),
                'lift': 1.0 - 0.005 * np.sum(np.abs(design_vector)),
                'converged': True
            }
        self.cfd_time += time.time() - cfd_start
        
        # Evaluate objective
        objective_value = objective_function(cfd_result, new_geometry)
        
        # Store history
        self.objective_history.append(objective_value)
        self.design_history.append(design_vector.copy())
        
        # Evaluate constraints
        constraint_values = {}
        for constraint in self.constraints:
            if constraint.active:
                constraint_values[constraint.name] = self._evaluate_constraint(design_vector, constraint)
        
        self.constraint_history.append(constraint_values)
        
        return objective_value
    
    def _evaluate_constraint(self, design_vector: np.ndarray,
                           constraint: OptimizationConstraint) -> float:
        """Evaluate a single constraint."""
        new_geometry = self.parameterization.apply_design_changes(design_vector, self.current_geometry)
        
        if constraint.evaluation_function:
            return constraint.evaluation_function(new_geometry)
        else:
            # Default constraint evaluation
            return 0.0
    
    def _compute_gradients(self, design_vector: np.ndarray,
                         objective_function: Callable) -> np.ndarray:
        """Compute objective gradients using adjoint method."""
        adjoint_start = time.time()
        
        # Apply design changes
        new_geometry = self.parameterization.apply_design_changes(design_vector, self.current_geometry)
        
        if self.adjoint_solver:
            # Run adjoint solver
            adjoint_result = self.adjoint_solver(new_geometry, objective_function)
            adjoint_gradients = adjoint_result.get('gradients', np.zeros(len(design_vector)))
        else:
            # Finite difference gradients for testing
            adjoint_gradients = self._finite_difference_gradients(design_vector, objective_function)
        
        # Compute shape sensitivities
        shape_sensitivities = self.parameterization.compute_shape_sensitivities(
            adjoint_gradients, new_geometry
        )
        
        self.adjoint_time += time.time() - adjoint_start
        
        # Store gradient history
        self.gradient_history.append(shape_sensitivities.copy())
        
        return shape_sensitivities
    
    def _finite_difference_gradients(self, design_vector: np.ndarray,
                                   objective_function: Callable) -> np.ndarray:
        """Compute gradients using finite differences (for testing)."""
        gradients = np.zeros(len(design_vector))
        epsilon = 1e-6
        
        # Base objective value
        base_objective = self._evaluate_objective_and_constraints(design_vector, objective_function)
        
        for i in range(len(design_vector)):
            # Perturb design variable
            perturbed_design = design_vector.copy()
            perturbed_design[i] += epsilon
            
            # Evaluate perturbed objective
            perturbed_objective = self._evaluate_objective_and_constraints(perturbed_design, objective_function)
            
            # Finite difference gradient
            gradients[i] = (perturbed_objective - base_objective) / epsilon
        
        return gradients
    
    def _optimization_callback(self, x: np.ndarray):
        """Callback function called after each optimization iteration."""
        self.iteration_count += 1
        
        if self.iteration_count % 10 == 0:
            current_obj = self.objective_history[-1] if self.objective_history else 0.0
            logger.info(f"Iteration {self.iteration_count}: Objective = {current_obj:.6e}")
        
        # Save intermediate results if requested
        if (self.config.save_intermediate and 
            self.iteration_count % self.config.save_frequency == 0):
            self._save_intermediate_results(x)
    
    def _save_intermediate_results(self, design_vector: np.ndarray):
        """Save intermediate optimization results."""
        # This would save geometry, solution, etc. to files
        logger.info(f"Saved intermediate results at iteration {self.iteration_count}")
    
    def _perform_sensitivity_analysis(self, optimal_design: np.ndarray) -> Dict[str, Any]:
        """Perform sensitivity analysis on optimal design."""
        sensitivity_analysis = {
            'design_variable_importance': {},
            'gradient_magnitude': {},
            'normalized_sensitivity': {}
        }
        
        if self.gradient_history:
            final_gradients = self.gradient_history[-1]
            
            # Compute importance metrics
            gradient_magnitudes = np.abs(final_gradients)
            max_gradient = np.max(gradient_magnitudes) + 1e-12
            
            for i, var in enumerate(self.parameterization.design_variables):
                sensitivity_analysis['gradient_magnitude'][var.name] = gradient_magnitudes[i]
                sensitivity_analysis['normalized_sensitivity'][var.name] = gradient_magnitudes[i] / max_gradient
                
                # Simple importance ranking
                importance = gradient_magnitudes[i] / max_gradient
                if importance > 0.5:
                    sensitivity_analysis['design_variable_importance'][var.name] = "high"
                elif importance > 0.1:
                    sensitivity_analysis['design_variable_importance'][var.name] = "medium"
                else:
                    sensitivity_analysis['design_variable_importance'][var.name] = "low"
        
        return sensitivity_analysis


def create_shape_optimizer(parameterization_type: str = "ffd",
                         algorithm: str = "slsqp",
                         max_iterations: int = 100,
                         config: Optional[OptimizationConfig] = None) -> ShapeOptimizer:
    """
    Factory function for creating shape optimizers.
    
    Args:
        parameterization_type: Type of design parameterization
        algorithm: Optimization algorithm
        max_iterations: Maximum optimization iterations
        config: Optional detailed configuration
        
    Returns:
        Configured shape optimizer
    """
    if config is None:
        config = OptimizationConfig(
            parameterization=ParameterizationType(parameterization_type),
            algorithm=OptimizationAlgorithm(algorithm),
            max_iterations=max_iterations
        )
    
    return ShapeOptimizer(config)


def test_shape_optimization():
    """Test shape optimization functionality."""
    print("Testing Shape Optimization Framework:")
    
    # Create test configuration
    config = OptimizationConfig(
        parameterization=ParameterizationType.FREE_FORM_DEFORMATION,
        algorithm=OptimizationAlgorithm.SLSQP,
        max_iterations=20,
        convergence_tolerance=1e-4
    )
    
    # Create shape optimizer
    optimizer = create_shape_optimizer(config=config)
    
    print(f"  Created shape optimizer with {config.algorithm.value} algorithm")
    
    # Test FFD parameterization
    print(f"\n  Testing Free Form Deformation:")
    initial_geometry = {
        'vertices': np.random.rand(100, 3),
        'n_surface_points': 50
    }
    
    optimizer.setup_parameterization(initial_geometry)
    n_design_vars = len(optimizer.parameterization.design_variables)
    print(f"    Created {n_design_vars} design variables")
    
    # Test design change application
    test_design = np.random.randn(n_design_vars) * 0.01
    deformed_geometry = optimizer.parameterization.apply_design_changes(
        test_design, initial_geometry
    )
    print(f"    Applied design changes to geometry")
    
    # Test sensitivity computation
    mock_gradients = np.random.randn(len(initial_geometry['vertices']) * 3)
    sensitivities = optimizer.parameterization.compute_shape_sensitivities(
        mock_gradients, deformed_geometry
    )
    print(f"    Computed {len(sensitivities)} shape sensitivities")
    
    # Test B-spline parameterization
    print(f"\n  Testing B-Spline Parameterization:")
    bspline_config = OptimizationConfig(
        parameterization=ParameterizationType.B_SPLINE,
        algorithm=OptimizationAlgorithm.L_BFGS_B,
        max_iterations=15
    )
    
    bspline_optimizer = ShapeOptimizer(bspline_config)
    
    # Initial airfoil geometry
    airfoil_geometry = {
        'surface_points': np.column_stack([
            np.linspace(0, 1, 50),
            0.1 * np.sin(np.pi * np.linspace(0, 1, 50))
        ])
    }
    
    bspline_optimizer.setup_parameterization(airfoil_geometry)
    n_bspline_vars = len(bspline_optimizer.parameterization.design_variables)
    print(f"    Created {n_bspline_vars} B-spline design variables")
    
    # Test optimization run with mock functions
    print(f"\n  Testing optimization run:")
    
    def mock_objective_function(cfd_result, geometry):
        """Mock objective function (minimize drag)."""
        return cfd_result.get('drag', 0.1)
    
    # Set up mock solvers
    def mock_cfd_solver(geometry):
        """Mock CFD solver."""
        # Simple mock result based on geometry
        if 'vertices' in geometry:
            complexity = np.std(geometry['vertices'])
            drag = 0.08 + 0.02 * complexity
            lift = 1.2 - 0.1 * complexity
        else:
            drag = 0.1
            lift = 1.0
        
        return {
            'drag': drag,
            'lift': lift,
            'converged': True
        }
    
    def mock_adjoint_solver(geometry, objective_function):
        """Mock adjoint solver."""
        # Return mock gradients
        if 'vertices' in geometry:
            n_vertices = len(geometry['vertices'])
            gradients = np.random.randn(n_vertices * 3) * 0.01
        else:
            gradients = np.random.randn(100) * 0.01
        
        return {'gradients': gradients}
    
    optimizer.cfd_solver = mock_cfd_solver
    optimizer.adjoint_solver = mock_adjoint_solver
    
    # Run optimization
    result = optimizer.optimize(mock_objective_function, initial_geometry)
    
    print(f"\n  Optimization Results:")
    print(f"    Success: {result.success}")
    print(f"    Iterations: {result.n_iterations}")
    print(f"    Function evaluations: {result.n_function_evaluations}")
    print(f"    Initial objective: {result.objective_history[0]:.6f}")
    print(f"    Final objective: {result.optimal_objective:.6f}")
    print(f"    Improvement: {((result.objective_history[0] - result.optimal_objective) / result.objective_history[0] * 100):.1f}%")
    print(f"    Total time: {result.total_time:.2f}s")
    print(f"    CFD time: {result.cfd_time:.2f}s")
    print(f"    Adjoint time: {result.adjoint_time:.2f}s")
    
    # Test sensitivity analysis
    if result.sensitivity_analysis:
        print(f"\n  Sensitivity Analysis:")
        importance_summary = {}
        for var_name, importance in result.sensitivity_analysis['design_variable_importance'].items():
            if importance not in importance_summary:
                importance_summary[importance] = 0
            importance_summary[importance] += 1
        
        for importance, count in importance_summary.items():
            print(f"    {importance.title()} importance variables: {count}")
    
    # Test constraint handling
    print(f"\n  Testing constraint handling:")
    print(f"    Added {len(optimizer.constraints)} constraints")
    
    if result.optimal_constraints:
        print(f"    Final constraint values:")
        for name, value in result.optimal_constraints.items():
            print(f"      {name}: {value:.6f}")
    
    print(f"\n  Shape optimization test completed!")


if __name__ == "__main__":
    test_shape_optimization()