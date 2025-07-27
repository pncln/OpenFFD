"""
3D Unstructured Mesh Data Structure for Supersonic CFD

Provides comprehensive mesh representation with:
- Support for mixed element types (tetrahedra, hexahedra, prisms, pyramids)
- Efficient connectivity storage and access
- Geometric properties computation
- Boundary condition management
- Memory-optimized data layout for cache efficiency
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import logging

logger = logging.getLogger(__name__)


class CellType(IntEnum):
    """Cell type enumeration matching VTK conventions for interoperability."""
    TETRAHEDRON = 10
    HEXAHEDRON = 12
    WEDGE = 13       # Prism
    PYRAMID = 14
    TRIANGLE = 5     # 2D boundary faces
    QUADRILATERAL = 9  # 2D boundary faces


class BoundaryCondition(Enum):
    """Boundary condition types for supersonic flows."""
    FARFIELD = "farfield"
    WALL = "wall"
    SYMMETRY = "symmetry"
    INLET = "inlet"
    OUTLET = "outlet"
    PRESSURE_OUTLET = "pressure_outlet"
    PERIODIC = "periodic"


@dataclass
class CellData:
    """Cell-centered data storage with memory layout optimization."""
    # Conservative variables [rho, rho*u, rho*v, rho*w, rho*E]
    conservatives: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    
    # Primitive variables [rho, u, v, w, p, T]
    primitives: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    
    # Residuals for time integration
    residuals: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    
    # Time step size for each cell
    dt: np.ndarray = field(default_factory=lambda: np.empty(0))
    
    # Cell volumes
    volumes: np.ndarray = field(default_factory=lambda: np.empty(0))
    
    # Cell centroids
    centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    
    # Gradient storage for high-order schemes
    gradients: Optional[np.ndarray] = None  # Shape: (n_cells, 5, 3)
    
    # Quality metrics
    aspect_ratio: np.ndarray = field(default_factory=lambda: np.empty(0))
    skewness: np.ndarray = field(default_factory=lambda: np.empty(0))
    
    def resize(self, n_cells: int) -> None:
        """Resize all arrays for given number of cells."""
        self.conservatives = np.zeros((n_cells, 5))
        self.primitives = np.zeros((n_cells, 6))
        self.residuals = np.zeros((n_cells, 5))
        self.dt = np.zeros(n_cells)
        self.volumes = np.zeros(n_cells)
        self.centroids = np.zeros((n_cells, 3))
        self.aspect_ratio = np.zeros(n_cells)
        self.skewness = np.zeros(n_cells)
        
        # Optional high-order gradient storage
        if self.gradients is not None:
            self.gradients = np.zeros((n_cells, 5, 3))


@dataclass 
class FaceData:
    """Face-centered data for flux computation."""
    # Face centers
    centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    
    # Face normal vectors (outward from owner cell)
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    
    # Face areas
    areas: np.ndarray = field(default_factory=lambda: np.empty(0))
    
    # Flux values [mass, momentum_x, momentum_y, momentum_z, energy]
    fluxes: np.ndarray = field(default_factory=lambda: np.empty((0, 5)))
    
    # Owner and neighbor cell indices (-1 for boundary faces)
    owner: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    neighbor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    
    # Boundary condition markers for boundary faces
    boundary_markers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    
    def resize(self, n_faces: int) -> None:
        """Resize all arrays for given number of faces."""
        self.centers = np.zeros((n_faces, 3))
        self.normals = np.zeros((n_faces, 3))
        self.areas = np.zeros(n_faces)
        self.fluxes = np.zeros((n_faces, 5))
        self.owner = np.full(n_faces, -1, dtype=np.int32)
        self.neighbor = np.full(n_faces, -1, dtype=np.int32)
        self.boundary_markers = np.zeros(n_faces, dtype=np.int32)


class UnstructuredMesh3D:
    """
    3D Unstructured mesh class optimized for supersonic CFD applications.
    
    Features:
    - Mixed element support (tetrahedra, hexahedra, prisms, pyramids)
    - Efficient face-based connectivity for finite volume method
    - Memory-optimized data layout for cache performance
    - Geometric properties computation
    - Boundary condition management
    - Parallel decomposition support
    """
    
    def __init__(self, 
                 points: Optional[np.ndarray] = None,
                 cells: Optional[Dict[CellType, np.ndarray]] = None,
                 boundary_patches: Optional[Dict[str, Dict]] = None):
        """
        Initialize unstructured mesh.
        
        Args:
            points: Nodal coordinates [n_points, 3]
            cells: Dictionary mapping cell types to connectivity arrays
            boundary_patches: Dictionary of boundary patch definitions
        """
        # Nodal data
        self.points = points if points is not None else np.empty((0, 3))
        self.n_points = len(self.points)
        
        # Cell connectivity: {CellType: connectivity_array}
        self.cells = cells if cells is not None else {}
        self.n_cells = sum(len(conn) for conn in self.cells.values())
        
        # Face connectivity and data
        self.faces: Dict[str, np.ndarray] = {}
        self.face_data = FaceData()
        self.n_faces = 0
        
        # Cell data
        self.cell_data = CellData()
        
        # Boundary management
        self.boundary_patches = boundary_patches if boundary_patches is not None else {}
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        
        # Connectivity matrices for efficient neighbor access
        self._cell_to_faces: Optional[List[List[int]]] = None
        self._face_to_cells: Optional[np.ndarray] = None
        
        # Geometric properties
        self._geometry_computed = False
        
        # Parallel decomposition info
        self.partition_id = 0
        self.n_partitions = 1
        self.ghost_cells: List[int] = []
        
        # Initialize if data provided
        if self.n_cells > 0:
            self._build_connectivity()
            self._compute_geometry()
    
    def add_cells(self, cell_type: CellType, connectivity: np.ndarray) -> None:
        """Add cells of specified type."""
        if cell_type in self.cells:
            self.cells[cell_type] = np.vstack([self.cells[cell_type], connectivity])
        else:
            self.cells[cell_type] = connectivity.copy()
        
        self.n_cells = sum(len(conn) for conn in self.cells.values())
        self._geometry_computed = False
    
    def add_boundary_patch(self, 
                          name: str, 
                          faces: np.ndarray,
                          bc_type: BoundaryCondition,
                          properties: Optional[Dict] = None) -> None:
        """Add boundary patch with specified boundary condition."""
        self.boundary_patches[name] = {
            'faces': faces,
            'type': bc_type,
            'properties': properties or {}
        }
        self.boundary_conditions[name] = bc_type
    
    def _build_connectivity(self) -> None:
        """Build face-based connectivity from cell data."""
        logger.info("Building face connectivity...")
        
        # Extract all unique faces from cells
        all_faces = []
        face_to_cell = []
        
        cell_idx = 0
        for cell_type, connectivity in self.cells.items():
            for i, cell_nodes in enumerate(connectivity):
                faces = self._extract_cell_faces(cell_type, cell_nodes)
                for face in faces:
                    # Sort face nodes to ensure uniqueness
                    sorted_face = tuple(sorted(face))
                    all_faces.append(sorted_face)
                    face_to_cell.append(cell_idx)
                cell_idx += 1
        
        # Find unique faces and build owner/neighbor relationships
        unique_faces = {}
        face_owners = []
        face_neighbors = []
        
        for i, face in enumerate(all_faces):
            if face in unique_faces:
                # Internal face - add neighbor
                face_idx = unique_faces[face]
                face_neighbors[face_idx] = face_to_cell[i]
            else:
                # New face - add as owner
                face_idx = len(unique_faces)
                unique_faces[face] = face_idx
                face_owners.append(face_to_cell[i])
                face_neighbors.append(-1)  # Boundary face initially
        
        self.n_faces = len(unique_faces)
        self.face_data.resize(self.n_faces)
        self.face_data.owner = np.array(face_owners, dtype=np.int32)
        self.face_data.neighbor = np.array(face_neighbors, dtype=np.int32)
        
        # Store face node connectivity
        self.faces['connectivity'] = np.array([list(face) for face in unique_faces.keys()])
        
        logger.info(f"Built connectivity: {self.n_faces} faces, {self.n_cells} cells")
    
    def _extract_cell_faces(self, cell_type: CellType, nodes: np.ndarray) -> List[Tuple]:
        """Extract face connectivity from cell nodes."""
        if cell_type == CellType.TETRAHEDRON:
            # 4 triangular faces
            return [
                (nodes[0], nodes[1], nodes[2]),
                (nodes[0], nodes[1], nodes[3]),
                (nodes[1], nodes[2], nodes[3]),
                (nodes[0], nodes[2], nodes[3])
            ]
        elif cell_type == CellType.HEXAHEDRON:
            # 6 quadrilateral faces
            return [
                (nodes[0], nodes[1], nodes[2], nodes[3]),  # Bottom
                (nodes[4], nodes[5], nodes[6], nodes[7]),  # Top
                (nodes[0], nodes[1], nodes[5], nodes[4]),  # Front
                (nodes[2], nodes[3], nodes[7], nodes[6]),  # Back
                (nodes[0], nodes[3], nodes[7], nodes[4]),  # Left
                (nodes[1], nodes[2], nodes[6], nodes[5])   # Right
            ]
        elif cell_type == CellType.WEDGE:
            # 3 quadrilateral + 2 triangular faces
            return [
                (nodes[0], nodes[1], nodes[2]),           # Bottom triangle
                (nodes[3], nodes[4], nodes[5]),           # Top triangle
                (nodes[0], nodes[1], nodes[4], nodes[3]), # Side quad
                (nodes[1], nodes[2], nodes[5], nodes[4]), # Side quad
                (nodes[0], nodes[2], nodes[5], nodes[3])  # Side quad
            ]
        elif cell_type == CellType.PYRAMID:
            # 1 quadrilateral + 4 triangular faces
            return [
                (nodes[0], nodes[1], nodes[2], nodes[3]), # Base quad
                (nodes[0], nodes[1], nodes[4]),           # Triangular face
                (nodes[1], nodes[2], nodes[4]),           # Triangular face
                (nodes[2], nodes[3], nodes[4]),           # Triangular face
                (nodes[0], nodes[3], nodes[4])            # Triangular face
            ]
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
    
    def _compute_geometry(self) -> None:
        """Compute geometric properties for all cells and faces."""
        logger.info("Computing geometric properties...")
        
        # Resize data arrays
        self.cell_data.resize(self.n_cells)
        
        # Compute cell properties
        cell_idx = 0
        for cell_type, connectivity in self.cells.items():
            for cell_nodes in connectivity:
                points = self.points[cell_nodes]
                
                # Compute cell volume and centroid
                volume, centroid = self._compute_cell_geometry(cell_type, points)
                self.cell_data.volumes[cell_idx] = volume
                self.cell_data.centroids[cell_idx] = centroid
                
                # Compute quality metrics
                self.cell_data.aspect_ratio[cell_idx] = self._compute_aspect_ratio(cell_type, points)
                self.cell_data.skewness[cell_idx] = self._compute_skewness(cell_type, points)
                
                cell_idx += 1
        
        # Compute face properties
        for i, face_nodes in enumerate(self.faces['connectivity']):
            points = self.points[face_nodes]
            
            # Face center
            self.face_data.centers[i] = np.mean(points, axis=0)
            
            # Face normal and area
            if len(face_nodes) == 3:  # Triangle
                v1 = points[1] - points[0]
                v2 = points[2] - points[0]
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                self.face_data.normals[i] = normal / (2 * area)
            elif len(face_nodes) == 4:  # Quadrilateral
                # Split into two triangles and average
                v1 = points[2] - points[0]
                v2 = points[3] - points[1]
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                self.face_data.normals[i] = normal / (2 * area)
            
            self.face_data.areas[i] = area
        
        self._geometry_computed = True
        logger.info("Geometric properties computed successfully")
    
    def _compute_cell_geometry(self, cell_type: CellType, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute volume and centroid for a cell."""
        if cell_type == CellType.TETRAHEDRON:
            # Volume = |det(v1, v2, v3)| / 6
            v1 = points[1] - points[0]
            v2 = points[2] - points[0]
            v3 = points[3] - points[0]
            volume = abs(np.linalg.det([v1, v2, v3])) / 6.0
            centroid = np.mean(points, axis=0)
            
        elif cell_type == CellType.HEXAHEDRON:
            # Decompose into tetrahedra for volume calculation
            # Simple centroid calculation
            centroid = np.mean(points, axis=0)
            
            # Approximate volume (can be improved with proper decomposition)
            # For now, use bounding box approximation
            bbox = np.max(points, axis=0) - np.min(points, axis=0)
            volume = np.prod(bbox)  # Rough approximation
            
        else:
            # Default fallback - bounding box volume
            centroid = np.mean(points, axis=0)
            bbox = np.max(points, axis=0) - np.min(points, axis=0)
            volume = np.prod(bbox)
        
        return volume, centroid
    
    def _compute_aspect_ratio(self, cell_type: CellType, points: np.ndarray) -> float:
        """Compute aspect ratio quality metric."""
        # Simple implementation based on edge length ratios
        edges = []
        n_points = len(points)
        for i in range(n_points):
            for j in range(i+1, n_points):
                edge_length = np.linalg.norm(points[j] - points[i])
                edges.append(edge_length)
        
        if len(edges) > 0:
            return max(edges) / min(edges)
        return 1.0
    
    def _compute_skewness(self, cell_type: CellType, points: np.ndarray) -> float:
        """Compute skewness quality metric."""
        # Simplified skewness based on centroid displacement
        centroid = np.mean(points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in points]
        if len(distances) > 1:
            return np.std(distances) / np.mean(distances)
        return 0.0
    
    def get_cell_neighbors(self, cell_id: int) -> List[int]:
        """Get neighboring cells for a given cell."""
        if self._cell_to_faces is None:
            self._build_cell_face_connectivity()
        
        neighbors = []
        for face_id in self._cell_to_faces[cell_id]:
            owner = self.face_data.owner[face_id]
            neighbor = self.face_data.neighbor[face_id]
            
            if owner == cell_id and neighbor >= 0:
                neighbors.append(neighbor)
            elif neighbor == cell_id and owner >= 0:
                neighbors.append(owner)
        
        return neighbors
    
    def _build_cell_face_connectivity(self) -> None:
        """Build cell-to-face connectivity matrix."""
        self._cell_to_faces = [[] for _ in range(self.n_cells)]
        
        for face_id in range(self.n_faces):
            owner = self.face_data.owner[face_id]
            neighbor = self.face_data.neighbor[face_id]
            
            if owner >= 0:
                self._cell_to_faces[owner].append(face_id)
            if neighbor >= 0:
                self._cell_to_faces[neighbor].append(face_id)
    
    def get_boundary_faces(self, patch_name: str) -> np.ndarray:
        """Get face indices for a boundary patch."""
        if patch_name in self.boundary_patches:
            return self.boundary_patches[patch_name]['faces']
        return np.array([], dtype=np.int32)
    
    def compute_mesh_quality(self) -> Dict[str, float]:
        """Compute overall mesh quality metrics."""
        if not self._geometry_computed:
            self._compute_geometry()
        
        return {
            'min_volume': np.min(self.cell_data.volumes),
            'max_volume': np.max(self.cell_data.volumes),
            'mean_volume': np.mean(self.cell_data.volumes),
            'volume_ratio': np.max(self.cell_data.volumes) / np.min(self.cell_data.volumes),
            'mean_aspect_ratio': np.mean(self.cell_data.aspect_ratio),
            'max_aspect_ratio': np.max(self.cell_data.aspect_ratio),
            'mean_skewness': np.mean(self.cell_data.skewness),
            'max_skewness': np.max(self.cell_data.skewness),
            'n_cells': self.n_cells,
            'n_faces': self.n_faces,
            'n_boundary_faces': np.sum(self.face_data.neighbor == -1)
        }
    
    def save_to_vtk(self, filename: str) -> None:
        """Save mesh to VTK format for visualization."""
        try:
            import pyvista as pv
            
            # Create PyVista mesh
            cells = []
            cell_types = []
            
            for cell_type, connectivity in self.cells.items():
                for cell_nodes in connectivity:
                    cells.extend([len(cell_nodes)] + cell_nodes.tolist())
                    cell_types.append(cell_type.value)
            
            mesh = pv.UnstructuredGrid(cells, cell_types, self.points)
            
            # Add cell data
            if self._geometry_computed:
                mesh.cell_data['Volume'] = self.cell_data.volumes
                mesh.cell_data['AspectRatio'] = self.cell_data.aspect_ratio
                mesh.cell_data['Skewness'] = self.cell_data.skewness
            
            mesh.save(filename)
            logger.info(f"Mesh saved to {filename}")
            
        except ImportError:
            logger.warning("PyVista not available - cannot save VTK file")
    
    def __repr__(self) -> str:
        """String representation of mesh."""
        cell_types = list(self.cells.keys())
        return (f"UnstructuredMesh3D(points={self.n_points}, cells={self.n_cells}, "
                f"faces={self.n_faces}, types={cell_types})")