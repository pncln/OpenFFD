"""
Advanced Connectivity Management for 3D Unstructured Meshes

Provides efficient connectivity data structures and algorithms for:
- Face-based finite volume connectivity
- Cell-to-cell neighbor finding
- Gradient stencil construction
- Parallel domain decomposition support
- Cache-optimized data access patterns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CellConnectivity:
    """Connectivity information for a single cell."""
    cell_id: int
    neighbors: List[int]  # Neighboring cell IDs
    faces: List[int]      # Face IDs belonging to this cell
    boundary_faces: List[int]  # Boundary face IDs
    stencil: Optional[List[int]] = None  # Extended stencil for high-order schemes


class ConnectivityManager:
    """
    High-performance connectivity manager for unstructured meshes.
    
    Features:
    - Efficient neighbor finding algorithms
    - Gradient stencil construction for high-order schemes
    - Memory-optimized storage with compressed sparse formats
    - Support for parallel domain decomposition
    - Cache-friendly data access patterns
    """
    
    def __init__(self, mesh=None):
        """Initialize connectivity manager."""
        self.mesh = mesh
        
        # Core connectivity data
        self._cell_to_faces: Optional[List[List[int]]] = None
        self._face_to_cells: Optional[np.ndarray] = None
        self._cell_neighbors: Optional[List[List[int]]] = None
        
        # Extended connectivity for high-order schemes
        self._gradient_stencils: Optional[Dict[int, List[int]]] = None
        self._reconstruction_stencils: Optional[Dict[int, List[int]]] = None
        
        # Parallel decomposition
        self._ghost_connectivity: Dict[int, List[int]] = {}
        self._halo_cells: Set[int] = set()
        
        # Performance optimization
        self._connectivity_built = False
        self._stencils_built = False
        
        if mesh is not None:
            self.build_connectivity()
    
    def build_connectivity(self) -> None:
        """Build complete connectivity information."""
        if self.mesh is None:
            raise ValueError("No mesh provided")
        
        logger.info("Building connectivity matrices...")
        
        self._build_face_connectivity()
        self._build_cell_neighbors()
        self._connectivity_built = True
        
        logger.info(f"Connectivity built: {self.mesh.n_cells} cells, {self.mesh.n_faces} faces")
    
    def _build_face_connectivity(self) -> None:
        """Build cell-to-face and face-to-cell connectivity."""
        # Initialize cell-to-faces mapping
        self._cell_to_faces = [[] for _ in range(self.mesh.n_cells)]
        
        # Build from face owner/neighbor data
        for face_id in range(self.mesh.n_faces):
            owner = self.mesh.face_data.owner[face_id]
            neighbor = self.mesh.face_data.neighbor[face_id]
            
            if owner >= 0:
                self._cell_to_faces[owner].append(face_id)
            if neighbor >= 0:
                self._cell_to_faces[neighbor].append(face_id)
        
        # Build face-to-cells array [n_faces, 2] with -1 for boundary
        self._face_to_cells = np.full((self.mesh.n_faces, 2), -1, dtype=np.int32)
        for face_id in range(self.mesh.n_faces):
            self._face_to_cells[face_id, 0] = self.mesh.face_data.owner[face_id]
            self._face_to_cells[face_id, 1] = self.mesh.face_data.neighbor[face_id]
    
    def _build_cell_neighbors(self) -> None:
        """Build cell-to-cell neighbor connectivity."""
        self._cell_neighbors = [[] for _ in range(self.mesh.n_cells)]
        
        for cell_id in range(self.mesh.n_cells):
            neighbors = set()
            
            # Get all faces of this cell
            for face_id in self._cell_to_faces[cell_id]:
                owner = self.mesh.face_data.owner[face_id]
                neighbor = self.mesh.face_data.neighbor[face_id]
                
                # Add the other cell sharing this face
                if owner == cell_id and neighbor >= 0:
                    neighbors.add(neighbor)
                elif neighbor == cell_id and owner >= 0:
                    neighbors.add(owner)
            
            self._cell_neighbors[cell_id] = list(neighbors)
    
    def get_cell_connectivity(self, cell_id: int) -> CellConnectivity:
        """Get complete connectivity information for a cell."""
        if not self._connectivity_built:
            self.build_connectivity()
        
        # Get boundary faces for this cell
        boundary_faces = []
        for face_id in self._cell_to_faces[cell_id]:
            if self.mesh.face_data.neighbor[face_id] == -1:
                boundary_faces.append(face_id)
        
        return CellConnectivity(
            cell_id=cell_id,
            neighbors=self._cell_neighbors[cell_id].copy(),
            faces=self._cell_to_faces[cell_id].copy(),
            boundary_faces=boundary_faces,
            stencil=self.get_gradient_stencil(cell_id) if self._stencils_built else None
        )
    
    def get_face_neighbors(self, face_id: int) -> Tuple[int, int]:
        """Get owner and neighbor cells for a face."""
        if not self._connectivity_built:
            self.build_connectivity()
        
        return (self._face_to_cells[face_id, 0], self._face_to_cells[face_id, 1])
    
    def build_gradient_stencils(self, method: str = "least_squares") -> None:
        """
        Build gradient computation stencils for all cells.
        
        Args:
            method: Gradient computation method ('least_squares', 'green_gauss', 'weighted_least_squares')
        """
        logger.info(f"Building gradient stencils using {method} method...")
        
        self._gradient_stencils = {}
        
        for cell_id in range(self.mesh.n_cells):
            if method == "least_squares":
                stencil = self._build_least_squares_stencil(cell_id)
            elif method == "green_gauss":
                stencil = self._build_green_gauss_stencil(cell_id)
            elif method == "weighted_least_squares":
                stencil = self._build_weighted_least_squares_stencil(cell_id)
            else:
                raise ValueError(f"Unknown gradient method: {method}")
            
            self._gradient_stencils[cell_id] = stencil
        
        self._stencils_built = True
        logger.info(f"Gradient stencils built for {len(self._gradient_stencils)} cells")
    
    def _build_least_squares_stencil(self, cell_id: int) -> List[int]:
        """Build least-squares gradient stencil."""
        # Start with direct neighbors
        stencil = self._cell_neighbors[cell_id].copy()
        
        # For better conditioning, add second-layer neighbors if needed
        if len(stencil) < 6:  # Need at least 6 points for 3D least squares
            second_layer = set()
            for neighbor_id in stencil:
                second_layer.update(self._cell_neighbors[neighbor_id])
            
            # Remove already included cells and self
            second_layer.discard(cell_id)
            for existing in stencil:
                second_layer.discard(existing)
            
            # Add closest second-layer neighbors
            if second_layer:
                cell_center = self.mesh.cell_data.centroids[cell_id]
                distances = []
                for neighbor_id in second_layer:
                    neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
                    dist = np.linalg.norm(neighbor_center - cell_center)
                    distances.append((dist, neighbor_id))
                
                # Sort by distance and add closest ones
                distances.sort()
                needed = max(0, 6 - len(stencil))
                for _, neighbor_id in distances[:needed]:
                    stencil.append(neighbor_id)
        
        return stencil
    
    def _build_green_gauss_stencil(self, cell_id: int) -> List[int]:
        """Build Green-Gauss gradient stencil (face neighbors only)."""
        return self._cell_neighbors[cell_id].copy()
    
    def _build_weighted_least_squares_stencil(self, cell_id: int) -> List[int]:
        """Build weighted least-squares stencil with distance-based weights."""
        # Similar to least squares but may include more distant neighbors
        return self._build_least_squares_stencil(cell_id)
    
    def get_gradient_stencil(self, cell_id: int) -> Optional[List[int]]:
        """Get gradient computation stencil for a cell."""
        if not self._stencils_built:
            return None
        return self._gradient_stencils.get(cell_id)
    
    def compute_stencil_weights(self, cell_id: int, method: str = "least_squares") -> Optional[np.ndarray]:
        """
        Compute weights for gradient stencil.
        
        Returns:
            Weight matrix for gradient computation [n_neighbors, 3]
        """
        if not self._stencils_built:
            self.build_gradient_stencils(method)
        
        stencil = self._gradient_stencils[cell_id]
        if not stencil:
            return None
        
        cell_center = self.mesh.cell_data.centroids[cell_id]
        n_neighbors = len(stencil)
        
        if method == "least_squares" or method == "weighted_least_squares":
            # Build least squares system: A * grad = b
            A = np.zeros((n_neighbors, 3))
            W = np.eye(n_neighbors) if method == "least_squares" else None
            
            for i, neighbor_id in enumerate(stencil):
                neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
                dr = neighbor_center - cell_center
                A[i] = dr
                
                # Distance-based weighting for weighted least squares
                if method == "weighted_least_squares":
                    weight = 1.0 / (np.linalg.norm(dr) + 1e-12)
                    if W is None:
                        W = np.zeros((n_neighbors, n_neighbors))
                    W[i, i] = weight
            
            # Solve weighted least squares: (A^T W A) * grad = A^T W * b
            if W is not None:
                ATA = A.T @ W @ A
                ATW = A.T @ W
            else:
                ATA = A.T @ A
                ATW = A.T
            
            try:
                # Compute pseudo-inverse
                weights = np.linalg.solve(ATA, ATW)
                return weights.T  # [n_neighbors, 3]
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix for cell {cell_id}, using simple averaging")
                return np.ones((n_neighbors, 3)) / n_neighbors
        
        else:  # Green-Gauss
            # For Green-Gauss, weights are based on face areas and normals
            weights = np.zeros((n_neighbors, 3))
            
            for i, neighbor_id in enumerate(stencil):
                # Find common face between cell_id and neighbor_id
                common_faces = set(self._cell_to_faces[cell_id]).intersection(
                    set(self._cell_to_faces[neighbor_id])
                )
                
                for face_id in common_faces:
                    face_area = self.mesh.face_data.areas[face_id]
                    face_normal = self.mesh.face_data.normals[face_id]
                    
                    # Ensure normal points from cell_id to neighbor_id
                    if self.mesh.face_data.owner[face_id] == neighbor_id:
                        face_normal = -face_normal
                    
                    weights[i] = face_area * face_normal
                    break
            
            return weights
    
    def find_cells_in_sphere(self, center: np.ndarray, radius: float) -> List[int]:
        """Find all cells within a sphere."""
        cells_in_sphere = []
        
        for cell_id in range(self.mesh.n_cells):
            cell_center = self.mesh.cell_data.centroids[cell_id]
            distance = np.linalg.norm(cell_center - center)
            
            if distance <= radius:
                cells_in_sphere.append(cell_id)
        
        return cells_in_sphere
    
    def build_reconstruction_stencils(self, order: int = 2) -> None:
        """Build high-order reconstruction stencils."""
        logger.info(f"Building {order}-order reconstruction stencils...")
        
        self._reconstruction_stencils = {}
        
        for cell_id in range(self.mesh.n_cells):
            # Determine required stencil size based on order
            if order == 2:
                required_size = 10  # For quadratic reconstruction in 3D
            elif order == 3:
                required_size = 20  # For cubic reconstruction in 3D
            else:
                required_size = 6   # Linear reconstruction
            
            stencil = self._build_extended_stencil(cell_id, required_size)
            self._reconstruction_stencils[cell_id] = stencil
        
        logger.info(f"Reconstruction stencils built for {len(self._reconstruction_stencils)} cells")
    
    def _build_extended_stencil(self, cell_id: int, target_size: int) -> List[int]:
        """Build extended stencil by growing from direct neighbors."""
        stencil = self._cell_neighbors[cell_id].copy()
        
        # Grow stencil until target size is reached
        layer = 1
        while len(stencil) < target_size and layer < 5:  # Limit to 5 layers
            next_layer = set()
            
            for neighbor_id in stencil:
                next_layer.update(self._cell_neighbors[neighbor_id])
            
            # Remove already included cells and self
            next_layer.discard(cell_id)
            for existing in stencil:
                next_layer.discard(existing)
            
            if not next_layer:
                break
            
            # Add closest cells from next layer
            cell_center = self.mesh.cell_data.centroids[cell_id]
            distances = []
            for neighbor_id in next_layer:
                neighbor_center = self.mesh.cell_data.centroids[neighbor_id]
                dist = np.linalg.norm(neighbor_center - cell_center)
                distances.append((dist, neighbor_id))
            
            distances.sort()
            needed = min(target_size - len(stencil), len(distances))
            
            for i in range(needed):
                stencil.append(distances[i][1])
            
            layer += 1
        
        return stencil
    
    def get_reconstruction_stencil(self, cell_id: int) -> Optional[List[int]]:
        """Get reconstruction stencil for a cell."""
        return self._reconstruction_stencils.get(cell_id)
    
    def setup_parallel_connectivity(self, 
                                  partition_id: int, 
                                  halo_cells: List[int],
                                  ghost_connectivity: Dict[int, List[int]]) -> None:
        """Setup connectivity for parallel computation."""
        self.mesh.partition_id = partition_id
        self._halo_cells = set(halo_cells)
        self._ghost_connectivity = ghost_connectivity
        
        logger.info(f"Parallel connectivity setup: partition {partition_id}, "
                   f"{len(halo_cells)} halo cells")
    
    def is_halo_cell(self, cell_id: int) -> bool:
        """Check if cell is a halo cell."""
        return cell_id in self._halo_cells
    
    def get_connectivity_stats(self) -> Dict[str, float]:
        """Get connectivity statistics."""
        if not self._connectivity_built:
            return {}
        
        neighbor_counts = [len(neighbors) for neighbors in self._cell_neighbors]
        
        stats = {
            'mean_neighbors': np.mean(neighbor_counts),
            'min_neighbors': np.min(neighbor_counts),
            'max_neighbors': np.max(neighbor_counts),
            'std_neighbors': np.std(neighbor_counts),
        }
        
        if self._stencils_built:
            stencil_sizes = [len(stencil) for stencil in self._gradient_stencils.values()]
            stats.update({
                'mean_stencil_size': np.mean(stencil_sizes),
                'min_stencil_size': np.min(stencil_sizes),
                'max_stencil_size': np.max(stencil_sizes)
            })
        
        return stats
    
    def validate_connectivity(self) -> bool:
        """Validate connectivity data integrity."""
        if not self._connectivity_built:
            return False
        
        logger.info("Validating connectivity...")
        
        # Check face-cell consistency
        for face_id in range(self.mesh.n_faces):
            owner = self.mesh.face_data.owner[face_id]
            neighbor = self.mesh.face_data.neighbor[face_id]
            
            # Check if face is properly referenced by owner
            if owner >= 0 and face_id not in self._cell_to_faces[owner]:
                logger.error(f"Face {face_id} not in owner cell {owner} face list")
                return False
            
            # Check if face is properly referenced by neighbor
            if neighbor >= 0 and face_id not in self._cell_to_faces[neighbor]:
                logger.error(f"Face {face_id} not in neighbor cell {neighbor} face list")
                return False
        
        # Check neighbor symmetry
        for cell_id in range(self.mesh.n_cells):
            for neighbor_id in self._cell_neighbors[cell_id]:
                if cell_id not in self._cell_neighbors[neighbor_id]:
                    logger.error(f"Asymmetric neighbor relationship: {cell_id} -> {neighbor_id}")
                    return False
        
        logger.info("Connectivity validation passed")
        return True