"""
Geometric Properties and Metric Calculations for 3D Unstructured Meshes

Provides comprehensive geometric computations including:
- Cell volumes, centroids, and quality metrics
- Face areas, normals, and geometric centers
- Metric tensors for coordinate transformations
- Distance functions and geometric queries
- High-precision geometric calculations for supersonic flows
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricTensor:
    """Metric tensor for coordinate transformations and grid quality assessment."""
    # Metric tensor components [3x3] for each cell
    g_ij: np.ndarray  # Covariant metric tensor
    g_contravariant: np.ndarray  # Contravariant metric tensor
    jacobian: float  # Jacobian determinant
    condition_number: float  # Condition number for grid quality
    
    # Grid skewness and orthogonality measures
    skewness_angle: float  # Maximum skewness angle in degrees
    orthogonality: float   # Orthogonality measure [0, 1]
    aspect_ratio: float    # Grid aspect ratio


class GeometricProperties:
    """
    High-precision geometric property calculator for unstructured meshes.
    
    Features:
    - Exact volume and area calculations
    - Robust normal vector computation
    - Metric tensor analysis
    - Grid quality assessment
    - Distance functions and spatial queries
    """
    
    def __init__(self, mesh=None, precision: str = "double"):
        """
        Initialize geometric properties calculator.
        
        Args:
            mesh: Unstructured mesh object
            precision: Numerical precision ("single", "double", "extended")
        """
        self.mesh = mesh
        self.precision = precision
        self._tolerance = self._get_tolerance(precision)
        
        # Cached geometric data
        self._cell_volumes: Optional[np.ndarray] = None
        self._cell_centroids: Optional[np.ndarray] = None
        self._face_areas: Optional[np.ndarray] = None
        self._face_normals: Optional[np.ndarray] = None
        self._face_centers: Optional[np.ndarray] = None
        
        # Quality metrics
        self._aspect_ratios: Optional[np.ndarray] = None
        self._skewness: Optional[np.ndarray] = None
        self._orthogonality: Optional[np.ndarray] = None
        
        # Metric tensors
        self._metric_tensors: Optional[List[MetricTensor]] = None
        
        self._geometry_computed = False
    
    def _get_tolerance(self, precision: str) -> float:
        """Get numerical tolerance based on precision."""
        if precision == "single":
            return 1e-6
        elif precision == "double":
            return 1e-12
        elif precision == "extended":
            return 1e-15
        else:
            return 1e-12
    
    def compute_all_properties(self) -> None:
        """Compute all geometric properties."""
        logger.info("Computing geometric properties...")
        
        self._compute_cell_properties()
        self._compute_face_properties()
        self._compute_quality_metrics()
        self._compute_metric_tensors()
        
        self._geometry_computed = True
        logger.info("Geometric properties computation complete")
    
    def _compute_cell_properties(self) -> None:
        """Compute volumes and centroids for all cells."""
        n_cells = self.mesh.n_cells
        self._cell_volumes = np.zeros(n_cells)
        self._cell_centroids = np.zeros((n_cells, 3))
        
        cell_idx = 0
        for cell_type, connectivity in self.mesh.cells.items():
            for cell_nodes in connectivity:
                points = self.mesh.points[cell_nodes]
                
                volume, centroid = self._compute_cell_geometry_exact(cell_type, points)
                self._cell_volumes[cell_idx] = volume
                self._cell_centroids[cell_idx] = centroid
                
                cell_idx += 1
    
    def _compute_cell_geometry_exact(self, cell_type, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute exact cell volume and centroid."""
        from .unstructured_mesh import CellType
        
        if cell_type == CellType.TETRAHEDRON:
            return self._tetrahedron_properties(points)
        elif cell_type == CellType.HEXAHEDRON:
            return self._hexahedron_properties(points)
        elif cell_type == CellType.WEDGE:
            return self._wedge_properties(points)
        elif cell_type == CellType.PYRAMID:
            return self._pyramid_properties(points)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
    
    def _tetrahedron_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Exact tetrahedron volume and centroid."""
        # Volume using scalar triple product
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]
        
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        centroid = np.mean(points, axis=0)
        
        return volume, centroid
    
    def _hexahedron_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Hexahedron volume using decomposition into tetrahedra."""
        # Decompose hexahedron into 6 tetrahedra from center
        center = np.mean(points, axis=0)
        
        # Face vertices (assuming standard hexahedron numbering)
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5]   # Right
        ]
        
        total_volume = 0.0
        weighted_centroid = np.zeros(3)
        
        for face_indices in faces:
            face_points = points[face_indices]
            
            # Split face into triangles and form tetrahedra with center
            for i in range(len(face_indices) - 2):
                tet_points = np.array([
                    center,
                    face_points[0],
                    face_points[i + 1], 
                    face_points[i + 2]
                ])
                
                tet_vol, tet_centroid = self._tetrahedron_properties(tet_points)
                total_volume += tet_vol
                weighted_centroid += tet_vol * tet_centroid
        
        if total_volume > self._tolerance:
            centroid = weighted_centroid / total_volume
        else:
            centroid = center
        
        return total_volume, centroid
    
    def _wedge_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Wedge (prism) volume using triangular cross-section."""
        # Wedge has two triangular faces and three quadrilateral faces
        # Decompose into tetrahedra
        
        # Use vertices 0, 1, 2 as one triangular face and 3, 4, 5 as the other
        total_volume = 0.0
        weighted_centroid = np.zeros(3)
        
        # Decompose into 3 tetrahedra
        tetrahedra = [
            [0, 1, 2, 3],
            [1, 2, 3, 4], 
            [2, 3, 4, 5]
        ]
        
        for tet_indices in tetrahedra:
            tet_points = points[tet_indices]
            tet_vol, tet_centroid = self._tetrahedron_properties(tet_points)
            
            total_volume += tet_vol
            weighted_centroid += tet_vol * tet_centroid
        
        if total_volume > self._tolerance:
            centroid = weighted_centroid / total_volume
        else:
            centroid = np.mean(points, axis=0)
        
        return total_volume, centroid
    
    def _pyramid_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Pyramid volume with quadrilateral base."""
        # Base vertices: 0, 1, 2, 3; apex: 4
        base_center = np.mean(points[:4], axis=0)
        apex = points[4]
        
        # Decompose into 4 tetrahedra from base center
        total_volume = 0.0
        weighted_centroid = np.zeros(3)
        
        for i in range(4):
            j = (i + 1) % 4
            tet_points = np.array([
                base_center,
                points[i],
                points[j],
                apex
            ])
            
            tet_vol, tet_centroid = self._tetrahedron_properties(tet_points)
            total_volume += tet_vol
            weighted_centroid += tet_vol * tet_centroid
        
        if total_volume > self._tolerance:
            centroid = weighted_centroid / total_volume
        else:
            centroid = np.mean(points, axis=0)
        
        return total_volume, centroid
    
    def _compute_face_properties(self) -> None:
        """Compute areas, normals, and centers for all faces."""
        n_faces = self.mesh.n_faces
        self._face_areas = np.zeros(n_faces)
        self._face_normals = np.zeros((n_faces, 3))
        self._face_centers = np.zeros((n_faces, 3))
        
        for i, face_nodes in enumerate(self.mesh.faces['connectivity']):
            points = self.mesh.points[face_nodes]
            
            area, normal, center = self._compute_face_geometry_exact(points)
            self._face_areas[i] = area
            self._face_normals[i] = normal
            self._face_centers[i] = center
    
    def _compute_face_geometry_exact(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute exact face area, normal, and center."""
        n_points = len(points)
        
        if n_points == 3:  # Triangle
            return self._triangle_properties(points)
        elif n_points == 4:  # Quadrilateral
            return self._quadrilateral_properties(points)
        else:
            # General polygon - decompose into triangles
            return self._polygon_properties(points)
    
    def _triangle_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Exact triangle properties."""
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        
        if area > self._tolerance:
            normal = cross / (2 * area)
        else:
            normal = np.array([0, 0, 1])  # Default normal
        
        center = np.mean(points, axis=0)
        
        return area, normal, center
    
    def _quadrilateral_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Quadrilateral properties using triangulation."""
        # Split into two triangles: (0,1,2) and (0,2,3)
        area1, normal1, _ = self._triangle_properties(points[[0, 1, 2]])
        area2, normal2, _ = self._triangle_properties(points[[0, 2, 3]])
        
        total_area = area1 + area2
        if total_area > self._tolerance:
            # Area-weighted average normal
            normal = (area1 * normal1 + area2 * normal2) / total_area
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])
        
        center = np.mean(points, axis=0)
        
        return total_area, normal, center
    
    def _polygon_properties(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """General polygon properties."""
        n_points = len(points)
        center = np.mean(points, axis=0)
        
        total_area = 0.0
        weighted_normal = np.zeros(3)
        
        # Fan triangulation from first vertex
        for i in range(1, n_points - 1):
            tri_points = points[[0, i, i + 1]]
            area, normal, _ = self._triangle_properties(tri_points)
            
            total_area += area
            weighted_normal += area * normal
        
        if total_area > self._tolerance:
            normal = weighted_normal / np.linalg.norm(weighted_normal)
        else:
            normal = np.array([0, 0, 1])
        
        return total_area, normal, center
    
    def _compute_quality_metrics(self) -> None:
        """Compute mesh quality metrics."""
        n_cells = self.mesh.n_cells
        self._aspect_ratios = np.zeros(n_cells)
        self._skewness = np.zeros(n_cells)
        self._orthogonality = np.zeros(n_cells)
        
        cell_idx = 0
        for cell_type, connectivity in self.mesh.cells.items():
            for cell_nodes in connectivity:
                points = self.mesh.points[cell_nodes]
                
                self._aspect_ratios[cell_idx] = self._compute_aspect_ratio(points)
                self._skewness[cell_idx] = self._compute_skewness_metric(points)
                self._orthogonality[cell_idx] = self._compute_orthogonality(points)
                
                cell_idx += 1
    
    def _compute_aspect_ratio(self, points: np.ndarray) -> float:
        """Compute aspect ratio as max/min edge length ratio."""
        edge_lengths = []
        n_points = len(points)
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edge_length = np.linalg.norm(points[j] - points[i])
                edge_lengths.append(edge_length)
        
        if len(edge_lengths) > 0:
            return max(edge_lengths) / (min(edge_lengths) + self._tolerance)
        return 1.0
    
    def _compute_skewness_metric(self, points: np.ndarray) -> float:
        """Compute skewness based on centroid displacement."""
        centroid = np.mean(points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in points]
        
        if len(distances) > 1:
            mean_dist = np.mean(distances)
            if mean_dist > self._tolerance:
                return np.std(distances) / mean_dist
        return 0.0
    
    def _compute_orthogonality(self, points: np.ndarray) -> float:
        """Compute orthogonality measure."""
        # Simple implementation based on dot products of edges
        edges = []
        n_points = len(points)
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edge = points[j] - points[i]
                edge_normalized = edge / (np.linalg.norm(edge) + self._tolerance)
                edges.append(edge_normalized)
        
        # Compute orthogonality as deviation from perpendicularity
        orthogonality_sum = 0.0
        count = 0
        
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                dot_product = abs(np.dot(edges[i], edges[j]))
                orthogonality_sum += dot_product
                count += 1
        
        if count > 0:
            return 1.0 - (orthogonality_sum / count)  # 1 = perfect orthogonality
        return 1.0
    
    def _compute_metric_tensors(self) -> None:
        """Compute metric tensors for coordinate transformation quality."""
        self._metric_tensors = []
        
        cell_idx = 0
        for cell_type, connectivity in self.mesh.cells.items():
            for cell_nodes in connectivity:
                points = self.mesh.points[cell_nodes]
                metric_tensor = self._compute_cell_metric_tensor(points)
                self._metric_tensors.append(metric_tensor)
                cell_idx += 1
    
    def _compute_cell_metric_tensor(self, points: np.ndarray) -> MetricTensor:
        """Compute metric tensor for a cell."""
        # Simplified metric tensor computation
        # For a more complete implementation, this would involve
        # coordinate transformation jacobians
        
        centroid = np.mean(points, axis=0)
        
        # Compute characteristic vectors
        vectors = points - centroid
        
        # Covariant metric tensor g_ij = sum(v_i * v_j)
        g_ij = np.zeros((3, 3))
        for v in vectors:
            g_ij += np.outer(v, v)
        
        g_ij /= len(vectors)
        
        # Contravariant metric tensor (inverse)
        try:
            g_contravariant = np.linalg.inv(g_ij)
            jacobian = np.linalg.det(g_ij)**0.5
            condition_number = np.linalg.cond(g_ij)
        except np.linalg.LinAlgError:
            g_contravariant = np.eye(3)
            jacobian = 1.0
            condition_number = 1.0
        
        # Compute quality measures
        eigenvals = np.linalg.eigvals(g_ij)
        aspect_ratio = max(eigenvals) / (min(eigenvals) + self._tolerance)
        
        skewness_angle = self._compute_skewness_angle(vectors)
        orthogonality = self._compute_orthogonality(points)
        
        return MetricTensor(
            g_ij=g_ij,
            g_contravariant=g_contravariant,
            jacobian=jacobian,
            condition_number=condition_number,
            skewness_angle=skewness_angle,
            orthogonality=orthogonality,
            aspect_ratio=aspect_ratio
        )
    
    def _compute_skewness_angle(self, vectors: np.ndarray) -> float:
        """Compute maximum skewness angle in degrees."""
        max_angle = 0.0
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1 = vectors[i] / (np.linalg.norm(vectors[i]) + self._tolerance)
                v2 = vectors[j] / (np.linalg.norm(vectors[j]) + self._tolerance)
                
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(abs(cos_angle))
                skew_angle = min(angle, np.pi/2 - angle)
                
                max_angle = max(max_angle, np.degrees(skew_angle))
        
        return max_angle
    
    # Public interface methods
    def get_cell_volume(self, cell_id: int) -> float:
        """Get volume of a specific cell."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._cell_volumes[cell_id]
    
    def get_cell_centroid(self, cell_id: int) -> np.ndarray:
        """Get centroid of a specific cell."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._cell_centroids[cell_id]
    
    def get_face_area(self, face_id: int) -> float:
        """Get area of a specific face."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._face_areas[face_id]
    
    def get_face_normal(self, face_id: int) -> np.ndarray:
        """Get normal vector of a specific face."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._face_normals[face_id]
    
    def get_face_center(self, face_id: int) -> np.ndarray:
        """Get center of a specific face."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._face_centers[face_id]
    
    def get_cell_metric_tensor(self, cell_id: int) -> MetricTensor:
        """Get metric tensor for a specific cell."""
        if not self._geometry_computed:
            self.compute_all_properties()
        return self._metric_tensors[cell_id]
    
    def compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Compute distance between two points."""
        return np.linalg.norm(point2 - point1)
    
    def compute_quality_summary(self) -> Dict[str, float]:
        """Compute mesh quality summary statistics."""
        if not self._geometry_computed:
            self.compute_all_properties()
        
        return {
            'min_volume': np.min(self._cell_volumes),
            'max_volume': np.max(self._cell_volumes),
            'mean_volume': np.mean(self._cell_volumes),
            'volume_ratio': np.max(self._cell_volumes) / (np.min(self._cell_volumes) + self._tolerance),
            'mean_aspect_ratio': np.mean(self._aspect_ratios),
            'max_aspect_ratio': np.max(self._aspect_ratios),
            'mean_skewness': np.mean(self._skewness),
            'max_skewness': np.max(self._skewness),
            'mean_orthogonality': np.mean(self._orthogonality),
            'min_orthogonality': np.min(self._orthogonality),
            'negative_volumes': np.sum(self._cell_volumes < 0),
            'zero_area_faces': np.sum(self._face_areas < self._tolerance) if self._face_areas is not None else 0
        }