#!/usr/bin/env python3
"""
Comprehensive OpenFOAM Mesh Connectivity Fix

This module provides improved cell connectivity extraction from OpenFOAM mesh data
that properly handles both structured and unstructured meshes for VTK export.

The key insight is that OpenFOAM cells are defined by faces, but VTK cells need
direct vertex connectivity. We need to reconstruct the 3D cell topology from
the face-cell relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenFOAMCellExtractor:
    """Extract proper 3D cell connectivity from OpenFOAM face-based mesh format."""
    
    def __init__(self):
        """Initialize cell extractor."""
        self.debug = False
        
    def extract_cells_with_proper_connectivity(self, points: np.ndarray,
                                               faces: List[List[int]], 
                                               owner: np.ndarray,
                                               neighbour: np.ndarray,
                                               n_cells: int,
                                               n_internal_faces: int) -> Dict[str, Any]:
        """
        Extract proper 3D cell connectivity from OpenFOAM face data.
        
        Args:
            points: Vertex coordinates [n_points, 3]
            faces: Face-vertex connectivity List[List[int]]
            owner: Owner cell for each face [n_faces]
            neighbour: Neighbour cell for internal faces [n_internal_faces]
            n_cells: Number of cells
            n_internal_faces: Number of internal faces
            
        Returns:
            Dictionary with proper cell connectivity data
        """
        logger.info(f"Extracting proper cell connectivity: {n_cells} cells, {len(faces)} faces")
        
        # Step 1: Build cell-face connectivity
        cells_faces = [[] for _ in range(n_cells)]
        
        # Add owner relationships
        for face_idx, owner_cell in enumerate(owner):
            if 0 <= owner_cell < n_cells:
                cells_faces[owner_cell].append(face_idx)
        
        # Add neighbour relationships for internal faces
        for face_idx, neighbour_cell in enumerate(neighbour):
            if 0 <= neighbour_cell < n_cells:
                cells_faces[neighbour_cell].append(face_idx)
        
        # Step 2: Determine mesh type based on face patterns
        mesh_type = self._detect_mesh_type(faces, cells_faces)
        logger.info(f"Detected mesh type: {mesh_type}")
        
        # Step 3: Extract proper cell topology
        if mesh_type == "structured_hex":
            cells_data = self._extract_structured_hex_cells(points, faces, cells_faces)
        elif mesh_type == "unstructured_tet":
            cells_data = self._extract_unstructured_tet_cells(points, faces, cells_faces)
        else:  # mixed or unknown
            cells_data = self._extract_mixed_cells(points, faces, cells_faces)
        
        # Step 4: Create VTK-compatible cell definitions
        vtk_cells, vtk_cell_types = self._create_vtk_cells(cells_data)
        
        return {
            'cells_vertices': cells_data['cells_vertices'],
            'cells_faces': cells_faces,
            'vtk_cells': vtk_cells,
            'vtk_cell_types': vtk_cell_types,
            'mesh_type': mesh_type,
            'cell_statistics': {
                'n_cells': len(cells_data['cells_vertices']),
                'n_valid_cells': cells_data['n_valid_cells'],
                'cell_type_counts': cells_data['cell_type_counts']
            }
        }
    
    def _detect_mesh_type(self, faces: List[List[int]], cells_faces: List[List[int]]) -> str:
        """Detect mesh type based on face and cell patterns."""
        if not faces or not cells_faces:
            return "unknown"
        
        # Analyze face types
        face_type_counts = {"triangle": 0, "quad": 0, "other": 0}
        for face in faces:
            if len(face) == 3:
                face_type_counts["triangle"] += 1
            elif len(face) == 4:
                face_type_counts["quad"] += 1
            else:
                face_type_counts["other"] += 1
        
        total_faces = len(faces)
        quad_ratio = face_type_counts["quad"] / total_faces if total_faces > 0 else 0
        tri_ratio = face_type_counts["triangle"] / total_faces if total_faces > 0 else 0
        
        # Analyze cell face counts
        cell_face_counts = {}
        for cell_faces in cells_faces:
            count = len(cell_faces)
            cell_face_counts[count] = cell_face_counts.get(count, 0) + 1
        
        logger.info(f"Face analysis: {quad_ratio:.1%} quads, {tri_ratio:.1%} triangles")
        logger.info(f"Cell face counts: {cell_face_counts}")
        
        # Classification logic
        if quad_ratio > 0.8 and 6 in cell_face_counts:  # Mostly quads, 6 faces per cell
            return "structured_hex"
        elif tri_ratio > 0.8 and 4 in cell_face_counts:  # Mostly triangles, 4 faces per cell
            return "unstructured_tet"
        else:
            return "mixed"
    
    def _extract_structured_hex_cells(self, points: np.ndarray, 
                                      faces: List[List[int]], 
                                      cells_faces: List[List[int]]) -> Dict[str, Any]:
        """Extract hexahedral cells from structured mesh."""
        logger.info("Extracting structured hexahedral cells...")
        
        cells_vertices = []
        cell_type_counts = {"hexahedron": 0, "degenerate": 0}
        n_valid_cells = 0
        
        for cell_idx, face_indices in enumerate(cells_faces):
            if len(face_indices) == 6:  # Hexahedron should have 6 faces
                hex_vertices = self._build_hexahedron_from_faces(faces, face_indices, points)
                if len(hex_vertices) == 8:
                    cells_vertices.append(hex_vertices)
                    cell_type_counts["hexahedron"] += 1
                    n_valid_cells += 1
                else:
                    # Degenerate hexahedron, create tetrahedra
                    tet_vertices = self._create_tetrahedra_from_vertices(hex_vertices)
                    cells_vertices.extend(tet_vertices)
                    cell_type_counts["degenerate"] += len(tet_vertices)
                    n_valid_cells += len(tet_vertices)
            else:
                # Non-hex cell, create tetrahedra
                cell_vertices = self._get_cell_vertices_from_faces(faces, face_indices)
                tet_vertices = self._create_tetrahedra_from_vertices(cell_vertices)
                cells_vertices.extend(tet_vertices)
                cell_type_counts["degenerate"] += len(tet_vertices)
                n_valid_cells += len(tet_vertices)
        
        return {
            'cells_vertices': cells_vertices,
            'cell_type_counts': cell_type_counts,
            'n_valid_cells': n_valid_cells
        }
    
    def _extract_unstructured_tet_cells(self, points: np.ndarray,
                                        faces: List[List[int]],
                                        cells_faces: List[List[int]]) -> Dict[str, Any]:
        """Extract tetrahedral cells from unstructured mesh."""
        logger.info("Extracting unstructured tetrahedral cells...")
        
        cells_vertices = []
        cell_type_counts = {"tetrahedron": 0, "other": 0}
        n_valid_cells = 0
        
        for cell_idx, face_indices in enumerate(cells_faces):
            if len(face_indices) == 4:  # Tetrahedron should have 4 faces
                tet_vertices = self._build_tetrahedron_from_faces(faces, face_indices)
                if len(tet_vertices) == 4:
                    cells_vertices.append(tet_vertices)
                    cell_type_counts["tetrahedron"] += 1
                    n_valid_cells += 1
                else:
                    # Invalid tetrahedron, skip or create fallback
                    cell_type_counts["other"] += 1
            else:
                # Non-tet cell, create tetrahedra
                cell_vertices = self._get_cell_vertices_from_faces(faces, face_indices)
                tet_vertices = self._create_tetrahedra_from_vertices(cell_vertices)
                cells_vertices.extend(tet_vertices)
                cell_type_counts["other"] += len(tet_vertices)
                n_valid_cells += len(tet_vertices)
        
        return {
            'cells_vertices': cells_vertices,
            'cell_type_counts': cell_type_counts,
            'n_valid_cells': n_valid_cells
        }
    
    def _extract_mixed_cells(self, points: np.ndarray,
                             faces: List[List[int]],
                             cells_faces: List[List[int]]) -> Dict[str, Any]:
        """Extract cells from mixed mesh by creating tetrahedra."""
        logger.info("Extracting mixed mesh cells (tetrahedralization)...")
        
        cells_vertices = []
        cell_type_counts = {"tetrahedron": 0}
        n_valid_cells = 0
        
        for cell_idx, face_indices in enumerate(cells_faces):
            # Get all vertices for this cell
            cell_vertices = self._get_cell_vertices_from_faces(faces, face_indices)
            
            # Create tetrahedra from cell vertices
            tet_vertices = self._create_tetrahedra_from_vertices(cell_vertices)
            cells_vertices.extend(tet_vertices)
            cell_type_counts["tetrahedron"] += len(tet_vertices)
            n_valid_cells += len(tet_vertices)
        
        return {
            'cells_vertices': cells_vertices,
            'cell_type_counts': cell_type_counts,
            'n_valid_cells': n_valid_cells
        }
    
    def _build_hexahedron_from_faces(self, faces: List[List[int]], 
                                     face_indices: List[int],
                                     points: np.ndarray) -> List[int]:
        """Build hexahedron vertex list from 6 faces with proper VTK ordering."""
        if len(face_indices) != 6:
            return []
        
        # Collect all vertices from faces
        all_vertices = set()
        face_vertices = []
        
        for face_idx in face_indices:
            if face_idx < len(faces):
                face = faces[face_idx]
                face_vertices.append(face)
                all_vertices.update(face)
        
        if len(all_vertices) != 8:
            return []  # Not a proper hexahedron
        
        # Convert to list for proper VTK hexahedron ordering
        vertex_list = list(all_vertices)
        
        # Implement proper VTK hexahedron vertex ordering
        return self._order_hexahedron_vertices_for_vtk(vertex_list, points, faces, face_indices)
    
    def _build_tetrahedron_from_faces(self, faces: List[List[int]], 
                                      face_indices: List[int]) -> List[int]:
        """Build tetrahedron vertex list from 4 faces."""
        if len(face_indices) != 4:
            return []
        
        # Collect all vertices from faces
        all_vertices = set()
        
        for face_idx in face_indices:
            if face_idx < len(faces):
                face = faces[face_idx]
                all_vertices.update(face)
        
        if len(all_vertices) != 4:
            return []  # Not a proper tetrahedron
        
        # Return sorted vertices
        return sorted(list(all_vertices))
    
    def _get_cell_vertices_from_faces(self, faces: List[List[int]], 
                                      face_indices: List[int]) -> List[int]:
        """Get all unique vertices from a cell's faces."""
        all_vertices = set()
        
        for face_idx in face_indices:
            if face_idx < len(faces):
                face = faces[face_idx]
                all_vertices.update(face)
        
        return sorted(list(all_vertices))
    
    def _create_tetrahedra_from_vertices(self, vertices: List[int]) -> List[List[int]]:
        """Create tetrahedra from a set of vertices using simple tetrahedralization."""
        if len(vertices) < 4:
            return []
        
        if len(vertices) == 4:
            return [vertices]
        
        # For more than 4 vertices, create multiple tetrahedra
        # Simple fan triangulation from first vertex
        tetrahedra = []
        base_vertex = vertices[0]
        
        # Create tetrahedra using combinations of remaining vertices
        for i in range(1, len(vertices) - 2):
            for j in range(i + 1, len(vertices) - 1):
                for k in range(j + 1, len(vertices)):
                    tet = [base_vertex, vertices[i], vertices[j], vertices[k]]
                    tetrahedra.append(tet)
        
        # Limit to avoid explosion
        return tetrahedra[:10]  # Max 10 tetrahedra per cell
    
    def _create_vtk_cells(self, cells_data: Dict[str, Any]) -> Tuple[List[List[int]], List[int]]:
        """Create VTK-compatible cell definitions."""
        vtk_cells = []
        vtk_cell_types = []
        
        # VTK cell type constants
        VTK_TETRA = 10
        VTK_HEXAHEDRON = 12
        
        for cell_vertices in cells_data['cells_vertices']:
            if len(cell_vertices) == 4:
                vtk_cells.append(cell_vertices)
                vtk_cell_types.append(VTK_TETRA)
            elif len(cell_vertices) == 8:
                vtk_cells.append(cell_vertices)
                vtk_cell_types.append(VTK_HEXAHEDRON)
            else:
                # Convert to tetrahedron by taking first 4 vertices
                if len(cell_vertices) >= 4:
                    vtk_cells.append(cell_vertices[:4])
                    vtk_cell_types.append(VTK_TETRA)
        
        return vtk_cells, vtk_cell_types
    
    def _order_hexahedron_vertices_for_vtk(self, vertices: List[int], 
                                          points: np.ndarray, 
                                          faces: List[List[int]], 
                                          face_indices: List[int]) -> List[int]:
        """
        Order hexahedron vertices according to VTK convention.
        
        VTK hexahedron vertex ordering:
        - Vertices 0-3: bottom face (counterclockwise when viewed from outside)
        - Vertices 4-7: top face (counterclockwise when viewed from outside)
        - Bottom and top faces should be parallel
        """
        if len(vertices) != 8:
            return sorted(vertices)  # Fallback
        
        # Get vertex coordinates
        vertex_coords = {v: points[v] for v in vertices}
        
        # Find bottom and top faces by analyzing z-coordinates
        z_coords = [vertex_coords[v][2] for v in vertices]
        z_min, z_max = min(z_coords), max(z_coords)
        z_mid = (z_min + z_max) / 2
        
        # Separate vertices into bottom and top
        bottom_vertices = [v for v in vertices if vertex_coords[v][2] < z_mid]
        top_vertices = [v for v in vertices if vertex_coords[v][2] >= z_mid]
        
        # Handle 2D case (z_min ≈ z_max) by using y-coordinates instead
        if abs(z_max - z_min) < 1e-10:
            y_coords = [vertex_coords[v][1] for v in vertices]
            y_min, y_max = min(y_coords), max(y_coords)
            y_mid = (y_min + y_max) / 2
            
            bottom_vertices = [v for v in vertices if vertex_coords[v][1] < y_mid]
            top_vertices = [v for v in vertices if vertex_coords[v][1] >= y_mid]
        
        # Ensure we have 4 vertices in each group
        if len(bottom_vertices) != 4 or len(top_vertices) != 4:
            return sorted(vertices)  # Fallback
        
        # Order vertices within each face counterclockwise
        bottom_ordered = self._order_face_vertices_ccw(bottom_vertices, vertex_coords)
        top_ordered = self._order_face_vertices_ccw(top_vertices, vertex_coords)
        
        # Return in VTK hexahedron order: bottom face (0-3), top face (4-7)
        return bottom_ordered + top_ordered
    
    def _order_face_vertices_ccw(self, face_vertices: List[int], vertex_coords: Dict[int, np.ndarray]) -> List[int]:
        """Order face vertices counterclockwise when viewed from outside."""
        if len(face_vertices) != 4:
            return sorted(face_vertices)
        
        # Find the centroid
        coords = [vertex_coords[v] for v in face_vertices]
        centroid = np.mean(coords, axis=0)
        
        # Calculate angles from centroid to each vertex
        angles = []
        for v in face_vertices:
            vec = vertex_coords[v][:2] - centroid[:2]  # Use x-y plane
            angle = np.arctan2(vec[1], vec[0])
            angles.append((angle, v))
        
        # Sort by angle to get counterclockwise order
        angles.sort()
        return [v for _, v in angles]


def test_cell_extractor():
    """Test the cell extractor with simple examples."""
    print("Testing OpenFOAM Cell Extractor...")
    
    # Simple cube mesh
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])
    
    # Define cube faces (6 faces, each with 4 vertices)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 7, 6, 5],  # top
        [0, 4, 5, 1],  # front
        [2, 6, 7, 3],  # back
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2]   # right
    ]
    
    owner = np.array([0, 0, 0, 0, 0, 0])  # All faces belong to cell 0
    neighbour = np.array([])  # No internal faces
    
    extractor = OpenFOAMCellExtractor()
    result = extractor.extract_cells_with_proper_connectivity(
        points, faces, owner, neighbour, n_cells=1, n_internal_faces=0
    )
    
    print(f"Mesh type: {result['mesh_type']}")
    print(f"Cells extracted: {len(result['cells_vertices'])}")
    print(f"VTK cells: {len(result['vtk_cells'])}")
    print(f"Cell statistics: {result['cell_statistics']}")
    
    return result


if __name__ == "__main__":
    test_cell_extractor()