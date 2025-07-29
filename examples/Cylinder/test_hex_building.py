#!/usr/bin/env python3
"""Test hexahedron building for cylinder mesh."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))

from src.openffd.cfd.openfoam_mesh_reader import OpenFOAMPolyMeshReader
from mesh_connectivity_fix import OpenFOAMCellExtractor
import numpy as np

def test_hex_building():
    reader = OpenFOAMPolyMeshReader()
    mesh_data = reader.read_polymesh('polyMesh')

    # Build cell-face connectivity
    cells_faces = [[] for _ in range(mesh_data.n_cells)]
    for face_idx, owner_cell in enumerate(mesh_data.owner):
        if owner_cell < mesh_data.n_cells:
            cells_faces[owner_cell].append(face_idx)
    for face_idx, neighbour_cell in enumerate(mesh_data.neighbour):
        if neighbour_cell < mesh_data.n_cells:
            cells_faces[neighbour_cell].append(face_idx)

    # Test hexahedron building for first few cells
    extractor = OpenFOAMCellExtractor()

    print('Testing hexahedron building for sample cells:')
    problem_found = False
    
    for cell_idx in range(min(10, len(cells_faces))):
        face_indices = cells_faces[cell_idx]
        
        # Get all vertices from this cell's faces
        all_vertices = set()
        for face_idx in face_indices:
            if face_idx < len(mesh_data.faces):
                face = mesh_data.faces[face_idx]
                all_vertices.update(face)
        
        hex_vertices = extractor._build_hexahedron_from_faces(mesh_data.faces, face_indices, mesh_data.points)
        
        print(f'  Cell {cell_idx}: {len(face_indices)} faces, {len(all_vertices)} unique vertices, hex_result: {len(hex_vertices)} vertices')
        
        if len(all_vertices) != 8:
            print(f'    PROBLEM: Cell {cell_idx} has {len(all_vertices)} vertices instead of 8!')
            print(f'    Face indices: {face_indices}')
            for i, face_idx in enumerate(face_indices):
                if face_idx < len(mesh_data.faces):
                    face = mesh_data.faces[face_idx]
                    print(f'      Face {i} (idx {face_idx}): {len(face)} vertices: {face}')
            problem_found = True
            break
    
    if not problem_found:
        print("All tested cells appear to be proper hexahedra with 8 vertices each.")
        
        # Test the full mesh connectivity fix
        print("\nRunning full mesh connectivity extraction...")
        result = extractor.extract_cells_with_proper_connectivity(
            mesh_data.points, mesh_data.faces, mesh_data.owner, mesh_data.neighbour, 
            mesh_data.n_cells, mesh_data.n_internal_faces
        )
        
        print(f"Mesh type: {result['mesh_type']}")
        print(f"Cell statistics: {result['cell_statistics']}")
        
        # Test vertex ordering for a few sample cells
        print("\nTesting VTK vertex ordering for sample cells:")
        vtk_cells = result['vtk_cells']
        for i in range(min(3, len(vtk_cells))):
            cell_vertices = vtk_cells[i]
            print(f"  Cell {i}: VTK vertices = {cell_vertices}")
            
            # Check if vertices are properly ordered by examining geometric properties
            if len(cell_vertices) == 8:
                coords = mesh_data.points[cell_vertices]
                print(f"    Bottom face (0-3): z = {[coords[j][2] for j in range(4)]}")
                print(f"    Top face (4-7): z = {[coords[j][2] for j in range(4, 8)]}")
                
                # Check if bottom face z-coords are consistent
                bottom_z = [coords[j][2] for j in range(4)]
                top_z = [coords[j][2] for j in range(4, 8)]
                
                bottom_consistent = max(bottom_z) - min(bottom_z) < 1e-6
                top_consistent = max(top_z) - min(top_z) < 1e-6
                proper_ordering = all(bz <= tz for bz in bottom_z for tz in top_z)
                
                print(f"    Bottom face z consistent: {bottom_consistent}")
                print(f"    Top face z consistent: {top_consistent}")
                print(f"    Proper z ordering: {proper_ordering}")

if __name__ == "__main__":
    test_hex_building()