#!/usr/bin/env python3

"""
Debug script to investigate face connectivity parsing issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from openffd.mesh.fluent_reader import FluentMeshReader
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_face_parsing():
    """Debug the face connectivity parsing to understand the issue."""
    
    print("=== DEBUGGING FACE CONNECTIVITY PARSING ===")
    
    # Create a sample face section data like what would be found for wedge_pos
    # Based on the actual data from the mesh file:
    # (13 (b 1551f4 1ff652 3 0)(
    # 4 2786c 27add 4492e 27a5b 6f24b 0 
    
    # Let's manually parse a few faces to understand the issue
    
    # Sample data from wedge_pos zone:
    sample_tokens = [
        "4", "2786c", "27add", "4492e", "27a5b", "6f24b", "0",
        "4", "66256", "665a6", "668f2", "66621", "aba5", "0",
        "4", "4355b", "2861d", "288b1", "2867d", "700c7", "0"
    ]
    
    print("\nSample tokens from wedge_pos face section:")
    for i, token in enumerate(sample_tokens):
        print(f"  [{i:2d}] {token}")
    
    print("\n=== CURRENT PARSER LOGIC ===")
    element_type = 4  # Quadrilateral
    nodes_per_face = 4
    nodes_plus_cells = nodes_per_face + 2  # nodes + left_cell + right_cell
    
    print(f"Element type: {element_type} (quad)")
    print(f"Nodes per face: {nodes_per_face}")
    print(f"Nodes + cells per face: {nodes_plus_cells}")
    
    # Parse using current logic
    faces = []
    i = 0
    face_num = 0
    
    while i < len(sample_tokens) and face_num < 3:
        print(f"\n--- Parsing Face {face_num} ---")
        print(f"Starting at token index {i}")
        
        if i + nodes_per_face - 1 < len(sample_tokens):
            face_nodes = []
            
            print("Raw tokens for this face:")
            for j in range(nodes_plus_cells):
                if i + j < len(sample_tokens):
                    print(f"  [{i+j}] {sample_tokens[i+j]}")
            
            # Convert hex node indices to decimal
            for j in range(nodes_per_face):
                hex_val = sample_tokens[i + j]
                node_idx = int(hex_val, 16) - 1  # Convert to 0-based indexing
                face_nodes.append(node_idx)
                print(f"  Node {j}: {hex_val} (hex) -> {int(hex_val, 16)} (dec) -> {node_idx} (0-based)")
            
            faces.append(face_nodes)
            print(f"  Face connectivity: {face_nodes}")
            
            # Skip past this face's data (nodes + cells)
            i += nodes_plus_cells
        else:
            break
        
        face_num += 1
    
    print(f"\n=== PARSED FACES ===")
    for i, face in enumerate(faces):
        print(f"Face {i}: {face}")
        # Check if nodes are reasonable (should be local, not spanning domain)
        node_range = max(face) - min(face)
        print(f"  Node range: {node_range} (min: {min(face)}, max: {max(face)})")
        if node_range > 100000:
            print(f"  ⚠️  SUSPICIOUS: Large node range suggests faces span domain")
        else:
            print(f"  ✅ OK: Reasonable node range")
    
    print(f"\n=== ANALYSIS ===")
    print("Current parser assumes format: node1 node2 node3 node4 left_cell right_cell")
    print("But looking at the data:")
    print("  4 2786c 27add 4492e 27a5b 6f24b 0")
    print("The first token '4' might be:")
    print("  1. Element type indicator (but we already know it's quad)")
    print("  2. Number of nodes (which is 4 for quad)")
    print("  3. Something else entirely")
    
    print("\n=== ALTERNATIVE PARSING APPROACHES ===")
    
    # Alternative 1: Skip the first token if it's the node count
    print("\nAlternative 1: Skip first token (assume it's node count)")
    sample_tokens_alt1 = sample_tokens[1:]  # Skip first '4'
    
    faces_alt1 = []
    i = 0
    face_num = 0
    
    while i < len(sample_tokens_alt1) and face_num < 3:
        if i + nodes_per_face - 1 < len(sample_tokens_alt1):
            face_nodes = []
            for j in range(nodes_per_face):
                hex_val = sample_tokens_alt1[i + j]
                node_idx = int(hex_val, 16) - 1
                face_nodes.append(node_idx)
            faces_alt1.append(face_nodes)
            i += nodes_plus_cells
        else:
            break
        face_num += 1
    
    print("Alternative 1 results:")
    for i, face in enumerate(faces_alt1):
        print(f"  Face {i}: {face}")
        node_range = max(face) - min(face)
        print(f"    Node range: {node_range}")
    
    # Alternative 2: Different format interpretation
    print("\nAlternative 2: Format is 'count node1 node2 node3 node4 left_cell right_cell'")
    i = 0
    faces_alt2 = []
    
    while i < len(sample_tokens):
        if i + 6 < len(sample_tokens):  # count + 4 nodes + 2 cells
            count = int(sample_tokens[i])
            if count == 4:  # Verify it's a quad
                face_nodes = []
                for j in range(1, 5):  # tokens 1-4 are nodes
                    hex_val = sample_tokens[i + j]
                    node_idx = int(hex_val, 16) - 1
                    face_nodes.append(node_idx)
                faces_alt2.append(face_nodes)
                i += 7  # count + 4 nodes + 2 cells
            else:
                i += 1
        else:
            break
    
    print("Alternative 2 results:")
    for i, face in enumerate(faces_alt2):
        print(f"  Face {i}: {face}")
        node_range = max(face) - min(face)
        print(f"    Node range: {node_range}")
    
    return faces, faces_alt1, faces_alt2

if __name__ == "__main__":
    debug_face_parsing()