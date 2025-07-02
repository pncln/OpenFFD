#!/usr/bin/env python3

"""
Debug script to understand the actual Fluent face format
"""

def analyze_fluent_format():
    """Analyze the Fluent face format based on documentation and actual data."""
    
    print("=== FLUENT FACE FORMAT ANALYSIS ===")
    
    # According to Fluent documentation and the file header:
    # faces: (13 (id start end type elemType)
    #         (v-0 v-1 .. v-n right-cell left-cell ...))
    
    print("\nFluent documentation states:")
    print("faces: (13 (id start end type elemType)")
    print("       (v-0 v-1 .. v-n right-cell left-cell ...))")
    
    print("\nActual data from wedge_pos (zone b):")
    print("Header: (13 (b 1551f4 1ff652 3 0)(")
    print("  - Zone ID: b (hex) = 11 (decimal)")
    print("  - Start: 1551f4 (hex) = 1397236 (decimal)")
    print("  - End: 1ff652 (hex) = 2087506 (decimal)")
    print("  - Type: 3 (boundary)")
    print("  - Element type: 0 (mixed)")
    
    print("\nData lines:")
    print("4 2786c 27add 4492e 27a5b 6f24b 0")
    print("4 66256 665a6 668f2 66621 aba5 0")
    
    print("\nInterpretation:")
    print("Since element type is 0 (mixed), each face line starts with:")
    print("  - Element type for this specific face")
    print("  - Followed by the nodes")
    print("  - Followed by left/right cells")
    
    print("\nSo format is:")
    print("element_type node1 node2 ... nodeN left_cell right_cell")
    
    print("\nFor a quad (element_type=4):")
    print("4 node1 node2 node3 node4 left_cell right_cell")
    
    # Let's test this interpretation
    sample_line = "4 2786c 27add 4492e 27a5b 6f24b 0"
    tokens = sample_line.split()
    
    print(f"\nTesting with line: {sample_line}")
    print(f"Tokens: {tokens}")
    
    element_type = int(tokens[0])
    print(f"Element type: {element_type}")
    
    if element_type == 4:  # Quad
        nodes = []
        for i in range(1, 5):
            hex_val = tokens[i]
            node_idx = int(hex_val, 16) - 1  # Convert to 0-based
            nodes.append(node_idx)
            print(f"  Node {i}: {hex_val} (hex) -> {int(hex_val, 16)} (1-based) -> {node_idx} (0-based)")
        
        left_cell = int(tokens[5], 16)
        right_cell = int(tokens[6], 16)
        
        print(f"Face nodes: {nodes}")
        print(f"Left cell: {left_cell}")
        print(f"Right cell: {right_cell}")
        
        node_range = max(nodes) - min(nodes)
        print(f"Node range: {node_range}")
        
        if node_range < 10000:
            print("✅ Reasonable node range - faces are local")
        else:
            print("⚠️  Large node range - still suspicious")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. The parser should handle mixed element types in face sections")
    print("2. Each face line should be parsed as:")
    print("   element_type node1 node2 ... nodeN left_cell right_cell")
    print("3. The number of nodes depends on the element_type in each line")
    print("4. Element type 4 = quad (4 nodes)")
    print("5. Element type 3 = triangle (3 nodes)")
    print("6. Element type 2 = line (2 nodes)")

if __name__ == "__main__":
    analyze_fluent_format()