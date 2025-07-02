#!/usr/bin/env python3

"""
Test the fixed element type format
"""

def test_fixed_format():
    """Test parsing with fixed element type 4 (quad)."""
    
    print("=== TESTING FIXED ELEMENT TYPE FORMAT ===")
    
    # Section 5: (13 (5 153adc 153e66 3 4)(
    # Element type = 4 (quad), so format should be:
    # node1 node2 node3 node4 left_cell right_cell
    
    sample_lines = [
        "8c1 aad9b aadd1 8c2 4a9cf 0",
        "8c7 aadd6 aad9a 8b6 4a9c9 0",
        "8c8 aad9c aadd7 8c9 4ab43 0"
    ]
    
    print("Sample lines from section 5 (fixed element type 4):")
    for line in sample_lines:
        print(f"  {line}")
    
    print("\nParsing as: node1 node2 node3 node4 left_cell right_cell")
    
    for i, line in enumerate(sample_lines):
        tokens = line.split()
        
        nodes = []
        for j in range(4):  # 4 nodes for quad
            hex_val = tokens[j]
            node_idx = int(hex_val, 16) - 1  # Convert to 0-based
            nodes.append(node_idx)
        
        left_cell = int(tokens[4], 16)
        right_cell = int(tokens[5], 16)
        
        node_range = max(nodes) - min(nodes)
        
        print(f"\nFace {i}:")
        print(f"  Nodes: {nodes}")
        print(f"  Left cell: {left_cell}, Right cell: {right_cell}")
        print(f"  Node range: {node_range}")
        
        if node_range < 1000:
            print(f"  ✅ Good: Local connectivity")
        else:
            print(f"  ⚠️  Large range: Still suspicious")

if __name__ == "__main__":
    test_fixed_format()