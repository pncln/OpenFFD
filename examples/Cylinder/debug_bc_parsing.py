#!/usr/bin/env python3
"""Debug BC parsing issues."""

import sys
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path.cwd().parent.parent / 'src'))

def debug_parsing():
    """Debug the boundary condition parsing."""
    
    with open('0_orig/U', 'r') as f:
        content = f.read()

    print('=== MANUAL BOUNDARY PARSING TEST ===')
    boundary_start = content.find('boundaryField')
    print(f'boundaryField found at position: {boundary_start}')

    if boundary_start != -1:
        brace_start = content.find('{', boundary_start)
        print(f'Opening brace at position: {brace_start}')
        
        # Find matching closing brace
        brace_count = 0
        brace_end = brace_start
        for i, char in enumerate(content[brace_start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = brace_start + i
                    break
                    
        print(f'Closing brace at position: {brace_end}')
        boundary_content = content[brace_start+1:brace_end]
        print(f'Boundary content length: {len(boundary_content)}')
        print('Boundary content:')
        print(repr(boundary_content))
        
        # Test line parsing
        lines = boundary_content.split('\n')
        print(f'Number of lines: {len(lines)}')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                print(f'Line {i}: {repr(stripped)}')
                # Check if this looks like a patch name
                if '{' in stripped and not stripped.startswith(('type', 'value', 'gradient')):
                    patch_name = stripped.split('{')[0].strip()
                    print(f'  -> Potential patch name: {repr(patch_name)}')
                    print(f'  -> Is alphanumeric: {patch_name.isalnum()}')

if __name__ == "__main__":
    debug_parsing()