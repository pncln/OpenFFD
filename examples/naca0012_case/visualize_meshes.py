#!/usr/bin/env python3
"""
Mesh Visualization Script for Optimization Analysis

This script helps you visualize and compare meshes from different optimization iterations.
You can load the meshes in ParaView or other visualization tools.

Usage:
    python3 visualize_meshes.py
"""

import os
from pathlib import Path

def list_available_meshes():
    """List all available mesh directories."""
    current_dir = Path.cwd()
    
    print("="*80)
    print("MESH VISUALIZATION GUIDE")
    print("="*80)
    
    # List optimization iteration meshes
    opt_mesh_dir = current_dir / "optimization_meshes"
    if opt_mesh_dir.exists():
        print("\n📁 OPTIMIZATION ITERATION MESHES:")
        print("   (These show how the mesh evolves during optimization)")
        
        iterations = sorted([d for d in opt_mesh_dir.iterdir() if d.is_dir()])
        for iteration_dir in iterations:
            design_vars_file = iteration_dir / "design_variables.txt"
            mesh_dir = iteration_dir / "polyMesh"
            
            if design_vars_file.exists():
                with open(design_vars_file, 'r') as f:
                    content = f.read()
                print(f"\n   📂 {iteration_dir.name}/")
                print(f"      Mesh: {mesh_dir}")
                for line in content.strip().split('\n'):
                    print(f"      {line}")
            else:
                print(f"\n   📂 {iteration_dir.name}/")
                print(f"      Mesh: {mesh_dir}")
    
    # List gradient computation meshes  
    grad_mesh_dir = current_dir / "gradient_meshes"
    if grad_mesh_dir.exists():
        print("\n📁 GRADIENT COMPUTATION MESHES:")
        print("   (These show perturbed meshes used for finite difference gradients)")
        
        components = sorted([d for d in grad_mesh_dir.iterdir() if d.is_dir()])
        for comp_dir in components:
            design_vars_file = comp_dir / "design_variables.txt"
            mesh_dir = comp_dir / "polyMesh"
            
            if design_vars_file.exists():
                with open(design_vars_file, 'r') as f:
                    content = f.read()
                print(f"\n   📂 {comp_dir.name}/")
                print(f"      Mesh: {mesh_dir}")
                for line in content.strip().split('\n'):
                    print(f"      {line}")
    
    # Original mesh
    original_mesh = current_dir / "constant" / "polyMesh"
    if original_mesh.exists():
        print(f"\n📁 CURRENT/ORIGINAL MESH:")
        print(f"   📂 constant/polyMesh/")
        print(f"      Mesh: {original_mesh}")
        print(f"      This is the current mesh state")

def generate_paraview_script():
    """Generate a ParaView Python script to load all meshes."""
    current_dir = Path.cwd()
    script_path = current_dir / "load_meshes_paraview.py"
    
    script_content = '''"""
ParaView Python script to load optimization meshes
Run this in ParaView's Python Shell or as a macro
"""

import paraview.simple as pv

# Clear any existing data
pv.Delete(pv.GetActiveSource())

sources = []

'''
    
    # Add optimization meshes
    opt_mesh_dir = current_dir / "optimization_meshes"
    if opt_mesh_dir.exists():
        iterations = sorted([d for d in opt_mesh_dir.iterdir() if d.is_dir()])
        for i, iteration_dir in enumerate(iterations):
            mesh_dir = iteration_dir / "polyMesh"
            if mesh_dir.exists():
                script_content += f'''
# Load {iteration_dir.name}
reader_{i} = pv.OpenFOAMReader(FileName=str(r"{mesh_dir.parent}"))
reader_{i}.MeshRegions = ['internalMesh']
reader_{i}.CellArrays = []
reader_{i}.PointArrays = []
sources.append(reader_{i})
pv.Show(reader_{i})
pv.ColorBy(reader_{i}, None)  # No coloring, just geometry
'''

    script_content += '''
# Arrange views
pv.ResetCamera()
pv.Render()

print("Loaded all optimization meshes!")
print("You can now:")
print("1. Toggle visibility of different iterations in the Pipeline Browser")
print("2. Apply filters like 'Extract Surface' to see surface meshes")
print("3. Use 'Plot Over Line' to compare mesh density")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n📄 PARAVIEW SCRIPT GENERATED:")
    print(f"   File: {script_path}")
    print(f"   Usage: Open ParaView, go to Tools > Python Shell, and run:")
    print(f"          exec(open('{script_path}').read())")

def main():
    """Main function."""
    list_available_meshes()
    
    print("\n" + "="*80)
    print("VISUALIZATION OPTIONS:")
    print("="*80)
    
    print("\n🔍 OPTION 1: ParaView (Recommended)")
    print("   1. Open ParaView")
    print("   2. File > Open > Navigate to any polyMesh directory above")
    print("   3. Select the parent directory (e.g., iteration_001)")
    print("   4. Choose 'OpenFOAM Reader'")
    print("   5. In Properties panel:")
    print("      - Mesh Regions: Check 'internalMesh'")
    print("      - Cell Arrays: Uncheck all (we just want geometry)")
    print("      - Click Apply")
    print("   6. Add 'Extract Surface' filter to see the surface mesh")
    
    print("\n🔍 OPTION 2: Command Line Tools")
    print("   - checkMesh: OpenFOAM utility to check mesh quality")
    print("   - postProcess: OpenFOAM utility for post-processing")
    
    print("\n🔍 OPTION 3: Python/VTK")
    print("   - Use PyFoam or OpenFOAM Python utilities")
    print("   - Load with VTK/PyVista for custom analysis")
    
    generate_paraview_script()
    
    print("\n" + "="*80)
    print("MESH COMPARISON TIPS:")
    print("="*80)
    print("📊 To see mesh evolution:")
    print("   1. Load iteration_001, iteration_002, iteration_003 in ParaView")
    print("   2. Apply 'Extract Surface' filter to each")
    print("   3. Toggle visibility to see differences")
    print("   4. Use 'Calculator' filter to compute mesh displacement")
    
    print("\n📊 To analyze gradient effects:")
    print("   1. Load baseline mesh (current) and gradient_component meshes")
    print("   2. Compare how each design variable perturbation affects geometry")
    print("   3. Look for regions of high deformation sensitivity")
    
    print("\n📊 Mesh quality analysis:")
    print("   1. Run 'checkMesh' in each iteration directory")
    print("   2. Compare aspect ratios, skewness, orthogonality")
    print("   3. Ensure optimization doesn't degrade mesh quality")

if __name__ == "__main__":
    main()