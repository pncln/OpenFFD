"""
ParaView Python script to load optimization meshes
Run this in ParaView's Python Shell or as a macro
"""

import paraview.simple as pv

# Clear any existing data
pv.Delete(pv.GetActiveSource())

sources = []


# Load iteration_001
reader_0 = pv.OpenFOAMReader(FileName=str(r"/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/naca0012_case/optimization_meshes/iteration_001"))
reader_0.MeshRegions = ['internalMesh']
reader_0.CellArrays = []
reader_0.PointArrays = []
sources.append(reader_0)
pv.Show(reader_0)
pv.ColorBy(reader_0, None)  # No coloring, just geometry

# Load iteration_002
reader_1 = pv.OpenFOAMReader(FileName=str(r"/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/naca0012_case/optimization_meshes/iteration_002"))
reader_1.MeshRegions = ['internalMesh']
reader_1.CellArrays = []
reader_1.PointArrays = []
sources.append(reader_1)
pv.Show(reader_1)
pv.ColorBy(reader_1, None)  # No coloring, just geometry

# Load iteration_003
reader_2 = pv.OpenFOAMReader(FileName=str(r"/Users/pncln/Documents/tubitak/verynew/ffd_gen/examples/naca0012_case/optimization_meshes/iteration_003"))
reader_2.MeshRegions = ['internalMesh']
reader_2.CellArrays = []
reader_2.PointArrays = []
sources.append(reader_2)
pv.Show(reader_2)
pv.ColorBy(reader_2, None)  # No coloring, just geometry

# Arrange views
pv.ResetCamera()
pv.Render()

print("Loaded all optimization meshes!")
print("You can now:")
print("1. Toggle visibility of different iterations in the Pipeline Browser")
print("2. Apply filters like 'Extract Surface' to see surface meshes")
print("3. Use 'Plot Over Line' to compare mesh density")
