# Hierarchical Free-Form Deformation (H-FFD)

## Overview

Hierarchical Free-Form Deformation (H-FFD) extends the traditional FFD approach by introducing multiple resolution levels with varying degrees of influence. This multi-resolution approach provides several significant advantages over standard FFD:

1. **Multi-scale control**: Coarse levels control global shape changes while finer levels enable detailed local deformations
2. **Flexible design**: Apply different deformation strategies at different scales
3. **Adaptive refinement**: Add detail only where needed, saving computational resources
4. **Improved precision**: Achieve more accurate and complex deformations with fewer control points

## Technical Background

H-FFD creates a hierarchy of FFD control lattices, each with its own resolution and influence weight:

- **Root level**: Coarse control lattice with global influence
- **Intermediate levels**: Medium-resolution lattices for regional deformations
- **Leaf levels**: Fine-resolution lattices for detailed local deformations

Each level has an associated weight factor that determines its degree of influence on the final deformation. The deformation process combines contributions from all levels based on these weights and the spatial influence of each level's control points.

## Mathematical Formulation

In traditional FFD, deformation at a point $P$ is computed as:

$$P' = \sum_{i=0}^{l} \sum_{j=0}^{m} \sum_{k=0}^{n} B_i^l(s) B_j^m(t) B_k^n(u) P_{ijk}$$

Where $B_i^l$, $B_j^m$, and $B_k^n$ are the Bernstein polynomials, and $P_{ijk}$ are the control points.

In H-FFD, this is extended to:

$$P' = P + \sum_{L=0}^{N_L} w_L \sum_{i=0}^{l_L} \sum_{j=0}^{m_L} \sum_{k=0}^{n_L} B_i^{l_L}(s) B_j^{m_L}(t) B_k^{n_L}(u) \Delta P_{ijk}^L$$

Where:
- $N_L$ is the number of hierarchical levels
- $w_L$ is the weight factor for level $L$
- $\Delta P_{ijk}^L$ is the displacement of control point $(i,j,k)$ at level $L$
- $(l_L, m_L, n_L)$ are the dimensions of the control lattice at level $L$

## Implementation in OpenFFD

### Core Components

- **HierarchicalLevel**: Represents a single level in the hierarchy with its own control lattice
- **HierarchicalFFD**: Manages the complete hierarchy and handles deformation

### Key Features

- **Automatic hierarchy generation**: Builds multiple levels from a base lattice
- **Influence-based deformation**: Each level influences points based on spatial proximity and weight factor
- **Level management**: Add or remove levels to customize the hierarchy
- **Parallel processing**: Utilizes parallel computing for large datasets

## Command Line Usage

### Basic Usage

```bash
python -m openffd.cli.app my_mesh.stl --hierarchical --base-dims 4 4 4 --max-depth 3 --subdivision-factor 2 --plot
```

### Key Parameters

- `--hierarchical`: Enable hierarchical FFD mode
- `--base-dims NX NY NZ`: Dimensions of the base (root) control lattice
- `--max-depth N`: Maximum depth of the hierarchy (number of levels)
- `--subdivision-factor N`: Factor for subdividing control lattices between levels
- `--show-levels [IDS]`: List of level IDs to show in visualization
- `--show-influence`: Show influence regions in visualization
- `--export-hierarchical DIR`: Export hierarchical FFD control points to a directory

### Example: Multi-resolution Wing Design

```bash
# Create a hierarchical FFD for wing design
python -m openffd.cli.app wing.stl --hierarchical --base-dims 6 4 3 --max-depth 3 --subdivision-factor 2 --export-hierarchical ./wing_hffd --plot --show-influence
```

## Programmatic Usage

```python
from openffd.core.hierarchical import create_hierarchical_ffd
from openffd.mesh.general import read_general_mesh
from openffd.utils.parallel import ParallelConfig
from openffd.visualization.hierarchical_viz import visualize_hierarchical_ffd_pyvista

# Load mesh
mesh_data = read_general_mesh("aircraft.stl")
mesh_points = mesh_data.points

# Create parallel configuration
parallel_config = ParallelConfig(enabled=True, n_workers=4)

# Create hierarchical FFD
hffd = create_hierarchical_ffd(
    mesh_points=mesh_points,
    base_dims=(6, 4, 3),           # Base level dimensions
    max_depth=3,                   # Number of hierarchical levels
    subdivision_factor=2,          # Factor for each subdivision
    parallel_config=parallel_config
)

# Print information about levels
for level_info in hffd.get_level_info():
    print(f"Level {level_info['level_id']} (depth {level_info['depth']}): "
          f"{level_info['dims']} dims, {level_info['num_control_points']} control points, "
          f"weight: {level_info['weight_factor']:.2f}")

# Create a deformation (move some control points)
deformed_control_points = {}
for level_id, level in hffd.levels.items():
    deformed_cp = level.control_points.copy()
    # Apply your deformation strategy here
    # Example: deformed_cp[10] += [0, 5.0, 0]  # Move control point 10 up by 5 units
    deformed_control_points[level_id] = deformed_cp

# Apply the deformation
deformed_points = hffd.deform_mesh(deformed_control_points, mesh_points)

# Visualize the hierarchical structure and deformation
visualize_hierarchical_ffd_pyvista(
    hffd=hffd,
    mesh_points=mesh_points,
    show_mesh=True,
    show_influence=True
)
```

## Advanced Features

### Custom Level Creation

You can add custom levels to specific regions:

```python
# Add a fine-resolution level to a specific region
region_bounds = [
    (x_min, x_max),
    (y_min, y_max),
    (z_min, z_max)
]
new_level_id = hffd.add_level(
    parent_level_id=0,  # Add as child of the root level
    dims=(16, 16, 16),  # High-resolution control lattice
    region=region_bounds  # Only affect this region
)
```

### Level Influence Analysis

Analyze how different levels influence the mesh:

```python
from openffd.visualization.hierarchical_viz import visualize_influence_distribution

# Visualize influence distribution
visualize_influence_distribution(
    hffd=hffd,
    mesh_points=mesh_points,
    save_path="influence_distribution.png"
)
```

## Performance Considerations

- Higher `max_depth` and `subdivision_factor` values create more detailed control but increase computational cost
- For large meshes, enable parallel processing with `--parallel`
- When visualizing complex hierarchies, use `--show-levels` to view specific levels
- For fine control over small features, consider adding targeted custom levels rather than increasing the global depth

## References

1. Forsey, D. R., & Bartels, R. H. (1988). Hierarchical B-spline refinement. *ACM SIGGRAPH Computer Graphics*, 22(4), 205-212.
2. Lee, S., Wolberg, G., & Shin, S. Y. (1997). Scattered data interpolation with multilevel B-splines. *IEEE Transactions on Visualization and Computer Graphics*, 3(3), 228-244.
3. Samareh, J. A. (2001). Novel multidisciplinary shape parameterization approach. *Journal of Aircraft*, 38(6), 1015-1024.
