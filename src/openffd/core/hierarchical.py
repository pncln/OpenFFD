"""
Hierarchical Free-Form Deformation (H-FFD) implementation for OpenFFD.

This module provides classes and functions for creating and manipulating
hierarchical FFD control boxes with multiple resolution levels of influence.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from openffd.core.control_box import create_ffd_box
from openffd.utils.parallel import (
    ParallelConfig, 
    parallel_process, 
    is_parallelizable
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalLevel:
    """Represents a single level in a hierarchical FFD structure.
    
    Each level has its own control lattice with different resolution.
    
    Attributes:
        level_id: Unique identifier for this level
        dims: Dimensions of the control lattice (nx, ny, nz)
        control_points: Array of control point coordinates with shape (nx*ny*nz, 3)
        weight_factor: Influence weight of this level (0.0-1.0)
        parent_level: Reference to parent level (None for root level)
        children: List of child levels (empty for leaf levels)
        bbox: Bounding box of this level's control points (min_coords, max_coords)
    """
    level_id: int
    dims: Tuple[int, int, int]
    control_points: np.ndarray
    weight_factor: float = 1.0
    parent_level: Optional['HierarchicalLevel'] = None
    children: List['HierarchicalLevel'] = field(default_factory=list)
    bbox: Tuple[np.ndarray, np.ndarray] = None
    
    def __post_init__(self):
        """Initialize computed attributes and optimization structures."""
        if self.bbox is None:
            # Calculate bounding box if not provided
            min_coords = np.min(self.control_points, axis=0)
            max_coords = np.max(self.control_points, axis=0)
            self.bbox = (min_coords, max_coords)
        
        # Pre-build KDTree for fast nearest neighbor searches
        self._kdtree = cKDTree(self.control_points)
        
        # Pre-calculate box properties for influence calculation
        min_coords, max_coords = self.bbox
        self._box_center = (min_coords + max_coords) / 2
        self._box_size = np.maximum(max_coords - min_coords, 1e-10)
        self._half_box_size = self._box_size / 2
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root level (no parent)."""
        return self.parent_level is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf level (no children)."""
        return len(self.children) == 0
    
    @property
    def depth(self) -> int:
        """Get the depth of this level in the hierarchy."""
        if self.is_root:
            return 0
        return self.parent_level.depth + 1
    
    def get_influence(self, points: np.ndarray) -> np.ndarray:
        """Calculate the influence of this level on the given points.
        
        OPTIMIZED: Vectorized calculation for maximum performance.
        
        Args:
            points: Array of point coordinates with shape (n, 3)
            
        Returns:
            Array of influence values with shape (n,)
        """
        # Use pre-calculated box properties for maximum speed
        
        # Use pre-calculated box properties for maximum speed
        # Vector from center to all points (broadcasting)
        vec = points - self._box_center
        
        # Normalize by box dimensions (broadcasting)
        norm_vec = vec / self._half_box_size
        
        # Distance in normalized space (max component for each point)
        normalized_dist = np.max(np.abs(norm_vec), axis=1)
        
        # Calculate influence based on normalized distance (vectorized)
        influence = np.maximum(0.0, 1.0 - normalized_dist)
        
        # Apply level weight factor
        influence *= self.weight_factor
        
        return influence
    
    def get_nearest_control_points(self, points: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Fast nearest neighbor search using pre-built KDTree.
        
        Args:
            points: Query points with shape (n, 3)
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        return self._kdtree.query(points, k=k)


class HierarchicalFFD:
    """Hierarchical Free-Form Deformation with multiple resolution levels.
    
    This class implements a hierarchical FFD structure with multiple levels
    of control lattices at different resolutions. Each level has a different
    degree of influence on the final deformation.
    
    Attributes:
        levels: Dictionary of hierarchical levels indexed by level_id
        root_level: The root level of the hierarchy
        mesh_points: Original mesh points
        parallel_config: Configuration for parallel processing
    """
    
    def __init__(
        self,
        mesh_points: np.ndarray,
        base_dims: Tuple[int, int, int] = (4, 4, 4),
        max_depth: int = 3,
        subdivision_factor: int = 2,
        parallel_config: Optional[ParallelConfig] = None,
        custom_dims: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None,
        margin: float = 0.05
    ):
        """Initialize a hierarchical FFD structure.
        
        Args:
            mesh_points: Array of mesh point coordinates with shape (n, 3)
            base_dims: Dimensions of the base (root) control lattice
            max_depth: Maximum depth of the hierarchy
            subdivision_factor: Factor for subdividing control lattices
            parallel_config: Configuration for parallel processing
        """
        self.mesh_points = mesh_points
        self.levels = {}
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Cache mesh bounding box for efficient level creation
        self._mesh_bbox = self._calculate_mesh_bbox(mesh_points)
        
        # Create root level
        start_time = time.time()
        logger.info(f"Creating hierarchical FFD with {max_depth} levels...")
        
        # OPTIMIZATION: Create all levels in parallel for maximum speed
        self._create_levels_parallel(base_dims, max_depth, subdivision_factor, margin, custom_dims)
        
        # Calculate level weights
        self._calculate_level_weights()
        
        end_time = time.time()
        logger.info(f"Created hierarchical FFD with {len(self.levels)} levels in {end_time - start_time:.2f} seconds")
    
    def _calculate_mesh_bbox(self, mesh_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate and cache mesh bounding box once."""
        min_coords = np.min(mesh_points, axis=0)
        max_coords = np.max(mesh_points, axis=0)
        return (min_coords, max_coords)
    
    def _create_levels_parallel(
        self,
        base_dims: Tuple[int, int, int],
        max_depth: int,
        subdivision_factor: int,
        margin: float,
        custom_dims: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None
    ) -> None:
        """OPTIMIZED: Create all hierarchy levels in parallel for maximum performance.
        
        This completely eliminates the sequential bottleneck by creating all levels
        simultaneously using multi-threading.
        """
        # Pre-calculate all level dimensions and relationships
        level_specs = []
        for depth in range(max_depth):
            factor = subdivision_factor ** depth
            dims = (base_dims[0] * factor, base_dims[1] * factor, base_dims[2] * factor)
            level_specs.append({
                'level_id': depth,
                'depth': depth,
                'dims': dims,
                'parent_id': depth - 1 if depth > 0 else None
            })
        
        # Create all FFD boxes in parallel using ThreadPoolExecutor
        def create_level_box(spec, parallel_config):
            """Create a single level's control box."""
            # Use cached mesh bbox for efficient box creation
            if spec['depth'] == 0:
                # Root level uses full mesh or custom dimensions
                region = custom_dims
            else:
                # Child levels use mesh bounding box (more efficient than parent bbox)
                min_coords, max_coords = self._mesh_bbox
                region = [(min_coords[i], max_coords[i]) for i in range(3)]
            
            control_points, bbox = create_ffd_box(
                self.mesh_points,
                control_dim=spec['dims'],
                margin=margin,
                custom_dims=region,
                parallel_config=parallel_config
            )
            
            return {
                'spec': spec,
                'control_points': control_points,
                'bbox': bbox
            }
        
        # PARALLEL EXECUTION: Create all levels simultaneously
        # Disable nested parallelization to prevent pool-within-pool overhead
        nested_parallel_config = ParallelConfig(enabled=False) if self.parallel_config.enabled else self.parallel_config
        
        def create_level_box_safe(spec):
            return create_level_box(spec, nested_parallel_config)
        
        with ThreadPoolExecutor(max_workers=min(max_depth, 4)) as executor:
            # Submit all level creation tasks with disabled nested parallelization
            future_to_spec = {executor.submit(create_level_box_safe, spec): spec for spec in level_specs}
            
            # Collect results as they complete
            level_results = {}
            for future in as_completed(future_to_spec):
                result = future.result()
                level_results[result['spec']['level_id']] = result
        
        # Build the hierarchy structure from parallel results
        for level_id in sorted(level_results.keys()):
            result = level_results[level_id]
            spec = result['spec']
            
            # Create level object
            level = HierarchicalLevel(
                level_id=spec['level_id'],
                dims=spec['dims'],
                control_points=result['control_points'],
                bbox=result['bbox'],
                parent_level=self.levels.get(spec['parent_id']),
                weight_factor=1.0  # Will be recalculated
            )
            
            # Add to levels dictionary
            self.levels[level_id] = level
            
            # Set up parent-child relationships
            if spec['parent_id'] is not None:
                parent = self.levels[spec['parent_id']]
                parent.children.append(level)
        
        # Set root level reference
        self.root_level = self.levels[0]
        
        logger.info(f"✅ PARALLEL: Created {len(self.levels)} levels simultaneously")
    
    def _calculate_level_weights(self) -> None:
        """Calculate weights for each level in the hierarchy.
        
        Weight assignment strategy:
        - Root level has highest weight
        - Deeper levels have progressively lower weights
        - Sum of all weights equals 1.0
        """
        max_depth = max(level.depth for level in self.levels.values())
        
        # Linear weight decay with depth
        for level_id, level in self.levels.items():
            # Weight decreases with depth
            level.weight_factor = 1.0 - (level.depth / (max_depth + 1))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(level.weight_factor for level in self.levels.values())
        for level in self.levels.values():
            level.weight_factor /= total_weight
    
    def get_all_control_points(self) -> Dict[int, np.ndarray]:
        """Get all control points for all levels.
        
        Returns:
            Dictionary mapping level_id to control points array
        """
        return {level_id: level.control_points for level_id, level in self.levels.items()}
    
    def deform_mesh(
        self,
        deformed_control_points: Dict[int, np.ndarray],
        points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """OPTIMIZED: Deform mesh points using hierarchical FFD with maximum performance.
        
        Uses vectorized operations and KDTree for O(log n) nearest neighbor searches.
        
        Args:
            deformed_control_points: Dictionary mapping level_id to deformed control points
            points: Points to deform (defaults to original mesh points)
            
        Returns:
            Deformed points
        """
        if points is None:
            points = self.mesh_points
        
        # Initialize deformation with zeros
        deformation = np.zeros_like(points)
        
        # Process each level with optimized operations
        for level_id, level in self.levels.items():
            # Skip levels without deformed control points
            if level_id not in deformed_control_points:
                continue
            
            # Get original and deformed control points for this level
            original_cp = level.control_points
            deformed_cp = deformed_control_points[level_id]
            
            # Calculate displacement of control points
            displacement = deformed_cp - original_cp
            
            # OPTIMIZED: Calculate influence using vectorized operations
            influence = level.get_influence(points)
            
            # Skip points with no influence (vectorized filtering)
            active_mask = influence > 0
            if not np.any(active_mask):
                continue
            
            active_points = points[active_mask]
            active_influence = influence[active_mask]
            
            # OPTIMIZED: Use KDTree for fast nearest neighbor search
            distances, nearest_indices = level.get_nearest_control_points(active_points, k=1)
            
            # Flatten indices for proper indexing
            nearest_indices = nearest_indices.flatten()
            
            # VECTORIZED: Apply displacement scaled by influence
            point_displacements = displacement[nearest_indices] * active_influence.reshape(-1, 1)
            
            # Apply deformations to the full deformation array
            deformation[active_mask] += point_displacements
        
        # Return deformed points
        return points + deformation
    
    def add_level(
        self,
        parent_level_id: int,
        dims: Tuple[int, int, int],
        region: Optional[List[Tuple[float, float]]] = None
    ) -> int:
        """OPTIMIZED: Add a new level to the hierarchy with efficient computation.
        
        Args:
            parent_level_id: ID of the parent level
            dims: Dimensions of the new level's control lattice
            region: Optional region bounds for the new level
            
        Returns:
            Level ID of the new level
        """
        parent_level = self.levels[parent_level_id]
        
        # Use cached mesh bounding box for better performance
        if region is None:
            min_coords, max_coords = self._mesh_bbox
            region = [(min_coords[i], max_coords[i]) for i in range(3)]
        
        # Create control box for the new level
        control_points, bbox = create_ffd_box(
            self.mesh_points,
            control_dim=dims,
            custom_dims=region,
            parallel_config=self.parallel_config
        )
        
        # Create new level
        new_level = HierarchicalLevel(
            level_id=len(self.levels),
            dims=dims,
            control_points=control_points,
            bbox=bbox,
            parent_level=parent_level,
            weight_factor=0.5  # Initial weight
        )
        
        # Add to parent and levels dictionary
        parent_level.children.append(new_level)
        self.levels[new_level.level_id] = new_level
        
        # Recalculate weights
        self._calculate_level_weights()
        
        return new_level.level_id
    
    def remove_level(self, level_id: int) -> None:
        """Remove a level from the hierarchy.
        
        Args:
            level_id: ID of the level to remove
        
        Raises:
            ValueError: If trying to remove the root level
            KeyError: If level_id doesn't exist
        """
        if level_id == 0:
            raise ValueError("Cannot remove root level")
        
        level = self.levels[level_id]
        parent = level.parent_level
        
        # Remove from parent's children
        parent.children = [child for child in parent.children if child.level_id != level_id]
        
        # Remove from levels dictionary
        del self.levels[level_id]
        
        # Recalculate weights
        self._calculate_level_weights()
    
    def get_level_info(self) -> List[Dict[str, Any]]:
        """Get information about all levels in the hierarchy.
        
        Returns:
            List of dictionaries with level information
        """
        info = []
        for level_id, level in sorted(self.levels.items()):
            info.append({
                'level_id': level_id,
                'depth': level.depth,
                'dims': level.dims,
                'num_control_points': len(level.control_points),
                'weight_factor': level.weight_factor,
                'bbox': level.bbox,
                'parent_id': level.parent_level.level_id if level.parent_level else None,
                'children_ids': [child.level_id for child in level.children]
            })
        return info


def create_hierarchical_ffd(
    mesh_points: np.ndarray,
    base_dims: Tuple[int, int, int] = (4, 4, 4),
    max_depth: int = 3,
    subdivision_factor: int = 2,
    parallel_config: Optional[ParallelConfig] = None,
    custom_dims: Optional[List[Optional[Tuple[Optional[float], Optional[float]]]]] = None,
    margin: float = 0.05
) -> HierarchicalFFD:
    """Create a hierarchical FFD structure.
    
    Args:
        mesh_points: Array of mesh point coordinates with shape (n, 3)
        base_dims: Dimensions of the base (root) control lattice
        max_depth: Maximum depth of the hierarchy
        subdivision_factor: Factor for subdividing control lattices
        parallel_config: Configuration for parallel processing
        
    Returns:
        HierarchicalFFD object
    """
    return HierarchicalFFD(
        mesh_points=mesh_points,
        base_dims=base_dims,
        max_depth=max_depth,
        subdivision_factor=subdivision_factor,
        parallel_config=parallel_config,
        custom_dims=custom_dims,
        margin=margin
    )
