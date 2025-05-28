"""
Hierarchical Free-Form Deformation (H-FFD) implementation for OpenFFD.

This module provides classes and functions for creating and manipulating
hierarchical FFD control boxes with multiple resolution levels of influence.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

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
        """Initialize computed attributes."""
        if self.bbox is None:
            # Calculate bounding box if not provided
            min_coords = np.min(self.control_points, axis=0)
            max_coords = np.max(self.control_points, axis=0)
            self.bbox = (min_coords, max_coords)
    
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
        
        The influence decreases with distance from the control points and
        is weighted by the level's weight factor.
        
        Args:
            points: Array of point coordinates with shape (n, 3)
            
        Returns:
            Array of influence values with shape (n,)
        """
        # Calculate normalized coordinates within the bounding box
        min_coords, max_coords = self.bbox
        box_size = max_coords - min_coords
        
        # Avoid division by zero
        box_size = np.maximum(box_size, 1e-10)
        
        # Calculate normalized distance from box center
        box_center = (min_coords + max_coords) / 2
        normalized_dist = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Vector from center to point
            vec = point - box_center
            
            # Normalize by box dimensions
            norm_vec = vec / (box_size / 2)
            
            # Distance in normalized space (max component)
            normalized_dist[i] = np.max(np.abs(norm_vec))
        
        # Calculate influence based on normalized distance
        # 1.0 at center, decreasing toward edges
        influence = np.maximum(0.0, 1.0 - normalized_dist)
        
        # Apply level weight factor
        influence *= self.weight_factor
        
        return influence


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
        
        # Create root level
        start_time = time.time()
        logger.info(f"Creating hierarchical FFD with {max_depth} levels...")
        
        # Create the root level control box
        control_points, bbox = create_ffd_box(
            mesh_points,
            control_dim=base_dims,
            margin=margin,
            custom_dims=custom_dims,
            parallel_config=self.parallel_config
        )
        
        # Create the root level
        root_level = HierarchicalLevel(
            level_id=0,
            dims=base_dims,
            control_points=control_points,
            bbox=bbox,
            weight_factor=1.0
        )
        
        self.levels[0] = root_level
        self.root_level = root_level
        
        # Create hierarchical levels
        self._create_hierarchy(root_level, max_depth, subdivision_factor)
        
        # Calculate level weights
        self._calculate_level_weights()
        
        end_time = time.time()
        logger.info(f"Created hierarchical FFD with {len(self.levels)} levels in {end_time - start_time:.2f} seconds")
    
    def _create_hierarchy(
        self,
        parent_level: HierarchicalLevel,
        max_depth: int,
        subdivision_factor: int
    ) -> None:
        """Recursively create the hierarchy of control lattices.
        
        Args:
            parent_level: Parent level to subdivide
            max_depth: Maximum depth of the hierarchy
            subdivision_factor: Factor for subdividing control lattices
        """
        if parent_level.depth >= max_depth - 1:
            return
        
        # Calculate dimensions for the child level
        nx, ny, nz = parent_level.dims
        child_dims = (
            nx * subdivision_factor,
            ny * subdivision_factor,
            nz * subdivision_factor
        )
        
        # Create child level control box
        child_control_points, child_bbox = create_ffd_box(
            self.mesh_points,
            control_dim=child_dims,
            custom_dims=[(coord[0], coord[1]) for coord in zip(*parent_level.bbox)],
            parallel_config=self.parallel_config
        )
        
        # Create child level
        child_level = HierarchicalLevel(
            level_id=len(self.levels),
            dims=child_dims,
            control_points=child_control_points,
            bbox=child_bbox,
            parent_level=parent_level,
            weight_factor=0.5  # Initial weight, will be recalculated later
        )
        
        # Add child to parent and to levels dictionary
        parent_level.children.append(child_level)
        self.levels[child_level.level_id] = child_level
        
        # Create next level recursively
        self._create_hierarchy(child_level, max_depth, subdivision_factor)
    
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
        """Deform mesh points using the hierarchical FFD.
        
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
        
        # Process each level
        for level_id, level in self.levels.items():
            # Skip levels without deformed control points
            if level_id not in deformed_control_points:
                continue
            
            # Get original and deformed control points for this level
            original_cp = level.control_points
            deformed_cp = deformed_control_points[level_id]
            
            # Calculate displacement of control points
            displacement = deformed_cp - original_cp
            
            # Calculate influence of this level on each point
            influence = level.get_influence(points)
            
            # TODO: Implement proper B-spline or trilinear interpolation
            # For now, use a simplified approach: find nearest control point
            for i, point in enumerate(points):
                # Skip points with no influence
                if influence[i] <= 0:
                    continue
                
                # Find nearest control point
                distances = np.sum((original_cp - point) ** 2, axis=1)
                nearest_idx = np.argmin(distances)
                
                # Apply displacement scaled by influence
                deformation[i] += displacement[nearest_idx] * influence[i]
        
        # Return deformed points
        return points + deformation
    
    def add_level(
        self,
        parent_level_id: int,
        dims: Tuple[int, int, int],
        region: Optional[List[Tuple[float, float]]] = None
    ) -> int:
        """Add a new level to the hierarchy.
        
        Args:
            parent_level_id: ID of the parent level
            dims: Dimensions of the new level's control lattice
            region: Optional region bounds for the new level
            
        Returns:
            Level ID of the new level
        """
        parent_level = self.levels[parent_level_id]
        
        # Use parent's bounding box if region not specified
        if region is None:
            region = [(coord[0], coord[1]) for coord in zip(*parent_level.bbox)]
        
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
