"""Mesh handling functionality for OpenFFD."""

from openffd.mesh.general import read_general_mesh, is_fluent_mesh, extract_patch_points
from openffd.mesh.zone_extractor import (
    ZoneExtractor, ZoneType, ZoneInfo, 
    extract_zones_parallel, read_mesh_with_zones
)

__all__ = [
    "read_general_mesh", 
    "is_fluent_mesh", 
    "extract_patch_points",
    "ZoneExtractor",
    "ZoneType",
    "ZoneInfo",
    "extract_zones_parallel",
    "read_mesh_with_zones"
]
