"""
OpenFFD Mesh Processing Module

This module provides comprehensive mesh processing capabilities for OpenFFD including:
- Advanced mesh deformation with FFD/HFFD control points
- Comprehensive mesh quality assessment and validation
- Multi-format mesh I/O support
- Mesh smoothing and repair algorithms
- Boundary layer preservation
- Parallel mesh processing capabilities
"""

# Import existing mesh functionality
from openffd.mesh.general import read_general_mesh, is_fluent_mesh, extract_patch_points
from openffd.mesh.zone_extractor import (
    ZoneExtractor, ZoneType, ZoneInfo, 
    extract_zones_parallel, read_mesh_with_zones
)

# Import new advanced mesh functionality
from .deformation import (
    MeshDeformationEngine,
    DeformationConfig,
    MeshQualityLimits,
    BoundaryLayerConfig,
    MeshQualityReport,
    MeshFormat as DeformationMeshFormat,
    MeshQualityMetric,
    SmoothingAlgorithm
)

from .quality import (
    MeshQualityAnalyzer,
    QualityThresholds,
    ElementType
)

from .formats import (
    MeshData,
    MeshFormat,
    MeshFormatRegistry,
    VTKCellType,
    read_mesh,
    write_mesh,
    convert_mesh,
    # Format-specific readers and writers
    OpenFOAMReader,
    OpenFOAMWriter,
    VTKLegacyReader,
    VTKLegacyWriter,
    VTKXMLReader,
    VTKXMLWriter,
    STLReader,
    STLWriter,
    GMSHReader,
    GMSHWriter,
    FluentReader,
    FluentWriter,
    CGNSReader,
    CGNSWriter
)

__all__ = [
    # Existing functionality
    "read_general_mesh", 
    "is_fluent_mesh", 
    "extract_patch_points",
    "ZoneExtractor",
    "ZoneType",
    "ZoneInfo",
    "extract_zones_parallel",
    "read_mesh_with_zones",
    
    # Advanced deformation functionality
    'MeshDeformationEngine',
    'DeformationConfig',
    'MeshQualityLimits',
    'BoundaryLayerConfig',
    'MeshQualityReport',
    'MeshQualityMetric',
    'SmoothingAlgorithm',
    
    # Quality analysis functionality
    'MeshQualityAnalyzer',
    'QualityThresholds',
    'ElementType',
    'VTKCellType',
    
    # Format I/O functionality
    'MeshData',
    'MeshFormat',
    'MeshFormatRegistry',
    'read_mesh',
    'write_mesh',
    'convert_mesh',
    'OpenFOAMReader',
    'OpenFOAMWriter',
    'VTKLegacyReader',
    'VTKLegacyWriter',
    'VTKXMLReader',
    'VTKXMLWriter',
    'STLReader',
    'STLWriter',
    'GMSHReader',
    'GMSHWriter',
    'FluentReader',
    'FluentWriter',
    'CGNSReader',
    'CGNSWriter'
]
