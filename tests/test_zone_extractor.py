#!/usr/bin/env python
"""
Test suite for the ZoneExtractor functionality.

This module contains tests for the ZoneExtractor class and related functionality.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

from openffd.mesh import (
    ZoneExtractor, ZoneType, ZoneInfo, 
    extract_zones_parallel, read_mesh_with_zones
)
from openffd.utils.parallel import ParallelConfig


def create_test_mesh():
    """Create a simple test mesh with multiple zones."""
    # Skip if meshio is not available
    if not MESHIO_AVAILABLE:
        return None
        
    # Create a simple cube mesh with two volumes and shared boundary
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1]
    ])
    
    # Two hexahedra (cubes)
    vol1_cells = np.array([[0, 1, 3, 2, 4, 5, 7, 6]])
    vol2_cells = np.array([[1, 8, 9, 3, 5, 10, 11, 7]])
    
    # Interface (shared face)
    interface_cells = np.array([[1, 3, 7, 5]])
    
    # Boundary faces
    bound1_cells = np.array([
        [0, 2, 6, 4],  # left face of vol1
        [0, 1, 5, 4],  # bottom face of vol1
        [0, 2, 3, 1],  # front face of vol1
        [2, 6, 7, 3],  # top face of vol1
        [4, 5, 7, 6]   # back face of vol1
    ])
    
    bound2_cells = np.array([
        [8, 9, 11, 10],  # right face of vol2
        [1, 8, 10, 5],   # bottom face of vol2
        [3, 9, 8, 1],    # front face of vol2
        [9, 3, 7, 11],   # top face of vol2
        [5, 10, 11, 7]   # back face of vol2
    ])
    
    # Create cells with different types
    cells = [
        meshio.CellBlock("hexahedron", np.vstack([vol1_cells, vol2_cells])),
        meshio.CellBlock("quad", np.vstack([interface_cells, bound1_cells, bound2_cells]))
    ]
    
    # Cell data for identifying zones
    hexahedron_zone_data = np.array([1, 2])  # 1 for vol1, 2 for vol2
    quad_zone_data = np.array([0, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])  # 0 for interface, 3 for bound1, 4 for bound2
    
    cell_data = {
        "zone": [hexahedron_zone_data, quad_zone_data]
    }
    
    # Create cell sets for better identification
    cell_sets = {
        "volume1": {"hexahedron": np.array([0])},
        "volume2": {"hexahedron": np.array([1])},
        "interface": {"quad": np.array([0])},
        "boundary1": {"quad": np.array([1, 2, 3, 4, 5])},
        "boundary2": {"quad": np.array([6, 7, 8, 9, 10])}
    }
    
    # Create the mesh
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
        cell_sets=cell_sets
    )
    
    return mesh


class TestZoneExtractor(unittest.TestCase):
    """Test the ZoneExtractor class and related functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Skip if meshio is not available
        if not MESHIO_AVAILABLE:
            raise unittest.SkipTest("meshio is not available")
            
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)
        
        # Create a test mesh
        cls.test_mesh = create_test_mesh()
        if cls.test_mesh is None:
            raise unittest.SkipTest("Failed to create test mesh")
            
        # Save the test mesh
        cls.test_mesh_path = cls.test_dir / "test_mesh.vtk"
        meshio.write(cls.test_mesh_path, cls.test_mesh)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        # Remove temporary directory
        cls.temp_dir.cleanup()
    
    def test_constructor(self):
        """Test the ZoneExtractor constructor."""
        extractor = ZoneExtractor(str(self.test_mesh_path))
        self.assertIsNotNone(extractor)
        self.assertEqual(extractor.mesh_file, str(self.test_mesh_path))
        self.assertFalse(extractor._is_fluent)
        self.assertFalse(extractor._loaded)
    
    def test_zone_detection(self):
        """Test the detection of zones in the mesh."""
        extractor = ZoneExtractor(str(self.test_mesh_path))
        zones = extractor.get_zone_info()
        
        # Check that we have the expected zones
        expected_zones = ["volume1", "volume2", "interface", "boundary1", "boundary2"]
        self.assertEqual(set(zones.keys()), set(expected_zones))
        
        # Check that the zones have the correct types
        self.assertEqual(zones["volume1"].zone_type, ZoneType.VOLUME)
        self.assertEqual(zones["volume2"].zone_type, ZoneType.VOLUME)
        self.assertEqual(zones["interface"].zone_type, ZoneType.BOUNDARY)
        self.assertEqual(zones["boundary1"].zone_type, ZoneType.BOUNDARY)
        self.assertEqual(zones["boundary2"].zone_type, ZoneType.BOUNDARY)
    
    def test_zone_extraction(self):
        """Test the extraction of zones from the mesh."""
        extractor = ZoneExtractor(str(self.test_mesh_path))
        
        # Extract a volume zone
        vol1_mesh = extractor.extract_zone_mesh("volume1")
        self.assertIsNotNone(vol1_mesh)
        self.assertEqual(len(vol1_mesh.points), 8)  # 8 points in the first cube
        
        # Extract a boundary zone
        bound1_mesh = extractor.extract_boundary_mesh("boundary1")
        self.assertIsNotNone(bound1_mesh)
        
        # Extract the interface
        interface_mesh = extractor.extract_zone_mesh("interface")
        self.assertIsNotNone(interface_mesh)
    
    def test_parallel_extraction(self):
        """Test parallel extraction of multiple zones."""
        extractor = ZoneExtractor(str(self.test_mesh_path), 
                                 ParallelConfig(enabled=True))
        
        # Extract multiple zones in parallel
        zones_to_extract = ["volume1", "volume2", "interface"]
        result = extractor.parallel_extract_zones(zones_to_extract)
        
        # Check that all zones were extracted
        self.assertEqual(set(result.keys()), set(zones_to_extract))
        for zone_name, zone_mesh in result.items():
            self.assertIsNotNone(zone_mesh)
    
    def test_zone_type_from_string(self):
        """Test conversion of strings to ZoneType."""
        # Test various volume strings
        self.assertEqual(ZoneType.from_string("volume"), ZoneType.VOLUME)
        self.assertEqual(ZoneType.from_string("vol"), ZoneType.VOLUME)
        self.assertEqual(ZoneType.from_string("cell"), ZoneType.VOLUME)
        
        # Test various boundary strings
        self.assertEqual(ZoneType.from_string("boundary"), ZoneType.BOUNDARY)
        self.assertEqual(ZoneType.from_string("bound"), ZoneType.BOUNDARY)
        self.assertEqual(ZoneType.from_string("surface"), ZoneType.BOUNDARY)
        
        # Test various interface strings
        self.assertEqual(ZoneType.from_string("interface"), ZoneType.INTERFACE)
        self.assertEqual(ZoneType.from_string("int"), ZoneType.INTERFACE)
        
        # Test unknown string
        self.assertEqual(ZoneType.from_string("unknown"), ZoneType.UNKNOWN)
    
    def test_zone_info_to_dict(self):
        """Test conversion of ZoneInfo to dictionary."""
        zone_info = ZoneInfo(
            name="test_zone",
            zone_type=ZoneType.VOLUME,
            cell_count=10,
            point_count=20,
            element_types={"hexahedron", "tetra"},
            dimensions=(2, 3, 4),
            metadata={"key": "value"}
        )
        
        zone_dict = zone_info.to_dict()
        self.assertEqual(zone_dict["name"], "test_zone")
        self.assertEqual(zone_dict["zone_type"], "VOLUME")
        self.assertEqual(zone_dict["cell_count"], 10)
        self.assertEqual(zone_dict["point_count"], 20)
        self.assertEqual(set(zone_dict["element_types"]), {"hexahedron", "tetra"})
        self.assertEqual(zone_dict["dimensions"], (2, 3, 4))
        self.assertEqual(zone_dict["metadata"], {"key": "value"})
    
    def test_export_zones_summary(self):
        """Test exporting a summary of zones to a JSON file."""
        extractor = ZoneExtractor(str(self.test_mesh_path))
        
        # Export zones summary
        summary_path = self.test_dir / "zones_summary.json"
        extractor.export_zones_summary(str(summary_path))
        
        # Check that the summary file exists
        self.assertTrue(summary_path.exists())
        
        # Load the summary and check its contents
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
        self.assertEqual(summary["mesh_file"], str(self.test_mesh_path))
        self.assertEqual(summary["format"], "General")
        self.assertEqual(set(summary["zones"].keys()), 
                        set(["volume1", "volume2", "interface", "boundary1", "boundary2"]))
    
    def test_save_zone_mesh(self):
        """Test saving a zone mesh to a file."""
        extractor = ZoneExtractor(str(self.test_mesh_path))
        
        # Extract and save a zone
        zone_path = self.test_dir / "volume1.vtk"
        extractor.save_zone_mesh("volume1", str(zone_path))
        
        # Check that the zone file exists
        self.assertTrue(zone_path.exists())
        
        # Load the saved zone and check its contents
        saved_mesh = meshio.read(zone_path)
        self.assertIsNotNone(saved_mesh)
        self.assertEqual(len(saved_mesh.points), 8)  # 8 points in the first cube
    
    def test_read_mesh_with_zones(self):
        """Test reading a mesh with zones."""
        mesh, zones = read_mesh_with_zones(str(self.test_mesh_path))
        
        # Check that the mesh and zones were loaded
        self.assertIsNotNone(mesh)
        self.assertIsNotNone(zones)
        self.assertEqual(set(zones.keys()), 
                        set(["volume1", "volume2", "interface", "boundary1", "boundary2"]))


if __name__ == "__main__":
    unittest.main()
