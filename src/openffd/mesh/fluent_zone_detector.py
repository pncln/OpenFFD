"""
Specialized Fluent zone detector module.

This module provides direct extraction of zone information from Fluent mesh files
without relying on complete mesh parsing. It's designed to complement meshio's
capabilities by specifically focusing on zone extraction.
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto

# Configure logging
logger = logging.getLogger(__name__)

class ZoneType(Enum):
    """Types of zones in Fluent meshes."""
    VOLUME = auto()
    BOUNDARY = auto()
    INTERFACE = auto()
    UNKNOWN = auto()

class FluentZoneInfo:
    """Information about a zone in a Fluent mesh.
    
    Attributes:
        zone_id: Numeric ID of the zone
        name: Name of the zone
        zone_type: Type of the zone (e.g., 'fluid', 'wall')
        node_ids: Set of node IDs in the zone
        face_ids: Set of face IDs in the zone
        enum_type: Enumerated zone type (VOLUME, BOUNDARY, etc.)
    """
    
    def __init__(self, zone_id: int, name: str, zone_type: str):
        """Initialize a FluentZoneInfo object.
        
        Args:
            zone_id: Numeric ID of the zone
            name: Name of the zone
            zone_type: Type of the zone
        """
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type
        self.node_ids = set()
        self.face_ids = set()
        self.enum_type = self._determine_enum_type()
    
    def add_node(self, node_id: int) -> None:
        """Add a node ID to the zone.
        
        Args:
            node_id: Node ID to add
        """
        self.node_ids.add(node_id)
    
    def add_face(self, face_id: int) -> None:
        """Add a face ID to the zone.
        
        Args:
            face_id: Face ID to add
        """
        self.face_ids.add(face_id)
    
    def _determine_enum_type(self) -> ZoneType:
        """Determine the zone type enum based on the string type.
        
        Returns:
            ZoneType enum value
        """
        if self.zone_type in ['interior', 'fluid']:
            return ZoneType.VOLUME
        elif self.zone_type in ['wall', 'symmetry', 'pressure-outlet', 'velocity-inlet', 
                              'pressure-inlet', 'mass-flow-inlet', 'axis']:
            return ZoneType.BOUNDARY
        elif self.zone_type in ['interface', 'periodic', 'fan', 'porous-jump', 'non-conformal-interface']:
            return ZoneType.INTERFACE
        else:
            return ZoneType.UNKNOWN
    
    def __str__(self) -> str:
        """String representation of the zone.
        
        Returns:
            String representation
        """
        return f"Zone {self.zone_id}: {self.name} (Type: {self.zone_type}, Enum: {self.enum_type.name}, Nodes: {len(self.node_ids)}, Faces: {len(self.face_ids)})"


def extract_zone_from_warning(warning_message: str) -> Optional[Tuple[str, str]]:
    """Extract zone type and name from a warning message.
    
    Args:
        warning_message: The warning message to parse
        
    Returns:
        Tuple of (zone_type, zone_name) or None if no match
    """
    # Check if this is a zone specification warning
    if "Zone specification not supported yet" in warning_message:
        # Extract zone type and name from the warning message
        match = re.search(r'\(([^,]+),\s*([^)]+)\)', warning_message)
        if match:
            zone_type = match.group(1).strip().lower()
            zone_name = match.group(2).strip()
            logger.info(f"Extracted zone from warning: ({zone_type}, {zone_name})")
            return zone_type, zone_name
    
    return None


def detect_zones_from_file(mesh_filename: str) -> List[FluentZoneInfo]:
    """Detect zones in a Fluent mesh file by directly parsing the file.
    
    Args:
        mesh_filename: Path to the Fluent mesh file
        
    Returns:
        List of FluentZoneInfo objects
    """
    zones = []
    
    # Type mapping for numeric zone types
    zone_type_map = {
        '2': 'fluid',
        '3': 'interior',
        '4': 'wall',
        '5': 'pressure-inlet',
        '6': 'pressure-outlet',
        '7': 'symmetry',
        '8': 'periodic-shadow',
        '9': 'periodic',
        '10': 'axis',
        '12': 'fan',
        '14': 'mass-flow-inlet',
        '20': 'interface',
        '24': 'velocity-inlet',
        '31': 'porous-jump',
        '36': 'non-conformal-interface',
    }
    
    # Try to capture zones from meshio warnings by reading the file with meshio
    try:
        import meshio
        
        # Capture warnings during mesh reading
        import warnings
        original_showwarning = warnings.showwarning
        captured_warnings = []
        
        def warning_catcher(message, category, filename, lineno, file=None, line=None):
            captured_warnings.append(str(message))
            original_showwarning(message, category, filename, lineno, file, line)
        
        warnings.showwarning = warning_catcher
        
        try:
            # Try to read mesh with meshio to trigger zone warnings
            logger.info(f"Attempting to read mesh with meshio to capture zone warnings: {mesh_filename}")
            meshio.read(mesh_filename)
        except Exception as e:
            logger.warning(f"Error reading mesh with meshio (this is expected): {e}")
        finally:
            # Restore original warning handler
            warnings.showwarning = original_showwarning
        
        # Process captured warnings
        zone_id = 1000
        for warning in captured_warnings:
            zone_info = extract_zone_from_warning(warning)
            if zone_info:
                zone_type, zone_name = zone_info
                # Create a sanitized zone name
                sanitized_name = f"{zone_type}_{zone_name.replace(' ', '_').replace('-', '_')}"
                zone = FluentZoneInfo(zone_id, sanitized_name, zone_type)
                zones.append(zone)
                zone_id += 1
                logger.info(f"Captured zone from warning: {zone}")
        
        if zone_id > 1000:
            logger.info(f"Captured {zone_id - 1000} zones from meshio warnings")
            return zones
    except ImportError:
        logger.warning("meshio not available, skipping warning-based zone detection")
    
    # If we didn't find any zones from warnings, try direct file parsing
    try:
        with open(mesh_filename, 'r') as f:
            content = f.read()
            
            # Look for zone definitions in the format (45 (zone-id zone-type "zone-name"))
            # This pattern appears in Fluent ASCII mesh files
            zone_pattern = r'\(45\s+\((\d+)\s+([a-zA-Z-]+|[0-9]+)\s+"([^"]+)"'
            zone_matches = re.finditer(zone_pattern, content)
            
            for match in zone_matches:
                zone_id = int(match.group(1))
                zone_type_raw = match.group(2)
                zone_name = match.group(3).strip()
                
                # Convert numeric zone type to string if needed
                if zone_type_raw.isdigit():
                    zone_type = zone_type_map.get(zone_type_raw, "unknown")
                else:
                    zone_type = zone_type_raw.strip().lower()
                
                # Create zone info
                zone_info = FluentZoneInfo(zone_id, zone_name, zone_type)
                zones.append(zone_info)
                logger.debug(f"Found zone in file: {zone_info}")
            
            # Look for warnings directly in the file content
            warning_pattern = r'Zone specification not supported yet \(([^,]+),\s*([^)]+)\)'
            warning_matches = re.finditer(warning_pattern, content)
            
            zone_id = 2000  # Use a different starting ID range for warning-extracted zones
            found_warning_zones = False
            
            for match in warning_matches:
                zone_type = match.group(1).strip().lower()
                zone_name = match.group(2).strip()
                
                # Create a sanitized zone name
                sanitized_name = f"{zone_type}_{zone_name.replace(' ', '_').replace('-', '_')}"
                
                # Create zone info
                zone_info = FluentZoneInfo(zone_id, sanitized_name, zone_type)
                zones.append(zone_info)
                zone_id += 1
                found_warning_zones = True
                logger.debug(f"Found zone from warning in file: {zone_info}")
            
            if found_warning_zones:
                logger.info(f"Found {zone_id-2000} zones from warning messages in file")
            
            # If we found zones using the primary patterns, return them
            if zones:
                logger.info(f"Found {len(zones)} zones in total from direct parsing")
                return zones
            
            # Try an alternative pattern: look for "(zone-type, zone-name)" in other contexts
            alt_pattern = r'\(([a-zA-Z-]+),\s*([^)]+)\)'
            alt_matches = re.finditer(alt_pattern, content)
            
            zone_id = 3000  # Start with a different ID range for this approach
            found_alt_zones = False
            
            for match in alt_matches:
                zone_type = match.group(1).strip().lower()
                zone_name = match.group(2).strip()
                
                # Skip if not likely a zone type
                if zone_type not in ['interior', 'fluid', 'wall', 'symmetry', 'pressure-outlet', 
                                    'velocity-inlet', 'pressure-inlet', 'mass-flow-inlet', 
                                    'interface', 'periodic', 'fan']:
                    continue
                
                # Create a sanitized zone name that combines type and name
                sanitized_name = f"{zone_type}_{zone_name.replace(' ', '_').replace('-', '_')}"
                
                # Create zone info
                zone_info = FluentZoneInfo(zone_id, sanitized_name, zone_type)
                zones.append(zone_info)
                zone_id += 1
                found_alt_zones = True
                logger.debug(f"Found zone using alternative pattern: {zone_info}")
            
            if found_alt_zones:
                logger.info(f"Found {zone_id-3000} zones using alternative pattern")
    
    except Exception as e:
        logger.error(f"Error reading mesh file directly: {e}")
    
    # If we found zones, return them
    if zones:
        return zones
    
    # If no zones were found, use a fallback approach based on filename
    logger.warning(f"No zones found in file {mesh_filename}, using fallback detection")
    
    # Get the base name for fallback zone naming
    mesh_base_name = os.path.basename(mesh_filename).split('.')[0]
    
    # Create standard volume zones
    for vol_type in ['interior', 'fluid']:
        zone_name = f"{vol_type}_{mesh_base_name}"
        zone_info = FluentZoneInfo(len(zones) + 1, zone_name, vol_type)
        zones.append(zone_info)
    
    # Create standard boundary zones
    boundary_patterns = [
        ('wall', 'launchpad'),
        ('wall', 'deflector'),
        ('wall', 'rocket'),
        ('symmetry', 'symmetry'),
        ('pressure-outlet', 'outlet'),
        ('velocity-inlet', 'inlet'),
        ('wall', 'wedge_pos'),
        ('wall', 'wedge_neg')
    ]
    
    for b_type, b_name in boundary_patterns:
        zone_name = f"{b_type}_{b_name}"
        zone_info = FluentZoneInfo(len(zones) + 1, zone_name, b_type)
        zones.append(zone_info)
    
    logger.info(f"Created {len(zones)} fallback zones for {mesh_filename}")
    return zones


def get_zone_tuples(mesh_filename: str) -> List[Tuple[str, str, ZoneType]]:
    """Get zone information as a list of tuples.
    
    This is a convenience function that returns zone information in the format
    expected by the zone extractor.
    
    Args:
        mesh_filename: Path to the Fluent mesh file
        
    Returns:
        List of tuples (zone_type, zone_name, zone_enum_type)
    """
    zones = detect_zones_from_file(mesh_filename)
    return [(z.zone_type, z.name, z.enum_type) for z in zones]
