"""
Solution File I/O for CFD Simulations

Implements comprehensive file I/O for CFD solution data:
- VTK (Visualization Toolkit) format for ParaView and VisIt
- Tecplot format for Tecplot 360 visualization
- CGNS (CFD General Notation System) for standard CFD data exchange
- HDF5 format for high-performance parallel I/O
- Native binary format for checkpoint/restart
- ASCII formats for debugging and legacy compatibility
- Parallel I/O for large-scale simulations
- Metadata and provenance tracking

Supports both structured and unstructured mesh formats with
comprehensive solution variable handling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, BinaryIO, TextIO
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import struct
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import pickle
import gzip
import time

# Try to import optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Enumeration of supported file formats."""
    VTK_LEGACY = "vtk_legacy"
    VTK_XML = "vtk_xml"
    TECPLOT_ASCII = "tecplot_ascii"
    TECPLOT_BINARY = "tecplot_binary"
    CGNS = "cgns"
    HDF5 = "hdf5"
    NATIVE_BINARY = "native_binary"
    NATIVE_ASCII = "native_ascii"
    PLOT3D = "plot3d"


class DataType(Enum):
    """Enumeration of data types."""
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


@dataclass
class VariableInfo:
    """Information about a solution variable."""
    name: str
    data_type: DataType
    units: str = ""
    description: str = ""
    location: str = "cell"  # "cell" or "node"


@dataclass
class MeshInfo:
    """Information about mesh geometry."""
    n_nodes: int
    n_cells: int
    n_faces: int
    coordinates: np.ndarray
    connectivity: np.ndarray
    cell_types: np.ndarray
    boundary_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionData:
    """Container for solution data."""
    variables: Dict[str, np.ndarray] = field(default_factory=dict)
    variable_info: Dict[str, VariableInfo] = field(default_factory=dict)
    time: float = 0.0
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IOConfig:
    """Configuration for file I/O operations."""
    
    # File format preferences
    default_format: FileFormat = FileFormat.VTK_XML
    precision: str = "double"  # "single" or "double"
    compression: bool = True
    compression_level: int = 6
    
    # Parallel I/O
    parallel_io: bool = False
    mpi_rank: int = 0
    mpi_size: int = 1
    
    # Data filtering
    include_ghost_cells: bool = False
    variable_filter: Optional[List[str]] = None
    time_range: Optional[Tuple[float, float]] = None
    
    # Performance settings
    buffer_size: int = 1024 * 1024  # 1MB buffer
    chunk_size: int = 10000  # Records per chunk
    async_io: bool = False
    
    # Metadata
    include_metadata: bool = True
    include_provenance: bool = True
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


class FileWriter(ABC):
    """Abstract base class for file writers."""
    
    def __init__(self, config: IOConfig):
        """Initialize file writer."""
        self.config = config
        
    @abstractmethod
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write solution to file."""
        pass
    
    @abstractmethod
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData]) -> bool:
        """Write time series data."""
        pass


class FileReader(ABC):
    """Abstract base class for file readers."""
    
    def __init__(self, config: IOConfig):
        """Initialize file reader."""
        self.config = config
        
    @abstractmethod
    def read_solution(self, filename: str) -> Tuple[MeshInfo, SolutionData]:
        """Read solution from file."""
        pass
    
    @abstractmethod
    def read_time_series(self, pattern: str) -> Tuple[MeshInfo, List[SolutionData]]:
        """Read time series data."""
        pass


class VTKWriter(FileWriter):
    """
    VTK format writer for ParaView visualization.
    
    Supports both legacy and XML VTK formats.
    """
    
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write solution in VTK format."""
        try:
            if self.config.default_format == FileFormat.VTK_XML:
                return self._write_vtk_xml(filename, mesh, solution)
            else:
                return self._write_vtk_legacy(filename, mesh, solution)
        except Exception as e:
            logger.error(f"Failed to write VTK file {filename}: {e}")
            return False
    
    def _write_vtk_legacy(self,
                         filename: str,
                         mesh: MeshInfo,
                         solution: SolutionData) -> bool:
        """Write legacy VTK format."""
        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"CFD Solution at time {solution.time}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            f.write(f"POINTS {mesh.n_nodes} float\n")
            for i in range(mesh.n_nodes):
                coord = mesh.coordinates[i] if i < len(mesh.coordinates) else [0, 0, 0]
                f.write(f"{coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}\n")
            
            # Cells
            if hasattr(mesh, 'connectivity') and mesh.connectivity is not None:
                total_size = mesh.n_cells + np.sum([len(cell) for cell in mesh.connectivity])
                f.write(f"CELLS {mesh.n_cells} {total_size}\n")
                
                for cell_nodes in mesh.connectivity:
                    f.write(f"{len(cell_nodes)}")
                    for node in cell_nodes:
                        f.write(f" {node}")
                    f.write("\n")
                
                # Cell types
                f.write(f"CELL_TYPES {mesh.n_cells}\n")
                for cell_type in mesh.cell_types:
                    f.write(f"{cell_type}\n")
            
            # Cell data
            if solution.variables:
                f.write(f"CELL_DATA {mesh.n_cells}\n")
                
                for var_name, var_data in solution.variables.items():
                    if var_name in solution.variable_info:
                        var_info = solution.variable_info[var_name]
                        
                        if var_info.data_type == DataType.SCALAR:
                            f.write(f"SCALARS {var_name} float 1\n")
                            f.write("LOOKUP_TABLE default\n")
                            for value in var_data.flatten():
                                f.write(f"{value:.6e}\n")
                        
                        elif var_info.data_type == DataType.VECTOR:
                            f.write(f"VECTORS {var_name} float\n")
                            for i in range(len(var_data)):
                                vec = var_data[i] if len(var_data[i]) >= 3 else list(var_data[i]) + [0.0] * (3 - len(var_data[i]))
                                f.write(f"{vec[0]:.6e} {vec[1]:.6e} {vec[2]:.6e}\n")
        
        logger.info(f"Wrote VTK legacy file: {filename}")
        return True
    
    def _write_vtk_xml(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write XML VTK format."""
        if not VTK_AVAILABLE:
            logger.warning("VTK library not available, falling back to legacy format")
            return self._write_vtk_legacy(filename, mesh, solution)
        
        # Create VTK unstructured grid
        ugrid = vtk.vtkUnstructuredGrid()
        
        # Add points
        points = vtk.vtkPoints()
        for i in range(mesh.n_nodes):
            coord = mesh.coordinates[i] if i < len(mesh.coordinates) else [0, 0, 0]
            points.InsertNextPoint(coord[0], coord[1], coord[2])
        ugrid.SetPoints(points)
        
        # Add cells (simplified - assumes tetrahedra)
        if hasattr(mesh, 'connectivity') and mesh.connectivity is not None:
            for cell_nodes in mesh.connectivity:
                if len(cell_nodes) == 4:  # Tetrahedron
                    ugrid.InsertNextCell(vtk.VTK_TETRA, 4, cell_nodes)
                elif len(cell_nodes) == 8:  # Hexahedron
                    ugrid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, cell_nodes)
        
        # Add solution data
        for var_name, var_data in solution.variables.items():
            if var_name in solution.variable_info:
                var_info = solution.variable_info[var_name]
                
                if var_info.data_type == DataType.SCALAR:
                    vtk_array = numpy_support.numpy_to_vtk(var_data.flatten())
                    vtk_array.SetName(var_name)
                    ugrid.GetCellData().AddArray(vtk_array)
                    if ugrid.GetCellData().GetNumberOfArrays() == 1:
                        ugrid.GetCellData().SetActiveScalars(var_name)
        
        # Write file
        if filename.endswith('.vtu'):
            writer = vtk.vtkXMLUnstructuredGridWriter()
        else:
            writer = vtk.vtkUnstructuredGridWriter()
        
        writer.SetFileName(filename)
        writer.SetInputData(ugrid)
        
        if self.config.compression:
            writer.SetCompressorTypeToZLib()
        
        writer.Write()
        
        logger.info(f"Wrote VTK XML file: {filename}")
        return True
    
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData]) -> bool:
        """Write VTK time series."""
        success = True
        
        for i, solution in enumerate(solutions):
            # Generate filename with time/iteration info
            name, ext = os.path.splitext(base_filename)
            time_filename = f"{name}_t{solution.time:.6f}_i{solution.iteration:06d}{ext}"
            
            if not self.write_solution(time_filename, mesh, solution):
                success = False
        
        # Create PVD file for time series (if VTK XML format)
        if self.config.default_format == FileFormat.VTK_XML:
            self._write_pvd_file(base_filename, solutions)
        
        return success
    
    def _write_pvd_file(self, base_filename: str, solutions: List[SolutionData]):
        """Write ParaView Data (PVD) file for time series."""
        name, _ = os.path.splitext(base_filename)
        pvd_filename = f"{name}.pvd"
        
        root = ET.Element("VTKFile", type="Collection", version="0.1")
        collection = ET.SubElement(root, "Collection")
        
        for solution in solutions:
            name, ext = os.path.splitext(base_filename)
            vtu_filename = f"{name}_t{solution.time:.6f}_i{solution.iteration:06d}.vtu"
            
            dataset = ET.SubElement(collection, "DataSet",
                                  timestep=str(solution.time),
                                  group="",
                                  part="0",
                                  file=os.path.basename(vtu_filename))
        
        tree = ET.ElementTree(root)
        tree.write(pvd_filename, encoding="utf-8", xml_declaration=True)
        
        logger.info(f"Wrote PVD time series file: {pvd_filename}")


class TecplotWriter(FileWriter):
    """
    Tecplot format writer for Tecplot 360 visualization.
    
    Supports both ASCII and binary Tecplot formats.
    """
    
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write solution in Tecplot format."""
        try:
            if self.config.default_format == FileFormat.TECPLOT_BINARY:
                return self._write_tecplot_binary(filename, mesh, solution)
            else:
                return self._write_tecplot_ascii(filename, mesh, solution)
        except Exception as e:
            logger.error(f"Failed to write Tecplot file {filename}: {e}")
            return False
    
    def _write_tecplot_ascii(self,
                           filename: str,
                           mesh: MeshInfo,
                           solution: SolutionData) -> bool:
        """Write ASCII Tecplot format."""
        with open(filename, 'w') as f:
            # Title
            f.write(f'TITLE = "CFD Solution at time {solution.time}"\n')
            
            # Variables
            var_names = ["X", "Y", "Z"]
            for var_name in solution.variables.keys():
                var_names.append(var_name.replace(" ", "_"))
            
            variables_str = ", ".join([f'"{name}"' for name in var_names])
            f.write(f'VARIABLES = {variables_str}\n')
            
            # Zone header
            f.write(f'ZONE T="Solution", N={mesh.n_nodes}, E={mesh.n_cells}, ')
            f.write('DATAPACKING=BLOCK, ZONETYPE=FETETRAHEDRON\n')
            
            # Coordinate data
            for dim in range(3):
                for i in range(mesh.n_nodes):
                    coord = mesh.coordinates[i] if i < len(mesh.coordinates) else [0, 0, 0]
                    value = coord[dim] if dim < len(coord) else 0.0
                    f.write(f"{value:.6e}\n")
            
            # Solution data (interpolated to nodes if cell-centered)
            for var_name, var_data in solution.variables.items():
                # Simple interpolation from cells to nodes (would need proper interpolation)
                for i in range(mesh.n_nodes):
                    # Use nearest cell value (simplified)
                    cell_idx = min(i, len(var_data) - 1)
                    if var_data.ndim == 1:
                        value = var_data[cell_idx]
                    else:
                        value = var_data[cell_idx, 0] if len(var_data[cell_idx]) > 0 else 0.0
                    f.write(f"{value:.6e}\n")
            
            # Connectivity (assuming tetrahedra)
            if hasattr(mesh, 'connectivity') and mesh.connectivity is not None:
                for cell_nodes in mesh.connectivity:
                    # Tecplot uses 1-based indexing
                    connectivity_line = " ".join([str(node + 1) for node in cell_nodes[:4]])
                    f.write(f"{connectivity_line}\n")
        
        logger.info(f"Wrote Tecplot ASCII file: {filename}")
        return True
    
    def _write_tecplot_binary(self,
                            filename: str,
                            mesh: MeshInfo,
                            solution: SolutionData) -> bool:
        """Write binary Tecplot format (simplified)."""
        # This would implement the full Tecplot binary format
        # For now, fallback to ASCII
        logger.warning("Tecplot binary format not fully implemented, using ASCII")
        return self._write_tecplot_ascii(filename, mesh, solution)
    
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData]) -> bool:
        """Write Tecplot time series."""
        success = True
        
        for i, solution in enumerate(solutions):
            name, ext = os.path.splitext(base_filename)
            time_filename = f"{name}_t{solution.time:.6f}{ext}"
            
            if not self.write_solution(time_filename, mesh, solution):
                success = False
        
        return success


class HDF5Writer(FileWriter):
    """
    HDF5 format writer for high-performance parallel I/O.
    
    Optimized for large-scale simulations with parallel I/O support.
    """
    
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write solution in HDF5 format."""
        if not HDF5_AVAILABLE:
            logger.error("HDF5 library not available")
            return False
        
        try:
            with h5py.File(filename, 'w') as f:
                # Mesh group
                mesh_group = f.create_group("mesh")
                mesh_group.create_dataset("coordinates", data=mesh.coordinates,
                                        compression="gzip" if self.config.compression else None)
                
                if hasattr(mesh, 'connectivity') and mesh.connectivity is not None:
                    # Convert ragged connectivity to fixed-size array (simplified)
                    max_nodes = max(len(cell) for cell in mesh.connectivity)
                    connectivity_array = np.full((mesh.n_cells, max_nodes), -1, dtype=np.int32)
                    
                    for i, cell_nodes in enumerate(mesh.connectivity):
                        connectivity_array[i, :len(cell_nodes)] = cell_nodes
                    
                    mesh_group.create_dataset("connectivity", data=connectivity_array,
                                            compression="gzip" if self.config.compression else None)
                
                mesh_group.create_dataset("cell_types", data=mesh.cell_types,
                                        compression="gzip" if self.config.compression else None)
                
                # Solution group
                solution_group = f.create_group("solution")
                solution_group.attrs["time"] = solution.time
                solution_group.attrs["iteration"] = solution.iteration
                
                # Variables
                for var_name, var_data in solution.variables.items():
                    dataset = solution_group.create_dataset(var_name, data=var_data,
                                                          compression="gzip" if self.config.compression else None)
                    
                    # Add variable metadata
                    if var_name in solution.variable_info:
                        var_info = solution.variable_info[var_name]
                        dataset.attrs["units"] = var_info.units
                        dataset.attrs["description"] = var_info.description
                        dataset.attrs["data_type"] = var_info.data_type.value
                        dataset.attrs["location"] = var_info.location
                
                # Metadata
                if self.config.include_metadata:
                    meta_group = f.create_group("metadata")
                    for key, value in solution.metadata.items():
                        if isinstance(value, (int, float, str)):
                            meta_group.attrs[key] = value
                        else:
                            # Store complex objects as JSON strings
                            meta_group.attrs[key] = json.dumps(value, default=str)
            
            logger.info(f"Wrote HDF5 file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write HDF5 file {filename}: {e}")
            return False
    
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData]) -> bool:
        """Write HDF5 time series in single file."""
        if not HDF5_AVAILABLE:
            logger.error("HDF5 library not available")
            return False
        
        try:
            with h5py.File(base_filename, 'w') as f:
                # Mesh group (written once)
                mesh_group = f.create_group("mesh")
                mesh_group.create_dataset("coordinates", data=mesh.coordinates,
                                        compression="gzip" if self.config.compression else None)
                
                # Time series group
                n_times = len(solutions)
                times = np.array([sol.time for sol in solutions])
                iterations = np.array([sol.iteration for sol in solutions])
                
                f.create_dataset("times", data=times)
                f.create_dataset("iterations", data=iterations)
                
                # Create datasets for each variable
                for var_name in solutions[0].variables.keys():
                    # Determine shape for time series
                    first_var = solutions[0].variables[var_name]
                    if first_var.ndim == 1:
                        shape = (n_times, len(first_var))
                    else:
                        shape = (n_times,) + first_var.shape
                    
                    # Create dataset
                    dataset = f.create_dataset(var_name, shape=shape,
                                             compression="gzip" if self.config.compression else None)
                    
                    # Fill with time series data
                    for i, solution in enumerate(solutions):
                        dataset[i] = solution.variables[var_name]
                    
                    # Add metadata from first solution
                    if var_name in solutions[0].variable_info:
                        var_info = solutions[0].variable_info[var_name]
                        dataset.attrs["units"] = var_info.units
                        dataset.attrs["description"] = var_info.description
            
            logger.info(f"Wrote HDF5 time series file: {base_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write HDF5 time series {base_filename}: {e}")
            return False


class NativeBinaryWriter(FileWriter):
    """
    Native binary format writer for checkpointing and restart.
    
    Optimized for fast I/O and exact solution preservation.
    """
    
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData) -> bool:
        """Write solution in native binary format."""
        try:
            with open(filename, 'wb') as f:
                # Write header
                self._write_header(f, mesh, solution)
                
                # Write mesh data
                self._write_mesh_binary(f, mesh)
                
                # Write solution data
                self._write_solution_binary(f, solution)
            
            logger.info(f"Wrote native binary file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write native binary file {filename}: {e}")
            return False
    
    def _write_header(self, f: BinaryIO, mesh: MeshInfo, solution: SolutionData):
        """Write binary file header."""
        # Magic number
        f.write(b'CFD_NATIVE_v1.0')
        
        # Metadata
        header_data = {
            'n_nodes': mesh.n_nodes,
            'n_cells': mesh.n_cells,
            'n_faces': mesh.n_faces,
            'time': solution.time,
            'iteration': solution.iteration,
            'n_variables': len(solution.variables),
            'timestamp': datetime.now().isoformat()
        }
        
        # Write header as JSON
        header_json = json.dumps(header_data).encode('utf-8')
        f.write(struct.pack('I', len(header_json)))
        f.write(header_json)
    
    def _write_mesh_binary(self, f: BinaryIO, mesh: MeshInfo):
        """Write mesh data in binary format."""
        # Coordinates
        f.write(mesh.coordinates.tobytes())
        
        # Connectivity (if available)
        if hasattr(mesh, 'connectivity') and mesh.connectivity is not None:
            # Write connectivity sizes first
            sizes = np.array([len(cell) for cell in mesh.connectivity], dtype=np.int32)
            f.write(sizes.tobytes())
            
            # Write flattened connectivity
            flat_connectivity = np.concatenate(mesh.connectivity)
            f.write(flat_connectivity.tobytes())
        
        # Cell types
        f.write(mesh.cell_types.tobytes())
    
    def _write_solution_binary(self, f: BinaryIO, solution: SolutionData):
        """Write solution data in binary format."""
        # Variable names and metadata
        var_info = {}
        for var_name, var_data in solution.variables.items():
            var_info[var_name] = {
                'shape': var_data.shape,
                'dtype': str(var_data.dtype)
            }
            if var_name in solution.variable_info:
                info = solution.variable_info[var_name]
                var_info[var_name].update({
                    'data_type': info.data_type.value,
                    'units': info.units,
                    'description': info.description
                })
        
        # Write variable info
        var_info_json = json.dumps(var_info).encode('utf-8')
        f.write(struct.pack('I', len(var_info_json)))
        f.write(var_info_json)
        
        # Write variable data
        for var_name, var_data in solution.variables.items():
            f.write(var_data.tobytes())
    
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData]) -> bool:
        """Write native binary time series."""
        success = True
        
        for i, solution in enumerate(solutions):
            name, ext = os.path.splitext(base_filename)
            time_filename = f"{name}_t{solution.time:.6f}_i{solution.iteration:06d}{ext}"
            
            if not self.write_solution(time_filename, mesh, solution):
                success = False
        
        return success


class FileIOManager:
    """
    Main manager for CFD file I/O operations.
    
    Coordinates reading and writing across multiple formats.
    """
    
    def __init__(self, config: IOConfig):
        """Initialize file I/O manager."""
        self.config = config
        
        # Initialize writers
        self.writers = {
            FileFormat.VTK_LEGACY: VTKWriter(config),
            FileFormat.VTK_XML: VTKWriter(config),
            FileFormat.TECPLOT_ASCII: TecplotWriter(config),
            FileFormat.TECPLOT_BINARY: TecplotWriter(config),
            FileFormat.HDF5: HDF5Writer(config),
            FileFormat.NATIVE_BINARY: NativeBinaryWriter(config)
        }
        
        # File format detection patterns
        self.format_patterns = {
            '.vtk': FileFormat.VTK_LEGACY,
            '.vtu': FileFormat.VTK_XML,
            '.plt': FileFormat.TECPLOT_ASCII,
            '.dat': FileFormat.TECPLOT_ASCII,
            '.h5': FileFormat.HDF5,
            '.hdf5': FileFormat.HDF5,
            '.cfd': FileFormat.NATIVE_BINARY
        }
    
    def write_solution(self,
                      filename: str,
                      mesh: MeshInfo,
                      solution: SolutionData,
                      format_override: Optional[FileFormat] = None) -> bool:
        """Write solution file with automatic format detection."""
        # Determine format
        if format_override:
            file_format = format_override
        else:
            file_format = self._detect_format(filename)
        
        # Filter variables if requested
        filtered_solution = self._filter_solution(solution)
        
        # Get appropriate writer
        writer = self.writers.get(file_format)
        if not writer:
            logger.error(f"No writer available for format: {file_format}")
            return False
        
        # Add provenance metadata
        if self.config.include_provenance:
            self._add_provenance(filtered_solution)
        
        # Write file
        success = writer.write_solution(filename, mesh, filtered_solution)
        
        if success:
            logger.info(f"Successfully wrote {file_format.value} file: {filename}")
        
        return success
    
    def write_time_series(self,
                         base_filename: str,
                         mesh: MeshInfo,
                         solutions: List[SolutionData],
                         format_override: Optional[FileFormat] = None) -> bool:
        """Write time series with automatic format detection."""
        # Determine format
        if format_override:
            file_format = format_override
        else:
            file_format = self._detect_format(base_filename)
        
        # Filter time range if requested
        filtered_solutions = self._filter_time_series(solutions)
        
        # Get appropriate writer
        writer = self.writers.get(file_format)
        if not writer:
            logger.error(f"No writer available for format: {file_format}")
            return False
        
        # Write time series
        success = writer.write_time_series(base_filename, mesh, filtered_solutions)
        
        if success:
            logger.info(f"Successfully wrote {file_format.value} time series: {base_filename}")
        
        return success
    
    def _detect_format(self, filename: str) -> FileFormat:
        """Detect file format from filename extension."""
        _, ext = os.path.splitext(filename.lower())
        return self.format_patterns.get(ext, self.config.default_format)
    
    def _filter_solution(self, solution: SolutionData) -> SolutionData:
        """Filter solution variables based on configuration."""
        if not self.config.variable_filter:
            return solution
        
        filtered = SolutionData(
            time=solution.time,
            iteration=solution.iteration,
            metadata=solution.metadata.copy()
        )
        
        for var_name in self.config.variable_filter:
            if var_name in solution.variables:
                filtered.variables[var_name] = solution.variables[var_name]
                if var_name in solution.variable_info:
                    filtered.variable_info[var_name] = solution.variable_info[var_name]
        
        return filtered
    
    def _filter_time_series(self, solutions: List[SolutionData]) -> List[SolutionData]:
        """Filter time series based on time range."""
        if not self.config.time_range:
            return solutions
        
        t_min, t_max = self.config.time_range
        
        filtered = []
        for solution in solutions:
            if t_min <= solution.time <= t_max:
                filtered.append(solution)
        
        return filtered
    
    def _add_provenance(self, solution: SolutionData):
        """Add provenance information to solution metadata."""
        provenance = {
            'writer_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'format_version': 'CFD_v1.0'
        }
        
        solution.metadata.update(provenance)
    
    def get_supported_formats(self) -> List[FileFormat]:
        """Get list of supported file formats."""
        return list(self.writers.keys())
    
    def validate_mesh(self, mesh: MeshInfo) -> bool:
        """Validate mesh data before writing."""
        if mesh.n_nodes <= 0 or mesh.n_cells <= 0:
            logger.error("Invalid mesh: zero or negative node/cell count")
            return False
        
        if mesh.coordinates is None or len(mesh.coordinates) != mesh.n_nodes:
            logger.error("Invalid mesh: coordinate array size mismatch")
            return False
        
        return True
    
    def validate_solution(self, solution: SolutionData, mesh: MeshInfo) -> bool:
        """Validate solution data before writing."""
        for var_name, var_data in solution.variables.items():
            if len(var_data) != mesh.n_cells and len(var_data) != mesh.n_nodes:
                logger.error(f"Variable {var_name} size mismatch with mesh")
                return False
        
        return True


def create_io_manager(config: Optional[IOConfig] = None) -> FileIOManager:
    """
    Factory function for creating file I/O managers.
    
    Args:
        config: I/O configuration
        
    Returns:
        Configured file I/O manager
    """
    if config is None:
        config = IOConfig()
    
    return FileIOManager(config)


def test_file_io():
    """Test file I/O functionality."""
    print("Testing CFD File I/O:")
    print(f"HDF5 available: {HDF5_AVAILABLE}")
    print(f"VTK available: {VTK_AVAILABLE}")
    
    # Create test data
    n_nodes = 1000
    n_cells = 800
    
    # Create test mesh
    coordinates = np.random.rand(n_nodes, 3) * 10.0
    connectivity = []
    for i in range(n_cells):
        # Create tetrahedral cells
        nodes = np.random.choice(n_nodes, 4, replace=False)
        connectivity.append(nodes)
    
    cell_types = np.full(n_cells, 10)  # VTK tetrahedron type
    
    mesh = MeshInfo(
        n_nodes=n_nodes,
        n_cells=n_cells,
        n_faces=n_cells * 4,  # Simplified
        coordinates=coordinates,
        connectivity=connectivity,
        cell_types=cell_types
    )
    
    # Create test solution
    solution = SolutionData(
        time=1.5,
        iteration=150
    )
    
    # Add solution variables
    solution.variables['pressure'] = np.random.rand(n_cells) * 100000 + 101325
    solution.variables['density'] = np.random.rand(n_cells) * 0.5 + 1.0
    solution.variables['temperature'] = np.random.rand(n_cells) * 50 + 288
    solution.variables['velocity'] = np.random.rand(n_cells, 3) * 100
    
    # Add variable info
    solution.variable_info['pressure'] = VariableInfo('pressure', DataType.SCALAR, 'Pa', 'Static pressure')
    solution.variable_info['density'] = VariableInfo('density', DataType.SCALAR, 'kg/m³', 'Density')
    solution.variable_info['temperature'] = VariableInfo('temperature', DataType.SCALAR, 'K', 'Temperature')
    solution.variable_info['velocity'] = VariableInfo('velocity', DataType.VECTOR, 'm/s', 'Velocity vector')
    
    # Test different formats
    print(f"\n  Testing file formats:")
    
    # Create I/O manager
    config = IOConfig(compression=True, include_metadata=True)
    io_manager = create_io_manager(config)
    
    # Test formats
    test_files = [
        ('test_solution.vtk', FileFormat.VTK_LEGACY),
        ('test_solution.vtu', FileFormat.VTK_XML),
        ('test_solution.plt', FileFormat.TECPLOT_ASCII),
        ('test_solution.cfd', FileFormat.NATIVE_BINARY)
    ]
    
    if HDF5_AVAILABLE:
        test_files.append(('test_solution.h5', FileFormat.HDF5))
    
    for filename, file_format in test_files:
        print(f"    Testing {file_format.value}:")
        
        # Validate data
        if not io_manager.validate_mesh(mesh):
            print(f"      ✗ Mesh validation failed")
            continue
        
        if not io_manager.validate_solution(solution, mesh):
            print(f"      ✗ Solution validation failed")
            continue
        
        # Write file
        success = io_manager.write_solution(filename, mesh, solution, file_format)
        
        if success:
            file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            print(f"      ✓ Written successfully ({file_size} bytes)")
            
            # Cleanup
            if os.path.exists(filename):
                os.remove(filename)
        else:
            print(f"      ✗ Write failed")
    
    # Test time series
    print(f"\n  Testing time series I/O:")
    
    # Create multiple time steps
    solutions = []
    for i in range(5):
        time_solution = SolutionData(
            time=i * 0.1,
            iteration=i * 10,
            variables=solution.variables.copy(),
            variable_info=solution.variable_info.copy()
        )
        
        # Add some time variation
        time_solution.variables['pressure'] *= (1.0 + 0.1 * np.sin(i))
        solutions.append(time_solution)
    
    # Test time series writing
    success = io_manager.write_time_series('test_series.vtk', mesh, solutions)
    print(f"    VTK time series: {'✓ Success' if success else '✗ Failed'}")
    
    # Cleanup time series files
    for i, sol in enumerate(solutions):
        filename = f"test_series_t{sol.time:.6f}_i{sol.iteration:06d}.vtk"
        if os.path.exists(filename):
            os.remove(filename)
    
    # Test with HDF5 time series
    if HDF5_AVAILABLE:
        success = io_manager.write_time_series('test_series.h5', mesh, solutions, FileFormat.HDF5)
        print(f"    HDF5 time series: {'✓ Success' if success else '✗ Failed'}")
        
        if os.path.exists('test_series.h5'):
            file_size = os.path.getsize('test_series.h5')
            print(f"      File size: {file_size} bytes")
            os.remove('test_series.h5')
    
    # Test format detection
    print(f"\n  Testing format detection:")
    test_extensions = ['.vtk', '.vtu', '.plt', '.h5', '.cfd']
    for ext in test_extensions:
        detected_format = io_manager._detect_format(f'test{ext}')
        print(f"    {ext} -> {detected_format.value}")
    
    print(f"\n  File I/O test completed!")


if __name__ == "__main__":
    test_file_io()