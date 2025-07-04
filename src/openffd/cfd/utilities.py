"""
CFD Utilities Module

This module provides utility classes and functions for CFD operations including:
- Mesh format conversion and processing
- Case setup and management
- Field data processing and analysis
- Post-processing and visualization utilities
- Parallel execution management
- Results extraction and formatting
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import psutil
import time

from .base import CFDConfig, CFDResults
from .openfoam import OpenFOAMConfig, SimulationResults, FieldData

logger = logging.getLogger(__name__)

@dataclass
class MeshConversionConfig:
    """Configuration for mesh format conversion."""
    input_format: str
    output_format: str
    scale_factor: float = 1.0
    unit_conversion: Optional[Dict[str, float]] = None
    merge_patches: bool = False
    patch_mapping: Dict[str, str] = field(default_factory=dict)
    quality_check: bool = True
    repair_mesh: bool = False

@dataclass
class PostProcessingConfig:
    """Configuration for post-processing operations."""
    extract_surfaces: bool = True
    surface_names: List[str] = field(default_factory=list)
    extract_lines: bool = False
    line_definitions: List[Dict[str, Any]] = field(default_factory=list)
    compute_forces: bool = True
    force_patches: List[str] = field(default_factory=list)
    field_statistics: bool = True
    statistical_fields: List[str] = field(default_factory=list)
    create_plots: bool = True
    export_vtk: bool = True
    export_csv: bool = True

class MeshConverter:
    """Utility for converting between different mesh formats."""
    
    def __init__(self):
        """Initialize mesh converter."""
        self.logger = logging.getLogger(f"{__name__}.MeshConverter")
        self.supported_formats = {
            'openfoam': ['.foam', '.points', '.faces', '.cells'],
            'vtk': ['.vtk', '.vtu', '.vtp'],
            'stl': ['.stl'],
            'cgns': ['.cgns'],
            'gmsh': ['.msh', '.geo'],
            'fluent': ['.cas', '.msh'],
            'tecplot': ['.plt', '.dat'],
            'nastran': ['.nas', '.bdf'],
            'abaqus': ['.inp']
        }
    
    def convert_mesh(self, input_path: Path, output_path: Path, 
                    config: MeshConversionConfig) -> bool:
        """Convert mesh from one format to another.
        
        Args:
            input_path: Input mesh file path
            output_path: Output mesh file path
            config: Conversion configuration
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Converting mesh from {config.input_format} to {config.output_format}")
            
            # Validate input file
            if not input_path.exists():
                self.logger.error(f"Input mesh file not found: {input_path}")
                return False
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform conversion based on formats
            if config.input_format == 'openfoam' and config.output_format == 'vtk':
                return self._convert_openfoam_to_vtk(input_path, output_path, config)
            elif config.input_format == 'vtk' and config.output_format == 'openfoam':
                return self._convert_vtk_to_openfoam(input_path, output_path, config)
            elif config.input_format == 'stl' and config.output_format == 'openfoam':
                return self._convert_stl_to_openfoam(input_path, output_path, config)
            elif config.input_format == 'gmsh' and config.output_format == 'openfoam':
                return self._convert_gmsh_to_openfoam(input_path, output_path, config)
            else:
                # Use universal converter (if available)
                return self._universal_conversion(input_path, output_path, config)
            
        except Exception as e:
            self.logger.error(f"Mesh conversion failed: {e}")
            return False
    
    def _convert_openfoam_to_vtk(self, input_path: Path, output_path: Path, 
                               config: MeshConversionConfig) -> bool:
        """Convert OpenFOAM mesh to VTK format."""
        try:
            # Use foamToVTK utility
            cmd = ['foamToVTK', '-case', str(input_path)]
            if config.extract_surfaces:
                cmd.append('-surfaceFields')
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=input_path)
            
            if result.returncode == 0:
                # Move VTK files to output location
                vtk_dir = input_path / "VTK"
                if vtk_dir.exists():
                    shutil.move(str(vtk_dir), str(output_path))
                    return True
            
            self.logger.error(f"foamToVTK failed: {result.stderr}")
            return False
            
        except Exception as e:
            self.logger.error(f"OpenFOAM to VTK conversion failed: {e}")
            return False
    
    def _convert_vtk_to_openfoam(self, input_path: Path, output_path: Path,
                               config: MeshConversionConfig) -> bool:
        """Convert VTK mesh to OpenFOAM format."""
        # This would require a custom VTK reader and OpenFOAM writer
        self.logger.warning("VTK to OpenFOAM conversion not fully implemented")
        return False
    
    def _convert_stl_to_openfoam(self, input_path: Path, output_path: Path,
                               config: MeshConversionConfig) -> bool:
        """Convert STL mesh to OpenFOAM format using snappyHexMesh."""
        try:
            # Create case directory
            case_dir = output_path
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy STL file to triSurface directory
            tri_surface_dir = case_dir / "constant" / "triSurface"
            tri_surface_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, tri_surface_dir / input_path.name)
            
            # Generate snappyHexMeshDict
            self._generate_snappy_hex_mesh_dict(case_dir, input_path.name, config)
            
            # Run blockMesh to create background mesh
            self._run_block_mesh(case_dir)
            
            # Run snappyHexMesh
            self._run_snappy_hex_mesh(case_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"STL to OpenFOAM conversion failed: {e}")
            return False
    
    def _convert_gmsh_to_openfoam(self, input_path: Path, output_path: Path,
                                config: MeshConversionConfig) -> bool:
        """Convert GMSH mesh to OpenFOAM format."""
        try:
            # Use gmshToFoam utility
            cmd = ['gmshToFoam', str(input_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_path)
            
            if result.returncode == 0:
                self.logger.info("GMSH to OpenFOAM conversion successful")
                return True
            else:
                self.logger.error(f"gmshToFoam failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"GMSH to OpenFOAM conversion failed: {e}")
            return False
    
    def _universal_conversion(self, input_path: Path, output_path: Path,
                            config: MeshConversionConfig) -> bool:
        """Universal mesh conversion using external tools."""
        # This could use tools like meshio, or other converters
        try:
            import meshio
            
            # Read mesh
            mesh = meshio.read(input_path)
            
            # Apply scaling if specified
            if config.scale_factor != 1.0:
                mesh.points *= config.scale_factor
            
            # Write mesh
            meshio.write(output_path, mesh)
            
            return True
            
        except ImportError:
            self.logger.warning("meshio not available for universal conversion")
            return False
        except Exception as e:
            self.logger.error(f"Universal conversion failed: {e}")
            return False
    
    def _generate_snappy_hex_mesh_dict(self, case_dir: Path, stl_name: str,
                                     config: MeshConversionConfig):
        """Generate snappyHexMeshDict for STL meshing."""
        dict_path = case_dir / "system" / "snappyHexMeshDict"
        
        with open(dict_path, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2112                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
""")
            f.write(f"    {stl_name.replace('.stl', '')}\n")
            f.write("    {\n")
            f.write("        type triSurfaceMesh;\n")
            f.write(f"        file \"{stl_name}\";\n")
            f.write("    }\n")
            f.write("}\n\n")
            
            # Add castellated mesh controls
            f.write("""castellatedMeshControls
{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    
    features
    (
    );
    
    refinementSurfaces
    {
""")
            f.write(f"        {stl_name.replace('.stl', '')}\n")
            f.write("        {\n")
            f.write("            level (1 1);\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("}\n\n")
            
            # Add snap controls
            f.write("""snapControls
{
    nSmoothPatch 3;
    tolerance 4.0;
    nSolveIter 30;
    nRelaxIter 5;
}

addLayersControls
{
    relativeSizes true;
    layers {};
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    relaxed {};
}

debug 0;
mergeTolerance 1e-6;

// ************************************************************************* //
""")
    
    def _run_block_mesh(self, case_dir: Path):
        """Run blockMesh to create background mesh."""
        # First create blockMeshDict
        self._generate_block_mesh_dict(case_dir)
        
        cmd = ['blockMesh']
        result = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"blockMesh failed: {result.stderr}")
    
    def _generate_block_mesh_dict(self, case_dir: Path):
        """Generate blockMeshDict for background mesh."""
        dict_path = case_dir / "system" / "blockMeshDict"
        
        with open(dict_path, 'w') as f:
            f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    (-5 -5 -5)
    ( 5 -5 -5)
    ( 5  5 -5)
    (-5  5 -5)
    (-5 -5  5)
    ( 5 -5  5)
    ( 5  5  5)
    (-5  5  5)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1)
);

edges ();

boundary
(
    movingWall
    {
        type wall;
        faces ((3 7 6 2));
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
        );
    }
    frontAndBack
    {
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

// ************************************************************************* //
""")
    
    def _run_snappy_hex_mesh(self, case_dir: Path):
        """Run snappyHexMesh."""
        cmd = ['snappyHexMesh', '-overwrite']
        result = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"snappyHexMesh failed: {result.stderr}")

class CaseGenerator:
    """Utility for generating CFD case setups."""
    
    def __init__(self):
        """Initialize case generator."""
        self.logger = logging.getLogger(f"{__name__}.CaseGenerator")
    
    def generate_case(self, config: OpenFOAMConfig, template_dir: Optional[Path] = None) -> bool:
        """Generate complete CFD case setup.
        
        Args:
            config: OpenFOAM configuration
            template_dir: Template case directory
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Generating case: {config.case_directory}")
            
            # Copy template if provided
            if template_dir and template_dir.exists():
                shutil.copytree(template_dir, config.case_directory, dirs_exist_ok=True)
            else:
                # Create case from scratch
                config.case_directory.mkdir(parents=True, exist_ok=True)
                (config.case_directory / "0").mkdir(exist_ok=True)
                (config.case_directory / "constant").mkdir(exist_ok=True)
                (config.case_directory / "system").mkdir(exist_ok=True)
            
            # Generate all required files
            self._generate_all_files(config)
            
            self.logger.info("Case generation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Case generation failed: {e}")
            return False
    
    def _generate_all_files(self, config: OpenFOAMConfig):
        """Generate all required OpenFOAM files."""
        # This would implement generation of all OpenFOAM case files
        # including boundary conditions, initial conditions, etc.
        pass
    
    def validate_case(self, case_dir: Path) -> Dict[str, bool]:
        """Validate OpenFOAM case setup.
        
        Args:
            case_dir: Case directory
            
        Returns:
            Validation results
        """
        validation_results = {}
        
        # Check required directories
        validation_results['has_0_dir'] = (case_dir / "0").exists()
        validation_results['has_constant_dir'] = (case_dir / "constant").exists()
        validation_results['has_system_dir'] = (case_dir / "system").exists()
        
        # Check required files
        validation_results['has_control_dict'] = (case_dir / "system" / "controlDict").exists()
        validation_results['has_fv_schemes'] = (case_dir / "system" / "fvSchemes").exists()
        validation_results['has_fv_solution'] = (case_dir / "system" / "fvSolution").exists()
        
        # Check mesh
        polymesh_dir = case_dir / "constant" / "polyMesh"
        validation_results['has_mesh'] = (
            polymesh_dir.exists() and
            (polymesh_dir / "points").exists() and
            (polymesh_dir / "faces").exists() and
            (polymesh_dir / "cells").exists()
        )
        
        # Check boundary conditions
        u_file = case_dir / "0" / "U"
        p_file = case_dir / "0" / "p"
        validation_results['has_u_bc'] = u_file.exists()
        validation_results['has_p_bc'] = p_file.exists()
        
        return validation_results

class FieldProcessor:
    """Utility for processing CFD field data."""
    
    def __init__(self):
        """Initialize field processor."""
        self.logger = logging.getLogger(f"{__name__}.FieldProcessor")
    
    def extract_field_data(self, case_dir: Path, field_name: str, 
                          time_value: Optional[str] = None) -> Optional[FieldData]:
        """Extract field data from OpenFOAM case.
        
        Args:
            case_dir: Case directory
            field_name: Field name (e.g., 'U', 'p', 'k')
            time_value: Time directory (latest if None)
            
        Returns:
            Field data or None if failed
        """
        try:
            # Find time directory
            if time_value is None:
                time_dirs = [d for d in case_dir.iterdir() 
                           if d.is_dir() and d.name.replace('.', '').isdigit()]
                if not time_dirs:
                    return None
                time_dir = max(time_dirs, key=lambda x: float(x.name))
            else:
                time_dir = case_dir / time_value
            
            field_file = time_dir / field_name
            if not field_file.exists():
                return None
            
            # Parse OpenFOAM field file
            field_data = self._parse_openfoam_field(field_file)
            
            return field_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract field data: {e}")
            return None
    
    def _parse_openfoam_field(self, field_file: Path) -> FieldData:
        """Parse OpenFOAM field file."""
        # This is a simplified implementation
        # Real implementation would parse the complete OpenFOAM field format
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Extract basic information
        field_name = field_file.name
        
        # Extract dimensions
        dim_match = re.search(r'dimensions\s+\[(.*?)\]', content)
        dimensions = dim_match.group(1) if dim_match else "unknown"
        
        # Extract internal field (simplified)
        internal_match = re.search(r'internalField\s+uniform\s+([^;]+);', content)
        if internal_match:
            value_str = internal_match.group(1).strip()
            if value_str.startswith('(') and value_str.endswith(')'):
                # Vector field
                values = np.array([float(x) for x in value_str[1:-1].split()])
            else:
                # Scalar field
                values = np.array([float(value_str)])
        else:
            values = np.array([])
        
        return FieldData(
            name=field_name,
            values=values,
            locations=np.array([]),  # Would extract mesh points
            dimensions=f"[{dimensions}]"
        )
    
    def compute_field_statistics(self, field_data: FieldData) -> Dict[str, float]:
        """Compute statistical measures for field data."""
        if len(field_data.values) == 0:
            return {}
        
        stats = {
            'min': float(np.min(field_data.values)),
            'max': float(np.max(field_data.values)),
            'mean': float(np.mean(field_data.values)),
            'std': float(np.std(field_data.values)),
            'rms': float(np.sqrt(np.mean(field_data.values**2))),
            'median': float(np.median(field_data.values))
        }
        
        if field_data.values.ndim > 1:
            # Vector field statistics
            magnitude = np.linalg.norm(field_data.values, axis=-1)
            stats.update({
                'magnitude_min': float(np.min(magnitude)),
                'magnitude_max': float(np.max(magnitude)),
                'magnitude_mean': float(np.mean(magnitude)),
                'magnitude_std': float(np.std(magnitude))
            })
        
        return stats

class PostProcessor:
    """Comprehensive post-processing utility."""
    
    def __init__(self, config: PostProcessingConfig = None):
        """Initialize post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessingConfig()
        self.logger = logging.getLogger(f"{__name__}.PostProcessor")
    
    def process_results(self, case_dir: Path) -> Dict[str, Any]:
        """Process CFD results comprehensively.
        
        Args:
            case_dir: Case directory
            
        Returns:
            Processing results
        """
        results = {}
        
        try:
            # Extract surface data
            if self.config.extract_surfaces:
                results['surface_data'] = self._extract_surface_data(case_dir)
            
            # Extract line data
            if self.config.extract_lines:
                results['line_data'] = self._extract_line_data(case_dir)
            
            # Compute forces and moments
            if self.config.compute_forces:
                results['forces'] = self._extract_forces(case_dir)
            
            # Compute field statistics
            if self.config.field_statistics:
                results['field_statistics'] = self._compute_field_statistics(case_dir)
            
            # Create visualizations
            if self.config.create_plots:
                results['plots'] = self._create_plots(case_dir, results)
            
            # Export data
            if self.config.export_vtk:
                self._export_vtk(case_dir)
            
            if self.config.export_csv:
                self._export_csv(case_dir, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            return {}
    
    def _extract_surface_data(self, case_dir: Path) -> Dict[str, Any]:
        """Extract surface data from CFD results."""
        surface_data = {}
        
        for surface_name in self.config.surface_names:
            # Use OpenFOAM utilities to extract surface data
            try:
                cmd = ['postProcess', '-func', f'surfaces', '-case', str(case_dir)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Process extracted surface files
                    surface_data[surface_name] = self._process_surface_files(case_dir, surface_name)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract surface {surface_name}: {e}")
        
        return surface_data
    
    def _extract_line_data(self, case_dir: Path) -> Dict[str, Any]:
        """Extract line data from CFD results."""
        line_data = {}
        
        for line_def in self.config.line_definitions:
            try:
                # Generate line sampling
                line_name = line_def.get('name', 'line')
                start_point = line_def.get('start', [0, 0, 0])
                end_point = line_def.get('end', [1, 1, 1])
                n_points = line_def.get('nPoints', 100)
                
                # Create sample dictionary
                self._create_sample_dict(case_dir, line_name, start_point, end_point, n_points)
                
                # Run postProcess
                cmd = ['postProcess', '-func', 'sample', '-case', str(case_dir)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    line_data[line_name] = self._process_line_files(case_dir, line_name)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract line data: {e}")
        
        return line_data
    
    def _extract_forces(self, case_dir: Path) -> Dict[str, Any]:
        """Extract force and moment data."""
        forces_data = {}
        
        # Look for force coefficient files
        postprocessing_dir = case_dir / "postProcessing"
        if postprocessing_dir.exists():
            for force_dir in postprocessing_dir.glob("force*"):
                if force_dir.is_dir():
                    force_name = force_dir.name
                    
                    # Find latest time directory
                    time_dirs = [d for d in force_dir.iterdir() if d.is_dir()]
                    if time_dirs:
                        latest_time = max(time_dirs, key=lambda x: float(x.name))
                        
                        # Read coefficient files
                        coeff_file = latest_time / "coefficient.dat"
                        if coeff_file.exists():
                            forces_data[force_name] = self._read_coefficient_file(coeff_file)
        
        return forces_data
    
    def _compute_field_statistics(self, case_dir: Path) -> Dict[str, Any]:
        """Compute statistics for specified fields."""
        field_stats = {}
        
        field_processor = FieldProcessor()
        
        for field_name in self.config.statistical_fields:
            field_data = field_processor.extract_field_data(case_dir, field_name)
            if field_data:
                field_stats[field_name] = field_processor.compute_field_statistics(field_data)
        
        return field_stats
    
    def _create_plots(self, case_dir: Path, results: Dict[str, Any]) -> Dict[str, Path]:
        """Create visualization plots."""
        plots = {}
        
        try:
            import matplotlib.pyplot as plt
            
            plot_dir = case_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            # Create convergence plots
            log_files = list(case_dir.glob("*.log"))
            if log_files:
                residual_data = self._parse_residuals(log_files[0])
                if residual_data:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for field, residuals in residual_data.items():
                        ax.semilogy(residuals, label=field)
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Residual')
                    ax.legend()
                    ax.grid(True)
                    plt.title('Convergence History')
                    
                    plot_path = plot_dir / "convergence.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plots['convergence'] = plot_path
            
            # Create force coefficient plots
            if 'forces' in results:
                for force_name, force_data in results['forces'].items():
                    if 'time' in force_data and 'cd' in force_data:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                        
                        ax1.plot(force_data['time'], force_data['cd'], label='Cd')
                        if 'cl' in force_data:
                            ax1.plot(force_data['time'], force_data['cl'], label='Cl')
                        ax1.set_ylabel('Force Coefficients')
                        ax1.legend()
                        ax1.grid(True)
                        
                        if 'cm' in force_data:
                            ax2.plot(force_data['time'], force_data['cm'], label='Cm')
                            ax2.set_ylabel('Moment Coefficient')
                            ax2.legend()
                            ax2.grid(True)
                        
                        ax2.set_xlabel('Time')
                        plt.title(f'Force Coefficients - {force_name}')
                        
                        plot_path = plot_dir / f"forces_{force_name}.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        plots[f'forces_{force_name}'] = plot_path
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Failed to create plots: {e}")
        
        return plots
    
    def _export_vtk(self, case_dir: Path):
        """Export results to VTK format."""
        try:
            cmd = ['foamToVTK', '-case', str(case_dir)]
            subprocess.run(cmd, cwd=case_dir, check=True, capture_output=True)
            self.logger.info("VTK export completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VTK export failed: {e}")
    
    def _export_csv(self, case_dir: Path, results: Dict[str, Any]):
        """Export results to CSV format."""
        csv_dir = case_dir / "csv_export"
        csv_dir.mkdir(exist_ok=True)
        
        # Export force data
        if 'forces' in results:
            for force_name, force_data in results['forces'].items():
                df = pd.DataFrame(force_data)
                csv_path = csv_dir / f"forces_{force_name}.csv"
                df.to_csv(csv_path, index=False)
        
        # Export field statistics
        if 'field_statistics' in results:
            stats_data = []
            for field_name, stats in results['field_statistics'].items():
                for stat_name, value in stats.items():
                    stats_data.append({
                        'field': field_name,
                        'statistic': stat_name,
                        'value': value
                    })
            
            if stats_data:
                df = pd.DataFrame(stats_data)
                csv_path = csv_dir / "field_statistics.csv"
                df.to_csv(csv_path, index=False)
    
    def _process_surface_files(self, case_dir: Path, surface_name: str) -> Dict[str, Any]:
        """Process extracted surface files."""
        # Placeholder implementation
        return {}
    
    def _process_line_files(self, case_dir: Path, line_name: str) -> Dict[str, Any]:
        """Process extracted line files."""
        # Placeholder implementation
        return {}
    
    def _create_sample_dict(self, case_dir: Path, line_name: str, 
                          start_point: List[float], end_point: List[float], n_points: int):
        """Create sampling dictionary for line extraction."""
        sample_dict_path = case_dir / "system" / "sampleDict"
        
        with open(sample_dict_path, 'w') as f:
            f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object sampleDict; }}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type sets;

interpolationScheme cellPoint;

setFormat raw;

sets
(
    {line_name}
    {{
        type uniform;
        axis distance;
        start ({start_point[0]} {start_point[1]} {start_point[2]});
        end ({end_point[0]} {end_point[1]} {end_point[2]});
        nPoints {n_points};
    }}
);

fields (U p k epsilon omega);

// ************************************************************************* //
""")
    
    def _read_coefficient_file(self, coeff_file: Path) -> Dict[str, List[float]]:
        """Read force coefficient file."""
        try:
            data = np.loadtxt(coeff_file, skiprows=1)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Typical columns: time, cd, cl, cm, cd_pressure, cd_viscous, cl_pressure, cl_viscous
            result = {
                'time': data[:, 0].tolist(),
                'cd': data[:, 1].tolist() if data.shape[1] > 1 else [],
                'cl': data[:, 2].tolist() if data.shape[1] > 2 else [],
                'cm': data[:, 3].tolist() if data.shape[1] > 3 else []
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to read coefficient file: {e}")
            return {}
    
    def _parse_residuals(self, log_file: Path) -> Dict[str, List[float]]:
        """Parse residual data from log file."""
        residual_data = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract residual patterns
            residual_pattern = re.compile(
                r'Solving for (\w+), Initial residual = ([\d.e-]+)',
                re.MULTILINE
            )
            
            for match in residual_pattern.finditer(content):
                field = match.group(1)
                residual = float(match.group(2))
                
                if field not in residual_data:
                    residual_data[field] = []
                residual_data[field].append(residual)
            
        except Exception as e:
            self.logger.error(f"Failed to parse residuals: {e}")
        
        return residual_data

class ParallelManager:
    """Utility for managing parallel CFD execution."""
    
    def __init__(self):
        """Initialize parallel manager."""
        self.logger = logging.getLogger(f"{__name__}.ParallelManager")
    
    def optimize_decomposition(self, case_dir: Path, n_processors: int, 
                             mesh_size: Optional[int] = None) -> Dict[str, Any]:
        """Optimize domain decomposition for parallel execution.
        
        Args:
            case_dir: Case directory
            n_processors: Number of processors
            mesh_size: Estimated mesh size
            
        Returns:
            Decomposition configuration
        """
        # Calculate optimal decomposition
        factors = self._factorize(n_processors)
        
        # Try to make decomposition as cubic as possible
        if len(factors) == 1:
            nx = ny = nz = factors[0]
        elif len(factors) == 2:
            nx, ny = factors
            nz = 1
        else:
            nx, ny, nz = factors[0], factors[1], factors[2]
            for i in range(3, len(factors)):
                # Distribute remaining factors
                if nx <= ny <= nz:
                    nx *= factors[i]
                elif ny <= nz:
                    ny *= factors[i]
                else:
                    nz *= factors[i]
        
        decomp_config = {
            'numberOfSubdomains': n_processors,
            'method': 'simple',
            'simpleCoeffs': {
                'n': [nx, ny, nz],
                'delta': 0.001
            }
        }
        
        return decomp_config
    
    def _factorize(self, n: int) -> List[int]:
        """Factorize number into approximately equal factors."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        
        # Try to balance factors
        while len(factors) > 3:
            factors.sort()
            factors[0] *= factors[1]
            factors.pop(1)
        
        return factors
    
    def monitor_parallel_performance(self, case_dir: Path) -> Dict[str, Any]:
        """Monitor parallel execution performance."""
        performance_data = {}
        
        # Check processor directories
        proc_dirs = list(case_dir.glob("processor*"))
        performance_data['n_processors'] = len(proc_dirs)
        
        # Analyze load balancing
        if proc_dirs:
            cell_counts = []
            for proc_dir in proc_dirs:
                owner_file = proc_dir / "constant" / "polyMesh" / "owner"
                if owner_file.exists():
                    with open(owner_file, 'r') as f:
                        content = f.read()
                        # Extract cell count (simplified)
                        lines = content.strip().split('\n')
                        try:
                            cell_count = int(lines[-2])  # Typical location
                            cell_counts.append(cell_count)
                        except:
                            pass
            
            if cell_counts:
                performance_data['cell_counts'] = cell_counts
                performance_data['load_balance'] = {
                    'min_cells': min(cell_counts),
                    'max_cells': max(cell_counts),
                    'mean_cells': np.mean(cell_counts),
                    'imbalance': (max(cell_counts) - min(cell_counts)) / np.mean(cell_counts)
                }
        
        return performance_data

class ResultsExtractor:
    """Utility for extracting and formatting CFD results."""
    
    def __init__(self):
        """Initialize results extractor."""
        self.logger = logging.getLogger(f"{__name__}.ResultsExtractor")
        self.field_processor = FieldProcessor()
        self.post_processor = PostProcessor()
    
    def extract_all_results(self, case_dir: Path) -> Dict[str, Any]:
        """Extract all available results from CFD case.
        
        Args:
            case_dir: Case directory
            
        Returns:
            Complete results dictionary
        """
        results = {}
        
        try:
            # Extract basic case information
            results['case_info'] = self._extract_case_info(case_dir)
            
            # Extract field data
            results['fields'] = self._extract_all_fields(case_dir)
            
            # Extract force coefficients
            results['forces'] = self._extract_all_forces(case_dir)
            
            # Extract residual history
            results['residuals'] = self._extract_residual_history(case_dir)
            
            # Extract mesh information
            results['mesh'] = self._extract_mesh_info(case_dir)
            
            # Post-process data
            results['post_processing'] = self.post_processor.process_results(case_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Results extraction failed: {e}")
            return {}
    
    def _extract_case_info(self, case_dir: Path) -> Dict[str, Any]:
        """Extract basic case information."""
        info = {'case_directory': str(case_dir)}
        
        # Read controlDict
        control_dict = case_dir / "system" / "controlDict"
        if control_dict.exists():
            with open(control_dict, 'r') as f:
                content = f.read()
                
                # Extract application
                app_match = re.search(r'application\s+(\w+);', content)
                if app_match:
                    info['solver'] = app_match.group(1)
                
                # Extract time settings
                end_time_match = re.search(r'endTime\s+([\d.e-]+);', content)
                if end_time_match:
                    info['end_time'] = float(end_time_match.group(1))
        
        return info
    
    def _extract_all_fields(self, case_dir: Path) -> Dict[str, FieldData]:
        """Extract all field data from latest time."""
        fields = {}
        
        # Find latest time directory
        time_dirs = [d for d in case_dir.iterdir() 
                   if d.is_dir() and d.name.replace('.', '').isdigit()]
        if not time_dirs:
            return fields
        
        latest_time_dir = max(time_dirs, key=lambda x: float(x.name))
        
        # Extract all field files
        for field_file in latest_time_dir.iterdir():
            if field_file.is_file() and field_file.name not in ['uniform']:
                field_data = self.field_processor.extract_field_data(
                    case_dir, field_file.name, latest_time_dir.name
                )
                if field_data:
                    fields[field_file.name] = field_data
        
        return fields
    
    def _extract_all_forces(self, case_dir: Path) -> Dict[str, Any]:
        """Extract all force and moment data."""
        return self.post_processor._extract_forces(case_dir)
    
    def _extract_residual_history(self, case_dir: Path) -> Dict[str, List[float]]:
        """Extract complete residual history."""
        log_files = list(case_dir.glob("*.log"))
        if log_files:
            return self.post_processor._parse_residuals(log_files[0])
        return {}
    
    def _extract_mesh_info(self, case_dir: Path) -> Dict[str, Any]:
        """Extract mesh information."""
        mesh_info = {}
        
        polymesh_dir = case_dir / "constant" / "polyMesh"
        if polymesh_dir.exists():
            # Count points
            points_file = polymesh_dir / "points"
            if points_file.exists():
                with open(points_file, 'r') as f:
                    content = f.read()
                    # Extract point count (simplified)
                    lines = content.strip().split('\n')
                    for line in lines:
                        if line.strip().isdigit():
                            mesh_info['n_points'] = int(line.strip())
                            break
            
            # Count faces
            faces_file = polymesh_dir / "faces"
            if faces_file.exists():
                with open(faces_file, 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    for line in lines:
                        if line.strip().isdigit():
                            mesh_info['n_faces'] = int(line.strip())
                            break
            
            # Count cells
            owner_file = polymesh_dir / "owner"
            if owner_file.exists():
                with open(owner_file, 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    try:
                        mesh_info['n_cells'] = int(lines[-2])
                    except:
                        pass
        
        return mesh_info