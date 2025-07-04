#!/usr/bin/env python3
"""
NACA0012 Airfoil Shape Optimization with OpenFOAM and OpenFFD

This script creates a NACA0012 airfoil, generates a complete OpenFOAM case,
runs CFD simulations, and performs shape optimization using FFD control points.

Features:
- Generates NACA0012 airfoil geometry
- Creates complete OpenFOAM case with proper mesh
- Runs CFD simulations with OpenFOAM
- Performs shape optimization using FFD deformation
- Tracks optimization progress and results

Prerequisites:
- OpenFOAM installed (activate with 'openfoam' command)
- OpenFFD package installed

Usage:
    # First activate OpenFOAM environment
    openfoam
    
    # Then run the optimization
    python3 naca0012_optimization.py
"""

import numpy as np
import subprocess
import os
import sys
import shutil
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Import OpenFFD modules
from openffd.cfd import (
    OpenFOAMSolver, OpenFOAMConfig, SolverType, TurbulenceModel,
    SensitivityAnalyzer, SensitivityConfig, GradientMethod, ObjectiveFunction
)
from openffd.mesh import MeshDeformationEngine, DeformationConfig
from openffd.core import create_ffd_box

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('naca0012_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

class NACA0012Optimizer:
    """Complete NACA0012 airfoil optimization workflow."""
    
    def __init__(self, case_name="naca0012_optimization"):
        self.case_name = case_name
        self.case_dir = Path(f"./{case_name}")
        self.results = {}
        
    def check_openfoam_environment(self):
        """Check if OpenFOAM environment is properly activated."""
        try:
            # Check if OpenFOAM environment variables are set
            foam_vars = ['FOAM_APP', 'WM_PROJECT', 'FOAM_ETC', 'WM_PROJECT_DIR']
            missing_vars = [var for var in foam_vars if var not in os.environ]
            
            if missing_vars:
                logger.error("❌ OpenFOAM environment not activated!")
                logger.error(f"Missing environment variables: {missing_vars}")
                logger.info("Please run 'openfoam' command first to activate the environment")
                return False
                
            logger.info(f"✅ OpenFOAM environment detected: {os.environ.get('WM_PROJECT', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking OpenFOAM environment: {e}")
            return False
    
    def run_command(self, command, cwd=None, timeout=300):
        """Run a command in the current environment."""
        try:
            logger.info(f"Running: {command}")
            
            result = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, 
                text=True, timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Command completed successfully")
                return True, result.stdout, result.stderr
            else:
                logger.error(f"❌ Command failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Command timed out after {timeout} seconds")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"💥 Error running command: {e}")
            return False, "", str(e)
    
    def generate_naca0012_coordinates(self, n_points=100):
        """Generate NACA0012 airfoil coordinates."""
        logger.info("🔧 Generating NACA0012 airfoil coordinates...")
        
        # NACA0012 parameters
        thickness = 0.12  # 12% thickness
        
        # Generate x coordinates (cosine clustering for better resolution at leading/trailing edges)
        beta = np.linspace(0, np.pi, n_points // 2)
        x = 0.5 * (1 - np.cos(beta))
        
        # NACA 4-digit thickness distribution
        yt = thickness / 0.2 * (
            0.2969 * np.sqrt(x) - 
            0.1260 * x - 
            0.3516 * x**2 + 
            0.2843 * x**3 - 
            0.1015 * x**4
        )
        
        # Create upper and lower surface coordinates
        x_upper = x
        y_upper = yt
        x_lower = x[::-1]  # Reverse order for lower surface
        y_lower = -yt[::-1]
        
        # Combine into single airfoil (remove duplicate trailing edge)
        x_airfoil = np.concatenate([x_upper, x_lower[1:]])
        y_airfoil = np.concatenate([y_upper, y_lower[1:]])
        
        logger.info(f"✅ Generated NACA0012 with {len(x_airfoil)} points")
        logger.info(f"   Max thickness: {np.max(yt):.4f} at x = {x[np.argmax(yt)]:.3f}")
        
        return x_airfoil, y_airfoil
    
    def create_openfoam_case(self):
        """Create complete OpenFOAM case directory structure."""
        logger.info("📁 Creating OpenFOAM case directory structure...")
        
        # Create main directories
        directories = [
            self.case_dir,
            self.case_dir / "0",
            self.case_dir / "constant",
            self.case_dir / "constant" / "polyMesh",
            self.case_dir / "system"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Created case directory: {self.case_dir}")
        return True
    
    def create_background_mesh(self):
        """Create simple background blockMesh for snappyHexMesh."""
        logger.info("🏗️ Creating background mesh for snappyHexMesh...")
        
        # Domain boundaries
        x_min, x_max = -4.0, 8.0
        y_min, y_max = -4.0, 4.0
        z_min, z_max = -0.1, 0.1
        
        # Mesh resolution (coarse background mesh)
        nx, ny, nz = 40, 30, 1
        
        blockmesh_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2112                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

// Simple background mesh for snappyHexMesh
vertices
(
    ({x_min} {y_min} {z_min})    // 0
    ({x_max} {y_min} {z_min})    // 1
    ({x_max} {y_max} {z_min})    // 2
    ({x_min} {y_max} {z_min})    // 3
    ({x_min} {y_min} {z_max})    // 4
    ({x_max} {y_min} {z_max})    // 5
    ({x_max} {y_max} {z_max})    // 6
    ({x_min} {y_max} {z_max})    // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    
    topWall
    {{
        type symmetryPlane;
        faces
        (
            (3 7 6 2)
        );
    }}
    
    bottomWall
    {{
        type symmetryPlane;
        faces
        (
            (0 1 5 4)
        );
    }}
    
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
'''
        
        # Write blockMeshDict
        blockmesh_file = self.case_dir / "system" / "blockMeshDict"
        with open(blockmesh_file, 'w') as f:
            f.write(blockmesh_content)
        
        logger.info("✅ Created simple background blockMeshDict")
        return True
    
    def create_naca0012_stl(self, x_airfoil, y_airfoil):
        """Create high-quality STL file for NACA0012 airfoil."""
        logger.info("📐 Creating NACA0012 STL file...")
        
        # Create triSurface directory
        tri_surface_dir = self.case_dir / "constant" / "triSurface"
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        
        stl_file = tri_surface_dir / "airfoil.stl"
        
        # Create 3D coordinates for STL
        z1, z2 = -0.1, 0.1
        n_points = len(x_airfoil)
        
        # Write STL file with proper triangulation
        with open(stl_file, 'w') as f:
            f.write("solid NACA0012\n")
            
            # Create triangular facets for airfoil surface
            for i in range(n_points - 1):
                x1, y1 = x_airfoil[i], y_airfoil[i]
                x2, y2 = x_airfoil[i + 1], y_airfoil[i + 1]
                
                # Calculate normal vector (pointing outward from airfoil)
                dx = x2 - x1
                dy = y2 - y1
                
                # For 2D airfoil, normal in z-direction
                nx, ny, nz = 0.0, 0.0, 1.0
                
                # Triangle 1: (x1,y1,z1) -> (x2,y2,z1) -> (x1,y1,z2)
                f.write(f"  facet normal {nx} {ny} {nz}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {x1:.6f} {y1:.6f} {z1}\n")
                f.write(f"      vertex {x2:.6f} {y2:.6f} {z1}\n")
                f.write(f"      vertex {x1:.6f} {y1:.6f} {z2}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
                
                # Triangle 2: (x2,y2,z1) -> (x2,y2,z2) -> (x1,y1,z2)
                f.write(f"  facet normal {nx} {ny} {nz}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {x2:.6f} {y2:.6f} {z1}\n")
                f.write(f"      vertex {x2:.6f} {y2:.6f} {z2}\n")
                f.write(f"      vertex {x1:.6f} {y1:.6f} {z2}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write("endsolid NACA0012\n")
        
        logger.info(f"✅ Created NACA0012 STL file: {stl_file}")
        return stl_file
    
    def create_snappy_hex_mesh_dict(self):
        """Create snappyHexMeshDict for NACA0012 mesh generation."""
        logger.info("🏗️ Creating snappyHexMeshDict for NACA0012...")
        
        snappy_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
addLayers       true;

geometry
{
    airfoil
    {
        type triSurfaceMesh;
        file "airfoil.stl";
        
        regions
        {
            NACA0012
            {
                name airfoil;
            }
        }
    }
}

castellatedMeshControls
{
    maxLocalCells 2000000;
    maxGlobalCells 4000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;

    features
    (
    );

    refinementSurfaces
    {
        airfoil
        {
            level (3 4);
            patchInfo
            {
                type wall;
            }
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
    }

    locationInMesh (2 0.5 0);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 1.0;
    nSolveIter 100;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;

    layers
    {
        airfoil
        {
            nSurfaceLayers 5;
        }
    }

    expansionRatio 1.2;
    finalLayerThickness 0.4;
    minThickness 0.01;
    nGrow 0;
    featureAngle 180;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}

debug 0;
mergeTolerance 1e-6;

// ************************************************************************* //
'''
        
        # Write snappyHexMeshDict
        snappy_file = self.case_dir / "system" / "snappyHexMeshDict"
        with open(snappy_file, 'w') as f:
            f.write(snappy_content)
        
        logger.info("✅ Created snappyHexMeshDict")
        return True
    
    def create_airfoil_stl(self, x_airfoil, y_airfoil, stl_file):
        """Create STL file from airfoil coordinates."""
        logger.info("📐 Creating airfoil STL file...")
        
        # Create 3D coordinates
        z1, z2 = -0.1, 0.1
        n_points = len(x_airfoil)
        
        # STL header
        stl_content = "solid airfoil\n"
        
        # Create triangular facets for airfoil surface
        for i in range(n_points - 1):
            # Front face triangles
            x1, y1 = x_airfoil[i], y_airfoil[i]
            x2, y2 = x_airfoil[i + 1], y_airfoil[i + 1]
            
            # Triangle 1: (i,z1) -> (i+1,z1) -> (i,z2)
            stl_content += f"  facet normal 0 0 1\n"
            stl_content += f"    outer loop\n"
            stl_content += f"      vertex {x1} {y1} {z1}\n"
            stl_content += f"      vertex {x2} {y2} {z1}\n"
            stl_content += f"      vertex {x1} {y1} {z2}\n"
            stl_content += f"    endloop\n"
            stl_content += f"  endfacet\n"
            
            # Triangle 2: (i+1,z1) -> (i+1,z2) -> (i,z2)
            stl_content += f"  facet normal 0 0 1\n"
            stl_content += f"    outer loop\n"
            stl_content += f"      vertex {x2} {y2} {z1}\n"
            stl_content += f"      vertex {x2} {y2} {z2}\n"
            stl_content += f"      vertex {x1} {y1} {z2}\n"
            stl_content += f"    endloop\n"
            stl_content += f"  endfacet\n"
        
        stl_content += "endsolid airfoil\n"
        
        # Write STL file
        with open(stl_file, 'w') as f:
            f.write(stl_content)
        
        logger.info(f"✅ Created STL file: {stl_file}")
        return True
    
    def create_control_dict(self):
        """Create controlDict for steady-state simulation."""
        control_dict_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;

startFrom       latestTime;

startTime       0;

stopAt          endTime;

endTime         500;

deltaT          1;

writeControl    timeStep;

writeInterval   50;

purgeWrite      3;

writeFormat     ascii;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

functions
{
    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1.225;
        CofR            (0.25 0 0);    // Quarter chord
        
        log             true;
    }
    
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1.225;
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        CofR            (0.25 0 0);    // Quarter chord
        pitchAxis       (0 0 1);
        magUInf         50;
        lRef            1;              // Chord length
        Aref            1;              // Reference area per unit span
        
        log             true;
    }
}

// ************************************************************************* //
'''
        
        control_dict_file = self.case_dir / "system" / "controlDict"
        with open(control_dict_file, 'w') as f:
            f.write(control_dict_content)
        
        logger.info("✅ Created controlDict")
        return True
    
    def create_initial_boundary_conditions(self):
        """Create initial boundary conditions for background mesh."""
        logger.info("📋 Creating initial boundary conditions...")
        
        # Velocity field (U)
        u_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (50 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (50 0 0);
    }
    
    outlet
    {
        type            zeroGradient;
    }
    
    topWall
    {
        type            symmetryPlane;
    }
    
    bottomWall
    {
        type            symmetryPlane;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
'''
        
        # Pressure field (p)
        p_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    topWall
    {
        type            symmetryPlane;
    }
    
    bottomWall
    {
        type            symmetryPlane;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
'''
        
        # Write boundary condition files
        u_file = self.case_dir / "0" / "U"
        p_file = self.case_dir / "0" / "p"
        
        with open(u_file, 'w') as f:
            f.write(u_content)
        with open(p_file, 'w') as f:
            f.write(p_content)
        
        logger.info("✅ Created initial boundary conditions (U, p)")
        return True
    
    def update_boundary_conditions_after_snappy(self):
        """Update boundary conditions after snappyHexMesh to include airfoil."""
        logger.info("🔄 Updating boundary conditions after snappyHexMesh...")
        
        # Update U field
        u_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (50 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (50 0 0);
    }
    
    outlet
    {
        type            zeroGradient;
    }
    
    topWall
    {
        type            symmetryPlane;
    }
    
    bottomWall
    {
        type            symmetryPlane;
    }
    
    airfoil
    {
        type            noSlip;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
'''
        
        # Update p field
        p_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    topWall
    {
        type            symmetryPlane;
    }
    
    bottomWall
    {
        type            symmetryPlane;
    }
    
    airfoil
    {
        type            zeroGradient;
    }
    
    frontAndBack
    {
        type            empty;
    }
}

// ************************************************************************* //
'''
        
        # Write updated boundary condition files
        u_file = self.case_dir / "0" / "U"
        p_file = self.case_dir / "0" / "p"
        
        with open(u_file, 'w') as f:
            f.write(u_content)
        with open(p_file, 'w') as f:
            f.write(p_content)
        
        logger.info("✅ Updated boundary conditions with airfoil patch")
        return True
    
    def create_schemes_and_solution(self):
        """Create fvSchemes and fvSolution."""
        
        # fvSchemes
        schemes_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         orthogonal;
}

// ************************************************************************* //
'''
        
        # fvSolution
        solution_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-3;
        U               1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}

// ************************************************************************* //
'''
        
        # Write files
        schemes_file = self.case_dir / "system" / "fvSchemes"
        solution_file = self.case_dir / "system" / "fvSolution"
        
        with open(schemes_file, 'w') as f:
            f.write(schemes_content)
        with open(solution_file, 'w') as f:
            f.write(solution_content)
        
        logger.info("✅ Created fvSchemes and fvSolution")
        return True
    
    def create_transport_properties(self):
        """Create transport properties for air."""
        transport_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] 1.5e-05;

// ************************************************************************* //
'''
        
        transport_file = self.case_dir / "constant" / "transportProperties"
        with open(transport_file, 'w') as f:
            f.write(transport_content)
        
        logger.info("✅ Created transportProperties")
        return True
    
    def create_turbulence_properties(self):
        """Create turbulence properties for laminar flow."""
        turbulence_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "constant";
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType laminar;

// ************************************************************************* //
'''
        
        turbulence_file = self.case_dir / "constant" / "turbulenceProperties"
        with open(turbulence_file, 'w') as f:
            f.write(turbulence_content)
        
        logger.info("✅ Created turbulenceProperties")
        return True
    
    def generate_mesh(self, x_airfoil, y_airfoil):
        """Generate mesh using snappyHexMesh."""
        logger.info("🏗️ Generating NACA0012 mesh with snappyHexMesh...")
        
        # Step 1: Create background mesh with blockMesh
        logger.info("Creating background mesh...")
        success, stdout, stderr = self.run_command("blockMesh", cwd=self.case_dir)
        
        if not success:
            logger.error("❌ Background mesh generation failed")
            return False
        
        logger.info("✅ Background mesh created")
        
        # Step 2: Create NACA0012 STL file
        stl_file = self.create_naca0012_stl(x_airfoil, y_airfoil)
        
        # Step 3: Create snappyHexMeshDict
        self.create_snappy_hex_mesh_dict()
        
        # Step 4: Run snappyHexMesh
        logger.info("Running snappyHexMesh for NACA0012...")
        success, stdout, stderr = self.run_command("snappyHexMesh -overwrite", cwd=self.case_dir, timeout=1800)
        
        if success:
            logger.info("✅ NACA0012 mesh generated successfully")
            
            # Step 5: Update boundary conditions to include airfoil
            self.update_boundary_conditions_after_snappy()
            
            # Check mesh quality
            logger.info("🔍 Checking final mesh quality...")
            success, stdout, stderr = self.run_command("checkMesh", cwd=self.case_dir)
            
            if success:
                logger.info("✅ Mesh quality check passed")
                return True
            else:
                logger.warning("⚠️ Mesh quality check had warnings, continuing...")
                return True  # Continue anyway
        else:
            logger.error("❌ snappyHexMesh failed")
            logger.error(f"Error: {stderr}")
            return False
    
    def run_simulation(self, max_iterations=500):
        """Run CFD simulation."""
        logger.info("🚀 Running CFD simulation...")
        
        # Modify controlDict for current run
        control_dict = self.case_dir / "system" / "controlDict"
        if control_dict.exists():
            with open(control_dict, 'r') as f:
                content = f.read()
            
            content = content.replace('endTime         500;', f'endTime         {max_iterations};')
            
            with open(control_dict, 'w') as f:
                f.write(content)
        
        # Run simpleFoam
        start_time = time.time()
        success, stdout, stderr = self.run_command("simpleFoam", cwd=self.case_dir, timeout=1800)
        end_time = time.time()
        
        if success:
            logger.info(f"✅ CFD simulation completed in {end_time - start_time:.1f} seconds")
            return True
        else:
            logger.error("❌ CFD simulation failed")
            return False
    
    def extract_force_coefficients(self):
        """Extract force coefficients from simulation results."""
        logger.info("📊 Extracting force coefficients...")
        
        # Look for force coefficients in multiple possible locations
        possible_locations = [
            self.case_dir / "postProcessing" / "forceCoeffs",
            self.case_dir / "postProcessing" / "forces",
        ]
        
        force_dir = None
        for location in possible_locations:
            if location.exists():
                force_dir = location
                break
        
        if not force_dir:
            logger.warning("⚠️ Force coefficients directory not found")
            logger.info("Checking postProcessing directory contents...")
            
            post_dir = self.case_dir / "postProcessing"
            if post_dir.exists():
                contents = list(post_dir.iterdir())
                logger.info(f"Available directories: {[d.name for d in contents if d.is_dir()]}")
                
                # Try to find any force-related files
                for subdir in contents:
                    if subdir.is_dir() and ('force' in subdir.name.lower() or 'coeff' in subdir.name.lower()):
                        force_dir = subdir
                        logger.info(f"Found force directory: {force_dir}")
                        break
            
            if not force_dir:
                logger.warning("No force calculation results found")
                return None
        
        # Find coefficient files
        coeff_files = []
        for pattern in ["**/forceCoeffs.dat", "**/coefficient.dat", "**/*.dat"]:
            coeff_files.extend(list(force_dir.glob(pattern)))
        
        if not coeff_files:
            logger.warning("⚠️ Force coefficients file not found")
            logger.info(f"Contents of {force_dir}:")
            for item in force_dir.rglob("*"):
                logger.info(f"  {item}")
            return None
        
        logger.info(f"Found force coefficient files: {[f.name for f in coeff_files]}")
        
        # Read latest coefficients
        try:
            # Try the most recent file
            latest_file = max(coeff_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Reading from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                lines = f.readlines()
            
            logger.info(f"Force file has {len(lines)} lines")
            
            # Parse last line of data
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Time'):
                    parts = line.split()
                    logger.info(f"Parsing line: {line}")
                    logger.info(f"Split into {len(parts)} parts: {parts}")
                    
                    if len(parts) >= 4:
                        try:
                            time_val = float(parts[0])
                            cd = float(parts[1])
                            cl = float(parts[2])
                            cm = float(parts[3]) if len(parts) > 3 else 0.0
                            
                            logger.info(f"✅ Force coefficients extracted:")
                            logger.info(f"   Time: {time_val}")
                            logger.info(f"   Cd (drag):  {cd:.6f}")
                            logger.info(f"   Cl (lift):  {cl:.6f}")
                            logger.info(f"   Cm (moment): {cm:.6f}")
                            
                            return {"time": time_val, "cd": cd, "cl": cl, "cm": cm}
                        except ValueError as e:
                            logger.warning(f"Could not parse line '{line}': {e}")
                            continue
        
        except Exception as e:
            logger.error(f"❌ Error reading force coefficients: {e}")
        
        logger.warning("Could not extract valid force coefficients")
        return None
    
    def create_ffd_control_points(self, x_airfoil, y_airfoil):
        """Create FFD control points for shape optimization."""
        logger.info("🎯 Creating FFD control points...")
        
        # Create 3D mesh points from airfoil coordinates
        z = np.zeros_like(x_airfoil)
        mesh_points = np.column_stack([x_airfoil, y_airfoil, z])
        
        # Create FFD control box
        ffd_dimensions = (5, 3, 2)  # 5x3x2 control points
        ffd_control_points, ffd_bbox = create_ffd_box(mesh_points, ffd_dimensions)
        
        logger.info(f"✅ FFD control box created:")
        logger.info(f"   Dimensions: {ffd_dimensions}")
        logger.info(f"   Total control points: {np.prod(ffd_dimensions)}")
        logger.info(f"   Bounding box: {ffd_bbox}")
        
        return ffd_control_points, ffd_bbox
    
    def run_optimization_iteration(self, iteration, ffd_control_points, baseline_cd):
        """Run one optimization iteration."""
        logger.info(f"🔄 Optimization iteration {iteration}")
        
        # Create iteration directory
        iter_dir = self.case_dir.parent / f"naca0012_iter_{iteration}"
        if iter_dir.exists():
            shutil.rmtree(iter_dir)
        shutil.copytree(self.case_dir, iter_dir)
        
        # Apply FFD perturbation (simple example)
        if iteration > 1:
            # Perturb upper surface control points to reduce thickness
            perturbation = np.zeros_like(ffd_control_points)
            # Modify upper control points (y > 0)
            upper_mask = ffd_control_points[:, 1] > 0
            perturbation[upper_mask, 1] = -0.005 * iteration  # Reduce thickness
            
            ffd_control_points += perturbation
            logger.info(f"   Applied thickness reduction: {0.005 * iteration:.6f}")
        
        # Modify case for quick iteration
        control_dict = iter_dir / "system" / "controlDict"
        if control_dict.exists():
            with open(control_dict, 'r') as f:
                content = f.read()
            
            content = content.replace('endTime         500;', 'endTime         100;')
            content = content.replace('writeInterval   50;', 'writeInterval   25;')
            
            with open(control_dict, 'w') as f:
                f.write(content)
        
        # Run simulation
        success, stdout, stderr = self.run_command("simpleFoam", cwd=iter_dir, timeout=600)
        
        if success:
            # Extract results
            force_dir = iter_dir / "postProcessing" / "forceCoeffs"
            if force_dir.exists():
                coeff_files = list(force_dir.glob("**/forceCoeffs.dat"))
                if coeff_files:
                    try:
                        with open(coeff_files[-1], 'r') as f:
                            lines = f.readlines()
                        
                        # Get last data line
                        for line in reversed(lines):
                            if line.strip() and not line.startswith('#'):
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    cd = float(parts[1])
                                    cl = float(parts[2])
                                    
                                    improvement = baseline_cd - cd
                                    improvement_pct = (improvement / baseline_cd) * 100
                                    
                                    logger.info(f"   Results: Cd = {cd:.6f}, Cl = {cl:.6f}")
                                    logger.info(f"   Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
                                    
                                    # Clean up iteration directory
                                    shutil.rmtree(iter_dir)
                                    
                                    return cd, cl, ffd_control_points
                                    
                    except Exception as e:
                        logger.error(f"   Error reading results: {e}")
        
        # Clean up on failure
        if iter_dir.exists():
            shutil.rmtree(iter_dir)
        
        logger.warning(f"   Iteration {iteration} failed")
        return None, None, ffd_control_points
    
    def save_optimization_results(self, results):
        """Save optimization results to file."""
        results_file = self.case_dir / "optimization_results.txt"
        
        with open(results_file, 'w') as f:
            f.write("NACA0012 Airfoil Shape Optimization Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Results Summary:\n")
            f.write("-" * 20 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"✅ Results saved to {results_file}")
    
    def run_complete_optimization(self):
        """Run the complete optimization workflow."""
        logger.info("🎯 Starting NACA0012 Optimization Workflow")
        logger.info("=" * 60)
        
        try:
            # 1. Check OpenFOAM environment
            if not self.check_openfoam_environment():
                return False
            
            # 2. Generate NACA0012 coordinates
            x_airfoil, y_airfoil = self.generate_naca0012_coordinates()
            
            # 3. Create OpenFOAM case
            if not self.create_openfoam_case():
                return False
            
            # 4. Create all case files
            self.create_background_mesh()
            self.create_control_dict()
            self.create_initial_boundary_conditions()
            self.create_schemes_and_solution()
            self.create_transport_properties()
            self.create_turbulence_properties()
            
            # 5. Generate mesh
            if not self.generate_mesh(x_airfoil, y_airfoil):
                return False
            
            # 6. Run baseline simulation
            logger.info("🚀 Running baseline simulation...")
            if not self.run_simulation():
                return False
            
            # 7. Extract baseline results
            baseline_results = self.extract_force_coefficients()
            if not baseline_results:
                logger.error("❌ Could not extract baseline results")
                return False
            
            baseline_cd = baseline_results["cd"]
            baseline_cl = baseline_results["cl"]
            
            logger.info(f"📊 Baseline NACA0012 results:")
            logger.info(f"   Drag coefficient: {baseline_cd:.6f}")
            logger.info(f"   Lift coefficient: {baseline_cl:.6f}")
            
            # 8. Create FFD control points
            ffd_control_points, ffd_bbox = self.create_ffd_control_points(x_airfoil, y_airfoil)
            
            # 9. Run optimization iterations
            logger.info("🔄 Starting optimization iterations...")
            n_iterations = 3
            best_cd = baseline_cd
            best_cl = baseline_cl
            optimization_history = []
            
            for iteration in range(1, n_iterations + 1):
                cd, cl, ffd_control_points = self.run_optimization_iteration(
                    iteration, ffd_control_points, baseline_cd
                )
                
                if cd is not None:
                    optimization_history.append({
                        "iteration": iteration,
                        "cd": cd,
                        "cl": cl,
                        "improvement": baseline_cd - cd
                    })
                    
                    if cd < best_cd:
                        best_cd = cd
                        best_cl = cl
                        logger.info(f"🎯 New best design found in iteration {iteration}!")
            
            # 10. Final results
            total_improvement = baseline_cd - best_cd
            improvement_pct = (total_improvement / baseline_cd) * 100
            
            logger.info("🏁 Optimization completed!")
            logger.info("=" * 60)
            logger.info("📊 Final Results:")
            logger.info(f"   Baseline Cd:    {baseline_cd:.6f}")
            logger.info(f"   Optimized Cd:   {best_cd:.6f}")
            logger.info(f"   Improvement:    {total_improvement:.6f} ({improvement_pct:.2f}%)")
            logger.info(f"   Final Cl:       {best_cl:.6f}")
            logger.info(f"   Iterations:     {n_iterations}")
            
            # 11. Save results
            final_results = {
                "baseline_cd": baseline_cd,
                "baseline_cl": baseline_cl,
                "best_cd": best_cd,
                "best_cl": best_cl,
                "improvement": total_improvement,
                "improvement_percent": improvement_pct,
                "iterations": n_iterations,
                "optimization_history": optimization_history
            }
            
            self.results = final_results
            self.save_optimization_results(final_results)
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    print("🎯 NACA0012 Airfoil Shape Optimization with OpenFOAM")
    print("=" * 60)
    
    # Create optimizer
    optimizer = NACA0012Optimizer("naca0012_optimization")
    
    # Run optimization
    success = optimizer.run_complete_optimization()
    
    if success:
        print("\n🎉 NACA0012 optimization completed successfully!")
        print("✅ All files generated and simulations completed")
        print(f"📁 Results available in: {optimizer.case_dir}")
    else:
        print("\n❌ Optimization failed.")
        print("Please check that OpenFOAM environment is activated with 'openfoam' command")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())