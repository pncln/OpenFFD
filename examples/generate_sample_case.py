#!/usr/bin/env python3
"""
Generate Sample OpenFOAM Case

This script generates a complete OpenFOAM case in a persistent directory
so you can inspect the generated files.
"""

import logging
from pathlib import Path
from openffd.cfd.openfoam import OpenFOAMConfig, SolverType, TurbulenceModel, OpenFOAMSolver

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_openfoam_case():
    """Create a complete sample OpenFOAM case."""
    
    # Create output directory in the examples folder
    output_dir = Path(__file__).parent / "sample_openfoam_case"
    
    # Remove existing directory if it exists
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    logger.info(f"Creating OpenFOAM case in: {output_dir}")
    
    # Create comprehensive configuration
    config = OpenFOAMConfig(
        case_directory=output_dir,
        solver_executable="simpleFoam",
        solver_type=SolverType.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        
        # Time settings
        end_time=1000.0,  # Steady state (large time)
        time_step=1.0,
        write_interval=100,
        max_iterations=2000,
        
        # Convergence settings
        convergence_tolerance={
            'p': 1e-6,
            'U': 1e-6, 
            'k': 1e-6,
            'omega': 1e-6
        },
        
        # Parallel execution
        parallel_execution=True,
        num_processors=4,
        decomposition_method="scotch",
        
        # Physical properties (air at 20°C)
        fluid_properties={
            'nu': 1.5e-5,  # kinematic viscosity (m²/s)
            'rho': 1.225,  # density (kg/m³)
        },
        
        # Comprehensive boundary conditions for airfoil case
        boundary_conditions={
            'inlet': {
                'U': {'type': 'fixedValue', 'value': 'uniform (50 0 0)'},  # 50 m/s inlet
                'p': {'type': 'zeroGradient'},
                'k': {'type': 'fixedValue', 'value': 'uniform 0.375'},    # 1% turbulence
                'omega': {'type': 'fixedValue', 'value': 'uniform 2.6'}   # Corresponding omega
            },
            'outlet': {
                'U': {'type': 'zeroGradient'},
                'p': {'type': 'fixedValue', 'value': 'uniform 0'},        # Reference pressure
                'k': {'type': 'zeroGradient'},
                'omega': {'type': 'zeroGradient'}
            },
            'airfoil': {
                'U': {'type': 'noSlip'},                                  # No-slip wall
                'p': {'type': 'zeroGradient'},
                'k': {'type': 'kqRWallFunction', 'value': 'uniform 1e-10'},
                'omega': {'type': 'omegaWallFunction', 'value': 'uniform 1e6'}
            },
            'farfield': {
                'U': {'type': 'symmetryPlane'},                          # Symmetry boundary
                'p': {'type': 'symmetryPlane'},
                'k': {'type': 'symmetryPlane'},
                'omega': {'type': 'symmetryPlane'}
            },
            'frontAndBack': {
                'U': {'type': 'empty'},                                  # 2D case
                'p': {'type': 'empty'},
                'k': {'type': 'empty'},
                'omega': {'type': 'empty'}
            }
        },
        
        # Force calculation for optimization
        force_calculation=True,
        force_patches=['airfoil'],
        reference_values={
            'rho': 1.225,    # Reference density
            'U': 50.0,       # Reference velocity
            'A': 1.0,        # Reference area (per unit span)
            'L': 1.0         # Reference length (chord)
        }
    )
    
    # Create mock solver (since OpenFOAM may not be installed)
    solver = OpenFOAMSolver.__new__(OpenFOAMSolver)
    solver.name = 'SampleCaseGenerator'
    solver.version = 'test-v2112'
    solver.logger = logging.getLogger('sample_case')
    solver.openfoam_root = Path('/sample/openfoam/path')
    solver._process = None
    
    # Create case directory structure
    logger.info("Creating directory structure...")
    success = solver.prepare_case_directory(config)
    if not success:
        logger.error("Failed to create case directory")
        return False
    
    # Generate all OpenFOAM files
    logger.info("Generating OpenFOAM configuration files...")
    
    try:
        # System files
        solver._write_control_dict(config)
        logger.info("✓ Generated system/controlDict")
        
        solver._write_fv_schemes(config)
        logger.info("✓ Generated system/fvSchemes")
        
        solver._write_fv_solution(config)
        logger.info("✓ Generated system/fvSolution")
        
        solver._setup_decomposition(config)
        logger.info("✓ Generated system/decomposeParDict")
        
        # Constant files
        solver._write_transport_properties(config)
        logger.info("✓ Generated constant/transportProperties")
        
        solver._write_turbulence_properties(config)
        logger.info("✓ Generated constant/turbulenceProperties")
        
        # Initial conditions (0 directory)
        solver._write_boundary_conditions(config)
        logger.info("✓ Generated 0/U and 0/p")
        
        # Add function objects and regenerate controlDict
        solver._add_function_objects(config)
        solver._write_control_dict(config)
        logger.info("✓ Added function objects to controlDict")
        
        # Generate additional turbulence fields
        generate_turbulence_fields(config)
        logger.info("✓ Generated 0/k and 0/omega")
        
        # Create sample scripts
        create_run_scripts(config)
        logger.info("✓ Generated run scripts")
        
    except Exception as e:
        logger.error(f"Failed to generate files: {e}")
        return False
    
    # Report file sizes and contents
    report_generated_files(output_dir)
    
    return True

def generate_turbulence_fields(config):
    """Generate turbulence field files (k and omega)."""
    
    # Generate k field
    k_file = config.case_directory / "0" / "k"
    with open(k_file, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class volScalarField; object k; }
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.375;

boundaryField
{
""")
        
        # Add boundary conditions from config
        for patch_name, bc_dict in config.boundary_conditions.items():
            if 'k' in bc_dict:
                f.write(f"    {patch_name}\n")
                f.write("    {\n")
                f.write(f"        type    {bc_dict['k'].get('type', 'zeroGradient')};\n")
                if 'value' in bc_dict['k']:
                    f.write(f"        value   {bc_dict['k']['value']};\n")
                f.write("    }\n")
        
        f.write("}\n\n// ************************************************************************* //\n")
    
    # Generate omega field
    omega_file = config.case_directory / "0" / "omega"
    with open(omega_file, 'w') as f:
        f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class volScalarField; object omega; }
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 2.6;

boundaryField
{
""")
        
        # Add boundary conditions from config
        for patch_name, bc_dict in config.boundary_conditions.items():
            if 'omega' in bc_dict:
                f.write(f"    {patch_name}\n")
                f.write("    {\n")
                f.write(f"        type    {bc_dict['omega'].get('type', 'zeroGradient')};\n")
                if 'value' in bc_dict['omega']:
                    f.write(f"        value   {bc_dict['omega']['value']};\n")
                f.write("    }\n")
        
        f.write("}\n\n// ************************************************************************* //\n")

def create_run_scripts(config):
    """Create sample run scripts."""
    
    # Create Allrun script
    allrun_script = config.case_directory / "Allrun"
    with open(allrun_script, 'w') as f:
        f.write("""#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

echo "Running OpenFOAM CFD case generated by OpenFFD"
echo "=============================================="

# Check if mesh exists
if [ ! -d "constant/polyMesh" ]; then
    echo "ERROR: No mesh found in constant/polyMesh"
    echo "Please copy your mesh files to constant/polyMesh before running"
    exit 1
fi

# Check mesh quality
echo "Checking mesh quality..."
runApplication checkMesh

# Decompose for parallel run
if [ -f "system/decomposeParDict" ]; then
    echo "Decomposing case for parallel execution..."
    runApplication decomposePar
    
    # Run solver in parallel
    echo "Running simpleFoam in parallel..."
    runParallel simpleFoam
    
    # Reconstruct case
    echo "Reconstructing case..."
    runApplication reconstructPar
else
    # Run serial
    echo "Running simpleFoam in serial..."
    runApplication simpleFoam
fi

echo "Simulation completed!"
echo "Check the log files for convergence information."
""")
    
    # Make script executable
    import stat
    allrun_script.chmod(allrun_script.stat().st_mode | stat.S_IEXEC)
    
    # Create Allclean script
    allclean_script = config.case_directory / "Allclean"
    with open(allclean_script, 'w') as f:
        f.write("""#!/bin/sh
cd ${0%/*} || exit 1    # Run from this directory

# Source tutorial run functions
. $WM_PROJECT_DIR/bin/tools/CleanFunctions

echo "Cleaning OpenFOAM case..."

# Remove time directories (except 0)
rm -rf [1-9]* 0.[0-9]* 0.0[1-9]*

# Remove processor directories
rm -rf processor*

# Remove log files
rm -f log.*

# Remove postProcessing
rm -rf postProcessing

# Remove dynamicCode
rm -rf dynamicCode

echo "Case cleaned!"
""")
    
    # Make script executable
    allclean_script.chmod(allclean_script.stat().st_mode | stat.S_IEXEC)

def report_generated_files(output_dir):
    """Report on all generated files."""
    
    logger.info(f"\n{'='*60}")
    logger.info("GENERATED OPENFOAM CASE FILES")
    logger.info(f"{'='*60}")
    logger.info(f"Case directory: {output_dir}")
    
    # List all generated files with sizes
    total_size = 0
    file_count = 0
    
    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_count += 1
            relative_path = file_path.relative_to(output_dir)
            logger.info(f"  {str(relative_path):<30} {size:>8} bytes")
    
    logger.info(f"{'-'*60}")
    logger.info(f"Total: {file_count} files, {total_size} bytes")
    
    # Show key file contents
    logger.info(f"\n{'='*60}")
    logger.info("SAMPLE FILE CONTENTS")
    logger.info(f"{'='*60}")
    
    # Show controlDict header
    control_dict = output_dir / "system" / "controlDict"
    if control_dict.exists():
        logger.info("\ncontrolDict (first 10 lines):")
        logger.info("-" * 40)
        with open(control_dict, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"  {line.rstrip()}")
    
    logger.info(f"\n{'='*60}")
    logger.info("USAGE INSTRUCTIONS")
    logger.info(f"{'='*60}")
    logger.info(f"""
To use this OpenFOAM case:

1. Copy your mesh files to: {output_dir}/constant/polyMesh/
   (You need: points, faces, cells, boundary files)

2. If OpenFOAM is installed, run:
   cd {output_dir}
   ./Allrun

3. Or run manually:
   cd {output_dir}
   checkMesh                    # Check mesh quality
   decomposePar                 # Decompose for parallel (optional)
   mpirun -np 4 simpleFoam -parallel  # Run parallel
   reconstructPar               # Reconstruct results

4. View results:
   paraFoam                     # Open in ParaView
   
5. Clean case:
   ./Allclean

Generated case includes:
✓ Complete solver configuration (simpleFoam, k-ω SST)
✓ Proper boundary conditions for airfoil analysis
✓ Force coefficient calculation setup
✓ Parallel execution configuration (4 processors)
✓ Run and clean scripts
""")

def main():
    """Main function."""
    logger.info("OpenFOAM Sample Case Generator")
    logger.info("=" * 50)
    
    success = create_sample_openfoam_case()
    
    if success:
        logger.info("\n🎉 OpenFOAM case generated successfully!")
        logger.info("✅ All files created with proper content")
        logger.info("✅ Ready for CFD simulation")
    else:
        logger.error("\n❌ Failed to generate OpenFOAM case")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())