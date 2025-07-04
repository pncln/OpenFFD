#!/usr/bin/env python3
"""
NACA0012 Airfoil Shape Optimization Example

This example demonstrates a complete NACA0012 airfoil optimization workflow
using the OpenFFD CFD optimization module. It shows how to:

1. Set up optimization configuration  
2. Run automated NACA0012 optimization
3. Analyze optimization results
4. Create professional OpenFOAM cases

Prerequisites:
- OpenFOAM installed (activate with 'openfoam' command)
- OpenFFD package installed with CFD module

Usage:
    # First activate OpenFOAM environment
    openfoam
    
    # Then run the optimization
    python3 naca0012_optimization_clean.py
"""

import numpy as np
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import OpenFFD optimization module
from openffd.cfd import (
    OptimizationManager,
    OptimizationConfig, 
    OptimizationResults,
    AirfoilGenerator,
    optimize_naca0012_airfoil,
    create_naca_airfoil_case
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('naca0012_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run NACA0012 optimization example."""
    logger.info("🎯 NACA0012 Airfoil Shape Optimization Example")
    logger.info("=" * 60)
    
    try:
        # 1. Check OpenFOAM environment first
        logger.info("🔧 Checking OpenFOAM availability...")
        
        # Simple check like the old working example
        openfoam_available = False
        
        # Check environment variables first
        foam_vars = ['FOAM_APP', 'WM_PROJECT', 'FOAM_ETC', 'WM_PROJECT_DIR']
        missing_vars = [var for var in foam_vars if var not in os.environ]
        
        if not missing_vars:
            openfoam_available = True
            logger.info("✅ OpenFOAM environment variables detected")
        else:
            # Check if commands are in PATH
            try:
                import subprocess
                result = subprocess.run(['which', 'blockMesh'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    openfoam_available = True
                    logger.info("✅ OpenFOAM commands found in PATH")
                else:
                    logger.info("📝 OpenFOAM commands not found in PATH")
            except:
                logger.info("📝 Could not check for OpenFOAM commands")
        
        if not openfoam_available:
            logger.warning("⚠️ OpenFOAM environment not available")
            logger.info("📝 Demonstration will continue with case creation only")
            
            # Just demonstrate capabilities without running simulations
            # demonstrate_airfoil_generation()
            demonstrate_case_creation()
            demonstrate_optimization_setup()
            return True
        
        logger.info("✅ OpenFOAM environment activated successfully")
        
        # 2. Configure optimization
        config = OptimizationConfig(
            objective="minimize_drag",
            max_iterations=3,  # Small number for demonstration
            convergence_tolerance=1e-6,
            initial_step_size=0.001,
            save_intermediate=True
        )
        
        logger.info("⚙️ Optimization Configuration:")
        logger.info(f"   Objective: {config.objective}")
        logger.info(f"   Max iterations: {config.max_iterations}")
        logger.info(f"   Convergence tolerance: {config.convergence_tolerance}")
        
        # 3. Run optimization using the convenience function
        case_dir = "./naca0012_optimization_clean"
        logger.info(f"📁 Case directory: {case_dir}")
        
        results = optimize_naca0012_airfoil(case_dir, config)
        
        # 4. Analyze results
        if results.success:
            logger.info("🎉 Optimization completed successfully!")
            logger.info("=" * 40)
            logger.info("📊 Final Results:")
            logger.info(f"   Initial Cd:    {results.initial_objective:.6f}")
            logger.info(f"   Final Cd:      {results.final_objective:.6f}")
            logger.info(f"   Improvement:   {results.improvement:.6f}")
            logger.info(f"   Improvement%:  {results.improvement_percent:.2f}%")
            logger.info(f"   Iterations:    {results.iterations}")
            logger.info(f"   Elapsed time:  {results.elapsed_time:.1f}s")
            
            # Save results
            save_optimization_results(results, case_dir)
            
            # Plot convergence history
            if results.convergence_history:
                plot_convergence_history(results, case_dir)
            
        else:
            logger.error("❌ Optimization failed!")
            logger.error(f"   Error: {results.message}")
            
            # Still demonstrate other capabilities
            demonstrate_airfoil_generation()
            demonstrate_case_creation()
            return False
        
        # 5. Demonstrate other capabilities
        demonstrate_airfoil_generation()
        demonstrate_case_creation()
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Example failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to demonstrate capabilities anyway
        try:
            demonstrate_airfoil_generation()
            demonstrate_case_creation()
        except:
            pass
        
        return False

def save_optimization_results(results: OptimizationResults, case_dir: str):
    """Save optimization results to file."""
    results_file = Path(case_dir) / "optimization_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("NACA0012 Airfoil Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Success:           {results.success}\n")
        f.write(f"Initial Cd:        {results.initial_objective:.6f}\n")
        f.write(f"Final Cd:          {results.final_objective:.6f}\n")
        f.write(f"Improvement:       {results.improvement:.6f}\n")
        f.write(f"Improvement%:      {results.improvement_percent:.2f}%\n")
        f.write(f"Iterations:        {results.iterations}\n")
        f.write(f"Elapsed Time:      {results.elapsed_time:.1f}s\n")
        f.write(f"Message:           {results.message}\n\n")
        
        if results.convergence_history:
            f.write("Convergence History:\n")
            for i, cd in enumerate(results.convergence_history):
                f.write(f"  Iteration {i+1}: Cd = {cd:.6f}\n")
    
    logger.info(f"✅ Results saved to {results_file}")

def plot_convergence_history(results: OptimizationResults, case_dir: str):
    """Plot and save convergence history."""
    try:
        plt.figure(figsize=(10, 6))
        
        iterations = range(1, len(results.convergence_history) + 1)
        plt.plot(iterations, results.convergence_history, 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=results.initial_objective, color='r', linestyle='--', 
                   label=f'Baseline Cd = {results.initial_objective:.6f}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Drag Coefficient (Cd)')
        plt.title('NACA0012 Optimization Convergence History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_file = Path(case_dir) / "convergence_history.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Convergence plot saved to {plot_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not create convergence plot: {e}")

def demonstrate_airfoil_generation():
    """Demonstrate airfoil generation capabilities."""
    logger.info("\n🔧 Demonstrating Airfoil Generation:")
    
    # Generate NACA0012
    x_coords, y_coords = AirfoilGenerator.generate_naca0012(n_points=100)
    logger.info(f"   NACA0012: {len(x_coords)} points generated")
    
    # Generate other NACA airfoils
    # naca_codes = ["2412", "4412", "6412"]
    # for code in naca_codes:
    #     try:
    #         x, y = AirfoilGenerator.generate_naca_4digit(code, n_points=100)
    #         logger.info(f"   NACA{code}: {len(x)} points generated")
    #     except Exception as e:
    #         logger.warning(f"   NACA{code}: Generation failed - {e}")
    
    # Create STL file
    stl_file = "./demo_naca0012.stl"
    success = AirfoilGenerator.create_stl_file(x_coords, y_coords, stl_file)
    if success:
        logger.info(f"   STL file created: {stl_file}")

def demonstrate_case_creation():
    """Demonstrate OpenFOAM case creation."""
    logger.info("\n🏗️ Demonstrating Case Creation:")
    
    # Create a sample case
    case_dir = "./demo_naca_case"
    success = create_naca_airfoil_case(
        case_dir=case_dir,
        naca_code="0012",
        config={
            "control": {
                "endTime": 500,
                "writeInterval": 50,
                "magUInf": 25
            },
            "mesh": {
                "nx": 80,
                "ny": 60,
                "surface_max_level": 3
            },
            "initial": {
                "U_inlet": [25, 0, 0]
            }
        }
    )
    
    if success:
        logger.info(f"   Demo case created: {case_dir}")
        logger.info("   Ready for OpenFOAM simulation!")
        
        # Validate STL file
        stl_file = Path(case_dir) / "constant" / "triSurface" / "airfoil.stl"
        if stl_file.exists():
            try:
                with open(stl_file, 'r') as f:
                    content = f.read()
                    if 'nan' in content.lower():
                        logger.warning("   ⚠️ STL file contains invalid normals")
                    else:
                        logger.info("   ✅ STL file validated - no NaN values")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not validate STL file: {e}")
    else:
        logger.warning("   Demo case creation failed")

def demonstrate_optimization_setup():
    """Demonstrate optimization setup without running simulations."""
    logger.info("\n🎯 Demonstrating Optimization Setup:")
    
    from openffd.cfd import OptimizationManager, OptimizationConfig
    
    # Show configuration options
    config = OptimizationConfig(
        objective="minimize_drag",
        max_iterations=10,
        convergence_tolerance=1e-6,
        initial_step_size=0.002,
        max_step_size=0.01,
        parallel_evaluation=False
    )
    
    logger.info("   Optimization configuration created:")
    logger.info(f"   - Objective: {config.objective}")
    logger.info(f"   - Max iterations: {config.max_iterations}")
    logger.info(f"   - Convergence tolerance: {config.convergence_tolerance}")
    logger.info(f"   - Step size range: {config.min_step_size} to {config.max_step_size}")
    
    # Show optimization manager setup
    try:
        optimizer = OptimizationManager(config)
        logger.info("   ✅ OptimizationManager created successfully")
        logger.info("   📝 Ready for optimization when OpenFOAM is available")
    except Exception as e:
        logger.warning(f"   ⚠️ OptimizationManager setup issue: {e}")
    
    # Demonstrate FFD control point creation
    try:
        from openffd.cfd import AirfoilGenerator
        x, y = AirfoilGenerator.generate_naca0012(n_points=50)
        
        # Create mesh points for FFD
        z = np.zeros_like(x)
        mesh_points = np.column_stack([x, y, z])
        
        from openffd.core import create_ffd_box
        ffd_dimensions = (5, 3, 2)
        ffd_control_points, ffd_bbox = create_ffd_box(mesh_points, ffd_dimensions)
        
        logger.info(f"   ✅ FFD control box created: {np.prod(ffd_dimensions)} control points")
        try:
            if hasattr(ffd_bbox, 'flatten'):
                bbox_str = [f'{x:.3f}' for x in ffd_bbox.flatten()]
            else:
                bbox_str = str(ffd_bbox)
            logger.info(f"   📐 Bounding box: {bbox_str}")
        except:
            logger.info(f"   📐 Bounding box: {ffd_bbox}")
        
    except Exception as e:
        logger.warning(f"   ⚠️ FFD setup issue: {e}")
    
    logger.info("   💡 To run full optimization, ensure OpenFOAM is installed and activated")

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'🎉' if success else '⚠️'} NACA0012 optimization example completed!")
    
    if success:
        print("\n📁 Generated files:")
        
        # Check what files actually exist
        files_to_check = [
            ("naca0012_optimization_clean/", "Complete OpenFOAM case (if OpenFOAM available)"),
            ("naca0012_optimization.log", "Detailed execution log"),
            ("optimization_results.txt", "Results summary (if optimization ran)"),
            ("convergence_history.png", "Convergence plot (if optimization ran)"),
            ("demo_naca0012.stl", "Sample STL file"),
            ("demo_naca_case/", "Demo OpenFOAM case")
        ]
        
        for filename, description in files_to_check:
            exists = Path(filename).exists()
            status = "✅" if exists else "📝"
            print(f"   {status} {filename:<35} {description}")
        
        print("\n🔍 To inspect results:")
        if Path("optimization_results.txt").exists():
            print("   cat optimization_results.txt")
        if Path("naca0012_optimization_clean/optimization_results.txt").exists():
            print("   cat naca0012_optimization_clean/optimization_results.txt")
        print("   cat naca0012_optimization.log")
        
        print("\n🚀 To run OpenFOAM cases manually (requires OpenFOAM):")
        for case_dir in ["demo_naca_case", "naca0012_optimization_clean"]:
            if Path(case_dir).exists():
                print(f"   cd {case_dir}")
                print("   openfoam")
                print("   blockMesh && snappyHexMesh -overwrite && simpleFoam")
                break
                
        print("\n💡 Notes:")
        print("   - Full optimization requires OpenFOAM installation")
        print("   - Demo capabilities work without OpenFOAM")
        print("   - STL files and case structure are always generated")
        
    else:
        print("\n📝 Demonstration completed with limited capabilities")
        print("   - Install OpenFOAM for full optimization workflow")
        print("   - Check log file: tail -20 naca0012_optimization.log")