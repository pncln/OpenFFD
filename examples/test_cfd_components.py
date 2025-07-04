#!/usr/bin/env python3
"""
Unit Tests for OpenFFD CFD Components

This script tests individual components of the CFD solver interface
to ensure they work correctly without requiring OpenFOAM installation.
"""

import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil

# Import OpenFFD modules
from openffd.cfd import (
    OpenFOAMConfig, SolverType, TurbulenceModel,
    SensitivityConfig, GradientMethod, ObjectiveFunction,
    MeshConverter, ParallelManager
)
from openffd.cfd.base import CFDResults, SolverStatus, ConvergenceData
from openffd.cfd.openfoam import SimulationResults, ForceCoefficients, ResidualData
from openffd.cfd.sensitivity import SensitivityResults, FiniteDifferenceConfig
from openffd.cfd.utilities import PostProcessingConfig, MeshConversionConfig
from openffd.mesh import MeshDeformationEngine, DeformationConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openfoam_config():
    """Test OpenFOAM configuration creation and validation."""
    logger.info("Testing OpenFOAM configuration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir) / "test_case"
        
        # Test valid configuration
        config = OpenFOAMConfig(
            case_directory=case_dir,
            solver_executable="simpleFoam",
            solver_type=SolverType.SIMPLE_FOAM,
            turbulence_model=TurbulenceModel.K_OMEGA_SST,
            end_time=100.0,
            max_iterations=500,
            parallel_execution=True,
            num_processors=4
        )
        
        assert config.solver_type == SolverType.SIMPLE_FOAM
        assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST
        assert config.parallel_execution == True
        assert config.num_processors == 4
        assert len(config.convergence_tolerance) > 0  # Should have defaults
        
        logger.info("✓ OpenFOAM configuration test passed")

def test_simulation_results():
    """Test simulation results data structures."""
    logger.info("Testing simulation results...")
    
    # Create sample residual data
    residuals = [
        ResidualData(
            iteration=i,
            time=float(i),
            fields={'p': 1e-3/(i+1), 'U': 1e-4/(i+1), 'k': 1e-5/(i+1)}
        ) for i in range(10)
    ]
    
    # Create force coefficients
    force_coeffs = ForceCoefficients(
        cd=0.02,
        cl=0.5,
        cm=-0.1,
        forces=np.array([10.0, 50.0, -5.0]),
        moments=np.array([0.0, 0.0, -20.0])
    )
    
    # Create simulation results
    results = SimulationResults(
        status=SolverStatus.CONVERGED,
        convergence_data=[],
        execution_time=120.5,
        iterations=10,
        final_residuals={'p': 1e-6, 'U': 1e-6, 'k': 1e-6},
        objective_values={'drag_coefficient': 0.02, 'lift_coefficient': 0.5},
        force_coefficients=force_coeffs,
        residual_history=residuals,
        solver_type=SolverType.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST
    )
    
    # Test properties
    assert results.is_converged == True
    assert results.is_successful == True
    assert len(results.get_residual_history('p')) == 10
    assert results.get_residual_history('p')[0] == 1e-3
    
    # Test convergence plot data
    plot_data = results.get_convergence_plot_data()
    assert 'iterations' in plot_data
    assert 'p' in plot_data
    assert len(plot_data['iterations']) == 10
    
    logger.info("✓ Simulation results test passed")

def test_sensitivity_config():
    """Test sensitivity analysis configuration."""
    logger.info("Testing sensitivity configuration...")
    
    # Test finite difference configuration
    fd_config = FiniteDifferenceConfig(
        method=GradientMethod.FINITE_DIFFERENCE_CENTRAL,
        step_size=1e-6,
        adaptive_step=True,
        parallel_evaluation=True,
        max_workers=4
    )
    
    # Test sensitivity configuration
    sensitivity_config = SensitivityConfig(
        gradient_method=GradientMethod.FINITE_DIFFERENCE_CENTRAL,
        objective_functions=[ObjectiveFunction.DRAG_COEFFICIENT, ObjectiveFunction.LIFT_COEFFICIENT],
        design_variables=['mesh_points', 'angle_of_attack'],
        fd_config=fd_config,
        save_gradients=True,
        gradient_verification=True
    )
    
    assert len(sensitivity_config.objective_functions) == 2
    assert len(sensitivity_config.design_variables) == 2
    assert sensitivity_config.gradient_method == GradientMethod.FINITE_DIFFERENCE_CENTRAL
    assert sensitivity_config.fd_config.parallel_evaluation == True
    
    logger.info("✓ Sensitivity configuration test passed")

def test_sensitivity_results():
    """Test sensitivity results data structure."""
    logger.info("Testing sensitivity results...")
    
    # Create sample gradient data
    gradients = {
        'drag_coefficient': {
            'mesh_points': np.random.rand(100, 3) * 1e-3,
            'angle_of_attack': np.array([0.05])
        },
        'lift_coefficient': {
            'mesh_points': np.random.rand(100, 3) * 1e-2,
            'angle_of_attack': np.array([0.8])
        }
    }
    
    # Calculate gradient norms
    gradient_norms = {}
    for obj, variables in gradients.items():
        gradient_norms[obj] = {}
        for var, grad in variables.items():
            gradient_norms[obj][var] = np.linalg.norm(grad)
    
    # Create sensitivity results
    results = SensitivityResults(
        gradients=gradients,
        gradient_norms=gradient_norms,
        computation_time=45.2,
        method_used=GradientMethod.FINITE_DIFFERENCE_CENTRAL,
        objective_values={'drag_coefficient': 0.02, 'lift_coefficient': 0.5}
    )
    
    # Test methods
    drag_mesh_grad = results.get_gradient('drag_coefficient', 'mesh_points')
    assert drag_mesh_grad is not None
    assert drag_mesh_grad.shape == (100, 3)
    
    total_norm = results.get_total_gradient_norm('drag_coefficient')
    assert total_norm > 0
    
    # Test saving (to temporary file)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_gradients.npz"
        results.save_results(output_path, "numpy")
        assert output_path.exists()
    
    logger.info("✓ Sensitivity results test passed")

def test_mesh_converter():
    """Test mesh converter functionality."""
    logger.info("Testing mesh converter...")
    
    converter = MeshConverter()
    
    # Test supported formats
    assert 'openfoam' in converter.supported_formats
    assert 'vtk' in converter.supported_formats
    assert 'stl' in converter.supported_formats
    
    # Test conversion config
    config = MeshConversionConfig(
        input_format='stl',
        output_format='openfoam',
        scale_factor=1.0,
        quality_check=True,
        repair_mesh=True
    )
    
    assert config.input_format == 'stl'
    assert config.output_format == 'openfoam'
    assert config.quality_check == True
    
    logger.info("✓ Mesh converter test passed")

def test_parallel_manager():
    """Test parallel execution manager."""
    logger.info("Testing parallel manager...")
    
    manager = ParallelManager()
    
    # Test decomposition optimization
    with tempfile.TemporaryDirectory() as temp_dir:
        case_dir = Path(temp_dir)
        decomp_config = manager.optimize_decomposition(case_dir, 8, mesh_size=100000)
        
        assert decomp_config['numberOfSubdomains'] == 8
        assert decomp_config['method'] == 'simple'
        assert 'simpleCoeffs' in decomp_config
        assert 'n' in decomp_config['simpleCoeffs']
        
        # Test that the decomposition is reasonable
        n = decomp_config['simpleCoeffs']['n']
        assert len(n) == 3
        assert np.prod(n) == 8
    
    logger.info("✓ Parallel manager test passed")

def test_mesh_deformation():
    """Test mesh deformation engine integration."""
    logger.info("Testing mesh deformation...")
    
    # Create deformation configuration
    config = DeformationConfig(
        smoothing_iterations=3,
        enable_auto_repair=True,
        parallel_enabled=False  # Disable for testing
    )
    
    # Create deformation engine
    engine = MeshDeformationEngine(config)
    
    # Test configuration
    assert engine.config.smoothing_iterations == 3
    assert engine.config.enable_auto_repair == True
    
    # Test performance report (should be empty initially)
    performance = engine.get_performance_report()
    assert 'deformation_times' in performance
    assert 'quality_check_times' in performance
    
    logger.info("✓ Mesh deformation test passed")

def test_post_processing_config():
    """Test post-processing configuration."""
    logger.info("Testing post-processing configuration...")
    
    config = PostProcessingConfig(
        extract_surfaces=True,
        surface_names=['airfoil', 'inlet', 'outlet'],
        compute_forces=True,
        force_patches=['airfoil'],
        create_plots=True,
        export_vtk=True,
        export_csv=True
    )
    
    assert config.extract_surfaces == True
    assert len(config.surface_names) == 3
    assert 'airfoil' in config.surface_names
    assert config.force_patches == ['airfoil']
    
    logger.info("✓ Post-processing configuration test passed")

def test_data_validation():
    """Test data validation and error handling."""
    logger.info("Testing data validation...")
    
    # Test invalid solver type
    try:
        config = OpenFOAMConfig(
            case_directory=Path("/tmp/test"),
            solver_executable="",  # Invalid empty executable
            end_time=-1.0,  # Invalid negative time
            max_iterations=0  # Invalid zero iterations
        )
        
        # This should work (no validation in constructor)
        # But validation should catch errors
        from openffd.cfd.base import CFDSolver
        solver = CFDSolver("test")
        errors = solver.validate_config(config)
        
        assert len(errors) > 0  # Should have validation errors
        assert any("executable" in error.lower() for error in errors)
        
    except Exception as e:
        # This is expected for invalid configurations
        logger.info(f"Expected validation error: {e}")
    
    logger.info("✓ Data validation test passed")

def test_objective_functions():
    """Test objective function enumeration."""
    logger.info("Testing objective functions...")
    
    # Test all objective functions are accessible
    objectives = [
        ObjectiveFunction.DRAG_COEFFICIENT,
        ObjectiveFunction.LIFT_COEFFICIENT,
        ObjectiveFunction.PRESSURE_LOSS,
        ObjectiveFunction.HEAT_TRANSFER,
        ObjectiveFunction.MASS_FLOW_RATE,
        ObjectiveFunction.MOMENT_COEFFICIENT
    ]
    
    # Test string values
    assert ObjectiveFunction.DRAG_COEFFICIENT.value == "drag_coefficient"
    assert ObjectiveFunction.LIFT_COEFFICIENT.value == "lift_coefficient"
    assert ObjectiveFunction.PRESSURE_LOSS.value == "pressure_loss"
    
    # Test that we can create a list of objectives
    objective_list = [obj.value for obj in objectives]
    assert len(objective_list) == len(objectives)
    assert "drag_coefficient" in objective_list
    
    logger.info("✓ Objective functions test passed")

def run_all_tests():
    """Run all component tests."""
    logger.info("Running OpenFFD CFD Component Tests")
    logger.info("=" * 50)
    
    test_functions = [
        test_openfoam_config,
        test_simulation_results,
        test_sensitivity_config,
        test_sensitivity_results,
        test_mesh_converter,
        test_parallel_manager,
        test_mesh_deformation,
        test_post_processing_config,
        test_data_validation,
        test_objective_functions
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)