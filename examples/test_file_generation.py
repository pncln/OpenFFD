#!/usr/bin/env python3
"""
Test OpenFOAM File Generation

This script thoroughly tests the OpenFOAM file generation capabilities
to ensure all case files are created correctly with proper content.
"""

import tempfile
import logging
from pathlib import Path
from openffd.cfd.openfoam import OpenFOAMConfig, SolverType, TurbulenceModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOpenFOAMFileGeneration:
    """Test class for OpenFOAM file generation."""
    
    def __init__(self):
        """Initialize test class."""
        self.logger = logging.getLogger(__name__)
    
    def create_mock_solver(self):
        """Create a mock OpenFOAM solver for testing."""
        from openffd.cfd.openfoam import OpenFOAMSolver
        
        # Create mock solver without requiring OpenFOAM installation
        solver = OpenFOAMSolver.__new__(OpenFOAMSolver)
        solver.name = 'MockOpenFOAMSolver'
        solver.version = 'test-v2112'
        solver.logger = logging.getLogger('mock_solver')
        solver.openfoam_root = Path('/mock/openfoam/path')
        solver._process = None
        
        return solver
    
    def test_control_dict_generation(self):
        """Test controlDict file generation."""
        logger.info("Testing controlDict generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "system").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                turbulence_model=TurbulenceModel.K_OMEGA_SST,
                end_time=100.0,
                time_step=1.0,
                max_iterations=1000,
                write_interval=50
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_control_dict(config)
            
            # Check file exists and has content
            control_dict_path = case_dir / "system" / "controlDict"
            assert control_dict_path.exists(), "controlDict file was not created"
            
            # Check file size
            file_size = control_dict_path.stat().st_size
            assert file_size > 0, f"controlDict file is empty (size: {file_size})"
            logger.info(f"✓ controlDict created with {file_size} bytes")
            
            # Check file content
            with open(control_dict_path, 'r') as f:
                content = f.read()
                
                # Check essential content
                required_content = [
                    "FoamFile",
                    "controlDict",
                    "simpleFoam",
                    f"endTime        {config.end_time}",
                    f"deltaT        {config.time_step}",
                    f"writeInterval        {config.write_interval}"
                ]
                
                for req_content in required_content:
                    assert req_content in content, f"Missing required content: {req_content}"
                
                logger.info("✓ controlDict content validation passed")
            
            return True
    
    def test_fv_schemes_generation(self):
        """Test fvSchemes file generation."""
        logger.info("Testing fvSchemes generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "system").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_fv_schemes(config)
            
            # Check file exists and has content
            fv_schemes_path = case_dir / "system" / "fvSchemes"
            assert fv_schemes_path.exists(), "fvSchemes file was not created"
            
            # Check file size
            file_size = fv_schemes_path.stat().st_size
            assert file_size > 0, f"fvSchemes file is empty (size: {file_size})"
            logger.info(f"✓ fvSchemes created with {file_size} bytes")
            
            # Check file content
            with open(fv_schemes_path, 'r') as f:
                content = f.read()
                
                # Check essential content
                required_content = [
                    "FoamFile",
                    "fvSchemes",
                    "ddtSchemes",
                    "gradSchemes",
                    "divSchemes",
                    "laplacianSchemes",
                    "Gauss linear"
                ]
                
                for req_content in required_content:
                    assert req_content in content, f"Missing required content: {req_content}"
                
                logger.info("✓ fvSchemes content validation passed")
            
            return True
    
    def test_fv_solution_generation(self):
        """Test fvSolution file generation."""
        logger.info("Testing fvSolution generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "system").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                convergence_tolerance={
                    'p': 1e-6,
                    'U': 1e-6,
                    'k': 1e-6,
                    'omega': 1e-6
                }
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_fv_solution(config)
            
            # Check file exists and has content
            fv_solution_path = case_dir / "system" / "fvSolution"
            assert fv_solution_path.exists(), "fvSolution file was not created"
            
            # Check file size
            file_size = fv_solution_path.stat().st_size
            assert file_size > 0, f"fvSolution file is empty (size: {file_size})"
            logger.info(f"✓ fvSolution created with {file_size} bytes")
            
            # Check file content
            with open(fv_solution_path, 'r') as f:
                content = f.read()
                
                # Check essential content
                required_content = [
                    "FoamFile",
                    "fvSolution",
                    "solvers",
                    "SIMPLE",
                    "relaxationFactors",
                    "GAMG",
                    "1e-06"  # convergence tolerance
                ]
                
                for req_content in required_content:
                    assert req_content in content, f"Missing required content: {req_content}"
                
                logger.info("✓ fvSolution content validation passed")
            
            return True
    
    def test_transport_properties_generation(self):
        """Test transportProperties file generation."""
        logger.info("Testing transportProperties generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "constant").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                fluid_properties={
                    'nu': 1.5e-5,
                    'rho': 1.225
                }
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_transport_properties(config)
            
            # Check file exists and has content
            transport_path = case_dir / "constant" / "transportProperties"
            assert transport_path.exists(), "transportProperties file was not created"
            
            # Check file size
            file_size = transport_path.stat().st_size
            assert file_size > 0, f"transportProperties file is empty (size: {file_size})"
            logger.info(f"✓ transportProperties created with {file_size} bytes")
            
            # Check file content
            with open(transport_path, 'r') as f:
                content = f.read()
                
                # Check essential content
                required_content = [
                    "FoamFile",
                    "transportProperties",
                    "transportModel",
                    "Newtonian",
                    "nu",
                    "1.5e-05"
                ]
                
                for req_content in required_content:
                    assert req_content in content, f"Missing required content: {req_content}"
                
                logger.info("✓ transportProperties content validation passed")
            
            return True
    
    def test_turbulence_properties_generation(self):
        """Test turbulenceProperties file generation."""
        logger.info("Testing turbulenceProperties generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "constant").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                turbulence_model=TurbulenceModel.K_OMEGA_SST
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_turbulence_properties(config)
            
            # Check file exists and has content
            turbulence_path = case_dir / "constant" / "turbulenceProperties"
            assert turbulence_path.exists(), "turbulenceProperties file was not created"
            
            # Check file size
            file_size = turbulence_path.stat().st_size
            assert file_size > 0, f"turbulenceProperties file is empty (size: {file_size})"
            logger.info(f"✓ turbulenceProperties created with {file_size} bytes")
            
            # Check file content
            with open(turbulence_path, 'r') as f:
                content = f.read()
                
                # Check essential content
                required_content = [
                    "FoamFile",
                    "turbulenceProperties",
                    "simulationType",
                    "RAS",
                    "RASModel",
                    "kOmegaSST"
                ]
                
                for req_content in required_content:
                    assert req_content in content, f"Missing required content: {req_content}"
                
                logger.info("✓ turbulenceProperties content validation passed")
            
            return True
    
    def test_boundary_conditions_generation(self):
        """Test boundary condition files generation."""
        logger.info("Testing boundary condition files generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "test_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "0").mkdir(exist_ok=True)
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                boundary_conditions={
                    'inlet': {
                        'U': {'type': 'fixedValue', 'value': 'uniform (10 0 0)'},
                        'p': {'type': 'zeroGradient'}
                    },
                    'outlet': {
                        'U': {'type': 'zeroGradient'},
                        'p': {'type': 'fixedValue', 'value': 'uniform 0'}
                    },
                    'walls': {
                        'U': {'type': 'noSlip'},
                        'p': {'type': 'zeroGradient'}
                    }
                }
            )
            
            # Create mock solver and test file generation
            solver = self.create_mock_solver()
            solver._write_boundary_conditions(config)
            
            # Check U file
            u_file = case_dir / "0" / "U"
            assert u_file.exists(), "U file was not created"
            
            u_size = u_file.stat().st_size
            assert u_size > 0, f"U file is empty (size: {u_size})"
            logger.info(f"✓ U file created with {u_size} bytes")
            
            # Check p file
            p_file = case_dir / "0" / "p"
            assert p_file.exists(), "p file was not created"
            
            p_size = p_file.stat().st_size
            assert p_size > 0, f"p file is empty (size: {p_size})"
            logger.info(f"✓ p file created with {p_size} bytes")
            
            # Check U file content
            with open(u_file, 'r') as f:
                u_content = f.read()
                required_u_content = [
                    "FoamFile",
                    "volVectorField",
                    "dimensions",
                    "internalField",
                    "boundaryField"
                ]
                
                for req_content in required_u_content:
                    assert req_content in u_content, f"Missing required U content: {req_content}"
            
            # Check p file content
            with open(p_file, 'r') as f:
                p_content = f.read()
                required_p_content = [
                    "FoamFile",
                    "volScalarField",
                    "dimensions",
                    "internalField",
                    "boundaryField"
                ]
                
                for req_content in required_p_content:
                    assert req_content in p_content, f"Missing required p content: {req_content}"
            
            logger.info("✓ Boundary condition files content validation passed")
            
            return True
    
    def test_complete_case_setup(self):
        """Test complete OpenFOAM case setup."""
        logger.info("Testing complete case setup...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "complete_test_case"
            
            config = OpenFOAMConfig(
                case_directory=case_dir,
                solver_executable="simpleFoam",
                solver_type=SolverType.SIMPLE_FOAM,
                turbulence_model=TurbulenceModel.K_OMEGA_SST,
                end_time=100.0,
                time_step=1.0,
                max_iterations=1000,
                parallel_execution=True,
                num_processors=4,
                force_calculation=True,
                force_patches=['airfoil'],
                boundary_conditions={
                    'inlet': {
                        'U': {'type': 'fixedValue', 'value': 'uniform (50 0 0)'},
                        'p': {'type': 'zeroGradient'}
                    },
                    'outlet': {
                        'U': {'type': 'zeroGradient'},
                        'p': {'type': 'fixedValue', 'value': 'uniform 0'}
                    },
                    'airfoil': {
                        'U': {'type': 'noSlip'},
                        'p': {'type': 'zeroGradient'}
                    }
                }
            )
            
            # Create mock solver
            solver = self.create_mock_solver()
            
            # Test case setup
            success = solver.prepare_case_directory(config)
            assert success, "Case directory preparation failed"
            
            # Generate all files
            solver._write_control_dict(config)
            solver._write_fv_schemes(config)
            solver._write_fv_solution(config)
            solver._write_transport_properties(config)
            solver._write_turbulence_properties(config)
            solver._write_boundary_conditions(config)
            solver._setup_decomposition(config)
            solver._add_function_objects(config)
            
            # Re-write controlDict with function objects
            solver._write_control_dict(config)
            
            # Check all expected files exist
            expected_files = [
                "system/controlDict",
                "system/fvSchemes", 
                "system/fvSolution",
                "system/decomposeParDict",
                "constant/transportProperties",
                "constant/turbulenceProperties",
                "0/U",
                "0/p"
            ]
            
            all_files_exist = True
            total_size = 0
            
            for file_path in expected_files:
                full_path = case_dir / file_path
                if not full_path.exists():
                    logger.error(f"❌ Missing file: {file_path}")
                    all_files_exist = False
                else:
                    file_size = full_path.stat().st_size
                    total_size += file_size
                    if file_size == 0:
                        logger.warning(f"⚠️ Empty file: {file_path}")
                    else:
                        logger.info(f"✓ {file_path}: {file_size} bytes")
            
            assert all_files_exist, "Some expected files are missing"
            assert total_size > 0, "All files are empty"
            
            logger.info(f"✓ Complete case setup successful - Total size: {total_size} bytes")
            
            return True
    
    def run_all_tests(self):
        """Run all file generation tests."""
        logger.info("Starting OpenFOAM File Generation Tests")
        logger.info("=" * 50)
        
        tests = [
            self.test_control_dict_generation,
            self.test_fv_schemes_generation,
            self.test_fv_solution_generation,
            self.test_transport_properties_generation,
            self.test_turbulence_properties_generation,
            self.test_boundary_conditions_generation,
            self.test_complete_case_setup
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                logger.error(f"❌ Test {test.__name__} failed: {e}")
                failed += 1
        
        logger.info(f"\nFile Generation Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            logger.info("🎉 All file generation tests passed!")
            logger.info("✅ OpenFOAM case files are created correctly with proper content!")
        else:
            logger.error(f"❌ {failed} file generation tests failed")
        
        return failed == 0

def main():
    """Main test function."""
    tester = TestOpenFOAMFileGeneration()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())