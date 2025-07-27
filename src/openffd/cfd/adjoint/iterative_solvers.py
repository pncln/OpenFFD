"""
Iterative Solvers for Adjoint Equations

Implements various iterative methods for solving large sparse linear systems
arising from discrete adjoint equations: (∂R/∂U)^T λ = -∂J/∂U
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab, cg
from scipy.sparse.linalg import spilu, factorized
from scipy.sparse import diags, identity

logger = logging.getLogger(__name__)


@dataclass
class IterativeSolverConfig:
    """Configuration for iterative solvers."""
    
    # Convergence criteria
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-12
    max_iterations: int = 1000
    
    # GMRES specific
    restart_frequency: int = 30
    
    # Preconditioning
    use_preconditioning: bool = True
    preconditioner_type: str = "ilu"  # "ilu", "jacobi", "block_jacobi", "none"
    ilu_fill_factor: float = 10.0
    ilu_drop_tolerance: float = 1e-4
    
    # Monitoring
    monitor_convergence: bool = True
    print_residuals: bool = False
    save_convergence_history: bool = True
    
    # Performance
    use_initial_guess: bool = True
    orthogonalization_method: str = "mgs"  # "mgs", "cgs"


class AdjointIterativeSolver(ABC):
    """Abstract base class for iterative adjoint solvers."""
    
    def __init__(self, config: Optional[IterativeSolverConfig] = None):
        """
        Initialize iterative solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config or IterativeSolverConfig()
        
        # Convergence history
        self.residual_history: List[float] = []
        self.convergence_history: List[float] = []
        
        # Solver statistics
        self.iterations_performed = 0
        self.final_residual = 0.0
        self.solve_time = 0.0
        self.is_converged = False
        
        # Preconditioner
        self.preconditioner: Optional[LinearOperator] = None
        
    @abstractmethod
    def solve(self,
              matrix: Union[csr_matrix, LinearOperator],
              rhs: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> bool:
        """
        Solve linear system Ax = b.
        
        Args:
            matrix: System matrix or linear operator
            rhs: Right-hand side vector
            initial_guess: Initial guess for solution
            
        Returns:
            True if converged, False otherwise
        """
        pass
    
    def setup_preconditioner(self, matrix: csr_matrix) -> None:
        """Setup preconditioner for the given matrix."""
        if not self.config.use_preconditioning:
            self.preconditioner = None
            return
        
        try:
            if self.config.preconditioner_type == "ilu":
                self.preconditioner = self._create_ilu_preconditioner(matrix)
            elif self.config.preconditioner_type == "jacobi":
                self.preconditioner = self._create_jacobi_preconditioner(matrix)
            elif self.config.preconditioner_type == "block_jacobi":
                self.preconditioner = self._create_block_jacobi_preconditioner(matrix)
            else:
                self.preconditioner = None
                
        except Exception as e:
            logger.warning(f"Failed to create preconditioner: {e}")
            self.preconditioner = None
    
    def _create_ilu_preconditioner(self, matrix: csr_matrix) -> LinearOperator:
        """Create ILU preconditioner."""
        ilu = spilu(matrix, 
                   fill_factor=self.config.ilu_fill_factor,
                   drop_tol=self.config.ilu_drop_tolerance)
        
        def matvec(x):
            return ilu.solve(x)
        
        return LinearOperator(matrix.shape, matvec=matvec)
    
    def _create_jacobi_preconditioner(self, matrix: csr_matrix) -> LinearOperator:
        """Create Jacobi (diagonal) preconditioner."""
        diagonal = matrix.diagonal()
        # Avoid division by zero
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        inv_diagonal = 1.0 / diagonal
        
        def matvec(x):
            return inv_diagonal * x
        
        return LinearOperator(matrix.shape, matvec=matvec)
    
    def _create_block_jacobi_preconditioner(self, matrix: csr_matrix, block_size: int = 5) -> LinearOperator:
        """Create block Jacobi preconditioner."""
        n = matrix.shape[0]
        n_blocks = n // block_size
        
        # Extract diagonal blocks and compute their inverses
        block_inverses = []
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n)
            
            # Extract diagonal block
            block = matrix[start_idx:end_idx, start_idx:end_idx].toarray()
            
            # Compute inverse
            try:
                block_inv = np.linalg.inv(block)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                block_inv = np.linalg.pinv(block)
            
            block_inverses.append(block_inv)
        
        def matvec(x):
            result = np.zeros_like(x)
            for i, block_inv in enumerate(block_inverses):
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, n)
                result[start_idx:end_idx] = block_inv @ x[start_idx:end_idx]
            return result
        
        return LinearOperator(matrix.shape, matvec=matvec)
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get comprehensive solver statistics."""
        return {
            'iterations_performed': self.iterations_performed,
            'final_residual': self.final_residual,
            'solve_time': self.solve_time,
            'is_converged': self.is_converged,
            'residual_history': self.residual_history.copy(),
            'convergence_rate': self._compute_convergence_rate(),
            'preconditioner_type': self.config.preconditioner_type,
            'relative_tolerance': self.config.relative_tolerance
        }
    
    def _compute_convergence_rate(self) -> float:
        """Compute average convergence rate."""
        if len(self.residual_history) < 2:
            return 0.0
        
        # Geometric mean of residual ratios
        ratios = []
        for i in range(1, len(self.residual_history)):
            if self.residual_history[i-1] > 1e-15:
                ratio = self.residual_history[i] / self.residual_history[i-1]
                ratios.append(ratio)
        
        if len(ratios) == 0:
            return 0.0
        
        return np.exp(np.mean(np.log(np.clip(ratios, 1e-15, 1.0))))


class GMRESAdjointSolver(AdjointIterativeSolver):
    """
    GMRES solver for adjoint equations.
    
    Generalized Minimal Residual method with restart capability.
    Well-suited for non-symmetric systems like adjoint equations.
    """
    
    def __init__(self, config: Optional[IterativeSolverConfig] = None):
        """Initialize GMRES solver."""
        super().__init__(config)
        self.solver_name = "GMRES"
        
    def solve(self,
              matrix: Union[csr_matrix, LinearOperator],
              rhs: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> bool:
        """Solve using GMRES method."""
        import time
        start_time = time.time()
        
        # Setup preconditioner
        if isinstance(matrix, csr_matrix):
            self.setup_preconditioner(matrix)
        
        # Prepare initial guess
        if initial_guess is None or not self.config.use_initial_guess:
            x0 = np.zeros_like(rhs)
        else:
            x0 = initial_guess.copy()
        
        # Callback for monitoring convergence
        residuals = []
        def callback(residual_norm):
            residuals.append(residual_norm)
            if self.config.print_residuals:
                logger.info(f"GMRES iteration {len(residuals)}: residual = {residual_norm:.2e}")
        
        # Solve using GMRES
        try:
            solution, info = gmres(
                matrix,
                rhs,
                x0=x0,
                tol=self.config.relative_tolerance,
                restart=self.config.restart_frequency,
                maxiter=self.config.max_iterations,
                M=self.preconditioner,
                callback=callback,
                atol=self.config.absolute_tolerance
            )
            
            # Store results
            self.iterations_performed = len(residuals)
            self.residual_history = residuals
            self.final_residual = residuals[-1] if residuals else 0.0
            self.is_converged = (info == 0)
            self.solve_time = time.time() - start_time
            
            # Store solution in initial_guess array
            if initial_guess is not None:
                initial_guess[:] = solution
            
            if self.config.monitor_convergence:
                logger.info(f"GMRES converged: {self.is_converged}, "
                           f"iterations: {self.iterations_performed}, "
                           f"final residual: {self.final_residual:.2e}")
            
            return self.is_converged
            
        except Exception as e:
            logger.error(f"GMRES solve failed: {e}")
            return False


class BiCGSTABAdjointSolver(AdjointIterativeSolver):
    """
    BiCGSTAB solver for adjoint equations.
    
    Biconjugate Gradient Stabilized method.
    Often faster than GMRES but may be less stable.
    """
    
    def __init__(self, config: Optional[IterativeSolverConfig] = None):
        """Initialize BiCGSTAB solver."""
        super().__init__(config)
        self.solver_name = "BiCGSTAB"
        
    def solve(self,
              matrix: Union[csr_matrix, LinearOperator],
              rhs: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> bool:
        """Solve using BiCGSTAB method."""
        import time
        start_time = time.time()
        
        # Setup preconditioner
        if isinstance(matrix, csr_matrix):
            self.setup_preconditioner(matrix)
        
        # Prepare initial guess
        if initial_guess is None or not self.config.use_initial_guess:
            x0 = np.zeros_like(rhs)
        else:
            x0 = initial_guess.copy()
        
        # Callback for monitoring
        residuals = []
        def callback(residual_norm):
            residuals.append(residual_norm)
            if self.config.print_residuals:
                logger.info(f"BiCGSTAB iteration {len(residuals)}: residual = {residual_norm:.2e}")
        
        # Solve using BiCGSTAB
        try:
            solution, info = bicgstab(
                matrix,
                rhs,
                x0=x0,
                tol=self.config.relative_tolerance,
                maxiter=self.config.max_iterations,
                M=self.preconditioner,
                callback=callback,
                atol=self.config.absolute_tolerance
            )
            
            # Store results
            self.iterations_performed = len(residuals)
            self.residual_history = residuals
            self.final_residual = residuals[-1] if residuals else 0.0
            self.is_converged = (info == 0)
            self.solve_time = time.time() - start_time
            
            # Store solution
            if initial_guess is not None:
                initial_guess[:] = solution
            
            if self.config.monitor_convergence:
                logger.info(f"BiCGSTAB converged: {self.is_converged}, "
                           f"iterations: {self.iterations_performed}, "
                           f"final residual: {self.final_residual:.2e}")
            
            return self.is_converged
            
        except Exception as e:
            logger.error(f"BiCGSTAB solve failed: {e}")
            return False


class PreconditionedSolver(AdjointIterativeSolver):
    """
    Preconditioned solver with multiple algorithm options.
    
    Automatically selects appropriate solver based on problem characteristics.
    """
    
    def __init__(self, 
                 solver_type: str = "auto",  # "auto", "gmres", "bicgstab", "cg"
                 config: Optional[IterativeSolverConfig] = None):
        """
        Initialize preconditioned solver.
        
        Args:
            solver_type: Type of iterative solver to use
            config: Solver configuration
        """
        super().__init__(config)
        self.solver_type = solver_type
        self.selected_solver = None
        
    def solve(self,
              matrix: Union[csr_matrix, LinearOperator],
              rhs: np.ndarray,
              initial_guess: Optional[np.ndarray] = None) -> bool:
        """Solve using automatically selected or specified solver."""
        # Select solver
        if self.solver_type == "auto":
            solver_type = self._select_solver(matrix, rhs)
        else:
            solver_type = self.solver_type
        
        # Create and configure solver
        if solver_type == "gmres":
            self.selected_solver = GMRESAdjointSolver(self.config)
        elif solver_type == "bicgstab":
            self.selected_solver = BiCGSTABAdjointSolver(self.config)
        else:
            # Default to GMRES
            self.selected_solver = GMRESAdjointSolver(self.config)
        
        # Solve
        success = self.selected_solver.solve(matrix, rhs, initial_guess)
        
        # Copy statistics
        self.iterations_performed = self.selected_solver.iterations_performed
        self.residual_history = self.selected_solver.residual_history
        self.final_residual = self.selected_solver.final_residual
        self.is_converged = self.selected_solver.is_converged
        self.solve_time = self.selected_solver.solve_time
        
        return success
    
    def _select_solver(self, matrix: Union[csr_matrix, LinearOperator], rhs: np.ndarray) -> str:
        """Automatically select appropriate solver."""
        # Simple heuristics for solver selection
        n = len(rhs)
        
        if n < 10000:
            # Small problems: use GMRES
            return "gmres"
        elif n < 100000:
            # Medium problems: try BiCGSTAB first
            return "bicgstab"
        else:
            # Large problems: use GMRES with larger restart
            self.config.restart_frequency = min(100, n // 1000)
            return "gmres"


class MultiLevelPreconditioner:
    """
    Multilevel preconditioner for adjoint equations.
    
    Uses multigrid-like approach for preconditioning large systems.
    """
    
    def __init__(self, 
                 matrix: csr_matrix,
                 n_levels: int = 3,
                 coarsening_ratio: float = 0.25):
        """
        Initialize multilevel preconditioner.
        
        Args:
            matrix: Fine-level matrix
            n_levels: Number of multigrid levels
            coarsening_ratio: Coarsening ratio between levels
        """
        self.matrix = matrix
        self.n_levels = n_levels
        self.coarsening_ratio = coarsening_ratio
        
        # Build multigrid hierarchy
        self.matrices = [matrix]
        self.restrictors = []
        self.prolongators = []
        
        self._build_hierarchy()
        
    def _build_hierarchy(self) -> None:
        """Build multigrid hierarchy."""
        current_matrix = self.matrix
        
        for level in range(1, self.n_levels):
            # Create restriction operator (simplified algebraic multigrid)
            n_current = current_matrix.shape[0]
            n_coarse = max(int(n_current * self.coarsening_ratio), 10)
            
            # Simple injection restriction
            restriction = self._create_injection_restriction(n_current, n_coarse)
            prolongation = restriction.T
            
            # Coarse matrix: R * A * P
            coarse_matrix = restriction @ current_matrix @ prolongation
            
            self.restrictors.append(restriction)
            self.prolongators.append(prolongation)
            self.matrices.append(coarse_matrix)
            
            current_matrix = coarse_matrix
    
    def _create_injection_restriction(self, n_fine: int, n_coarse: int) -> csr_matrix:
        """Create injection restriction operator."""
        step = n_fine // n_coarse
        rows = []
        cols = []
        data = []
        
        for i in range(n_coarse):
            fine_index = min(i * step, n_fine - 1)
            rows.append(i)
            cols.append(fine_index)
            data.append(1.0)
        
        return csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))
    
    def apply_preconditioner(self, residual: np.ndarray) -> np.ndarray:
        """Apply V-cycle multigrid preconditioner."""
        return self._v_cycle(residual, 0)
    
    def _v_cycle(self, residual: np.ndarray, level: int) -> np.ndarray:
        """Recursive V-cycle implementation."""
        if level == self.n_levels - 1:
            # Coarsest level: direct solve
            try:
                return spilu(self.matrices[level]).solve(residual)
            except:
                return np.linalg.solve(self.matrices[level].toarray(), residual)
        
        # Pre-smoothing (simplified Jacobi)
        solution = self._smooth(self.matrices[level], residual, np.zeros_like(residual))
        
        # Compute residual
        new_residual = residual - self.matrices[level] @ solution
        
        # Restrict to coarse level
        coarse_residual = self.restrictors[level] @ new_residual
        
        # Solve on coarse level
        coarse_correction = self._v_cycle(coarse_residual, level + 1)
        
        # Prolongate correction
        fine_correction = self.prolongators[level] @ coarse_correction
        
        # Add correction
        solution += fine_correction
        
        # Post-smoothing
        solution = self._smooth(self.matrices[level], residual, solution)
        
        return solution
    
    def _smooth(self, matrix: csr_matrix, rhs: np.ndarray, solution: np.ndarray, n_iter: int = 2) -> np.ndarray:
        """Apply smoothing iterations (weighted Jacobi)."""
        diagonal = matrix.diagonal()
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        
        omega = 0.7  # Relaxation parameter
        
        for _ in range(n_iter):
            residual = rhs - matrix @ solution
            solution += omega * (residual / diagonal)
        
        return solution


def create_adjoint_solver(solver_type: str = "gmres",
                         config: Optional[IterativeSolverConfig] = None) -> AdjointIterativeSolver:
    """
    Factory function for creating adjoint solvers.
    
    Args:
        solver_type: Type of solver ("gmres", "bicgstab", "preconditioned")
        config: Solver configuration
        
    Returns:
        Configured adjoint solver
    """
    if solver_type == "gmres":
        return GMRESAdjointSolver(config)
    elif solver_type == "bicgstab":
        return BiCGSTABAdjointSolver(config)
    elif solver_type == "preconditioned":
        return PreconditionedSolver("auto", config)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def test_iterative_solvers():
    """Test iterative solvers on synthetic problems."""
    print("Testing Iterative Solvers:")
    
    # Create test problem
    n = 1000
    np.random.seed(42)
    
    # Create SPD matrix for testing
    A_dense = np.random.rand(n, n)
    A_dense = A_dense + A_dense.T + n * np.eye(n)  # Make SPD
    A = csr_matrix(A_dense)
    
    # Right-hand side
    x_exact = np.random.rand(n)
    b = A @ x_exact
    
    # Test different solvers
    solvers = {
        "GMRES": GMRESAdjointSolver(),
        "BiCGSTAB": BiCGSTABAdjointSolver(),
        "Preconditioned": PreconditionedSolver("auto")
    }
    
    config = IterativeSolverConfig(
        relative_tolerance=1e-8,
        max_iterations=100,
        use_preconditioning=True
    )
    
    for name, solver in solvers.items():
        solver.config = config
        x_initial = np.zeros(n)
        
        print(f"\n  Testing {name}:")
        success = solver.solve(A, b, x_initial)
        
        if success:
            error = np.linalg.norm(x_initial - x_exact)
            stats = solver.get_solver_statistics()
            
            print(f"    Converged: {success}")
            print(f"    Iterations: {stats['iterations_performed']}")
            print(f"    Final residual: {stats['final_residual']:.2e}")
            print(f"    Solution error: {error:.2e}")
            print(f"    Solve time: {stats['solve_time']:.3f}s")
        else:
            print(f"    Failed to converge")


if __name__ == "__main__":
    test_iterative_solvers()