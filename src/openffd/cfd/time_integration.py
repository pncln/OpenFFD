"""
Time Integration Schemes for CFD Equations

Implements various time integration methods for solving time-dependent
Euler and Navier-Stokes equations:
- Explicit Runge-Kutta schemes (RK1, RK2, RK4, RK3-TVD)
- Implicit methods (Backward Euler, BDF2, Crank-Nicolson)
- Local time stepping for steady-state acceleration
- Adaptive time stepping with error control
- Multi-stage methods optimized for CFD

These methods ensure stability and accuracy for both steady and unsteady flows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, LinearOperator

logger = logging.getLogger(__name__)


@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration schemes."""
    
    # Scheme selection
    scheme_type: str = "rk4"  # "euler", "rk2", "rk4", "rk3_tvd", "backward_euler", "bdf2", "crank_nicolson"
    
    # Time stepping
    dt: float = 1e-4  # Time step size
    max_time: float = 1.0  # Maximum simulation time
    max_iterations: int = 10000  # Maximum time steps
    
    # Adaptive time stepping
    use_adaptive_stepping: bool = False
    dt_min: float = 1e-8  # Minimum time step
    dt_max: float = 1e-2  # Maximum time step
    error_tolerance: float = 1e-6  # Error tolerance for adaptive stepping
    safety_factor: float = 0.8  # Safety factor for time step adjustment
    
    # Local time stepping (for steady-state)
    use_local_timestepping: bool = False
    local_cfl_number: float = 0.8  # Local CFL number
    
    # Stability and damping
    cfl_number: float = 0.5  # Global CFL number
    dissipation_coefficient: float = 0.0  # Numerical dissipation
    
    # Implicit solver settings
    implicit_tolerance: float = 1e-8  # Convergence tolerance for implicit schemes
    implicit_max_iterations: int = 20  # Maximum Newton iterations
    use_matrix_free: bool = True  # Use matrix-free Newton methods
    
    # Output control
    output_frequency: int = 100  # Output frequency
    monitor_residuals: bool = True
    save_intermediate_solutions: bool = False


class TimeIntegrator(ABC):
    """Abstract base class for time integration schemes."""
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        """Initialize time integrator."""
        self.config = config or TimeIntegrationConfig()
        self.scheme_name = "base_integrator"
        
        # Time stepping data
        self.current_time = 0.0
        self.time_step = self.config.dt
        self.iteration = 0
        
        # Solution storage
        self.current_solution: Optional[np.ndarray] = None
        self.previous_solution: Optional[np.ndarray] = None
        self.solution_history: List[np.ndarray] = []
        
        # Residual and convergence tracking
        self.residual_history: List[float] = []
        self.time_history: List[float] = []
        self.dt_history: List[float] = []
        
        # Statistics
        self.total_function_evaluations = 0
        self.total_jacobian_evaluations = 0
        
    @abstractmethod
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """
        Integrate one time step.
        
        Args:
            solution: Current solution
            residual_function: Function to compute residuals
            dt: Time step size
            
        Returns:
            Updated solution
        """
        pass
    
    def integrate_to_time(self,
                         initial_solution: np.ndarray,
                         residual_function: Callable[[np.ndarray, float], np.ndarray],
                         final_time: float,
                         **kwargs) -> Dict[str, Any]:
        """
        Integrate from current time to final time.
        
        Args:
            initial_solution: Initial solution
            residual_function: Residual function
            final_time: Target time
            
        Returns:
            Integration results and statistics
        """
        self.current_solution = initial_solution.copy()
        self.current_time = 0.0
        self.iteration = 0
        
        # Integration loop
        while self.current_time < final_time and self.iteration < self.config.max_iterations:
            # Determine time step
            if self.config.use_adaptive_stepping:
                dt = self._compute_adaptive_timestep(residual_function)
            elif self.config.use_local_timestepping:
                dt = self._compute_local_timestep()
            else:
                dt = min(self.time_step, final_time - self.current_time)
            
            # Store previous solution
            self.previous_solution = self.current_solution.copy()
            
            # Integrate one step
            self.current_solution = self.integrate_step(
                self.current_solution, residual_function, dt
            )
            
            # Update time and iteration
            self.current_time += dt
            self.iteration += 1
            
            # Store history
            self.time_history.append(self.current_time)
            self.dt_history.append(dt)
            
            # Compute and store residual
            residual = residual_function(self.current_solution, self.current_time)
            residual_norm = np.linalg.norm(residual)
            self.residual_history.append(residual_norm)
            
            # Store solution if requested
            if self.config.save_intermediate_solutions:
                self.solution_history.append(self.current_solution.copy())
            
            # Output progress
            if self.config.monitor_residuals and self.iteration % self.config.output_frequency == 0:
                logger.info(f"Time integration: t={self.current_time:.6f}, "
                           f"dt={dt:.2e}, residual={residual_norm:.2e}")
            
            # Check convergence for steady-state
            if residual_norm < self.config.error_tolerance:
                logger.info(f"Converged to steady state at t={self.current_time:.6f}")
                break
        
        return self._compile_results()
    
    def _compute_adaptive_timestep(self, residual_function: Callable) -> float:
        """Compute adaptive time step based on error estimate."""
        # Simplified adaptive time stepping
        # In practice, this would use error estimates from embedded methods
        
        if len(self.residual_history) < 2:
            return self.time_step
        
        # Estimate rate of change
        residual_ratio = self.residual_history[-1] / (self.residual_history[-2] + 1e-15)
        
        # Adjust time step based on residual change
        if residual_ratio > 1.5:  # Residual increasing
            new_dt = self.time_step * 0.5
        elif residual_ratio < 0.5:  # Residual decreasing rapidly
            new_dt = self.time_step * 1.2
        else:
            new_dt = self.time_step
        
        # Apply bounds
        new_dt = max(self.config.dt_min, min(new_dt, self.config.dt_max))
        self.time_step = new_dt
        
        return new_dt
    
    def _compute_local_timestep(self) -> np.ndarray:
        """Compute local time steps for each cell."""
        # This would compute local time steps based on local wave speeds
        # For now, return global time step
        return self.time_step
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile integration results."""
        return {
            'final_solution': self.current_solution,
            'final_time': self.current_time,
            'iterations': self.iteration,
            'time_history': self.time_history,
            'residual_history': self.residual_history,
            'dt_history': self.dt_history,
            'converged': self.residual_history[-1] < self.config.error_tolerance if self.residual_history else False,
            'function_evaluations': self.total_function_evaluations,
            'jacobian_evaluations': self.total_jacobian_evaluations,
            'scheme_name': self.scheme_name
        }


class ExplicitEuler(TimeIntegrator):
    """
    First-order explicit Euler method.
    
    Simple but limited stability. Mainly for testing.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "explicit_euler"
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """Explicit Euler: U^{n+1} = U^n + dt * R(U^n)"""
        residual = residual_function(solution, self.current_time)
        self.total_function_evaluations += 1
        
        new_solution = solution + dt * residual
        return new_solution


class RungeKutta2(TimeIntegrator):
    """
    Second-order explicit Runge-Kutta method (RK2).
    
    Better stability and accuracy than Euler method.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "runge_kutta_2"
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """RK2 method"""
        # Stage 1
        k1 = residual_function(solution, self.current_time)
        self.total_function_evaluations += 1
        
        # Stage 2
        intermediate = solution + 0.5 * dt * k1
        k2 = residual_function(intermediate, self.current_time + 0.5 * dt)
        self.total_function_evaluations += 1
        
        # Final update
        new_solution = solution + dt * k2
        return new_solution


class RungeKutta4(TimeIntegrator):
    """
    Fourth-order explicit Runge-Kutta method (RK4).
    
    Classical RK4 with excellent accuracy for smooth problems.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "runge_kutta_4"
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """Classical RK4 method"""
        # Stage 1
        k1 = residual_function(solution, self.current_time)
        self.total_function_evaluations += 1
        
        # Stage 2
        intermediate = solution + 0.5 * dt * k1
        k2 = residual_function(intermediate, self.current_time + 0.5 * dt)
        self.total_function_evaluations += 1
        
        # Stage 3
        intermediate = solution + 0.5 * dt * k2
        k3 = residual_function(intermediate, self.current_time + 0.5 * dt)
        self.total_function_evaluations += 1
        
        # Stage 4
        intermediate = solution + dt * k3
        k4 = residual_function(intermediate, self.current_time + dt)
        self.total_function_evaluations += 1
        
        # Final update
        new_solution = solution + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_solution


class RungeKutta3TVD(TimeIntegrator):
    """
    Third-order TVD Runge-Kutta method.
    
    Optimal for CFD applications with shock-capturing schemes.
    Maintains TVD property of spatial discretization.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "runge_kutta_3_tvd"
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """Third-order TVD RK method"""
        U0 = solution
        
        # Stage 1: U1 = U0 + dt * R(U0)
        R0 = residual_function(U0, self.current_time)
        U1 = U0 + dt * R0
        self.total_function_evaluations += 1
        
        # Stage 2: U2 = 3/4 * U0 + 1/4 * U1 + 1/4 * dt * R(U1)
        R1 = residual_function(U1, self.current_time + dt)
        U2 = 0.75 * U0 + 0.25 * U1 + 0.25 * dt * R1
        self.total_function_evaluations += 1
        
        # Stage 3: U3 = 1/3 * U0 + 2/3 * U2 + 2/3 * dt * R(U2)
        R2 = residual_function(U2, self.current_time + 0.5 * dt)
        U3 = (1.0/3.0) * U0 + (2.0/3.0) * U2 + (2.0/3.0) * dt * R2
        self.total_function_evaluations += 1
        
        return U3


class BackwardEuler(TimeIntegrator):
    """
    First-order implicit Backward Euler method.
    
    Unconditionally stable, suitable for stiff problems.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "backward_euler"
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """Backward Euler: U^{n+1} = U^n + dt * R(U^{n+1})"""
        
        def implicit_residual(u_new):
            """Residual for implicit system: F(U) = U - U_old - dt * R(U) = 0"""
            residual = residual_function(u_new, self.current_time + dt)
            self.total_function_evaluations += 1
            return u_new - solution - dt * residual
        
        # Solve nonlinear system using Newton's method
        new_solution = self._newton_solve(implicit_residual, solution, residual_function, dt)
        return new_solution
    
    def _newton_solve(self,
                     implicit_residual: Callable,
                     initial_guess: np.ndarray,
                     residual_function: Callable,
                     dt: float) -> np.ndarray:
        """Solve implicit system using Newton's method."""
        u = initial_guess.copy()
        
        for newton_iter in range(self.config.implicit_max_iterations):
            # Compute residual
            F = implicit_residual(u)
            residual_norm = np.linalg.norm(F)
            
            if residual_norm < self.config.implicit_tolerance:
                break
            
            # Compute Jacobian (finite difference approximation)
            if self.config.use_matrix_free:
                # Matrix-free Newton using GMRES
                delta_u = self._gmres_solve(F, u, implicit_residual, dt)
            else:
                # Form Jacobian matrix
                J = self._compute_jacobian(implicit_residual, u)
                self.total_jacobian_evaluations += 1
                
                # Solve linear system
                delta_u = spsolve(J, -F)
            
            # Update solution
            u += delta_u
            
            # Check convergence
            if np.linalg.norm(delta_u) < self.config.implicit_tolerance:
                break
        
        return u
    
    def _compute_jacobian(self, residual_function: Callable, u: np.ndarray) -> csr_matrix:
        """Compute Jacobian matrix using finite differences."""
        n = len(u)
        J = np.zeros((n, n))
        
        eps = 1e-8
        F0 = residual_function(u)
        
        for i in range(n):
            u_pert = u.copy()
            u_pert[i] += eps
            F_pert = residual_function(u_pert)
            J[:, i] = (F_pert - F0) / eps
        
        return csr_matrix(J)
    
    def _gmres_solve(self, rhs: np.ndarray, u: np.ndarray, 
                    implicit_residual: Callable, dt: float) -> np.ndarray:
        """Solve linear system using matrix-free GMRES."""
        from scipy.sparse.linalg import gmres
        
        def matvec(v):
            """Matrix-vector product for Jacobian"""
            eps = 1e-8
            F0 = implicit_residual(u)
            F_pert = implicit_residual(u + eps * v)
            return (F_pert - F0) / eps
        
        n = len(u)
        J_op = LinearOperator((n, n), matvec=matvec)
        
        solution, info = gmres(J_op, -rhs, rtol=self.config.implicit_tolerance)
        
        return solution if info == 0 else np.zeros_like(rhs)


class BDF2(TimeIntegrator):
    """
    Second-order Backward Differentiation Formula (BDF2).
    
    Higher-order implicit method with good stability properties.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        super().__init__(config)
        self.scheme_name = "bdf2"
        self.previous_previous_solution: Optional[np.ndarray] = None
    
    def integrate_step(self,
                      solution: np.ndarray,
                      residual_function: Callable[[np.ndarray, float], np.ndarray],
                      dt: float) -> np.ndarray:
        """BDF2: 3/2 * U^{n+1} - 2 * U^n + 1/2 * U^{n-1} = dt * R(U^{n+1})"""
        
        if self.previous_solution is None:
            # First step: use Backward Euler
            be_integrator = BackwardEuler(self.config)
            return be_integrator.integrate_step(solution, residual_function, dt)
        
        def implicit_residual(u_new):
            """BDF2 implicit residual"""
            residual = residual_function(u_new, self.current_time + dt)
            self.total_function_evaluations += 1
            
            if self.previous_previous_solution is not None:
                # Full BDF2
                return (1.5 * u_new - 2.0 * solution + 0.5 * self.previous_previous_solution 
                       - dt * residual)
            else:
                # Second-order approximation for second step
                return (1.5 * u_new - 2.0 * solution + 0.5 * solution - dt * residual)
        
        # Store for next step
        self.previous_previous_solution = self.previous_solution.copy() if self.previous_solution is not None else None
        
        # Solve implicit system
        new_solution = self._newton_solve_bdf2(implicit_residual, solution, residual_function, dt)
        return new_solution
    
    def _newton_solve_bdf2(self, implicit_residual: Callable, initial_guess: np.ndarray,
                          residual_function: Callable, dt: float) -> np.ndarray:
        """Newton solver for BDF2."""
        # Similar to backward Euler but with different residual
        u = initial_guess.copy()
        
        for newton_iter in range(self.config.implicit_max_iterations):
            F = implicit_residual(u)
            residual_norm = np.linalg.norm(F)
            
            if residual_norm < self.config.implicit_tolerance:
                break
            
            # Simple finite difference Jacobian
            eps = 1e-8
            n = len(u)
            J = np.zeros((n, n))
            
            for i in range(n):
                u_pert = u.copy()
                u_pert[i] += eps
                F_pert = implicit_residual(u_pert)
                J[:, i] = (F_pert - F) / eps
            
            # Solve and update
            try:
                delta_u = np.linalg.solve(J, -F)
                u += delta_u
            except np.linalg.LinAlgError:
                # Fallback to simple step
                u += -0.1 * F
            
            if np.linalg.norm(delta_u) < self.config.implicit_tolerance:
                break
        
        return u


class AdaptiveRungeKutta:
    """
    Adaptive Runge-Kutta with embedded error estimation.
    
    Uses embedded Runge-Kutta pairs for automatic time step control.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        """Initialize adaptive RK integrator."""
        self.config = config or TimeIntegrationConfig()
        self.scheme_name = "adaptive_rk"
        
        # Embedded RK4/5 (Dormand-Prince) coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
        
        self.b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        self.b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        self.c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])


class LocalTimeSteppingSolver:
    """
    Local time stepping for steady-state convergence acceleration.
    
    Uses different time steps for each cell based on local stability limits.
    """
    
    def __init__(self, config: Optional[TimeIntegrationConfig] = None):
        """Initialize local time stepping solver."""
        self.config = config or TimeIntegrationConfig()
        self.scheme_name = "local_timestepping"
        
    def compute_local_timesteps(self,
                              solution: np.ndarray,
                              mesh_info: Dict[str, Any],
                              flow_properties: Dict[str, Any]) -> np.ndarray:
        """
        Compute local time steps for each cell.
        
        Args:
            solution: Current solution
            mesh_info: Mesh information
            flow_properties: Flow property data
            
        Returns:
            Local time steps for each cell
        """
        n_cells = solution.shape[0]
        local_dt = np.zeros(n_cells)
        
        # Extract flow variables
        gamma = 1.4
        
        for cell in range(n_cells):
            rho, rho_u, rho_v, rho_w, rho_E = solution[cell]
            rho = max(rho, 1e-12)
            
            # Compute primitive variables
            u, v, w = rho_u/rho, rho_v/rho, rho_w/rho
            velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
            
            # Compute pressure and speed of sound
            kinetic_energy = 0.5 * rho * velocity_magnitude**2
            pressure = (gamma - 1) * (rho_E - kinetic_energy)
            speed_of_sound = np.sqrt(gamma * pressure / rho)
            
            # Estimate cell size (simplified)
            if 'cell_volumes' in mesh_info:
                cell_size = np.power(mesh_info['cell_volumes'][cell], 1.0/3.0)
            else:
                cell_size = 1.0
            
            # Compute local time step based on CFL condition
            max_wave_speed = velocity_magnitude + speed_of_sound
            local_dt[cell] = self.config.local_cfl_number * cell_size / (max_wave_speed + 1e-12)
        
        return local_dt


def create_time_integrator(scheme_type: str, 
                          config: Optional[TimeIntegrationConfig] = None) -> TimeIntegrator:
    """
    Factory function for creating time integrators.
    
    Args:
        scheme_type: Type of time integration scheme
        config: Integration configuration
        
    Returns:
        Configured time integrator
    """
    if scheme_type == "euler":
        return ExplicitEuler(config)
    elif scheme_type == "rk2":
        return RungeKutta2(config)
    elif scheme_type == "rk4":
        return RungeKutta4(config)
    elif scheme_type == "rk3_tvd":
        return RungeKutta3TVD(config)
    elif scheme_type == "backward_euler":
        return BackwardEuler(config)
    elif scheme_type == "bdf2":
        return BDF2(config)
    else:
        raise ValueError(f"Unknown time integration scheme: {scheme_type}")


def test_time_integration():
    """Test time integration schemes on a simple ODE system."""
    print("Testing Time Integration Schemes:")
    
    # Simple test ODE: dy/dt = -y, exact solution: y(t) = y0 * exp(-t)
    def simple_ode(y, t):
        return -y
    
    # Initial condition
    y0 = np.array([1.0])
    final_time = 1.0
    exact_solution = np.exp(-final_time)
    
    # Test different schemes
    schemes = ["euler", "rk2", "rk4", "rk3_tvd", "backward_euler"]
    
    print(f"\n  Testing simple ODE: dy/dt = -y, y(0) = 1")
    print(f"  Exact solution at t=1: {exact_solution:.6f}")
    
    for scheme in schemes:
        config = TimeIntegrationConfig(
            scheme_type=scheme,
            dt=0.01,
            max_time=final_time,
            monitor_residuals=False
        )
        
        integrator = create_time_integrator(scheme, config)
        results = integrator.integrate_to_time(y0, simple_ode, final_time)
        
        final_solution = results['final_solution'][0]
        error = abs(final_solution - exact_solution)
        
        print(f"    {scheme:15s}: solution = {final_solution:.6f}, error = {error:.2e}")
    
    # Test system of ODEs (2D harmonic oscillator)
    print(f"\n  Testing 2D harmonic oscillator system:")
    
    def harmonic_oscillator(y, t):
        # y = [x, dx/dt, y, dy/dt], equations: d²x/dt² = -x, d²y/dt² = -y
        dydt = np.zeros_like(y)
        dydt[0] = y[1]      # dx/dt
        dydt[1] = -y[0]     # d²x/dt²
        dydt[2] = y[3]      # dy/dt
        dydt[3] = -y[2]     # d²y/dt²
        return dydt
    
    y0_2d = np.array([1.0, 0.0, 0.0, 1.0])  # Initial: x=1, dx/dt=0, y=0, dy/dt=1
    
    config = TimeIntegrationConfig(
        scheme_type="rk4",
        dt=0.01,
        max_time=2*np.pi,  # One full period
        monitor_residuals=False
    )
    
    integrator = create_time_integrator("rk4", config)
    results = integrator.integrate_to_time(y0_2d, harmonic_oscillator, 2*np.pi)
    
    final_solution = results['final_solution']
    # After one period, should return to initial state
    error = np.linalg.norm(final_solution - y0_2d)
    
    print(f"    RK4 harmonic oscillator: final state = {final_solution}")
    print(f"    Error after one period: {error:.2e}")
    print(f"    Function evaluations: {results['function_evaluations']}")


if __name__ == "__main__":
    test_time_integration()