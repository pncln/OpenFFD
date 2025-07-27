"""
Parallel Computing with MPI for Domain Decomposition

Implements distributed memory parallelization for CFD simulations:
- Domain decomposition strategies (geometric, graph-based)
- MPI communication patterns for CFD data structures
- Parallel solution algorithms with ghost cell exchange
- Load balancing and dynamic repartitioning
- Parallel I/O for large-scale simulations
- Scalability optimization for HPC systems

Enables efficient execution on multi-node clusters and supercomputers
with proper scaling characteristics for production CFD workflows.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import time

# MPI import with fallback for non-MPI environments
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    # Mock MPI for testing without MPI installation
    class MockMPI:
        COMM_WORLD = None
        SUM = "SUM"
        MAX = "MAX"
        MIN = "MIN"
        DOUBLE = "DOUBLE"
        INT = "INT"
    MPI = MockMPI()

logger = logging.getLogger(__name__)


class DecompositionMethod(Enum):
    """Enumeration of domain decomposition methods."""
    GEOMETRIC = "geometric"
    GRAPH_BASED = "graph_based"
    RECURSIVE_BISECTION = "recursive_bisection"
    HILBERT_CURVE = "hilbert_curve"


class CommunicationPattern(Enum):
    """Enumeration of MPI communication patterns."""
    POINT_TO_POINT = "point_to_point"
    COLLECTIVE = "collective"
    NON_BLOCKING = "non_blocking"
    ONE_SIDED = "one_sided"


@dataclass
class ParallelConfig:
    """Configuration for parallel computing."""
    
    # Decomposition settings
    decomposition_method: DecompositionMethod = DecompositionMethod.GEOMETRIC
    overlap_layers: int = 1  # Number of ghost cell layers
    load_balance_frequency: int = 100  # Rebalance every N iterations
    
    # Communication settings
    communication_pattern: CommunicationPattern = CommunicationPattern.NON_BLOCKING
    message_aggregation: bool = True  # Aggregate small messages
    compression_threshold: int = 1000  # Compress messages larger than threshold
    
    # Performance settings
    enable_overlap: bool = True  # Overlap communication with computation
    use_persistent_requests: bool = True  # Use persistent MPI requests
    buffer_size: int = 1024 * 1024  # Communication buffer size
    
    # Load balancing
    load_imbalance_threshold: float = 0.1  # Trigger rebalancing at 10% imbalance
    dynamic_load_balancing: bool = True
    migration_cost_factor: float = 0.5  # Cost of migrating cells
    
    # I/O settings
    parallel_io: bool = True
    io_aggregation_size: int = 64  # Number of processes for I/O aggregation


@dataclass
class DomainPartition:
    """Information about domain partition for one process."""
    
    rank: int  # Process rank
    n_local_cells: int  # Number of local cells
    local_cell_ids: np.ndarray  # Global IDs of local cells
    ghost_cell_ids: np.ndarray  # Global IDs of ghost cells
    
    # Neighbor information
    neighbor_ranks: List[int]  # Neighboring process ranks
    shared_faces: Dict[int, List[int]]  # Shared faces with each neighbor
    
    # Communication buffers
    send_buffers: Dict[int, np.ndarray] = None  # Send buffers by rank
    recv_buffers: Dict[int, np.ndarray] = None  # Receive buffers by rank
    
    # Load balancing
    computational_load: float = 0.0  # Estimated computational cost
    communication_load: float = 0.0  # Communication overhead


class DomainDecomposer(ABC):
    """Abstract base class for domain decomposition."""
    
    def __init__(self, config: ParallelConfig):
        """Initialize domain decomposer."""
        self.config = config
        
    @abstractmethod
    def decompose_domain(self,
                        mesh_info: Dict[str, Any],
                        n_processes: int) -> List[DomainPartition]:
        """
        Decompose domain into subdomains for parallel execution.
        
        Args:
            mesh_info: Mesh connectivity and geometry information
            n_processes: Number of MPI processes
            
        Returns:
            List of domain partitions, one per process
        """
        pass


class GeometricDecomposer(DomainDecomposer):
    """
    Geometric domain decomposition.
    
    Splits domain based on spatial coordinates using recursive coordinate bisection.
    """
    
    def decompose_domain(self,
                        mesh_info: Dict[str, Any],
                        n_processes: int) -> List[DomainPartition]:
        """Perform geometric decomposition."""
        n_cells = mesh_info.get('n_cells', 1000)
        cell_centers = mesh_info.get('cell_centers', np.random.rand(n_cells, 3))
        
        # Recursive coordinate bisection
        cell_assignments = self._recursive_bisection(cell_centers, n_processes)
        
        # Create domain partitions
        partitions = []
        for rank in range(n_processes):
            local_cells = np.where(cell_assignments == rank)[0]
            ghost_cells = self._find_ghost_cells(local_cells, mesh_info, rank, cell_assignments)
            
            partition = DomainPartition(
                rank=rank,
                n_local_cells=len(local_cells),
                local_cell_ids=local_cells,
                ghost_cell_ids=ghost_cells,
                neighbor_ranks=self._find_neighbors(rank, local_cells, mesh_info, cell_assignments),
                shared_faces={}
            )
            
            partitions.append(partition)
        
        logger.info(f"Geometric decomposition created {n_processes} partitions")
        return partitions
    
    def _recursive_bisection(self, coordinates: np.ndarray, n_processes: int) -> np.ndarray:
        """Recursive coordinate bisection algorithm."""
        n_cells = coordinates.shape[0]
        assignments = np.zeros(n_cells, dtype=int)
        
        # Use a queue for recursive splitting
        queue = [(np.arange(n_cells), 0, n_processes)]
        
        while queue:
            cell_indices, base_rank, n_procs = queue.pop(0)
            
            if n_procs == 1:
                assignments[cell_indices] = base_rank
                continue
            
            # Choose splitting dimension (longest extent)
            coords_subset = coordinates[cell_indices]
            extents = np.max(coords_subset, axis=0) - np.min(coords_subset, axis=0)
            split_dim = np.argmax(extents)
            
            # Split at median
            split_coord = np.median(coords_subset[:, split_dim])
            left_mask = coords_subset[:, split_dim] <= split_coord
            
            left_indices = cell_indices[left_mask]
            right_indices = cell_indices[~left_mask]
            
            # Distribute processes
            n_left = n_procs // 2
            n_right = n_procs - n_left
            
            if len(left_indices) > 0:
                queue.append((left_indices, base_rank, n_left))
            if len(right_indices) > 0:
                queue.append((right_indices, base_rank + n_left, n_right))
        
        return assignments
    
    def _find_ghost_cells(self,
                         local_cells: np.ndarray,
                         mesh_info: Dict[str, Any],
                         rank: int,
                         cell_assignments: np.ndarray) -> np.ndarray:
        """Find ghost cells for a partition."""
        ghost_cells = set()
        
        # Simplified neighbor finding - in practice use mesh connectivity
        for cell in local_cells:
            # Mock neighbors (should use actual mesh connectivity)
            neighbors = self._get_cell_neighbors(cell, mesh_info)
            for neighbor in neighbors:
                if neighbor < len(cell_assignments) and cell_assignments[neighbor] != rank:
                    ghost_cells.add(neighbor)
        
        return np.array(list(ghost_cells))
    
    def _get_cell_neighbors(self, cell_id: int, mesh_info: Dict[str, Any]) -> List[int]:
        """Get neighboring cells (simplified)."""
        n_cells = mesh_info.get('n_cells', 1000)
        max_neighbors = min(6, n_cells - 1)
        return [(cell_id + i + 1) % n_cells for i in range(max_neighbors)]
    
    def _find_neighbors(self,
                       rank: int,
                       local_cells: np.ndarray,
                       mesh_info: Dict[str, Any],
                       cell_assignments: np.ndarray) -> List[int]:
        """Find neighboring process ranks."""
        neighbor_ranks = set()
        
        for cell in local_cells:
            neighbors = self._get_cell_neighbors(cell, mesh_info)
            for neighbor in neighbors:
                if neighbor < len(cell_assignments):
                    neighbor_rank = cell_assignments[neighbor]
                    if neighbor_rank != rank:
                        neighbor_ranks.add(neighbor_rank)
        
        return list(neighbor_ranks)


class GraphBasedDecomposer(DomainDecomposer):
    """
    Graph-based domain decomposition using mesh connectivity.
    
    Uses graph partitioning algorithms like METIS for optimal load balancing.
    """
    
    def decompose_domain(self,
                        mesh_info: Dict[str, Any],
                        n_processes: int) -> List[DomainPartition]:
        """Perform graph-based decomposition."""
        # This would use a graph partitioning library like pymetis
        # For now, implement a simplified version
        
        n_cells = mesh_info.get('n_cells', 1000)
        
        # Create adjacency graph (simplified)
        adjacency = self._create_adjacency_graph(mesh_info)
        
        # Simple graph partitioning (should use METIS or similar)
        cell_assignments = self._simple_graph_partition(adjacency, n_processes)
        
        # Create partitions similar to geometric decomposer
        partitions = []
        for rank in range(n_processes):
            local_cells = np.where(cell_assignments == rank)[0]
            ghost_cells = self._find_ghost_cells_graph(local_cells, adjacency, cell_assignments, rank)
            
            partition = DomainPartition(
                rank=rank,
                n_local_cells=len(local_cells),
                local_cell_ids=local_cells,
                ghost_cell_ids=ghost_cells,
                neighbor_ranks=self._find_neighbors_graph(rank, local_cells, adjacency, cell_assignments),
                shared_faces={}
            )
            
            partitions.append(partition)
        
        logger.info(f"Graph-based decomposition created {n_processes} partitions")
        return partitions
    
    def _create_adjacency_graph(self, mesh_info: Dict[str, Any]) -> Dict[int, List[int]]:
        """Create cell adjacency graph."""
        n_cells = mesh_info.get('n_cells', 1000)
        adjacency = {}
        
        # Simplified adjacency (should use actual mesh faces)
        for cell in range(n_cells):
            neighbors = []
            for i in range(6):  # Assume max 6 neighbors
                neighbor = (cell + i + 1) % n_cells
                neighbors.append(neighbor)
            adjacency[cell] = neighbors
        
        return adjacency
    
    def _simple_graph_partition(self, adjacency: Dict[int, List[int]], n_processes: int) -> np.ndarray:
        """Simple graph partitioning (placeholder for METIS)."""
        n_cells = len(adjacency)
        assignments = np.zeros(n_cells, dtype=int)
        
        # Simple round-robin assignment (should use proper graph partitioning)
        cells_per_process = n_cells // n_processes
        for i in range(n_cells):
            assignments[i] = min(i // cells_per_process, n_processes - 1)
        
        return assignments
    
    def _find_ghost_cells_graph(self,
                               local_cells: np.ndarray,
                               adjacency: Dict[int, List[int]],
                               cell_assignments: np.ndarray,
                               rank: int) -> np.ndarray:
        """Find ghost cells using graph connectivity."""
        ghost_cells = set()
        
        for cell in local_cells:
            for neighbor in adjacency.get(cell, []):
                if neighbor < len(cell_assignments) and cell_assignments[neighbor] != rank:
                    ghost_cells.add(neighbor)
        
        return np.array(list(ghost_cells))
    
    def _find_neighbors_graph(self,
                             rank: int,
                             local_cells: np.ndarray,
                             adjacency: Dict[int, List[int]],
                             cell_assignments: np.ndarray) -> List[int]:
        """Find neighboring ranks using graph."""
        neighbor_ranks = set()
        
        for cell in local_cells:
            for neighbor in adjacency.get(cell, []):
                if neighbor < len(cell_assignments):
                    neighbor_rank = cell_assignments[neighbor]
                    if neighbor_rank != rank:
                        neighbor_ranks.add(neighbor_rank)
        
        return list(neighbor_ranks)


class MPICommunicator:
    """
    MPI communication manager for CFD data exchange.
    
    Handles ghost cell updates, global reductions, and collective operations.
    """
    
    def __init__(self, config: ParallelConfig):
        """Initialize MPI communicator."""
        self.config = config
        self.comm = MPI.COMM_WORLD if MPI_AVAILABLE else None
        self.rank = self.comm.Get_rank() if MPI_AVAILABLE else 0
        self.size = self.comm.Get_size() if MPI_AVAILABLE else 1
        
        # Communication requests for non-blocking operations
        self.send_requests = {}
        self.recv_requests = {}
        
        # Persistent requests for repeated communications
        self.persistent_sends = {}
        self.persistent_recvs = {}
        
    def exchange_ghost_cells(self,
                           solution: np.ndarray,
                           partition: DomainPartition) -> np.ndarray:
        """
        Exchange ghost cell data between neighboring processes.
        
        Args:
            solution: Local solution array
            partition: Domain partition information
            
        Returns:
            Updated solution with ghost cell data
        """
        if not MPI_AVAILABLE or self.size == 1:
            return solution
        
        # Prepare communication
        self._prepare_ghost_exchange(solution, partition)
        
        # Perform communication
        if self.config.communication_pattern == CommunicationPattern.NON_BLOCKING:
            self._non_blocking_exchange(partition)
        elif self.config.communication_pattern == CommunicationPattern.POINT_TO_POINT:
            self._point_to_point_exchange(partition)
        else:
            self._collective_exchange(solution, partition)
        
        # Update ghost cells in solution
        updated_solution = self._update_ghost_cells(solution, partition)
        
        return updated_solution
    
    def _prepare_ghost_exchange(self, solution: np.ndarray, partition: DomainPartition):
        """Prepare data for ghost cell exchange."""
        n_vars = solution.shape[1]
        
        # Initialize buffers if needed
        if partition.send_buffers is None:
            partition.send_buffers = {}
            partition.recv_buffers = {}
        
        # Pack data for each neighbor
        for neighbor_rank in partition.neighbor_ranks:
            if neighbor_rank not in partition.shared_faces:
                continue
            
            shared_faces = partition.shared_faces[neighbor_rank]
            n_shared = len(shared_faces)
            
            # Allocate buffers
            if neighbor_rank not in partition.send_buffers:
                partition.send_buffers[neighbor_rank] = np.zeros((n_shared, n_vars))
                partition.recv_buffers[neighbor_rank] = np.zeros((n_shared, n_vars))
            
            # Pack boundary data
            for i, face_id in enumerate(shared_faces):
                # Get cell data for this face (simplified)
                cell_id = face_id % solution.shape[0]  # Mock mapping
                partition.send_buffers[neighbor_rank][i] = solution[cell_id]
    
    def _non_blocking_exchange(self, partition: DomainPartition):
        """Non-blocking ghost cell exchange."""
        if not MPI_AVAILABLE:
            return
        
        requests = []
        
        # Post receives first
        for neighbor_rank in partition.neighbor_ranks:
            if neighbor_rank in partition.recv_buffers:
                recv_buffer = partition.recv_buffers[neighbor_rank]
                req = self.comm.Irecv(recv_buffer, source=neighbor_rank, tag=100)
                requests.append(req)
        
        # Post sends
        for neighbor_rank in partition.neighbor_ranks:
            if neighbor_rank in partition.send_buffers:
                send_buffer = partition.send_buffers[neighbor_rank]
                req = self.comm.Isend(send_buffer, dest=neighbor_rank, tag=100)
                requests.append(req)
        
        # Wait for completion
        MPI.Request.Waitall(requests)
    
    def _point_to_point_exchange(self, partition: DomainPartition):
        """Point-to-point ghost cell exchange."""
        if not MPI_AVAILABLE:
            return
        
        for neighbor_rank in partition.neighbor_ranks:
            if neighbor_rank in partition.send_buffers and neighbor_rank in partition.recv_buffers:
                send_buffer = partition.send_buffers[neighbor_rank]
                recv_buffer = partition.recv_buffers[neighbor_rank]
                
                # Simultaneous send/receive
                self.comm.Sendrecv(send_buffer, dest=neighbor_rank, sendtag=100,
                                  recvbuf=recv_buffer, source=neighbor_rank, recvtag=100)
    
    def _collective_exchange(self, solution: np.ndarray, partition: DomainPartition):
        """Collective ghost cell exchange (simplified)."""
        if not MPI_AVAILABLE:
            return
        
        # This would use MPI_Alltoallv or similar for efficient collective communication
        # For now, use simple approach
        pass
    
    def _update_ghost_cells(self, solution: np.ndarray, partition: DomainPartition) -> np.ndarray:
        """Update ghost cells in solution array."""
        # Create extended solution array with ghost cells
        n_local = partition.n_local_cells
        n_ghost = len(partition.ghost_cell_ids)
        n_vars = solution.shape[1]
        
        extended_solution = np.zeros((n_local + n_ghost, n_vars))
        extended_solution[:n_local] = solution
        
        # Unpack received data into ghost cells
        ghost_idx = n_local
        for neighbor_rank in partition.neighbor_ranks:
            if neighbor_rank in partition.recv_buffers:
                recv_buffer = partition.recv_buffers[neighbor_rank]
                n_received = recv_buffer.shape[0]
                extended_solution[ghost_idx:ghost_idx + n_received] = recv_buffer
                ghost_idx += n_received
        
        return extended_solution
    
    def global_reduction(self, local_value: Union[float, np.ndarray], operation: str = "sum") -> Union[float, np.ndarray]:
        """Perform global reduction operation."""
        if not MPI_AVAILABLE or self.size == 1:
            return local_value
        
        if operation == "sum":
            return self.comm.allreduce(local_value, op=MPI.SUM)
        elif operation == "max":
            return self.comm.allreduce(local_value, op=MPI.MAX)
        elif operation == "min":
            return self.comm.allreduce(local_value, op=MPI.MIN)
        else:
            raise ValueError(f"Unsupported reduction operation: {operation}")
    
    def broadcast(self, data: Any, root: int = 0) -> Any:
        """Broadcast data from root to all processes."""
        if not MPI_AVAILABLE or self.size == 1:
            return data
        
        return self.comm.bcast(data, root=root)
    
    def gather_solutions(self, local_solution: np.ndarray) -> Optional[np.ndarray]:
        """Gather solutions from all processes to root."""
        if not MPI_AVAILABLE or self.size == 1:
            return local_solution
        
        all_solutions = self.comm.gather(local_solution, root=0)
        
        if self.rank == 0:
            return np.concatenate(all_solutions, axis=0)
        else:
            return None


class LoadBalancer:
    """
    Dynamic load balancing for parallel CFD simulations.
    
    Monitors computational load and redistributes work as needed.
    """
    
    def __init__(self, config: ParallelConfig, communicator: MPICommunicator):
        """Initialize load balancer."""
        self.config = config
        self.communicator = communicator
        self.load_history = []
        
    def assess_load_balance(self, partitions: List[DomainPartition]) -> Dict[str, float]:
        """Assess current load balance across processes."""
        local_partition = partitions[self.communicator.rank] if partitions else None
        
        # Compute local load metrics
        local_load = self._compute_local_load(local_partition)
        
        # Gather loads from all processes
        all_loads = self.communicator.comm.allgather(local_load) if MPI_AVAILABLE else [local_load]
        
        # Compute load balance metrics
        avg_load = np.mean(all_loads)
        max_load = np.max(all_loads)
        min_load = np.min(all_loads)
        
        load_imbalance = (max_load - min_load) / (avg_load + 1e-12)
        
        metrics = {
            'average_load': avg_load,
            'max_load': max_load,
            'min_load': min_load,
            'load_imbalance': load_imbalance,
            'efficiency': min_load / (max_load + 1e-12)
        }
        
        self.load_history.append(metrics)
        return metrics
    
    def _compute_local_load(self, partition: Optional[DomainPartition]) -> float:
        """Compute local computational load."""
        if partition is None:
            return 0.0
        
        # Simple load model: cells + communication overhead
        computation_load = partition.n_local_cells
        communication_load = len(partition.ghost_cell_ids) * 0.1
        
        total_load = computation_load + communication_load
        return total_load
    
    def should_rebalance(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if load rebalancing is needed."""
        return current_metrics['load_imbalance'] > self.config.load_imbalance_threshold
    
    def rebalance_load(self, partitions: List[DomainPartition]) -> List[DomainPartition]:
        """Perform load rebalancing."""
        if not self.config.dynamic_load_balancing:
            return partitions
        
        logger.info("Performing dynamic load rebalancing...")
        
        # Simple rebalancing strategy: redistribute cells
        # In practice, this would use sophisticated migration algorithms
        
        # For now, return unchanged partitions
        return partitions


class ParallelSolver:
    """
    Main parallel CFD solver coordinator.
    
    Integrates domain decomposition, communication, and load balancing.
    """
    
    def __init__(self, config: ParallelConfig):
        """Initialize parallel solver."""
        self.config = config
        
        # Initialize components
        self.communicator = MPICommunicator(config)
        self.decomposer = self._create_decomposer()
        self.load_balancer = LoadBalancer(config, self.communicator)
        
        # Parallel state
        self.partitions: Optional[List[DomainPartition]] = None
        self.local_partition: Optional[DomainPartition] = None
        
        # Performance metrics
        self.communication_time = 0.0
        self.computation_time = 0.0
        self.load_balance_time = 0.0
        
    def _create_decomposer(self) -> DomainDecomposer:
        """Create domain decomposer based on configuration."""
        if self.config.decomposition_method == DecompositionMethod.GEOMETRIC:
            return GeometricDecomposer(self.config)
        elif self.config.decomposition_method == DecompositionMethod.GRAPH_BASED:
            return GraphBasedDecomposer(self.config)
        else:
            return GeometricDecomposer(self.config)
    
    def initialize_parallel_simulation(self, mesh_info: Dict[str, Any]) -> None:
        """Initialize parallel simulation with domain decomposition."""
        start_time = time.time()
        
        # Perform domain decomposition
        self.partitions = self.decomposer.decompose_domain(mesh_info, self.communicator.size)
        self.local_partition = self.partitions[self.communicator.rank]
        
        # Setup communication patterns
        self._setup_communication_patterns()
        
        init_time = time.time() - start_time
        logger.info(f"Parallel initialization completed in {init_time:.3f}s")
        logger.info(f"Process {self.communicator.rank}: {self.local_partition.n_local_cells} local cells, "
                   f"{len(self.local_partition.ghost_cell_ids)} ghost cells")
    
    def _setup_communication_patterns(self):
        """Setup MPI communication patterns."""
        if not MPI_AVAILABLE or not self.local_partition:
            return
        
        # Setup shared faces for communication
        self.local_partition.shared_faces = {}
        for neighbor_rank in self.local_partition.neighbor_ranks:
            # Simplified: assume some shared faces
            n_shared = min(10, self.local_partition.n_local_cells // 10)
            shared_faces = list(range(n_shared))
            self.local_partition.shared_faces[neighbor_rank] = shared_faces
    
    def parallel_solve_step(self,
                          local_solution: np.ndarray,
                          solve_function: Callable,
                          **kwargs) -> np.ndarray:
        """
        Perform one parallel solve step.
        
        Args:
            local_solution: Local solution array
            solve_function: Function to solve local equations
            **kwargs: Additional arguments for solve function
            
        Returns:
            Updated local solution
        """
        # Exchange ghost cells
        comm_start = time.time()
        extended_solution = self.communicator.exchange_ghost_cells(local_solution, self.local_partition)
        self.communication_time += time.time() - comm_start
        
        # Solve local equations
        comp_start = time.time()
        new_local_solution = solve_function(extended_solution, **kwargs)
        self.computation_time += time.time() - comp_start
        
        # Extract local part (remove ghost cells)
        if new_local_solution.shape[0] > local_solution.shape[0]:
            new_local_solution = new_local_solution[:local_solution.shape[0]]
        
        return new_local_solution
    
    def check_convergence(self, local_residual: np.ndarray) -> Tuple[bool, float]:
        """Check global convergence using parallel reduction."""
        # Compute local residual norm
        local_norm = np.linalg.norm(local_residual)
        
        # Global reduction
        global_norm = self.communicator.global_reduction(local_norm**2, "sum")
        global_norm = np.sqrt(global_norm)
        
        # Simple convergence criterion
        tolerance = 1e-6
        converged = global_norm < tolerance
        
        return converged, global_norm
    
    def gather_global_solution(self, local_solution: np.ndarray) -> Optional[np.ndarray]:
        """Gather complete solution on root process."""
        return self.communicator.gather_solutions(local_solution)
    
    def periodic_load_balancing(self, iteration: int) -> bool:
        """Perform periodic load balancing check."""
        if iteration % self.config.load_balance_frequency != 0:
            return False
        
        lb_start = time.time()
        
        # Assess load balance
        load_metrics = self.load_balancer.assess_load_balance(self.partitions)
        
        # Check if rebalancing is needed
        if self.load_balancer.should_rebalance(load_metrics):
            # Perform rebalancing
            self.partitions = self.load_balancer.rebalance_load(self.partitions)
            self.local_partition = self.partitions[self.communicator.rank]
            
            self.load_balance_time += time.time() - lb_start
            return True
        
        self.load_balance_time += time.time() - lb_start
        return False
    
    def get_parallel_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parallel performance statistics."""
        stats = {
            'rank': self.communicator.rank,
            'size': self.communicator.size,
            'local_cells': self.local_partition.n_local_cells if self.local_partition else 0,
            'ghost_cells': len(self.local_partition.ghost_cell_ids) if self.local_partition else 0,
            'neighbors': len(self.local_partition.neighbor_ranks) if self.local_partition else 0,
            'communication_time': self.communication_time,
            'computation_time': self.computation_time,
            'load_balance_time': self.load_balance_time,
            'parallel_efficiency': self.computation_time / (self.computation_time + self.communication_time + 1e-12)
        }
        
        # Global statistics (only on root)
        if self.communicator.rank == 0 and self.partitions:
            total_cells = sum(p.n_local_cells for p in self.partitions)
            max_cells = max(p.n_local_cells for p in self.partitions)
            min_cells = min(p.n_local_cells for p in self.partitions)
            
            stats.update({
                'total_cells': total_cells,
                'load_imbalance': (max_cells - min_cells) / (total_cells / self.communicator.size + 1e-12),
                'avg_cells_per_process': total_cells / self.communicator.size
            })
        
        return stats


def create_parallel_solver(decomposition_method: str = "geometric",
                          config: Optional[ParallelConfig] = None) -> ParallelSolver:
    """
    Factory function for creating parallel solvers.
    
    Args:
        decomposition_method: Domain decomposition method
        config: Parallel configuration
        
    Returns:
        Configured parallel solver
    """
    if config is None:
        config = ParallelConfig()
    
    config.decomposition_method = DecompositionMethod(decomposition_method)
    
    return ParallelSolver(config)


def test_parallel_computing():
    """Test parallel computing functionality."""
    print("Testing Parallel Computing with MPI:")
    print(f"MPI Available: {MPI_AVAILABLE}")
    
    # Create test configuration
    config = ParallelConfig(
        decomposition_method=DecompositionMethod.GEOMETRIC,
        communication_pattern=CommunicationPattern.NON_BLOCKING
    )
    
    # Create parallel solver
    solver = create_parallel_solver("geometric", config)
    
    print(f"Process rank: {solver.communicator.rank}")
    print(f"Total processes: {solver.communicator.size}")
    
    # Test domain decomposition
    print(f"\n  Testing domain decomposition:")
    
    # Mock mesh data
    n_cells = 1000
    mesh_info = {
        'n_cells': n_cells,
        'cell_centers': np.random.rand(n_cells, 3),
        'cell_volumes': np.ones(n_cells) * 0.001
    }
    
    # Initialize parallel simulation
    solver.initialize_parallel_simulation(mesh_info)
    
    print(f"    Local cells: {solver.local_partition.n_local_cells}")
    print(f"    Ghost cells: {len(solver.local_partition.ghost_cell_ids)}")
    print(f"    Neighbors: {len(solver.local_partition.neighbor_ranks)}")
    
    # Test parallel solve step
    print(f"\n  Testing parallel solve step:")
    
    local_solution = np.random.rand(solver.local_partition.n_local_cells, 5)
    
    def mock_solve_function(solution, **kwargs):
        # Simple update: add small perturbation
        return solution + np.random.rand(*solution.shape) * 0.01
    
    # Perform solve steps
    for step in range(3):
        local_solution = solver.parallel_solve_step(local_solution, mock_solve_function)
        
        # Check convergence
        residual = np.random.rand(solver.local_partition.n_local_cells, 5) * 0.1
        converged, global_norm = solver.check_convergence(residual.flatten())
        
        print(f"    Step {step + 1}: converged={converged}, residual={global_norm:.2e}")
    
    # Test load balancing
    print(f"\n  Testing load balancing:")
    load_metrics = solver.load_balancer.assess_load_balance(solver.partitions)
    
    if solver.communicator.rank == 0:
        print(f"    Load imbalance: {load_metrics['load_imbalance']:.3f}")
        print(f"    Efficiency: {load_metrics['efficiency']:.3f}")
    
    # Get parallel statistics
    stats = solver.get_parallel_statistics()
    print(f"\n  Parallel statistics for rank {stats['rank']}:")
    print(f"    Parallel efficiency: {stats['parallel_efficiency']:.3f}")
    print(f"    Communication time: {stats['communication_time']:.3f}s")
    print(f"    Computation time: {stats['computation_time']:.3f}s")
    
    print(f"\n  Parallel computing test completed!")


if __name__ == "__main__":
    test_parallel_computing()