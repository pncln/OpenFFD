"""
Generate synthetic mesh data for benchmarking FFD and HFFD algorithms.
"""

import numpy as np
import time
from typing import Tuple, Optional
from pathlib import Path

class MeshGenerator:
    """Generate various types of synthetic mesh data for benchmarking."""
    
    @staticmethod
    def generate_random_points(n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random 3D points in unit cube.
        
        Args:
            n_points: Number of points to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 3) with random 3D coordinates
        """
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.rand(n_points, 3)
    
    @staticmethod
    def generate_sphere_surface(n_points: int, radius: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate points on sphere surface.
        
        Args:
            n_points: Number of points to generate
            radius: Sphere radius
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 3) with sphere surface coordinates
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate uniform points on sphere using normal distribution
        points = np.random.randn(n_points, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms * radius
        
        return points
    
    @staticmethod
    def generate_cylinder_surface(n_points: int, radius: float = 1.0, height: float = 2.0, 
                                seed: Optional[int] = None) -> np.ndarray:
        """Generate points on cylinder surface.
        
        Args:
            n_points: Number of points to generate
            radius: Cylinder radius
            height: Cylinder height
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 3) with cylinder surface coordinates
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate angles and heights
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        z = np.random.uniform(-height/2, height/2, n_points)
        
        # Convert to Cartesian coordinates
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def generate_wing_profile(n_points: int, chord_length: float = 1.0, 
                            thickness: float = 0.12, seed: Optional[int] = None) -> np.ndarray:
        """Generate NACA-like airfoil profile points.
        
        Args:
            n_points: Number of points to generate
            chord_length: Wing chord length
            thickness: Maximum thickness as fraction of chord
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 3) with wing profile coordinates
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate x coordinates along chord
        x = np.linspace(0, chord_length, n_points // 2)
        
        # NACA symmetric airfoil equation (simplified)
        t = thickness
        y_upper = t * chord_length * (0.2969 * np.sqrt(x/chord_length) 
                                    - 0.1260 * (x/chord_length)
                                    - 0.3516 * (x/chord_length)**2
                                    + 0.2843 * (x/chord_length)**3
                                    - 0.1015 * (x/chord_length)**4)
        
        # Create upper and lower surfaces
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, -y_upper[::-1]])
        z_coords = np.random.uniform(-0.1, 0.1, len(x_coords))  # Small random z variation
        
        # Pad or trim to exact n_points
        if len(x_coords) > n_points:
            indices = np.linspace(0, len(x_coords)-1, n_points, dtype=int)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            z_coords = z_coords[indices]
        elif len(x_coords) < n_points:
            # Add random points to reach n_points
            extra_points = n_points - len(x_coords)
            extra_x = np.random.uniform(0, chord_length, extra_points)
            extra_y = np.random.uniform(-t*chord_length, t*chord_length, extra_points)
            extra_z = np.random.uniform(-0.1, 0.1, extra_points)
            
            x_coords = np.concatenate([x_coords, extra_x])
            y_coords = np.concatenate([y_coords, extra_y])
            z_coords = np.concatenate([z_coords, extra_z])
        
        return np.column_stack([x_coords, y_coords, z_coords])
    
    @staticmethod
    def generate_complex_geometry(n_points: int, geometry_type: str = "random", 
                                seed: Optional[int] = None) -> np.ndarray:
        """Generate complex geometry for realistic benchmarking.
        
        Args:
            n_points: Number of points to generate
            geometry_type: Type of geometry ("random", "sphere", "cylinder", "wing")
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_points, 3) with mesh coordinates
        """
        if geometry_type == "random":
            return MeshGenerator.generate_random_points(n_points, seed)
        elif geometry_type == "sphere":
            return MeshGenerator.generate_sphere_surface(n_points, seed=seed)
        elif geometry_type == "cylinder":
            return MeshGenerator.generate_cylinder_surface(n_points, seed=seed)
        elif geometry_type == "wing":
            return MeshGenerator.generate_wing_profile(n_points, seed=seed)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
    
    @staticmethod
    def save_mesh_data(mesh_points: np.ndarray, filepath: Path, 
                      metadata: Optional[dict] = None) -> None:
        """Save mesh data to file with metadata.
        
        Args:
            mesh_points: Mesh coordinates array
            filepath: Output file path
            metadata: Optional metadata dictionary
        """
        data_to_save = {'points': mesh_points}
        if metadata:
            data_to_save['metadata'] = metadata
        
        np.savez_compressed(filepath, **data_to_save)
    
    @staticmethod
    def load_mesh_data(filepath: Path) -> Tuple[np.ndarray, Optional[dict]]:
        """Load mesh data from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Tuple of (mesh_points, metadata)
        """
        data = np.load(filepath, allow_pickle=True)
        mesh_points = data['points']
        metadata = data.get('metadata', {}).item() if 'metadata' in data else None
        return mesh_points, metadata

def generate_benchmark_datasets(config, force_regenerate: bool = False) -> dict:
    """Generate all benchmark datasets according to configuration.
    
    Args:
        config: BenchmarkConfig instance
        force_regenerate: Whether to regenerate existing datasets
        
    Returns:
        Dictionary mapping (geometry_type, n_points) to file paths
    """
    datasets = {}
    geometry_types = ["random", "sphere", "cylinder", "wing"]
    
    for geometry_type in geometry_types:
        for n_points in config.mesh_sizes:
            filename = f"{geometry_type}_{n_points}_points.npz"
            filepath = config.data_dir / filename
            
            if filepath.exists() and not force_regenerate:
                print(f"Using existing dataset: {filepath}")
            else:
                print(f"Generating {geometry_type} dataset with {n_points:,} points...")
                
                start_time = time.time()
                mesh_points = MeshGenerator.generate_complex_geometry(
                    n_points, geometry_type, seed=42
                )
                generation_time = time.time() - start_time
                
                metadata = {
                    'geometry_type': geometry_type,
                    'n_points': n_points,
                    'generation_time': generation_time,
                    'seed': 42,
                    'bounds': {
                        'min': mesh_points.min(axis=0).tolist(),
                        'max': mesh_points.max(axis=0).tolist()
                    }
                }
                
                MeshGenerator.save_mesh_data(mesh_points, filepath, metadata)
                print(f"  Generated in {generation_time:.2f}s, saved to {filepath}")
            
            datasets[(geometry_type, n_points)] = filepath
    
    return datasets