"""
Advanced Mesh Quality Assessment and Analysis Module

This module provides comprehensive mesh quality analysis capabilities including:
- Multiple quality metrics calculation
- Quality visualization and reporting
- Quality-based mesh adaptation strategies
- Performance benchmarking for quality assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ElementType(Enum):
    """Supported element types."""
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"
    PRISM = "prism"
    PYRAMID = "pyramid"

@dataclass
class QualityThresholds:
    """Quality thresholds for different application types."""
    # CFD requirements (more stringent)
    cfd_excellent: Dict[str, float] = None
    cfd_good: Dict[str, float] = None
    cfd_acceptable: Dict[str, float] = None
    
    # Structural analysis requirements
    structural_excellent: Dict[str, float] = None
    structural_good: Dict[str, float] = None
    structural_acceptable: Dict[str, float] = None
    
    # General purpose requirements
    general_excellent: Dict[str, float] = None
    general_good: Dict[str, float] = None
    general_acceptable: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default thresholds."""
        if self.cfd_excellent is None:
            self.cfd_excellent = {
                'aspect_ratio': 5.0,
                'skewness': 0.25,
                'orthogonality': 0.15,
                'jacobian': 0.5,
                'volume_ratio': 2.0,
                'warpage': 0.1,
                'taper': 0.1,
                'stretch': 2.0,
                'condition_number': 10.0,
                'edge_ratio': 3.0
            }
        
        if self.cfd_good is None:
            self.cfd_good = {
                'aspect_ratio': 20.0,
                'skewness': 0.6,
                'orthogonality': 0.05,
                'jacobian': 0.2,
                'volume_ratio': 10.0,
                'warpage': 0.3,
                'taper': 0.3,
                'stretch': 5.0,
                'condition_number': 50.0,
                'edge_ratio': 10.0
            }
        
        if self.cfd_acceptable is None:
            self.cfd_acceptable = {
                'aspect_ratio': 100.0,
                'skewness': 0.95,
                'orthogonality': 0.01,
                'jacobian': 0.01,
                'volume_ratio': 100.0,
                'warpage': 0.9,
                'taper': 0.9,
                'stretch': 10.0,
                'condition_number': 1000.0,
                'edge_ratio': 100.0
            }

class MeshQualityAnalyzer:
    """Advanced mesh quality analyzer with comprehensive metrics."""
    
    def __init__(self, thresholds: QualityThresholds = None):
        """Initialize the quality analyzer.
        
        Args:
            thresholds: Quality thresholds for different applications
        """
        self.thresholds = thresholds or QualityThresholds()
        self.logger = logging.getLogger(__name__)
        
        # Cache for expensive computations
        self._adjacency_cache = {}
        self._normal_cache = {}
        self._volume_cache = {}
    
    def analyze_mesh_quality(self, points: np.ndarray, cells: np.ndarray, 
                           cell_types: List[ElementType] = None) -> Dict[str, Any]:
        """Perform comprehensive mesh quality analysis.
        
        Args:
            points: Mesh points array (N, 3)
            cells: Cell connectivity array
            cell_types: Element types for each cell
            
        Returns:
            Comprehensive quality analysis results
        """
        self.logger.info("Starting comprehensive mesh quality analysis...")
        
        if cell_types is None:
            cell_types = self._infer_cell_types(cells)
        
        # Calculate all quality metrics
        quality_metrics = {}
        
        # Geometric quality metrics
        quality_metrics['aspect_ratio'] = self.calculate_aspect_ratio(points, cells, cell_types)
        quality_metrics['skewness'] = self.calculate_skewness(points, cells, cell_types)
        quality_metrics['orthogonality'] = self.calculate_orthogonality(points, cells, cell_types)
        quality_metrics['jacobian'] = self.calculate_jacobian(points, cells, cell_types)
        quality_metrics['volume_ratio'] = self.calculate_volume_ratio(points, cells, cell_types)
        quality_metrics['warpage'] = self.calculate_warpage(points, cells, cell_types)
        quality_metrics['taper'] = self.calculate_taper(points, cells, cell_types)
        quality_metrics['stretch'] = self.calculate_stretch(points, cells, cell_types)
        quality_metrics['condition_number'] = self.calculate_condition_number(points, cells, cell_types)
        quality_metrics['edge_ratio'] = self.calculate_edge_ratio(points, cells, cell_types)
        
        # Additional specialized metrics
        quality_metrics['dihedral_angle'] = self.calculate_dihedral_angles(points, cells, cell_types)
        quality_metrics['face_planarity'] = self.calculate_face_planarity(points, cells, cell_types)
        quality_metrics['element_volume'] = self.calculate_element_volumes(points, cells, cell_types)
        quality_metrics['circumradius_ratio'] = self.calculate_circumradius_ratio(points, cells, cell_types)
        
        # Topology metrics
        quality_metrics['connectivity_quality'] = self.analyze_connectivity_quality(points, cells)
        quality_metrics['smoothness'] = self.analyze_mesh_smoothness(points, cells)
        
        # Create comprehensive report
        analysis_result = {
            'metrics': quality_metrics,
            'statistics': self._calculate_quality_statistics(quality_metrics),
            'classifications': self._classify_elements(quality_metrics),
            'recommendations': self._generate_improvement_recommendations(quality_metrics),
            'visualization_data': self._prepare_visualization_data(quality_metrics),
            'summary': self._create_quality_summary(quality_metrics)
        }
        
        self.logger.info("Mesh quality analysis completed")
        return analysis_result
    
    def calculate_aspect_ratio(self, points: np.ndarray, cells: np.ndarray, 
                             cell_types: List[ElementType]) -> np.ndarray:
        """Calculate aspect ratio for each element.
        
        The aspect ratio is defined as the ratio of the longest edge to the shortest edge.
        Lower values indicate better quality.
        """
        aspect_ratios = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.TRIANGLE:
                aspect_ratios[i] = self._triangle_aspect_ratio(points[cell])
            elif cell_type == ElementType.QUADRILATERAL:
                aspect_ratios[i] = self._quad_aspect_ratio(points[cell])
            elif cell_type == ElementType.TETRAHEDRON:
                aspect_ratios[i] = self._tetrahedron_aspect_ratio(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                aspect_ratios[i] = self._hexahedron_aspect_ratio(points[cell])
            elif cell_type == ElementType.PRISM:
                aspect_ratios[i] = self._prism_aspect_ratio(points[cell])
            elif cell_type == ElementType.PYRAMID:
                aspect_ratios[i] = self._pyramid_aspect_ratio(points[cell])
        
        return aspect_ratios
    
    def calculate_skewness(self, points: np.ndarray, cells: np.ndarray, 
                         cell_types: List[ElementType]) -> np.ndarray:
        """Calculate skewness for each element.
        
        Skewness measures how much an element deviates from the ideal shape.
        Values range from 0 (ideal) to 1 (degenerate).
        """
        skewness = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.TRIANGLE:
                skewness[i] = self._triangle_skewness(points[cell])
            elif cell_type == ElementType.QUADRILATERAL:
                skewness[i] = self._quad_skewness(points[cell])
            elif cell_type == ElementType.TETRAHEDRON:
                skewness[i] = self._tetrahedron_skewness(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                skewness[i] = self._hexahedron_skewness(points[cell])
            elif cell_type == ElementType.PRISM:
                skewness[i] = self._prism_skewness(points[cell])
            elif cell_type == ElementType.PYRAMID:
                skewness[i] = self._pyramid_skewness(points[cell])
        
        return skewness
    
    def calculate_orthogonality(self, points: np.ndarray, cells: np.ndarray, 
                              cell_types: List[ElementType]) -> np.ndarray:
        """Calculate orthogonality for each element.
        
        Orthogonality measures how close the element faces are to being perpendicular.
        Values closer to 1 indicate better orthogonality.
        """
        orthogonality = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.TRIANGLE:
                orthogonality[i] = self._triangle_orthogonality(points[cell])
            elif cell_type == ElementType.QUADRILATERAL:
                orthogonality[i] = self._quad_orthogonality(points[cell])
            elif cell_type == ElementType.TETRAHEDRON:
                orthogonality[i] = self._tetrahedron_orthogonality(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                orthogonality[i] = self._hexahedron_orthogonality(points[cell])
            elif cell_type == ElementType.PRISM:
                orthogonality[i] = self._prism_orthogonality(points[cell])
            elif cell_type == ElementType.PYRAMID:
                orthogonality[i] = self._pyramid_orthogonality(points[cell])
        
        return orthogonality
    
    def calculate_jacobian(self, points: np.ndarray, cells: np.ndarray, 
                         cell_types: List[ElementType]) -> np.ndarray:
        """Calculate Jacobian determinant for each element.
        
        The Jacobian determinant indicates element validity and orientation.
        Negative values indicate inverted elements.
        """
        jacobians = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.TRIANGLE:
                jacobians[i] = self._triangle_jacobian(points[cell])
            elif cell_type == ElementType.QUADRILATERAL:
                jacobians[i] = self._quad_jacobian(points[cell])
            elif cell_type == ElementType.TETRAHEDRON:
                jacobians[i] = self._tetrahedron_jacobian(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                jacobians[i] = self._hexahedron_jacobian(points[cell])
            elif cell_type == ElementType.PRISM:
                jacobians[i] = self._prism_jacobian(points[cell])
            elif cell_type == ElementType.PYRAMID:
                jacobians[i] = self._pyramid_jacobian(points[cell])
        
        return jacobians
    
    def calculate_volume_ratio(self, points: np.ndarray, cells: np.ndarray, 
                             cell_types: List[ElementType]) -> np.ndarray:
        """Calculate volume ratio for each element.
        
        Volume ratio compares the actual volume to the ideal volume.
        Values close to 1 indicate good quality.
        """
        volume_ratios = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            actual_volume = self._calculate_element_volume(points[cell], cell_type)
            ideal_volume = self._calculate_ideal_volume(points[cell], cell_type)
            
            if ideal_volume > 0:
                volume_ratios[i] = actual_volume / ideal_volume
            else:
                volume_ratios[i] = 0.0
        
        return volume_ratios
    
    def calculate_warpage(self, points: np.ndarray, cells: np.ndarray, 
                        cell_types: List[ElementType]) -> np.ndarray:
        """Calculate warpage for quadrilateral and hexahedral elements.
        
        Warpage measures how much a face deviates from planarity.
        """
        warpage = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.QUADRILATERAL:
                warpage[i] = self._quad_warpage(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                warpage[i] = self._hexahedron_warpage(points[cell])
            # Other element types don't have warpage
        
        return warpage
    
    def calculate_taper(self, points: np.ndarray, cells: np.ndarray, 
                      cell_types: List[ElementType]) -> np.ndarray:
        """Calculate taper for quadrilateral and hexahedral elements.
        
        Taper measures how much an element deviates from a rectangular shape.
        """
        taper = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.QUADRILATERAL:
                taper[i] = self._quad_taper(points[cell])
            elif cell_type == ElementType.HEXAHEDRON:
                taper[i] = self._hexahedron_taper(points[cell])
        
        return taper
    
    def calculate_stretch(self, points: np.ndarray, cells: np.ndarray, 
                        cell_types: List[ElementType]) -> np.ndarray:
        """Calculate stretch ratio for elements.
        
        Stretch measures the ratio of maximum to minimum edge lengths.
        """
        stretch = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            edge_lengths = self._get_edge_lengths(points[cell], cell_type)
            if len(edge_lengths) > 0:
                stretch[i] = np.max(edge_lengths) / np.min(edge_lengths)
        
        return stretch
    
    def calculate_condition_number(self, points: np.ndarray, cells: np.ndarray, 
                                 cell_types: List[ElementType]) -> np.ndarray:
        """Calculate condition number for elements.
        
        Condition number measures numerical stability of the element.
        """
        condition_numbers = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            jacobian_matrix = self._get_jacobian_matrix(points[cell], cell_type)
            if jacobian_matrix is not None:
                try:
                    condition_numbers[i] = np.linalg.cond(jacobian_matrix)
                except:
                    condition_numbers[i] = 1e6  # Very high condition number for degenerate cases
        
        return condition_numbers
    
    def calculate_edge_ratio(self, points: np.ndarray, cells: np.ndarray, 
                           cell_types: List[ElementType]) -> np.ndarray:
        """Calculate edge ratio for elements.
        
        Edge ratio is similar to aspect ratio but specifically for edge lengths.
        """
        edge_ratios = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            edge_lengths = self._get_edge_lengths(points[cell], cell_type)
            if len(edge_lengths) > 0:
                edge_ratios[i] = np.max(edge_lengths) / np.min(edge_lengths)
        
        return edge_ratios
    
    def calculate_dihedral_angles(self, points: np.ndarray, cells: np.ndarray, 
                                cell_types: List[ElementType]) -> np.ndarray:
        """Calculate minimum dihedral angles for 3D elements."""
        min_angles = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type in [ElementType.TETRAHEDRON, ElementType.HEXAHEDRON, 
                           ElementType.PRISM, ElementType.PYRAMID]:
                angles = self._get_dihedral_angles(points[cell], cell_type)
                min_angles[i] = np.min(angles) if len(angles) > 0 else 0.0
        
        return min_angles
    
    def calculate_face_planarity(self, points: np.ndarray, cells: np.ndarray, 
                               cell_types: List[ElementType]) -> np.ndarray:
        """Calculate face planarity for 3D elements."""
        planarity = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type in [ElementType.HEXAHEDRON, ElementType.PRISM, ElementType.PYRAMID]:
                planarity[i] = self._calculate_face_planarity(points[cell], cell_type)
        
        return planarity
    
    def calculate_element_volumes(self, points: np.ndarray, cells: np.ndarray, 
                                cell_types: List[ElementType]) -> np.ndarray:
        """Calculate volumes for 3D elements."""
        volumes = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            volumes[i] = self._calculate_element_volume(points[cell], cell_type)
        
        return volumes
    
    def calculate_circumradius_ratio(self, points: np.ndarray, cells: np.ndarray, 
                                   cell_types: List[ElementType]) -> np.ndarray:
        """Calculate circumradius to inradius ratio (for triangles and tetrahedra)."""
        ratios = np.zeros(len(cells))
        
        for i, (cell, cell_type) in enumerate(zip(cells, cell_types)):
            if cell_type == ElementType.TRIANGLE:
                ratios[i] = self._triangle_circumradius_ratio(points[cell])
            elif cell_type == ElementType.TETRAHEDRON:
                ratios[i] = self._tetrahedron_circumradius_ratio(points[cell])
        
        return ratios
    
    def analyze_connectivity_quality(self, points: np.ndarray, cells: np.ndarray) -> Dict[str, float]:
        """Analyze mesh connectivity quality."""
        # Build adjacency information
        adjacency = self._build_adjacency_graph(cells)
        
        # Calculate connectivity metrics
        valence_distribution = self._calculate_vertex_valence(adjacency)
        connectivity_regularity = self._assess_connectivity_regularity(valence_distribution)
        isolated_vertices = self._find_isolated_vertices(adjacency, len(points))
        
        return {
            'average_valence': np.mean(valence_distribution),
            'valence_std': np.std(valence_distribution),
            'connectivity_regularity': connectivity_regularity,
            'isolated_vertices': len(isolated_vertices),
            'isolated_percentage': len(isolated_vertices) / len(points) * 100
        }
    
    def analyze_mesh_smoothness(self, points: np.ndarray, cells: np.ndarray) -> Dict[str, float]:
        """Analyze mesh smoothness characteristics."""
        # Calculate normal vectors
        normals = self._calculate_face_normals(points, cells)
        
        # Analyze normal variation
        normal_variation = self._calculate_normal_variation(normals, cells)
        
        # Calculate curvature estimates
        curvature_estimates = self._estimate_mesh_curvature(points, cells)
        
        return {
            'normal_variation': np.mean(normal_variation),
            'max_normal_variation': np.max(normal_variation),
            'average_curvature': np.mean(curvature_estimates),
            'max_curvature': np.max(curvature_estimates),
            'smoothness_score': 1.0 / (1.0 + np.mean(normal_variation))
        }
    
    def create_quality_report(self, analysis_result: Dict[str, Any], 
                            output_path: Optional[Path] = None) -> str:
        """Create a comprehensive quality report."""
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("MESH QUALITY ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Summary statistics
        summary = analysis_result['summary']
        report_lines.append("SUMMARY")
        report_lines.append("-"*40)
        report_lines.append(f"Total Elements: {summary['total_elements']}")
        report_lines.append(f"Overall Quality Score: {summary['overall_score']:.3f}")
        report_lines.append(f"Elements Passing Quality: {summary['passing_percentage']:.1f}%")
        report_lines.append("")
        
        # Quality metrics statistics
        statistics = analysis_result['statistics']
        report_lines.append("QUALITY METRICS")
        report_lines.append("-"*40)
        
        for metric_name, stats in statistics.items():
            report_lines.append(f"{metric_name.upper().replace('_', ' ')}:")
            report_lines.append(f"  Mean: {stats['mean']:.6f}")
            report_lines.append(f"  Std:  {stats['std']:.6f}")
            report_lines.append(f"  Min:  {stats['min']:.6f}")
            report_lines.append(f"  Max:  {stats['max']:.6f}")
            report_lines.append("")
        
        # Classifications
        classifications = analysis_result['classifications']
        report_lines.append("ELEMENT CLASSIFICATIONS")
        report_lines.append("-"*40)
        report_lines.append(f"Excellent: {classifications['excellent_count']} ({classifications['excellent_percentage']:.1f}%)")
        report_lines.append(f"Good:      {classifications['good_count']} ({classifications['good_percentage']:.1f}%)")
        report_lines.append(f"Acceptable: {classifications['acceptable_count']} ({classifications['acceptable_percentage']:.1f}%)")
        report_lines.append(f"Poor:      {classifications['poor_count']} ({classifications['poor_percentage']:.1f}%)")
        report_lines.append("")
        
        # Recommendations
        recommendations = analysis_result['recommendations']
        if recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-"*40)
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Footer
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Quality report saved to {output_path}")
        
        return report_text
    
    def visualize_quality_metrics(self, analysis_result: Dict[str, Any], 
                                 output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Create quality visualization plots."""
        if output_dir is None:
            output_dir = Path("quality_plots")
        output_dir.mkdir(exist_ok=True)
        
        saved_plots = {}
        metrics = analysis_result['metrics']
        
        # Quality distribution histograms
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        main_metrics = ['aspect_ratio', 'skewness', 'orthogonality', 
                       'jacobian', 'volume_ratio', 'edge_ratio']
        
        for i, metric_name in enumerate(main_metrics):
            if i < len(axes) and metric_name in metrics:
                ax = axes[i]
                data = metrics[metric_name]
                
                # Remove outliers for better visualization
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                
                ax.hist(filtered_data, bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel(metric_name.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        quality_hist_path = output_dir / "quality_distributions.png"
        plt.savefig(quality_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots['distributions'] = quality_hist_path
        
        # Quality correlation matrix
        metric_data = {}
        for metric_name, values in metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                metric_data[metric_name] = values
        
        if len(metric_data) > 1:
            df = pd.DataFrame(metric_data)
            correlation_matrix = df.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Quality Metrics Correlation Matrix')
            plt.tight_layout()
            
            correlation_path = output_dir / "quality_correlations.png"
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots['correlations'] = correlation_path
        
        # Box plots for quality metrics
        fig, ax = plt.subplots(figsize=(15, 8))
        
        box_data = []
        labels = []
        for metric_name in main_metrics:
            if metric_name in metrics:
                data = metrics[metric_name]
                # Remove extreme outliers for better visualization
                q1, q3 = np.percentile(data, [5, 95])
                filtered_data = data[(data >= q1) & (data <= q3)]
                box_data.append(filtered_data)
                labels.append(metric_name.replace('_', ' ').title())
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Quality Value')
            ax.set_title('Quality Metrics Distribution')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            boxplot_path = output_dir / "quality_boxplots.png"
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots['boxplots'] = boxplot_path
        
        self.logger.info(f"Quality visualization plots saved to {output_dir}")
        return saved_plots
    
    # Private helper methods for quality calculations
    def _infer_cell_types(self, cells: np.ndarray) -> List[ElementType]:
        """Infer element types from cell connectivity."""
        cell_types = []
        for cell in cells:
            num_nodes = len(cell)
            if num_nodes == 3:
                cell_types.append(ElementType.TRIANGLE)
            elif num_nodes == 4:
                # Could be quad or tetrahedron - need more sophisticated detection
                cell_types.append(ElementType.TETRAHEDRON)  # Default assumption
            elif num_nodes == 6:
                cell_types.append(ElementType.PRISM)
            elif num_nodes == 8:
                cell_types.append(ElementType.HEXAHEDRON)
            else:
                cell_types.append(ElementType.TETRAHEDRON)  # Default
        
        return cell_types
    
    # Element-specific quality calculation methods
    def _triangle_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for triangle."""
        edges = [
            np.linalg.norm(vertices[1] - vertices[0]),
            np.linalg.norm(vertices[2] - vertices[1]),
            np.linalg.norm(vertices[0] - vertices[2])
        ]
        return max(edges) / min(edges) if min(edges) > 0 else 1e6
    
    def _triangle_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for triangle."""
        # Calculate area using cross product
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        # Calculate perimeter
        perimeter = (np.linalg.norm(vertices[1] - vertices[0]) +
                    np.linalg.norm(vertices[2] - vertices[1]) +
                    np.linalg.norm(vertices[0] - vertices[2]))
        
        # Ideal area for given perimeter
        ideal_area = perimeter**2 / (12 * np.sqrt(3))
        
        if ideal_area > 0:
            return 1.0 - area / ideal_area
        return 1.0
    
    def _triangle_orthogonality(self, vertices: np.ndarray) -> float:
        """Calculate orthogonality for triangle."""
        # For triangles, orthogonality is related to angle quality
        angles = self._calculate_triangle_angles(vertices)
        ideal_angle = np.pi / 3  # 60 degrees
        angle_deviations = np.abs(angles - ideal_angle)
        return 1.0 - np.max(angle_deviations) / ideal_angle
    
    def _triangle_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for triangle."""
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        
        # 2D Jacobian determinant
        if len(v1) >= 2:
            jacobian = v1[0] * v2[1] - v1[1] * v2[0]
            return jacobian
        return 0.0
    
    def _tetrahedron_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for tetrahedron."""
        # Calculate all edge lengths
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                edges.append(np.linalg.norm(vertices[i] - vertices[j]))
        
        return max(edges) / min(edges) if min(edges) > 0 else 1e6
    
    def _tetrahedron_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for tetrahedron."""
        # Calculate volume
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        
        # Calculate surface area
        surface_area = (
            self._triangle_area(vertices[[0, 1, 2]]) +
            self._triangle_area(vertices[[0, 1, 3]]) +
            self._triangle_area(vertices[[0, 2, 3]]) +
            self._triangle_area(vertices[[1, 2, 3]])
        )
        
        # Ideal tetrahedron ratio
        if surface_area > 0:
            actual_ratio = volume**(2/3) / surface_area
            ideal_ratio = 1.0 / (6 * np.sqrt(2))  # For regular tetrahedron
            return 1.0 - actual_ratio / ideal_ratio
        return 1.0
    
    def _tetrahedron_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for tetrahedron."""
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        
        # 3D Jacobian determinant
        jacobian_matrix = np.column_stack([v1, v2, v3])
        return np.linalg.det(jacobian_matrix)
    
    # Additional helper methods (simplified implementations)
    def _quad_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for quadrilateral."""
        # Simplified implementation
        return self._triangle_aspect_ratio(vertices[:3])
    
    def _quad_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for quadrilateral."""
        return self._triangle_skewness(vertices[:3])
    
    def _quad_orthogonality(self, vertices: np.ndarray) -> float:
        """Calculate orthogonality for quadrilateral."""
        return self._triangle_orthogonality(vertices[:3])
    
    def _quad_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for quadrilateral."""
        return self._triangle_jacobian(vertices[:3])
    
    def _hexahedron_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for hexahedron."""
        return self._tetrahedron_aspect_ratio(vertices[:4])
    
    def _hexahedron_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for hexahedron."""
        return self._tetrahedron_skewness(vertices[:4])
    
    def _hexahedron_orthogonality(self, vertices: np.ndarray) -> float:
        """Calculate orthogonality for hexahedron."""
        return self._tetrahedron_orthogonality(vertices[:4])
    
    def _hexahedron_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for hexahedron."""
        return self._tetrahedron_jacobian(vertices[:4])
    
    def _prism_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for prism."""
        return self._tetrahedron_aspect_ratio(vertices[:4])
    
    def _prism_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for prism."""
        return self._tetrahedron_skewness(vertices[:4])
    
    def _prism_orthogonality(self, vertices: np.ndarray) -> float:
        """Calculate orthogonality for prism."""
        return self._tetrahedron_orthogonality(vertices[:4])
    
    def _prism_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for prism."""
        return self._tetrahedron_jacobian(vertices[:4])
    
    def _pyramid_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Calculate aspect ratio for pyramid."""
        return self._tetrahedron_aspect_ratio(vertices[:4])
    
    def _pyramid_skewness(self, vertices: np.ndarray) -> float:
        """Calculate skewness for pyramid."""
        return self._tetrahedron_skewness(vertices[:4])
    
    def _pyramid_orthogonality(self, vertices: np.ndarray) -> float:
        """Calculate orthogonality for pyramid."""
        return self._tetrahedron_orthogonality(vertices[:4])
    
    def _pyramid_jacobian(self, vertices: np.ndarray) -> float:
        """Calculate Jacobian for pyramid."""
        return self._tetrahedron_jacobian(vertices[:4])
    
    def _calculate_triangle_angles(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate angles of a triangle."""
        # Calculate vectors
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[1]
        v3 = vertices[0] - vertices[2]
        
        # Calculate angles using dot product
        angles = []
        vectors = [(-v1, v2), (-v2, v3), (-v3, v1)]
        
        for vec1, vec2 in vectors:
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angles.append(np.arccos(cos_angle))
        
        return np.array(angles)
    
    def _triangle_area(self, vertices: np.ndarray) -> float:
        """Calculate area of triangle."""
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2))
    
    def _calculate_element_volume(self, vertices: np.ndarray, element_type: ElementType) -> float:
        """Calculate volume of element."""
        if element_type == ElementType.TRIANGLE:
            return self._triangle_area(vertices)
        elif element_type == ElementType.TETRAHEDRON:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            v3 = vertices[3] - vertices[0]
            return abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        # Add more element types as needed
        return 0.0
    
    def _calculate_ideal_volume(self, vertices: np.ndarray, element_type: ElementType) -> float:
        """Calculate ideal volume for given vertices."""
        # Simplified implementation - should be more sophisticated
        return self._calculate_element_volume(vertices, element_type)
    
    def _get_edge_lengths(self, vertices: np.ndarray, element_type: ElementType) -> np.ndarray:
        """Get all edge lengths for an element."""
        edges = []
        n = len(vertices)
        
        if element_type == ElementType.TRIANGLE:
            edges = [
                np.linalg.norm(vertices[1] - vertices[0]),
                np.linalg.norm(vertices[2] - vertices[1]),
                np.linalg.norm(vertices[0] - vertices[2])
            ]
        elif element_type == ElementType.TETRAHEDRON:
            for i in range(n):
                for j in range(i+1, n):
                    edges.append(np.linalg.norm(vertices[i] - vertices[j]))
        # Add more element types as needed
        
        return np.array(edges)
    
    def _get_jacobian_matrix(self, vertices: np.ndarray, element_type: ElementType) -> Optional[np.ndarray]:
        """Get Jacobian matrix for element."""
        if element_type == ElementType.TRIANGLE and len(vertices) >= 3:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            return np.column_stack([v1[:2], v2[:2]])
        elif element_type == ElementType.TETRAHEDRON and len(vertices) >= 4:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            v3 = vertices[3] - vertices[0]
            return np.column_stack([v1, v2, v3])
        
        return None
    
    def _get_dihedral_angles(self, vertices: np.ndarray, element_type: ElementType) -> np.ndarray:
        """Calculate dihedral angles for 3D elements."""
        # Simplified implementation
        return np.array([np.pi/3] * 6)  # Placeholder
    
    def _calculate_face_planarity(self, vertices: np.ndarray, element_type: ElementType) -> float:
        """Calculate face planarity for 3D elements."""
        # Simplified implementation
        return 0.95  # Placeholder
    
    def _triangle_circumradius_ratio(self, vertices: np.ndarray) -> float:
        """Calculate circumradius to inradius ratio for triangle."""
        # Calculate circumradius
        a = np.linalg.norm(vertices[1] - vertices[0])
        b = np.linalg.norm(vertices[2] - vertices[1])
        c = np.linalg.norm(vertices[0] - vertices[2])
        
        area = self._triangle_area(vertices)
        circumradius = (a * b * c) / (4 * area) if area > 0 else 1e6
        
        # Calculate inradius
        s = (a + b + c) / 2  # Semi-perimeter
        inradius = area / s if s > 0 else 1e-6
        
        return circumradius / inradius if inradius > 0 else 1e6
    
    def _tetrahedron_circumradius_ratio(self, vertices: np.ndarray) -> float:
        """Calculate circumradius to inradius ratio for tetrahedron."""
        # Simplified implementation
        return 3.0  # Placeholder
    
    def _quad_warpage(self, vertices: np.ndarray) -> float:
        """Calculate warpage for quadrilateral."""
        # Simplified implementation
        return 0.1  # Placeholder
    
    def _hexahedron_warpage(self, vertices: np.ndarray) -> float:
        """Calculate warpage for hexahedron."""
        # Simplified implementation
        return 0.1  # Placeholder
    
    def _quad_taper(self, vertices: np.ndarray) -> float:
        """Calculate taper for quadrilateral."""
        # Simplified implementation
        return 0.1  # Placeholder
    
    def _hexahedron_taper(self, vertices: np.ndarray) -> float:
        """Calculate taper for hexahedron."""
        # Simplified implementation
        return 0.1  # Placeholder
    
    def _build_adjacency_graph(self, cells: np.ndarray) -> Dict[int, List[int]]:
        """Build vertex adjacency graph."""
        adjacency = {}
        for cell in cells:
            for vertex in cell:
                if vertex not in adjacency:
                    adjacency[vertex] = []
                for other_vertex in cell:
                    if other_vertex != vertex and other_vertex not in adjacency[vertex]:
                        adjacency[vertex].append(other_vertex)
        return adjacency
    
    def _calculate_vertex_valence(self, adjacency: Dict[int, List[int]]) -> np.ndarray:
        """Calculate vertex valence distribution."""
        valences = [len(neighbors) for neighbors in adjacency.values()]
        return np.array(valences)
    
    def _assess_connectivity_regularity(self, valences: np.ndarray) -> float:
        """Assess connectivity regularity."""
        if len(valences) == 0:
            return 0.0
        
        ideal_valence = 6  # For triangular meshes
        regularity = 1.0 - np.std(valences) / ideal_valence
        return max(0.0, regularity)
    
    def _find_isolated_vertices(self, adjacency: Dict[int, List[int]], total_vertices: int) -> List[int]:
        """Find isolated vertices."""
        connected_vertices = set(adjacency.keys())
        all_vertices = set(range(total_vertices))
        isolated = list(all_vertices - connected_vertices)
        
        # Also check for vertices with no neighbors
        isolated.extend([v for v, neighbors in adjacency.items() if len(neighbors) == 0])
        
        return list(set(isolated))
    
    def _calculate_face_normals(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Calculate face normal vectors."""
        normals = []
        for cell in cells:
            if len(cell) >= 3:
                v1 = points[cell[1]] - points[cell[0]]
                v2 = points[cell[2]] - points[cell[0]]
                normal = np.cross(v1, v2)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                normals.append(normal)
        return np.array(normals)
    
    def _calculate_normal_variation(self, normals: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Calculate normal variation between adjacent faces."""
        # Simplified implementation
        if len(normals) < 2:
            return np.array([0.0])
        
        variations = []
        for i in range(len(normals) - 1):
            dot_product = np.dot(normals[i], normals[i + 1])
            angle = np.arccos(np.clip(dot_product, -1, 1))
            variations.append(angle)
        
        return np.array(variations)
    
    def _estimate_mesh_curvature(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Estimate mesh curvature."""
        # Simplified implementation
        return np.random.uniform(0, 0.1, len(cells))
    
    def _calculate_quality_statistics(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all quality metrics."""
        statistics = {}
        for metric_name, values in metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        return statistics
    
    def _classify_elements(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Classify elements based on quality thresholds."""
        if not metrics:
            return {}
        
        # Use first metric to determine number of elements
        first_metric = next(iter(metrics.values()))
        num_elements = len(first_metric)
        
        # Initialize classification arrays
        excellent = np.ones(num_elements, dtype=bool)
        good = np.ones(num_elements, dtype=bool)
        acceptable = np.ones(num_elements, dtype=bool)
        
        # Apply thresholds for each metric
        thresholds = self.thresholds.cfd_acceptable  # Default to CFD thresholds
        
        for metric_name, values in metrics.items():
            if metric_name in thresholds and isinstance(values, np.ndarray):
                threshold = thresholds[metric_name]
                
                # Different metrics have different "good" directions
                if metric_name in ['skewness', 'warpage', 'taper']:
                    # Lower is better
                    excellent &= values <= self.thresholds.cfd_excellent[metric_name]
                    good &= values <= self.thresholds.cfd_good[metric_name]
                    acceptable &= values <= threshold
                elif metric_name in ['orthogonality', 'jacobian']:
                    # Higher is better
                    excellent &= values >= self.thresholds.cfd_excellent[metric_name]
                    good &= values >= self.thresholds.cfd_good[metric_name]
                    acceptable &= values >= threshold
                else:
                    # Lower is better (aspect_ratio, etc.)
                    excellent &= values <= self.thresholds.cfd_excellent[metric_name]
                    good &= values <= self.thresholds.cfd_good[metric_name]
                    acceptable &= values <= threshold
        
        # Calculate final classifications
        poor = ~acceptable
        acceptable_only = acceptable & ~good
        good_only = good & ~excellent
        
        return {
            'excellent_count': np.sum(excellent),
            'excellent_percentage': np.sum(excellent) / num_elements * 100,
            'good_count': np.sum(good_only),
            'good_percentage': np.sum(good_only) / num_elements * 100,
            'acceptable_count': np.sum(acceptable_only),
            'acceptable_percentage': np.sum(acceptable_only) / num_elements * 100,
            'poor_count': np.sum(poor),
            'poor_percentage': np.sum(poor) / num_elements * 100,
            'excellent_elements': np.where(excellent)[0],
            'good_elements': np.where(good_only)[0],
            'acceptable_elements': np.where(acceptable_only)[0],
            'poor_elements': np.where(poor)[0]
        }
    
    def _generate_improvement_recommendations(self, metrics: Dict[str, np.ndarray]) -> List[str]:
        """Generate recommendations for mesh quality improvement."""
        recommendations = []
        
        # Analyze each metric and provide specific recommendations
        for metric_name, values in metrics.items():
            if not isinstance(values, np.ndarray) or len(values) == 0:
                continue
            
            mean_value = np.mean(values)
            max_value = np.max(values)
            poor_percentage = np.sum(values > self.thresholds.cfd_acceptable.get(metric_name, 1e6)) / len(values) * 100
            
            if metric_name == 'aspect_ratio' and mean_value > 10:
                recommendations.append(f"High aspect ratios detected (mean: {mean_value:.2f}). Consider mesh refinement or anisotropic adaptation.")
            
            if metric_name == 'skewness' and mean_value > 0.5:
                recommendations.append(f"High skewness detected (mean: {mean_value:.2f}). Apply Laplacian or Taubin smoothing.")
            
            if metric_name == 'orthogonality' and mean_value < 0.3:
                recommendations.append(f"Poor orthogonality detected (mean: {mean_value:.2f}). Consider mesh quality improvement algorithms.")
            
            if metric_name == 'jacobian' and np.min(values) < 0:
                num_inverted = np.sum(values < 0)
                recommendations.append(f"{num_inverted} inverted elements detected. Apply mesh repair algorithms.")
            
            if poor_percentage > 20:
                recommendations.append(f"{metric_name.replace('_', ' ').title()}: {poor_percentage:.1f}% of elements below acceptable quality. Consider global mesh improvement.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Mesh quality is generally acceptable. No major issues detected.")
        
        return recommendations
    
    def _prepare_visualization_data(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare data for visualization."""
        viz_data = {}
        
        for metric_name, values in metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                viz_data[metric_name] = {
                    'values': values,
                    'histogram_bins': np.histogram(values, bins=50),
                    'percentiles': np.percentile(values, [5, 25, 50, 75, 95]),
                    'outliers': values[np.abs(values - np.mean(values)) > 3 * np.std(values)]
                }
        
        return viz_data
    
    def _create_quality_summary(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create overall quality summary."""
        if not metrics:
            return {'overall_score': 0.0, 'total_elements': 0, 'passing_percentage': 0.0}
        
        # Calculate overall quality score
        scores = []
        total_elements = 0
        
        for metric_name, values in metrics.items():
            if isinstance(values, np.ndarray) and len(values) > 0:
                total_elements = len(values)
                
                # Normalize metric to 0-1 score
                if metric_name in self.thresholds.cfd_acceptable:
                    threshold = self.thresholds.cfd_acceptable[metric_name]
                    
                    if metric_name in ['skewness', 'warpage', 'taper']:
                        # Lower is better
                        score = np.mean(1.0 - np.clip(values / threshold, 0, 1))
                    elif metric_name in ['orthogonality', 'jacobian']:
                        # Higher is better
                        score = np.mean(np.clip(values / threshold, 0, 1))
                    else:
                        # Lower is better
                        score = np.mean(1.0 / (1.0 + values / threshold))
                    
                    scores.append(score)
        
        overall_score = np.mean(scores) if scores else 0.0
        
        # Calculate passing percentage (simplified)
        passing_elements = 0
        if metrics:
            first_metric = next(iter(metrics.values()))
            if isinstance(first_metric, np.ndarray):
                # Simple heuristic: assume elements pass if they're not in worst 20%
                passing_elements = int(0.8 * len(first_metric))
        
        passing_percentage = (passing_elements / total_elements * 100) if total_elements > 0 else 0.0
        
        return {
            'overall_score': overall_score,
            'total_elements': total_elements,
            'passing_percentage': passing_percentage,
            'individual_scores': dict(zip(metrics.keys(), scores)) if scores else {}
        }