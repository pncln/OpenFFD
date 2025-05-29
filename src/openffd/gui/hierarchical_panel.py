"""
Hierarchical FFD panel for the OpenFFD GUI.

This module provides the hierarchical FFD panel for the GUI,
allowing users to create and manipulate hierarchical FFD control boxes.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QGroupBox,
    QDoubleSpinBox, QComboBox, QCheckBox,
    QSplitter
)

from openffd.core.hierarchical import HierarchicalFFD, HierarchicalLevel, create_hierarchical_ffd
from openffd.utils.parallel import ParallelConfig
from openffd.visualization.hierarchical_viz import visualize_hierarchical_ffd_pyvista

# Configure logging
logger = logging.getLogger(__name__)


class HierarchicalFFDPanel(QWidget):
    """Panel for managing hierarchical FFD control boxes."""
    
    # Signal for when hierarchical FFD is updated
    hierarchical_ffd_updated = pyqtSignal(object)  # Emits HierarchicalFFD object
    
    def __init__(self, parent=None):
        """Initialize the hierarchical FFD panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.hierarchical_ffd = None
        self.mesh_points = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Hierarchical FFD Control")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(title_label)
        
        # Create hierarchical FFD group
        create_group = QGroupBox("Create Hierarchical FFD")
        create_layout = QVBoxLayout(create_group)
        
        # Base dimensions
        base_dims_layout = QHBoxLayout()
        base_dims_layout.addWidget(QLabel("Base Dimensions:"))
        
        self.base_dims_nx = QSpinBox()
        self.base_dims_nx.setRange(2, 20)
        self.base_dims_nx.setValue(4)
        self.base_dims_nx.setToolTip("Number of control points in X direction")
        
        self.base_dims_ny = QSpinBox()
        self.base_dims_ny.setRange(2, 20)
        self.base_dims_ny.setValue(4)
        self.base_dims_ny.setToolTip("Number of control points in Y direction")
        
        self.base_dims_nz = QSpinBox()
        self.base_dims_nz.setRange(2, 20)
        self.base_dims_nz.setValue(4)
        self.base_dims_nz.setToolTip("Number of control points in Z direction")
        
        base_dims_layout.addWidget(self.base_dims_nx)
        base_dims_layout.addWidget(self.base_dims_ny)
        base_dims_layout.addWidget(self.base_dims_nz)
        create_layout.addLayout(base_dims_layout)
        
        # Hierarchy depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 5)
        self.max_depth.setValue(3)
        self.max_depth.setToolTip("Maximum depth of the hierarchical FFD")
        depth_layout.addWidget(self.max_depth)
        
        depth_layout.addWidget(QLabel("Subdivision:"))
        self.subdiv_factor = QSpinBox()
        self.subdiv_factor.setRange(2, 4)
        self.subdiv_factor.setValue(2)
        self.subdiv_factor.setToolTip("Subdivision factor between hierarchy levels")
        depth_layout.addWidget(self.subdiv_factor)
        
        create_layout.addLayout(depth_layout)
        
        # Margin for control box
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Margin:"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 0.5)
        self.margin.setValue(0.05)
        self.margin.setSingleStep(0.01)
        self.margin.setToolTip("Margin around mesh for control box")
        margin_layout.addWidget(self.margin)
        create_layout.addLayout(margin_layout)
        
        # Create button
        self.create_button = QPushButton("Create Hierarchical FFD")
        self.create_button.setEnabled(False)
        self.create_button.clicked.connect(self._create_hierarchical_ffd)
        create_layout.addWidget(self.create_button)
        
        main_layout.addWidget(create_group)
        
        # Levels management group
        levels_group = QGroupBox("Levels Management")
        levels_layout = QVBoxLayout(levels_group)
        
        # Tree for showing levels
        self.levels_tree = QTreeWidget()
        self.levels_tree.setHeaderLabels(["Level", "Dimensions", "Weight"])
        self.levels_tree.setColumnWidth(0, 80)
        self.levels_tree.setColumnWidth(1, 100)
        levels_layout.addWidget(self.levels_tree)
        
        # Level controls
        level_controls_layout = QHBoxLayout()
        
        self.add_level_button = QPushButton("Add Level")
        self.add_level_button.setEnabled(False)
        self.add_level_button.clicked.connect(self._add_level)
        level_controls_layout.addWidget(self.add_level_button)
        
        self.remove_level_button = QPushButton("Remove Level")
        self.remove_level_button.setEnabled(False)
        self.remove_level_button.clicked.connect(self._remove_level)
        level_controls_layout.addWidget(self.remove_level_button)
        
        levels_layout.addLayout(level_controls_layout)
        
        main_layout.addWidget(levels_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout(viz_group)
        
        # Show mesh checkbox
        self.show_mesh_cb = QCheckBox("Show Mesh")
        self.show_mesh_cb.setChecked(True)
        viz_layout.addWidget(self.show_mesh_cb)
        
        # Color by level checkbox
        self.color_by_level_cb = QCheckBox("Color by Level")
        self.color_by_level_cb.setChecked(True)
        viz_layout.addWidget(self.color_by_level_cb)
        
        # Show influence regions
        self.show_influence_cb = QCheckBox("Show Influence Regions")
        self.show_influence_cb.setChecked(False)
        viz_layout.addWidget(self.show_influence_cb)
        
        # Visualization button
        self.visualize_button = QPushButton("Update Visualization")
        self.visualize_button.setEnabled(False)
        self.visualize_button.clicked.connect(self._update_visualization)
        viz_layout.addWidget(self.visualize_button)
        
        main_layout.addWidget(viz_group)
        
        # Set the layout
        self.setLayout(main_layout)
    
    def set_mesh_points(self, mesh_points: np.ndarray):
        """Set the mesh points for the hierarchical FFD.
        
        Args:
            mesh_points: Numpy array of mesh point coordinates
        """
        self.mesh_points = mesh_points
        self.create_button.setEnabled(mesh_points is not None and len(mesh_points) > 0)
    
    def _create_hierarchical_ffd(self):
        """Create a hierarchical FFD from the current settings."""
        if self.mesh_points is None or len(self.mesh_points) == 0:
            logger.error("No mesh points available for hierarchical FFD")
            return
        
        try:
            # Get configuration from UI
            base_dims = (
                self.base_dims_nx.value(),
                self.base_dims_ny.value(),
                self.base_dims_nz.value()
            )
            max_depth = self.max_depth.value()
            subdivision_factor = self.subdiv_factor.value()
            margin = self.margin.value()
            
            # Create the hierarchical FFD
            logger.info(f"Creating hierarchical FFD with base dims {base_dims}, depth {max_depth}, subdivision {subdivision_factor}")
            self.hierarchical_ffd = create_hierarchical_ffd(
                mesh_points=self.mesh_points,
                base_dims=base_dims,
                max_depth=max_depth,
                subdivision_factor=subdivision_factor,
                margin=margin
            )
            
            # Update UI
            self._update_levels_tree()
            self.add_level_button.setEnabled(True)
            self.remove_level_button.setEnabled(True)
            self.visualize_button.setEnabled(True)
            
            # Send signal
            self.hierarchical_ffd_updated.emit(self.hierarchical_ffd)
            
            # Update visualization
            self._update_visualization()
            
        except Exception as e:
            logger.error(f"Error creating hierarchical FFD: {str(e)}")
    
    def _update_levels_tree(self):
        """Update the levels tree with the current hierarchical FFD."""
        self.levels_tree.clear()
        
        if self.hierarchical_ffd is None:
            return
        
        # Get level information
        level_info = self.hierarchical_ffd.get_level_info()
        
        # Add levels to tree
        for level in level_info:
            item = QTreeWidgetItem(self.levels_tree)
            item.setText(0, f"Level {level['level_id']}")
            item.setText(1, f"{level['dims'][0]}×{level['dims'][1]}×{level['dims'][2]}")
            item.setText(2, f"{level['weight_factor']:.2f}")
            
            # Store level_id as data
            item.setData(0, Qt.ItemDataRole.UserRole, level['level_id'])
            
            # Add child information if available
            if level['children']:
                children_str = ", ".join([str(child) for child in level['children']])
                child_item = QTreeWidgetItem(item)
                child_item.setText(0, "Children")
                child_item.setText(1, children_str)
            
            if level['parent_level'] is not None:
                parent_item = QTreeWidgetItem(item)
                parent_item.setText(0, "Parent")
                parent_item.setText(1, str(level['parent_level']))
        
        self.levels_tree.expandAll()
    
    def _add_level(self):
        """Add a new level to the hierarchical FFD."""
        # This would require a more complex dialog to configure the new level
        # For now, we'll show a placeholder message
        logger.info("Adding new levels requires additional configuration dialog")
        
        # Implementation would involve creating a dialog to get:
        # - Parent level ID
        # - New level dimensions
        # - Region bounds
        
    def _remove_level(self):
        """Remove the selected level from the hierarchical FFD."""
        selected_items = self.levels_tree.selectedItems()
        
        if not selected_items:
            return
        
        # Get the level_id from the selected item
        item = selected_items[0]
        level_id = item.data(0, Qt.ItemDataRole.UserRole)
        
        if level_id is None:
            return
        
        try:
            # Prevent removing the root level (level_id 0)
            if level_id == 0:
                logger.warning("Cannot remove the root level")
                return
            
            # Remove the level
            self.hierarchical_ffd.remove_level(level_id)
            
            # Update UI
            self._update_levels_tree()
            
            # Send signal
            self.hierarchical_ffd_updated.emit(self.hierarchical_ffd)
            
            # Update visualization
            self._update_visualization()
            
        except Exception as e:
            logger.error(f"Error removing level: {str(e)}")
    
    def _update_visualization(self):
        """Update the hierarchical FFD visualization."""
        if self.hierarchical_ffd is None:
            return
        
        # Signal to the main application that the visualization should be updated
        self.hierarchical_ffd_updated.emit(self.hierarchical_ffd)
        
        # Note: The actual visualization is handled by the main window
        # which will integrate with the PyVista visualization component
