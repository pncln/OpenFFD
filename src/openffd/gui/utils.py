"""Utility functions for the OpenFFD GUI.

This module provides helper functions for the GUI components.
"""

import os
import logging
from typing import Optional

from PyQt6.QtWidgets import QMessageBox


def setup_logger() -> None:
    """Configure the logger for the GUI application."""
    logger = logging.getLogger("openffd")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)


def get_icon_path(icon_name: str) -> str:
    """Get the path to an icon file.
    
    Args:
        icon_name: Name of the icon file
        
    Returns:
        Full path to the icon file
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Icons directory is in the same directory as this file
    icons_dir = os.path.join(current_dir, "icons")
    
    # Create icons directory if it doesn't exist
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
    
    # Return full path to icon
    return os.path.join(icons_dir, icon_name)


def show_error_dialog(title: str, message: str) -> None:
    """Show an error dialog with the specified title and message.
    
    Args:
        title: Dialog title
        message: Error message
    """
    QMessageBox.critical(
        None,
        title,
        message
    )


def show_warning_dialog(title: str, message: str) -> None:
    """Show a warning dialog with the specified title and message.
    
    Args:
        title: Dialog title
        message: Warning message
    """
    QMessageBox.warning(
        None,
        title,
        message
    )


def show_info_dialog(title: str, message: str) -> None:
    """Show an information dialog with the specified title and message.
    
    Args:
        title: Dialog title
        message: Information message
    """
    QMessageBox.information(
        None,
        title,
        message
    )
