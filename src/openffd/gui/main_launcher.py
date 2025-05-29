#!/usr/bin/env python3
"""Launch script for the OpenFFD GUI application.

This script provides a simple way to launch the OpenFFD GUI
and can be used as a standalone entry point.
"""

import sys
import os
import logging
from pathlib import Path


def setup_environment():
    """Set up the environment for the GUI application."""
    # Add the parent directory to the path if running as a script
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent.parent
    
    if parent_dir not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logging.info(f"OpenFFD GUI launcher started")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Running from: {os.path.abspath(__file__)}")


def main():
    """Main entry point for the launcher."""
    setup_environment()
    
    try:
        # Import and launch the GUI
        from openffd.gui.main import launch_gui
        return launch_gui()
    except ImportError as e:
        logging.error(f"Failed to import the GUI module: {e}")
        logging.error("Make sure PyQt6 and other dependencies are installed.")
        print("\nError: Failed to start the OpenFFD GUI.")
        print("Make sure you have installed the required dependencies:")
        print("  pip install PyQt6 pyvista numpy matplotlib scipy")
        return 1
    except Exception as e:
        logging.exception(f"Error starting the GUI: {e}")
        print(f"\nError starting the OpenFFD GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
