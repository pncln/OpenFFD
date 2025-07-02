#!/usr/bin/env python3
"""Quick launcher for the OpenFFD GUI.

This script makes it easy to launch the OpenFFD GUI during development.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from openffd.gui.main import launch_gui
    sys.exit(launch_gui())
except ImportError as e:
    print(f"Error importing OpenFFD GUI: {e}")
    print("\nMake sure you have the required dependencies installed:")
    print("  pip install PyQt6 pyvista numpy matplotlib scipy")
    sys.exit(1)
