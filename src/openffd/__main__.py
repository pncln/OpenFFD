"""Main entry point for running OpenFFD as a module."""

import sys

if __name__ == "__main__":
    # Check if GUI mode is requested
    if "--gui" in sys.argv:
        from openffd.gui.main import launch_gui
        sys.exit(launch_gui())
    else:
        # Default to CLI mode
        from openffd.cli.app import main
        sys.exit(main())
