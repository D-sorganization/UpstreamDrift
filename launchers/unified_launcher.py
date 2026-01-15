"""Unified launcher interface wrapping PyQt GolfLauncher.

This module provides a consistent interface for launch_golf_suite.py
that wraps the PyQt-based GolfLauncher implementation.

The launcher now features:
- Async startup with background worker thread
- Real progress updates during splash screen
- Lazy loading of heavy modules (MuJoCo, Drake, etc.)
- Pre-loaded resources passed to main window (no duplicate loading)
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .golf_launcher import GolfLauncher

try:
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QApplication = None  # type: ignore

logger = logging.getLogger(__name__)


class UnifiedLauncher:
    """Unified launcher interface compatible with launch_golf_suite.py.

    This class wraps the PyQt GolfLauncher to provide a consistent
    interface with a mainloop() method as expected by the CLI launcher.

    The mainloop() method now delegates to the golf_launcher.main() function
    which implements async startup with splash screen for optimal UX.
    """

    def __init__(self) -> None:
        """Initialize the unified launcher.

        Note: QApplication is created lazily in mainloop() to allow
        the async startup system to manage the application lifecycle.
        """
        if not PYQT_AVAILABLE:
            raise ImportError(
                "PyQt6 is required to run the launcher. Install it with: pip install PyQt6"
            )

    def mainloop(self) -> int:
        """Start the launcher main loop with async startup.

        This method delegates to golf_launcher.main() which implements:
        - Immediate splash screen display
        - Background worker for heavy initialization
        - Real progress updates during startup
        - Pre-loaded resources passed to main window

        Returns:
            Exit code from the application
        """
        from .golf_launcher import main as golf_main

        return golf_main()

    def show_status(self) -> None:
        """Display suite status information.

        Shows available engines, their status, and configuration.
        """
        from shared.python.engine_manager import EngineManager

        manager = EngineManager()

        print("\n" + "=" * 60)
        print("Golf Modeling Suite - Status Report")
        print("=" * 60 + "\n")

        # Show available engines
        print("Available Engines:")
        print("-" * 60)

        engines = manager.get_available_engines()
        if engines:
            for engine in engines:
                print(f"  [OK] {engine.value.upper()}")
        else:
            print("  [MISSING] No engines available")

        print()

        # Show suite root
        from shared.python import SUITE_ROOT

        print(f"Suite Root: {SUITE_ROOT}")
        print()

        # Show launcher paths
        print("Launcher Paths:")
        print("-" * 60)

        launcher_dir = Path(__file__).parent
        for launcher_file in launcher_dir.glob("*_launcher.py"):
            if launcher_file.name != "unified_launcher.py":
                print(f"  • {launcher_file.name}")

        print()

        # Show engine directories
        print("Engine Directories:")
        print("-" * 60)

        engines_dir = SUITE_ROOT / "engines"
        if engines_dir.exists():
            for engine_dir in engines_dir.iterdir():
                if engine_dir.is_dir() and not engine_dir.name.startswith("."):
                    print(f"  • {engine_dir.name}")

        print("\n" + "=" * 60 + "\n")

    def get_version(self) -> str:
        """Get suite version from package metadata.

        Returns:
            Version string (e.g., "1.0.0-beta")

        Note:
            Primary source: Package metadata (installed package)
            Fallback: shared.__version__ (development mode)
            Last resort: Hardcoded default
        """
        # Try package metadata first (installed package)
        try:
            from importlib.metadata import PackageNotFoundError, version

            return version("golf-modeling-suite")
        except (PackageNotFoundError, ImportError):
            pass

        # Try shared package (development mode)
        try:
            from shared.python import __version__

            return __version__
        except (ImportError, AttributeError):
            pass

        # Last resort fallback
        return "1.0.0-beta"


# Convenience function for CLI usage
def launch() -> int:
    """Launch the Golf Modeling Suite GUI with async startup.

    This is the recommended entry point for launching the GUI.
    It uses the async startup system for optimal performance:
    - Splash screen appears immediately
    - Heavy modules loaded in background
    - Progress updates shown during loading
    - No duplicate resource loading

    Returns:
        Exit code
    """
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required. Install with: pip install PyQt6")
        return 1

    # Delegate directly to golf_launcher.main() for async startup
    from .golf_launcher import main as golf_main

    return golf_main()


def show_status() -> None:
    """Show suite status without launching GUI."""
    launcher = UnifiedLauncher()
    launcher.show_status()


if __name__ == "__main__":
    # Allow running this module directly
    sys.exit(launch())
