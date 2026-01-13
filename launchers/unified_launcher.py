"""Unified launcher interface wrapping PyQt GolfLauncher.

This module provides a consistent interface for launch_golf_suite.py
that wraps the PyQt-based GolfLauncher implementation.
"""

import logging
import sys
from pathlib import Path

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
    """

    def __init__(self) -> None:
        """Initialize the unified launcher."""
        if not PYQT_AVAILABLE:
            raise ImportError(
                "PyQt6 is required to run the launcher. Install it with: pip install PyQt6"
            )



        # Create QApplication if it doesn't exist
        # Check if QApplication is None (if import failed, but we raised above)
        # or if instance() returns None.
        if QApplication is None:
            raise ImportError("PyQt6 not properly imported.")

        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Create the actual launcher
        self.launcher = None

    def mainloop(self) -> int:
        """Start the launcher main loop.

        Returns:
            Exit code from the application
        """
        if self.launcher is None:
            from .golf_launcher import GolfLauncher

            self.launcher = GolfLauncher()

        self.launcher.show()
        if self.app is None:
            logger.error("QApplication failed to initialize.")
            return 1
        return int(self.app.exec())

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
    """Launch the Golf Modeling Suite GUI.

    Returns:
        Exit code
    """
    launcher = UnifiedLauncher()
    return launcher.mainloop()


def show_status() -> None:
    """Show suite status without launching GUI."""
    launcher = UnifiedLauncher()
    launcher.show_status()


if __name__ == "__main__":
    # Allow running this module directly
    sys.exit(launch())
