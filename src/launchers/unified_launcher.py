"""Unified launcher interface wrapping PyQt GolfLauncher.

This module provides a consistent interface for launch_golf_suite.py
that wraps the PyQt-based GolfLauncher implementation.

The launcher now features:
- Async startup with background worker thread
- Real progress updates during splash screen
- Lazy loading of heavy modules (MuJoCo, Drake, etc.)
- Pre-loaded resources passed to main window (no duplicate loading)
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    pass

if PYQT6_AVAILABLE:
    from PyQt6.QtWidgets import QApplication
else:
    QApplication = None  # type: ignore

logger = get_logger(__name__)


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
        if not PYQT6_AVAILABLE:
            raise ImportError(
                "PyQt6 is required to run the launcher. Install it with: pip install PyQt6"
            )

    def mainloop(self) -> None:
        """Start the launcher main loop with async startup.

        This method delegates to golf_launcher.main() which implements:
        - Immediate splash screen display
        - Background worker for heavy initialization
        - Real progress updates during startup
        - Pre-loaded resources passed to main window

        Does not return, calls sys.exit().
        """
        from .golf_launcher import main as golf_main

        golf_main()

    def show_status(self) -> None:
        """Display suite status information.

        Shows available engines, their status, and configuration.
        """
        from src.shared.python.engine_manager import EngineManager

        manager = EngineManager()

        # Show available engines

        engines = manager.get_available_engines()
        if engines:
            for _engine in engines:
                pass
        else:
            pass

        # Show suite root
        from src.shared.python import SUITE_ROOT

        # Show launcher paths

        launcher_dir = Path(__file__).parent
        for launcher_file in launcher_dir.glob("*_launcher.py"):
            if launcher_file.name != "unified_launcher.py":
                pass

        # Show engine directories

        engines_dir = SUITE_ROOT / "engines"
        if engines_dir.exists():
            for engine_dir in engines_dir.iterdir():
                if engine_dir.is_dir() and not engine_dir.name.startswith("."):
                    pass

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
            from src.shared.python import __version__

            version_str: str = __version__  # type: ignore[assignment]
            return version_str
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
    if not PYQT6_AVAILABLE:
        return 1

    # Delegate directly to golf_launcher.main() for async startup
    from .golf_launcher import main as golf_main

    golf_main()
    return 0


def show_status() -> None:
    """Show suite status without launching GUI."""
    launcher = UnifiedLauncher()
    launcher.show_status()


if __name__ == "__main__":
    # Allow running this module directly
    sys.exit(launch())
