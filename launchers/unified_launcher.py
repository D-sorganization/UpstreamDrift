"""Unified launcher interface wrapping PyQt GolfLauncher.

This module provides a consistent interface for launch_golf_suite.py
that wraps the PyQt-based GolfLauncher implementation.
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication


class UnifiedLauncher:
    """Unified launcher interface compatible with launch_golf_suite.py.

    This class wraps the PyQt GolfLauncher to provide a consistent
    interface with a mainloop() method as expected by the CLI launcher.
    """

    def __init__(self) -> None:
        """Initialize the unified launcher."""
        # Import here to avoid circular dependencies
        from .golf_launcher import GolfLauncher

        # Create QApplication if it doesn't exist
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Create the actual launcher
        self.launcher = GolfLauncher()

    def mainloop(self) -> int:
        """Start the launcher main loop.

        Returns:
            Exit code from the application
        """
        self.launcher.show()
        return self.app.exec()

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
                print(f"  ✅ {engine.value.upper()}")
        else:
            print("  ❌ No engines available")

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
        """Get suite version.

        Returns:
            Version string
        """
        # TODO: Read from version file or package metadata
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
