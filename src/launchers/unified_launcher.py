"""Unified launcher interface wrapping PyQt GolfLauncher.

This module provides a consistent interface for launch_golf_suite.py
that wraps the PyQt-based GolfLauncher implementation.

The launcher now features:
- Async startup with background worker thread
- Real progress updates during splash screen
- Lazy loading of heavy modules (MuJoCo, Drake, etc.)
- Pre-loaded resources passed to main window (no duplicate loading)
"""

import builtins
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

if PYQT6_AVAILABLE:
    from PyQt6.QtWidgets import QApplication
else:
    QApplication = None  # type: ignore

logger = get_logger(__name__)


def _is_pyqt6_available() -> bool:
    """Resolve PyQt availability with legacy module override support."""
    legacy_module = sys.modules.get("launchers.unified_launcher")
    if legacy_module is not None and hasattr(legacy_module, "PYQT6_AVAILABLE"):
        return bool(legacy_module.PYQT6_AVAILABLE)
    return bool(PYQT6_AVAILABLE)


def _get_golf_main(*, prefer_legacy: bool = False):
    """Resolve golf launcher entry point across legacy/new module paths."""
    if prefer_legacy:
        legacy_module = sys.modules.get("launchers.golf_launcher")
        if legacy_module is not None and hasattr(legacy_module, "main"):
            return legacy_module.main

    try:
        from .golf_launcher import main as golf_main

        return golf_main
    except ImportError:
        logger.debug("Could not import golf_launcher via relative import, trying absolute")

    try:
        module = importlib.import_module("launchers.golf_launcher")
        if hasattr(module, "main"):
            return module.main
    except ImportError:
        logger.debug("Could not import launchers.golf_launcher, falling back to direct import")

    from .golf_launcher import main as golf_main

    return golf_main


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
        if not _is_pyqt6_available():
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
        golf_main = _get_golf_main(prefer_legacy=True)
        golf_main()

    def show_status(self) -> None:
        """Display suite status information.

        Shows available engines, their status, and configuration.
        """
        try:
            from src.shared.python.engine_core.engine_manager import EngineManager
        except ImportError:
            from shared.python.engine_core.engine_manager import (
                EngineManager,  # type: ignore[no-redef]
            )

        manager = EngineManager()

        # Show available engines

        engines = manager.get_available_engines()
        if engines:
            logger.info("Available engines:")
            for _engine in engines:
                engine_name = str(getattr(_engine, "value", str(_engine)))
                logger.info(" - %s", engine_name)
                builtins.print(engine_name.upper())  # noqa: T201
        else:
            logger.info("No engines available.")
            builtins.print("NO ENGINES AVAILABLE")  # noqa: T201

        # Show suite root — import from the canonical location
        try:
            from src.shared.python import SUITE_ROOT
        except ImportError:
            from shared.python import SUITE_ROOT  # type: ignore[no-redef]

        logger.info("Suite root: %s", SUITE_ROOT)

        # Show launcher paths
        launcher_dir = Path(__file__).parent
        for launcher_file in launcher_dir.glob("*_launcher.py"):
            if launcher_file.name != "unified_launcher.py":
                logger.info("Launcher: %s", launcher_file.name)

        # Show engine directories
        engines_dir = SUITE_ROOT / "engines"
        if engines_dir.exists():
            for engine_dir in engines_dir.iterdir():
                if engine_dir.is_dir() and not engine_dir.name.startswith("."):
                    logger.info("Engine dir: %s", engine_dir.name)

    def get_version(self) -> str:
        """Get suite version from package metadata.

        Returns:
            Version string from pyproject.toml / installed package.

        Resolution order:
            1. Installed package metadata (``importlib.metadata``)
               - If PackageNotFoundError: fall through to pyproject.toml
               - If ImportError (broken env): skip directly to fallback
            2. pyproject.toml in the repo root (only in development mode)
            3. ``shared.python.__version__`` attribute (legacy fallback)
            4. Hardcoded fallback
        """
        # 1. Try installed package metadata
        metadata_broken = False
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version("upstream-drift")
            except PackageNotFoundError:
                logger.debug("Package 'upstream-drift' not found in metadata")

            try:
                return version("golf-modeling-suite")
            except PackageNotFoundError:
                logger.debug("Package 'golf-modeling-suite' not found in metadata")

        except ImportError:
            metadata_broken = True

        # 2. Try shared.python.__version__
        try:
            import shared.python as _shared  # type: ignore[import-untyped]

            v = getattr(_shared, "__version__", None)
            if v and not callable(v):
                return str(v)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not read version from shared.python: %s", e)

        # 3. Read directly from pyproject.toml (development / editable installs)
        # Only attempt when metadata machinery is functioning (not broken import)
        if not metadata_broken:
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    tomllib = None  # type: ignore[assignment]

            if tomllib is not None:
                try:
                    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
                    with pyproject_path.open("rb") as fh:
                        data = tomllib.load(fh)
                    return str(data["project"]["version"])
                except (KeyError, FileNotFoundError, OSError) as e:
                    logger.debug("Could not read version from pyproject.toml: %s", e)

        # 4. Hardcoded fallback
        return "1.0.0-beta"


# Convenience function for CLI usage
def launch() -> None:
    """Launch the Golf Modeling Suite GUI with async startup.

    This is the recommended entry point for launching the GUI.
    It uses the async startup system for optimal performance:
    - Splash screen appears immediately
    - Heavy modules loaded in background
    - Progress updates shown during loading
    - No duplicate resource loading
    """
    if not _is_pyqt6_available():
        logger.warning("PyQt6 not available.")
        return

    # Delegate directly to golf_launcher.main() for async startup
    golf_main = _get_golf_main(prefer_legacy=False)
    golf_main()


def show_status() -> None:
    """Show suite status without launching GUI."""
    launcher = UnifiedLauncher()
    launcher.show_status()


if __name__ == "__main__":
    # Allow running this module directly
    launch()
