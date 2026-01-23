#!/usr/bin/env python3
"""Golf Modeling Suite - Main Launcher Script.

This script provides a command-line interface to launch any component
of the Golf Modeling Suite.

Refactored to address DRY violations (Pragmatic Programmer principle).
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent / "shared" / "python"))

from shared.python.common_utils import GolfModelingError, setup_logging
from shared.python.constants import (
    C3D_VIEWER_SCRIPT,
    DRAKE_LAUNCHER_SCRIPT,
    GUI_LAUNCHER_SCRIPT,
    LOCAL_LAUNCHER_SCRIPT,
    MUJOCO_LAUNCHER_SCRIPT,
    PINOCCHIO_LAUNCHER_SCRIPT,
    URDF_GENERATOR_SCRIPT,
)

logger = setup_logging(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for launching a physics engine.

    This dataclass encapsulates all engine-specific launch parameters,
    eliminating duplicate code across engine launchers (DRY principle).
    """

    name: str
    script_path: Path
    module_name: str | None = None  # If set, run as module instead of script
    needs_validation: bool = True
    work_dir_offset: int = 2  # How many parents up from script to get work_dir


# Engine configuration registry - centralizes all engine launch parameters
ENGINE_CONFIGS: dict[str, EngineConfig] = {
    "mujoco": EngineConfig(
        name="MuJoCo",
        script_path=MUJOCO_LAUNCHER_SCRIPT,
        module_name=None,
        needs_validation=True,
        work_dir_offset=2,
    ),
    "drake": EngineConfig(
        name="Drake",
        script_path=DRAKE_LAUNCHER_SCRIPT,
        module_name="src.drake_gui_app",
        needs_validation=True,
        work_dir_offset=2,
    ),
    "pinocchio": EngineConfig(
        name="Pinocchio",
        script_path=PINOCCHIO_LAUNCHER_SCRIPT,
        module_name="pinocchio_golf.gui",
        needs_validation=True,
        work_dir_offset=2,
    ),
}


def _validate_and_get_workdir(script_path: Path, offset: int = 2) -> Path:
    """Validate script existence and return working directory.

    Args:
        script_path: Path to the script to launch.
        offset: Number of parent directories to traverse for work_dir.

    Returns:
        Path to the validated working directory.

    Raises:
        GolfModelingError: If script or workdir does not exist.
    """
    if not script_path.exists():
        raise GolfModelingError(f"Script not found: {script_path}")

    work_dir = script_path
    for _ in range(offset):
        work_dir = work_dir.parent

    if not work_dir.exists():
        raise GolfModelingError(f"Working directory not found: {work_dir}")

    return work_dir


def _run_subprocess(
    script_path: Path,
    work_dir: Path,
    module_name: str | None = None,
    env: dict | None = None,
) -> None:
    """Run a subprocess with consistent error handling.

    Args:
        script_path: Path to the script to run.
        work_dir: Working directory for the subprocess.
        module_name: If provided, run as Python module instead of script.
        env: Optional environment variables.
    """
    if module_name:
        cmd = [sys.executable, "-m", module_name]
    else:
        cmd = [sys.executable, str(script_path)]

    subprocess.run(cmd, cwd=str(work_dir), env=env)


def _validate_engine(engine_key: str) -> bool:
    """Validate that a physics engine is ready to launch.

    Args:
        engine_key: Engine identifier (mujoco, drake, pinocchio).

    Returns:
        True if engine is ready, False otherwise.
    """
    from shared.python.engine_manager import EngineManager, EngineType

    suite_root = Path(__file__).parent
    manager = EngineManager(suite_root)

    engine_type_map = {
        "mujoco": EngineType.MUJOCO,
        "drake": EngineType.DRAKE,
        "pinocchio": EngineType.PINOCCHIO,
    }

    engine_type = engine_type_map.get(engine_key)
    if not engine_type:
        logger.error(f"Unknown engine type: {engine_key}")
        return False

    probe_result = manager.get_probe_result(engine_type)
    if not probe_result.is_available():
        config = ENGINE_CONFIGS[engine_key]
        logger.error(f"{config.name} not ready:\n{probe_result.diagnostic_message}")
        logger.info(f"Fix: {probe_result.get_fix_instructions()}")
        return False

    return True


def _launch_physics_engine(engine_key: str) -> bool:
    """Generic physics engine launcher.

    This function eliminates duplicate code across launch_mujoco, launch_drake,
    and launch_pinocchio by using configuration-driven launching.

    Args:
        engine_key: Engine identifier (mujoco, drake, pinocchio).

    Returns:
        True if launch succeeded, False otherwise.
    """
    config = ENGINE_CONFIGS.get(engine_key)
    if not config:
        logger.error(f"Unknown engine: {engine_key}")
        return False

    try:
        suite_root = Path(__file__).parent

        # Validate engine if required
        if config.needs_validation and not _validate_engine(engine_key):
            return False

        script_path = suite_root / config.script_path
        work_dir = _validate_and_get_workdir(script_path, config.work_dir_offset)

        logger.info(f"Launching {config.name} engine...")
        _run_subprocess(script_path, work_dir, config.module_name)
        return True

    except Exception as e:
        logger.error(f"Error launching {config.name}: {e}")
        return False


def launch_gui_launcher() -> int | None:
    """Launch the GUI-based unified launcher."""
    try:
        from launchers.unified_launcher import UnifiedLauncher

        logger.info("Starting GUI launcher...")
        app = UnifiedLauncher()
        return app.mainloop()
    except ImportError as e:
        logger.error(f"Could not import GUI launcher: {e}")
        logger.info("Try: pip install PyQt6")
        return None
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        return None


def launch_local_launcher() -> bool:
    """Launch the local Python launcher."""
    try:
        from launchers.golf_suite_launcher import main

        logger.info("Starting local launcher...")
        main()
        return True
    except ImportError as e:
        logger.error(f"Could not import local launcher: {e}")
        logger.info("Try: pip install PyQt6")
        return False
    except Exception as e:
        logger.error(f"Error launching local GUI: {e}")
        return False


# Thin wrappers for backwards compatibility
def launch_mujoco() -> bool:
    """Launch MuJoCo engine directly with validation."""
    return _launch_physics_engine("mujoco")


def launch_drake() -> bool:
    """Launch Drake engine directly with validation."""
    return _launch_physics_engine("drake")


def launch_pinocchio() -> bool:
    """Launch Pinocchio engine directly with validation."""
    return _launch_physics_engine("pinocchio")


def launch_urdf_generator() -> bool:
    """Launch the Graphic URDF Generator."""
    try:
        suite_root = Path(__file__).parent
        generator_script = suite_root / URDF_GENERATOR_SCRIPT

        logger.info("Launching URDF Generator...")
        _run_subprocess(generator_script, suite_root)
        return True
    except Exception as e:
        logger.error(f"Error launching URDF Generator: {e}")
        return False


def launch_c3d_viewer() -> bool:
    """Launch the C3D Motion Viewer."""
    try:
        suite_root = Path(__file__).parent
        c3d_script = suite_root / C3D_VIEWER_SCRIPT

        # For relative imports to work (from .core.models), we must run as module
        src_root = c3d_script.parent.parent
        module_name = "apps.c3d_viewer"

        # Ensure repo root is in PYTHONPATH for 'shared' imports
        env = os.environ.copy()
        env["PYTHONPATH"] = str(suite_root) + os.pathsep + env.get("PYTHONPATH", "")

        logger.info("Launching C3D Motion Viewer...")
        _run_subprocess(c3d_script, src_root, module_name, env)
        return True
    except Exception as e:
        logger.error(f"Error launching C3D Viewer: {e}")
        return False


def _show_basic_status() -> None:
    """Show basic status (fallback)."""
    suite_root = Path(__file__).parent

    # Define component paths for status checks - consolidates duplicate dict patterns
    component_checks = {
        "Available Engines": {
            "MuJoCo": suite_root / "engines" / "physics_engines" / "mujoco",
            "Drake": suite_root / "engines" / "physics_engines" / "drake",
            "Pinocchio": suite_root / "engines" / "physics_engines" / "pinocchio",
            "2D MATLAB": suite_root
            / "engines"
            / "Simscape_Multibody_Models"
            / "2D_Golf_Model",
            "3D MATLAB": suite_root
            / "engines"
            / "Simscape_Multibody_Models"
            / "3D_Golf_Model",
            "Pendulum": suite_root / "engines" / "pendulum_models",
        },
        "Launchers": {
            "GUI Launcher": suite_root / GUI_LAUNCHER_SCRIPT,
            "Local Launcher": suite_root / LOCAL_LAUNCHER_SCRIPT,
        },
        "Shared Components": {
            "Python Utils": suite_root / "shared" / "python",
            "MATLAB Utils": suite_root / "shared" / "matlab",
            "Requirements": suite_root / "shared" / "python" / "requirements.txt",
        },
    }

    logger.info("")
    logger.info("=== Golf Modeling Suite Status ===")
    logger.info(f"Suite Root: {suite_root}")

    for section_name, components in component_checks.items():
        logger.info(f"\n{section_name}:")
        for name, path in components.items():
            status = "[OK]" if path.exists() else "[MISSING]"
            logger.info(f"  {status} {name}: {path}")


def show_status() -> None:
    """Show Golf Modeling Suite status."""
    try:
        from launchers.unified_launcher import UnifiedLauncher

        launcher = UnifiedLauncher()
        launcher.show_status()
    except Exception as e:
        logger.warning(f"Could not use UnifiedLauncher for status: {e}")
        _show_basic_status()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_golf_suite.py                    # Launch GUI launcher
  python launch_golf_suite.py --local           # Launch local launcher
  python launch_golf_suite.py --engine mujoco   # Launch MuJoCo directly
  python launch_golf_suite.py --urdf-generator  # Launch URDF Generator
  python launch_golf_suite.py --c3d-viewer      # Launch C3D Motion Viewer
  python launch_golf_suite.py --status          # Show suite status
        """,
    )

    parser.add_argument(
        "--local", action="store_true", help="Launch local Python launcher (no Docker)"
    )

    parser.add_argument(
        "--engine",
        choices=list(ENGINE_CONFIGS.keys()),
        help="Launch specific physics engine directly",
    )

    parser.add_argument(
        "--urdf-generator", action="store_true", help="Launch Graphic URDF Generator"
    )

    parser.add_argument(
        "--c3d-viewer", action="store_true", help="Launch C3D Motion Viewer"
    )

    parser.add_argument(
        "--status", action="store_true", help="Show Golf Modeling Suite status"
    )

    args = parser.parse_args()

    try:
        if args.status:
            show_status()
        elif args.urdf_generator:
            launch_urdf_generator()
        elif args.c3d_viewer:
            launch_c3d_viewer()
        elif args.engine:
            _launch_physics_engine(args.engine)
        elif args.local:
            launch_local_launcher()
        else:
            launch_gui_launcher()

    except KeyboardInterrupt:
        logger.info("Launcher interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
