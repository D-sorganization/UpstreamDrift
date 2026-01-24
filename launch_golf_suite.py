#!/usr/bin/env python3
"""Golf Modeling Suite - Main Launcher Script.

This script provides a command-line interface to launch any component
of the Golf Modeling Suite.

Refactored to address DRY and Orthogonality violations (Pragmatic Programmer).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Fix path to include src to find shared
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared.python.common_utils import GolfModelingError
from shared.python.constants import (
    DRAKE_LAUNCHER_SCRIPT,
    GUI_LAUNCHER_SCRIPT,
    LOCAL_LAUNCHER_SCRIPT,
    MUJOCO_LAUNCHER_SCRIPT,
    PINOCCHIO_LAUNCHER_SCRIPT,
    URDF_GENERATOR_SCRIPT,
)
from shared.python.launcher_utils import get_repo_root
from shared.python.logging_config import get_logger, setup_logging
from shared.python.subprocess_utils import run_command

setup_logging()
logger = get_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for launching a physics engine.

    Encapsulates engine-specific parameters to eliminate duplication (DRY).
    """

    name: str
    script_path: Path
    module_name: str | None = None
    needs_validation: bool = True
    work_dir_offset: int = 2


ENGINE_CONFIGS: dict[str, EngineConfig] = {
    "mujoco": EngineConfig(
        name="MuJoCo",
        script_path=MUJOCO_LAUNCHER_SCRIPT,
        needs_validation=True,
    ),
    "drake": EngineConfig(
        name="Drake",
        script_path=DRAKE_LAUNCHER_SCRIPT,
        module_name="src.drake_gui_app",
        needs_validation=True,
    ),
    "pinocchio": EngineConfig(
        name="Pinocchio",
        script_path=PINOCCHIO_LAUNCHER_SCRIPT,
        module_name="pinocchio_golf.gui",
        needs_validation=True,
    ),
}


def _get_workdir(script_path: Path, offset: int = 2) -> Path:
    """Resolve and validate working directory.

    Orthogonality: Decouples path resolution from process execution.
    """
    if not script_path.exists():
        raise GolfModelingError(f"Script not found: {script_path}")

    work_dir = script_path
    for _ in range(offset):
        work_dir = work_dir.parent

    if not work_dir.exists():
        raise GolfModelingError(f"Working directory not found: {work_dir}")

    return work_dir


def _validate_engine_availability(engine_key: str) -> bool:
    """Check if physics engine is correctly installed and ready.

    Args:
        engine_key: Engine identifier.

    Returns:
        True if ready, False otherwise.
    """
    try:
        from shared.python.engine_manager import EngineManager, EngineType

        suite_root = get_repo_root()
        manager = EngineManager(suite_root)

        type_map = {
            "mujoco": EngineType.MUJOCO,
            "drake": EngineType.DRAKE,
            "pinocchio": EngineType.PINOCCHIO,
        }

        engine_type = type_map.get(engine_key)
        if not engine_type:
            return False

        probe = manager.get_probe_result(engine_type)
        if not probe.is_available():
            logger.error(
                f"{ENGINE_CONFIGS[engine_key].name} not ready: {probe.diagnostic_message}"
            )
            logger.info(f"Fix instructions: {probe.get_fix_instructions()}")
            return False

        return True
    except ImportError:
        logger.warning(
            f"Could not perform deep validation for {engine_key}. Proceeding anyway."
        )
        return True


def _launch_engine(engine_key: str) -> bool:
    """Generic engine launcher (Consolidated for DRY)."""
    config = ENGINE_CONFIGS.get(engine_key)
    if not config:
        return False

    try:
        repo_root = get_repo_root()
        if config.needs_validation and not _validate_engine_availability(engine_key):
            return False

        script_path = repo_root / config.script_path
        work_dir = _get_workdir(script_path, config.work_dir_offset)

        logger.info(f"Launching {config.name} engine...")
        cmd = [sys.executable]
        if config.module_name:
            cmd.extend(["-m", config.module_name])
        else:
            cmd.append(str(script_path))

        subprocess_run = run_command(cmd, cwd=work_dir, capture_output=False)
        return subprocess_run.returncode == 0 if subprocess_run else False

    except Exception as e:
        logger.error(f"Error launching {config.name}: {e}")
        return False


def launch_gui() -> int:
    """Launch the main GUI launcher."""
    try:
        from launchers.unified_launcher import UnifiedLauncher

        logger.info("Starting GUI launcher...")
        app = UnifiedLauncher()
        return int(app.mainloop())
    except (ImportError, Exception) as e:
        logger.error(f"GUI launch failed: {e}")
        return 1


def launch_urdf_generator() -> bool:
    """Launch URDF Generator."""
    repo_root = get_repo_root()
    script = repo_root / URDF_GENERATOR_SCRIPT
    logger.info("Launching URDF Generator...")
    res = run_command(
        [sys.executable, str(script)], cwd=repo_root, capture_output=False
    )
    return res.returncode == 0 if res else False


def _print_status_section(title: str, components: dict[str, Path]) -> None:
    """Print a formatted status section.

    Orthogonality: Decouples reporting from status checking.
    """
    logger.info(f"\n{title}:")
    for name, path in components.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        logger.info(f"  {status} {name}: {path}")


def show_basic_status() -> None:
    """Fallback status reporter (Decomposed for Orthogonality)."""
    root = get_repo_root()
    logger.info(f"=== Golf Modeling Suite Status ===\nRoot: {root}")

    engines = {
        "MuJoCo": root / "engines/physics_engines/mujoco",
        "Drake": root / "engines/physics_engines/drake",
        "Pinocchio": root / "engines/physics_engines/pinocchio",
        "MATLAB 2D": root / "engines/Simscape_Multibody_Models/2D_Golf_Model",
        "MATLAB 3D": root / "engines/Simscape_Multibody_Models/3D_Golf_Model",
    }
    _print_status_section("Engines", engines)

    launchers = {
        "GUI": root / GUI_LAUNCHER_SCRIPT,
        "Local": root / LOCAL_LAUNCHER_SCRIPT,
    }
    _print_status_section("Launchers", launchers)


def main() -> int:
    """CLI Entry point (Decomposed for Orthogonality)."""
    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite - Unified Launcher"
    )
    parser.add_argument("--local", action="store_true", help="Launch local launcher")
    parser.add_argument(
        "--engine", choices=list(ENGINE_CONFIGS.keys()), help="Launch specific engine"
    )
    parser.add_argument(
        "--urdf-generator", action="store_true", help="Launch URDF Generator"
    )
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    try:
        if args.status:
            show_basic_status()
        elif args.urdf_generator:
            launch_urdf_generator()
        elif args.engine:
            _launch_engine(args.engine)
        elif args.local:
            # Simple wrapper for local launcher
            from launchers.golf_suite_launcher import main as local_main

            local_main()
        else:
            return launch_gui()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Launcher error: {e}")
        return 1

    return 0


# Backward compatibility wrappers for testing
def launch_c3d_viewer():
    logger.warning("launch_c3d_viewer is deprecated and removed.")
    return False

def launch_gui_launcher():
    return launch_gui()

def launch_local_launcher():
    from launchers.golf_suite_launcher import main as local_main
    try:
        local_main()
        return True
    except Exception:
        return False

def launch_mujoco():
    return _launch_engine("mujoco")

def launch_drake():
    return _launch_engine("drake")

def launch_pinocchio():
    return _launch_engine("pinocchio")


if __name__ == "__main__":
    sys.exit(main())
