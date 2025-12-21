#!/usr/bin/env python3
"""Golf Modeling Suite - Main Launcher Script.

This script provides a command-line interface to launch any component
of the Golf Modeling Suite.
"""

import argparse
import sys
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent / "shared" / "python"))

from shared.python.common_utils import GolfModelingError, setup_logging

logger = setup_logging(__name__)


def launch_gui_launcher():
    """Launch the GUI-based unified launcher."""
    try:
        from launchers.unified_launcher import UnifiedLauncher

        logger.info("Starting GUI launcher...")
        app = UnifiedLauncher()
        return app.mainloop()
    except ImportError as e:
        logger.error(f"Could not import GUI launcher: {e}")
        logger.info("Try: pip install PyQt6")
        return False
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        return False


def launch_local_launcher():
    """Launch the local Python launcher."""
    try:
        from launchers.golf_suite_launcher import main

        logger.info("Starting local launcher...")
        main()
    except ImportError as e:
        logger.error(f"Could not import local launcher: {e}")
        logger.info("Try: pip install PyQt6")
        return False
    except Exception as e:
        logger.error(f"Error launching local GUI: {e}")
        return False
    return True


def _validate_and_get_workdir(script_path: Path) -> Path:
    """Validate script existence and return working directory.

    Args:
        script_path: Path to the script to launch.

    Returns:
        Path to the validated working directory (parent of parent).

    Raises:
        GolfModelingError: If script or workdir does not exist.
    """
    if not script_path.exists():
        raise GolfModelingError(f"Script not found: {script_path}")

    work_dir = script_path.parent.parent
    if not work_dir.exists():
        raise GolfModelingError(f"Working directory not found: {work_dir}")

    return work_dir


def launch_mujoco():
    """Launch MuJoCo engine directly with validation."""
    try:
        import subprocess

        from shared.python.engine_manager import EngineManager, EngineType

        suite_root = Path(__file__).parent
        manager = EngineManager(suite_root)

        # Validate engine is ready
        probe_result = manager.get_probe_result(EngineType.MUJOCO)
        if not probe_result.is_available():
            logger.error(f"MuJoCo not ready:\n{probe_result.diagnostic_message}")
            logger.info(f"Fix: {probe_result.get_fix_instructions()}")
            return False

        mujoco_script = (
            suite_root
            / "engines"
            / "physics_engines"
            / "mujoco"
            / "python"
            / "mujoco_humanoid_golf"
            / "advanced_gui.py"
        )

        work_dir = _validate_and_get_workdir(mujoco_script)

        logger.info("Launching MuJoCo engine...")
        subprocess.run([sys.executable, str(mujoco_script)], cwd=str(work_dir))
    except Exception as e:
        logger.error(f"Error launching MuJoCo: {e}")
        return False
    return True


def launch_drake():
    """Launch Drake engine directly with validation."""
    try:
        import subprocess

        from shared.python.engine_manager import EngineManager, EngineType

        suite_root = Path(__file__).parent
        manager = EngineManager(suite_root)

        # Validate engine is ready
        probe_result = manager.get_probe_result(EngineType.DRAKE)
        if not probe_result.is_available():
            logger.error(f"Drake not ready:\n{probe_result.diagnostic_message}")
            logger.info(f"Fix: {probe_result.get_fix_instructions()}")
            return False

        drake_script = (
            suite_root
            / "engines"
            / "physics_engines"
            / "drake"
            / "python"
            / "src"
            / "golf_gui.py"
        )

        work_dir = _validate_and_get_workdir(drake_script)

        logger.info("Launching Drake engine...")
        subprocess.run([sys.executable, str(drake_script)], cwd=str(work_dir))
    except Exception as e:
        logger.error(f"Error launching Drake: {e}")
        return False
    return True


def launch_pinocchio():
    """Launch Pinocchio engine directly with validation."""
    try:
        import subprocess

        from shared.python.engine_manager import EngineManager, EngineType

        suite_root = Path(__file__).parent
        manager = EngineManager(suite_root)

        # Validate engine is ready
        probe_result = manager.get_probe_result(EngineType.PINOCCHIO)
        if not probe_result.is_available():
            logger.error(f"Pinocchio not ready:\n{probe_result.diagnostic_message}")
            logger.info(f"Fix: {probe_result.get_fix_instructions()}")
            return False

        pinocchio_script = (
            suite_root
            / "engines"
            / "physics_engines"
            / "pinocchio"
            / "python"
            / "pinocchio_golf"
            / "gui.py"
        )

        work_dir = _validate_and_get_workdir(pinocchio_script)

        logger.info("Launching Pinocchio engine...")
        subprocess.run([sys.executable, str(pinocchio_script)], cwd=str(work_dir))
    except Exception as e:
        logger.error(f"Error launching Pinocchio: {e}")
        return False
    return True


def show_status():
    """Show Golf Modeling Suite status."""
    try:
        from launchers.unified_launcher import UnifiedLauncher

        launcher = UnifiedLauncher()
        launcher.show_status()
    except Exception as e:
        # Fallback to basic status if UnifiedLauncher fails
        logger.warning(f"Could not use UnifiedLauncher for status: {e}")
        _show_basic_status()


def _show_basic_status():
    """Show basic status (fallback)."""
    suite_root = Path(__file__).parent

    lines = ["", "=== Golf Modeling Suite Status ===", f"Suite Root: {suite_root}"]

    # Check engines
    engines = {
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
    }

    lines.append("\nAvailable Engines:")
    for name, path in engines.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        lines.append(f"  {status} {name}: {path}")

    # Check launchers
    launchers = {
        "GUI Launcher": suite_root / "launchers" / "golf_launcher.py",
        "Local Launcher": suite_root / "launchers" / "golf_suite_launcher.py",
    }

    lines.append("\nLaunchers:")
    for name, path in launchers.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        lines.append(f"  {status} {name}: {path}")

    lines.append("\nShared Components:")
    shared_components = {
        "Python Utils": suite_root / "shared" / "python",
        "MATLAB Utils": suite_root / "shared" / "matlab",
        "Requirements": suite_root / "shared" / "python" / "requirements.txt",
    }

    for name, path in shared_components.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        lines.append(f"  {status} {name}: {path}")

    # Log the complete report
    for line in lines:
        logger.info(line)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Golf Modeling Suite - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_golf_suite.py                    # Launch GUI launcher
  python launch_golf_suite.py --local           # Launch local launcher
  python launch_golf_suite.py --engine mujoco   # Launch MuJoCo directly
  python launch_golf_suite.py --status          # Show suite status
        """,
    )

    parser.add_argument(
        "--local", action="store_true", help="Launch local Python launcher (no Docker)"
    )

    parser.add_argument(
        "--engine",
        choices=["mujoco", "drake", "pinocchio"],
        help="Launch specific physics engine directly",
    )

    parser.add_argument(
        "--status", action="store_true", help="Show Golf Modeling Suite status"
    )

    args = parser.parse_args()

    try:
        if args.status:
            show_status()
        elif args.engine:
            if args.engine == "mujoco":
                launch_mujoco()
            elif args.engine == "drake":
                launch_drake()
            elif args.engine == "pinocchio":
                launch_pinocchio()
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
