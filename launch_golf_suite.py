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

from shared.python.common_utils import setup_logging, GolfModelingError

logger = setup_logging(__name__)


def launch_gui_launcher():
    """Launch the GUI-based unified launcher."""
    try:
        from launchers.golf_launcher import UnifiedLauncher
        logger.info("Starting GUI launcher...")
        app = UnifiedLauncher()
        app.mainloop()
    except ImportError as e:
        logger.error(f"Could not import GUI launcher: {e}")
        logger.info("Try: pip install tkinter")
        return False
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        return False
    return True


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


def launch_mujoco():
    """Launch MuJoCo engine directly."""
    try:
        import subprocess
        mujoco_script = Path(__file__).parent / "engines" / "physics_engines" / "mujoco" / "python" / "mujoco_golf_pendulum" / "advanced_gui.py"
        if not mujoco_script.exists():
            raise GolfModelingError(f"MuJoCo script not found: {mujoco_script}")
        
        logger.info("Launching MuJoCo engine...")
        subprocess.run([sys.executable, str(mujoco_script)], 
                      cwd=mujoco_script.parent.parent)
    except Exception as e:
        logger.error(f"Error launching MuJoCo: {e}")
        return False
    return True


def launch_drake():
    """Launch Drake engine directly."""
    try:
        import subprocess
        drake_script = Path(__file__).parent / "engines" / "physics_engines" / "drake" / "python" / "src" / "golf_gui.py"
        if not drake_script.exists():
            raise GolfModelingError(f"Drake script not found: {drake_script}")
        
        logger.info("Launching Drake engine...")
        subprocess.run([sys.executable, str(drake_script)], 
                      cwd=drake_script.parent.parent)
    except Exception as e:
        logger.error(f"Error launching Drake: {e}")
        return False
    return True


def launch_pinocchio():
    """Launch Pinocchio engine directly."""
    try:
        import subprocess
        pinocchio_script = Path(__file__).parent / "engines" / "physics_engines" / "pinocchio" / "python" / "pinocchio_golf" / "gui.py"
        if not pinocchio_script.exists():
            raise GolfModelingError(f"Pinocchio script not found: {pinocchio_script}")
        
        logger.info("Launching Pinocchio engine...")
        subprocess.run([sys.executable, str(pinocchio_script)], 
                      cwd=pinocchio_script.parent.parent)
    except Exception as e:
        logger.error(f"Error launching Pinocchio: {e}")
        return False
    return True


def show_status():
    """Show Golf Modeling Suite status."""
    suite_root = Path(__file__).parent
    
    print("\n=== Golf Modeling Suite Status ===")
    print(f"Suite Root: {suite_root}")
    
    # Check engines
    engines = {
        "MuJoCo": suite_root / "engines" / "physics_engines" / "mujoco",
        "Drake": suite_root / "engines" / "physics_engines" / "drake", 
        "Pinocchio": suite_root / "engines" / "physics_engines" / "pinocchio",
        "2D MATLAB": suite_root / "engines" / "Simscape_Multibody_Models" / "2D_Golf_Model",
        "3D MATLAB": suite_root / "engines" / "Simscape_Multibody_Models" / "3D_Golf_Model",
        "Pendulum": suite_root / "engines" / "pendulum_models"
    }
    
    print("\nAvailable Engines:")
    for name, path in engines.items():
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}: {path}")
    
    # Check launchers
    launchers = {
        "GUI Launcher": suite_root / "launchers" / "golf_launcher.py",
        "Local Launcher": suite_root / "launchers" / "golf_suite_launcher.py"
    }
    
    print("\nLaunchers:")
    for name, path in launchers.items():
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}: {path}")
    
    print("\nShared Components:")
    shared_components = {
        "Python Utils": suite_root / "shared" / "python",
        "MATLAB Utils": suite_root / "shared" / "matlab",
        "Requirements": suite_root / "shared" / "python" / "requirements.txt"
    }
    
    for name, path in shared_components.items():
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}: {path}")
    
    print()


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
        """
    )
    
    parser.add_argument(
        "--local", 
        action="store_true",
        help="Launch local Python launcher (no Docker)"
    )
    
    parser.add_argument(
        "--engine",
        choices=["mujoco", "drake", "pinocchio"],
        help="Launch specific physics engine directly"
    )
    
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Show Golf Modeling Suite status"
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