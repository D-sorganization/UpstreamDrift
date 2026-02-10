#!/usr/bin/env python3
"""
Unified MuJoCo Launcher

Hub for accessing MuJoCo Humanoid Simulation and Analysis Dashboard.
Refactored to use BaseLauncher to eliminate DRY violations.
"""

import subprocess
import sys

from src.launchers.base import REPO_ROOT, BaseLauncher, LaunchItem, run_launcher


class MujocoUnifiedLauncher(BaseLauncher):
    """Launcher for MuJoCo simulation modes."""

    WINDOW_TITLE = "MuJoCo Golf Engine"
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 350
    GRID_COLUMNS = 1  # Vertical layout

    def get_items(self) -> list[LaunchItem]:
        """Return the list of MuJoCo launch modes."""
        return [
            LaunchItem(
                name="🏃 Humanoid Simulation",
                description="Full-body biomechanics simulation with muscle dynamics.",
                path="src/engines/physics_engines/mujoco/python/humanoid_launcher.py",
                item_type="tool",
                action=lambda: self._launch_python_script(
                    "src/engines/physics_engines/mujoco/python/humanoid_launcher.py"
                ),
            ),
            LaunchItem(
                name="📊 Analysis Dashboard",
                description="Real-time plotting, video analysis, and data visualization.",
                item_type="tool",
                action=lambda: self._launch_python_module(
                    "mujoco_humanoid_golf",
                    cwd_suffix="src/engines/physics_engines/mujoco/python",
                ),
            ),
        ]

    def _launch_python_script(self, relative_path: str) -> None:
        """Launch a Python script in a new process.

        Args:
            relative_path: Path relative to REPO_ROOT
        """
        script_path = REPO_ROOT / relative_path
        if not script_path.exists():
            self.show_error("Script Not Found", f"Script not found:\n{script_path}")
            return

        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=REPO_ROOT)
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.show_error("Launch Error", str(e))

    def _launch_python_module(
        self, module_name: str, cwd_suffix: str | None = None
    ) -> None:
        """Launch a Python module in a new process.

        Args:
            module_name: Name of the module to run (e.g., "mujoco_humanoid_golf")
            cwd_suffix: Optional path suffix for working directory
        """
        cwd = REPO_ROOT
        if cwd_suffix:
            cwd = REPO_ROOT / cwd_suffix

        try:
            subprocess.Popen(
                [sys.executable, "-m", module_name],
                cwd=cwd,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            self.show_error("Launch Error", str(e))


def main() -> int:
    """Entry point for the MuJoCo launcher."""
    return run_launcher(MujocoUnifiedLauncher)


if __name__ == "__main__":
    raise SystemExit(main())
