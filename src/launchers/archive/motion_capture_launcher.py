#!/usr/bin/env python3
"""
Motion Capture & Analysis Launcher

Central hub for C3D visualization and Markerless Pose Estimation.
Refactored to use BaseLauncher to eliminate DRY violations.
"""

import os
import sys
from pathlib import Path

from src.launchers.base import REPO_ROOT, BaseLauncher, LaunchItem, run_launcher
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
)


def _make_subprocess_env(suite_root: Path) -> dict[str, str]:
    """Create environment dict with suite_root in PYTHONPATH."""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    root_str = str(suite_root)
    if root_str not in pythonpath:
        env["PYTHONPATH"] = (
            f"{root_str}{os.pathsep}{pythonpath}" if pythonpath else root_str
        )
    return env


def _spawn_process(command: list[str], cwd: object) -> None:
    """Spawn a launcher subprocess through the secure wrapper."""
    env = _make_subprocess_env(REPO_ROOT)
    secure_popen(command, cwd=cwd, suite_root=REPO_ROOT, env=env)


class MoCapLauncher(BaseLauncher):
    """Launcher for motion capture and pose estimation tools."""

    WINDOW_TITLE = "Motion Capture"
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 450
    GRID_COLUMNS = 1  # Vertical layout for this launcher

    def get_items(self) -> list[LaunchItem]:
        """Return the list of motion capture tools."""
        return [
            LaunchItem(
                name="📈 C3D Motion Viewer",
                description="Visualize and analyze optical motion capture data (.c3d).",
                path="src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py",
                item_type="tool",
                action=lambda: self._launch_python_script(
                    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
                ),
            ),
            LaunchItem(
                name="🎥 OpenPose Analysis",
                description="High-accuracy body pose estimation (Academic License).",
                path="src/shared/python/pose_estimation/openpose_gui.py",
                item_type="tool",
                action=lambda: self._launch_python_script(
                    "src/shared/python/pose_estimation/openpose_gui.py"
                ),
            ),
            LaunchItem(
                name="⚡ MediaPipe Analysis",
                description="Fast, permissive license pose estimation (Apache 2.0).",
                path="src/shared/python/pose_estimation/mediapipe_gui.py",
                item_type="tool",
                action=lambda: self._launch_python_script(
                    "src/shared/python/pose_estimation/mediapipe_gui.py"
                ),
            ),
        ]

    def _launch_python_script(self, relative_path: str) -> None:
        """Launch a Python script in a new process.

        Args:
            relative_path: Path relative to REPO_ROOT
        """
        if relative_path is None:
            raise ValueError("relative_path must be provided")
        script_path = REPO_ROOT / relative_path
        if not script_path.exists():
            self.show_error("Script Not Found", f"Script not found:\n{script_path}")
            return

        try:
            _spawn_process([sys.executable, str(script_path)], cwd=REPO_ROOT)
        except (
            FileNotFoundError,
            PermissionError,
            OSError,
            SecureSubprocessError,
        ) as e:
            self.show_error("Launch Error", str(e))


def main() -> int:
    """Entry point for the Motion Capture launcher."""
    return run_launcher(MoCapLauncher)


if __name__ == "__main__":
    raise SystemExit(main())
