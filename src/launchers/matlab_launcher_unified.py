#!/usr/bin/env python3
"""
Matlab Unified Launcher

Consolidates Simscape Models and Analysis Tools into a single interface.
Refactored to use BaseLauncher to eliminate DRY violations.
"""

from src.launchers.base import BaseLauncher, LaunchItem, run_launcher


class MatlabLauncher(BaseLauncher):
    """Launcher for MATLAB Simscape models and analysis tools."""

    WINDOW_TITLE = "Matlab Models & Tools"
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 500
    GRID_COLUMNS = 2

    def get_items(self) -> list[LaunchItem]:
        """Return the list of MATLAB models and tools."""
        return [
            LaunchItem(
                name="Simscape 2D Model",
                description="2D Simscape Multibody Golf Swing Model (.slx)",
                path="src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/GolfSwingZVCF.slx",
                item_type="model",
            ),
            LaunchItem(
                name="Simscape 3D Model",
                description="3D Simscape Multibody Golf Swing Model (.slx)",
                path="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx",
                item_type="model",
            ),
            LaunchItem(
                name="Dataset Generator",
                description="Forward Dynamics Dataset Generator GUI (.m)",
                path="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m",
                item_type="tool",
            ),
            LaunchItem(
                name="Analysis GUI",
                description="Golf Swing Analysis & Plotting Suite (.m)",
                path="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m",
                item_type="tool",
            ),
        ]


def main() -> int:
    """Entry point for the MATLAB launcher."""
    return run_launcher(MatlabLauncher)


if __name__ == "__main__":
    raise SystemExit(main())
