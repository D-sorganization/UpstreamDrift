from typing import Any

"""Motion training module for inverse kinematics from club trajectory data.

This module provides tools to:
1. Parse club trajectory data from motion capture (Excel/CSV)
2. Solve inverse kinematics to determine body configurations
3. Visualize the resulting motion with the humanoid following the club
4. Export trajectories for use in other physics engines (MuJoCo, Drake)

Example Usage:
    >>> from motion_training import ClubTrajectoryParser, create_ik_solver
    >>>
    >>> # Parse club trajectory
    >>> parser = ClubTrajectoryParser("data/Wiffle_ProV1_club_3D_data.xlsx")
    >>> trajectory = parser.parse(sheet_name="TW_wiffle")
    >>>
    >>> # Solve IK
    >>> solver = create_ik_solver("models/golfer_ik.urdf")
    >>> result = solver.solve_trajectory(trajectory)
    >>>
    >>> # Export for MuJoCo
    >>> from motion_training import export_for_mujoco
    >>> export_for_mujoco(result, "output/trajectory.json", trajectory)
"""

# Lazy imports to avoid requiring all dependencies
__all__ = [
    # Parser
    "ClubTrajectory",
    "ClubTrajectoryParser",
    "ClubFrame",
    "SwingEventMarkers",
    "compute_hand_positions",
    # IK Solver
    "DualHandIKSolver",
    "DualHandIKSolverFallback",
    "IKSolverSettings",
    "IKResult",
    "TrajectoryIKResult",
    "create_ik_solver",
    # Visualization
    "MotionVisualizer",
    "MatplotlibVisualizer",
    "VisualizerSettings",
    # Pipeline
    "MotionTrainingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_motion_training",
    # Export
    "TrajectoryExporter",
    "export_for_mujoco",
    "export_for_drake",
]


def __getattr__(name: str) -> Any:
    """Lazy import for module components."""
    if name in (
        "ClubTrajectory",
        "ClubTrajectoryParser",
        "ClubFrame",
        "SwingEventMarkers",
        "compute_hand_positions",
    ):
        pass

        return locals()[name]

    if name in (
        "DualHandIKSolver",
        "DualHandIKSolverFallback",
        "IKSolverSettings",
        "IKResult",
        "TrajectoryIKResult",
        "create_ik_solver",
    ):
        pass

        return locals()[name]

    if name in ("MotionVisualizer", "MatplotlibVisualizer", "VisualizerSettings"):
        pass

        return locals()[name]

    if name in (
        "MotionTrainingPipeline",
        "PipelineConfig",
        "PipelineResult",
        "run_motion_training",
    ):
        pass

        return locals()[name]

    if name in ("TrajectoryExporter", "export_for_mujoco", "export_for_drake"):
        pass

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
