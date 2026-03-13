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


def __getattr__(name: str):
    """Lazy import for module components.

    Each group of names is loaded from its respective sub-module only when
    first accessed, keeping import cost low for callers that only need a
    subset of the package.

    Raises:
        AttributeError: If *name* is not a public export of this package.
        ImportError: Propagated if the underlying sub-module cannot be loaded
            (e.g. missing optional dependency such as pinocchio or meshcat).
    """
    if name in (
        "ClubTrajectory",
        "ClubTrajectoryParser",
        "ClubFrame",
        "SwingEventMarkers",
        "compute_hand_positions",
    ):
        from . import club_trajectory_parser as _parser_mod  # noqa: PLC0415

        return getattr(_parser_mod, name)

    if name in (
        "DualHandIKSolver",
        "DualHandIKSolverFallback",
        "IKSolverSettings",
        "IKResult",
        "TrajectoryIKResult",
        "create_ik_solver",
    ):
        from . import dual_hand_ik_solver as _ik_mod  # noqa: PLC0415

        return getattr(_ik_mod, name)

    if name in ("MotionVisualizer", "MatplotlibVisualizer", "VisualizerSettings"):
        from . import motion_visualizer as _viz_mod  # noqa: PLC0415

        return getattr(_viz_mod, name)

    if name in (
        "MotionTrainingPipeline",
        "PipelineConfig",
        "PipelineResult",
        "run_motion_training",
    ):
        from . import training_pipeline as _pipeline_mod  # noqa: PLC0415

        return getattr(_pipeline_mod, name)

    if name in ("TrajectoryExporter", "export_for_mujoco", "export_for_drake"):
        from . import trajectory_exporter as _exporter_mod  # noqa: PLC0415

        return getattr(_exporter_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
