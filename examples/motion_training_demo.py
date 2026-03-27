#!/usr/bin/env python3
"""Demonstration script for motion training from club trajectory.

This script shows how to:
1. Parse club trajectory data from motion capture
2. Solve inverse kinematics to generate body configurations
3. Visualize the motion (club + humanoid)
4. Export trajectories for other physics engines

Usage:
    python examples/motion_training_demo.py

Or with custom options:
    python examples/motion_training_demo.py --sheet TW_ProV1 --visualize --playback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the motion_training module to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "engines" / "physics_engines" / "pinocchio" / "python"),
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Motion Training Demo - Generate body motion from club trajectory",
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        default=str(PROJECT_ROOT / "data/Wiffle_ProV1_club_3D_data.xlsx"),
        help="Path to Excel file with club trajectory",
    )
    parser.add_argument(
        "--sheet",
        "-s",
        default="TW_wiffle",
        choices=["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"],
        help="Sheet name in Excel file",
    )
    parser.add_argument(
        "--urdf",
        "-u",
        default=str(
            PROJECT_ROOT
            / "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf",
        ),
        help="Path to golfer URDF",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(PROJECT_ROOT / "output/motion_training_demo"),
        help="Output directory",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Enable visualization (requires meshcat)",
    )
    parser.add_argument(
        "--playback",
        "-p",
        action="store_true",
        help="Enable playback animation",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=10,
        help="Subsample factor (use every Nth frame)",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export to all supported formats",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots (skip IK)",
    )
    return parser.parse_args()


def run_trajectory_analysis(trajectory_path: Path, sheet_name: str, output_dir: Path):
    """Run trajectory analysis and generate plots."""
    if not (trajectory_path is not None):
        raise ValueError("trajectory_path required")
    if not (sheet_name):
        raise ValueError("sheet_name required")
    if not (output_dir is not None):
        raise ValueError("output_dir required")
    from motion_training.club_trajectory_parser import ClubTrajectoryParser

    parser = ClubTrajectoryParser(trajectory_path)
    trajectory = parser.parse(sheet_name=sheet_name)

    # Generate 3D plot
    try:
        from motion_training.motion_visualizer import MatplotlibVisualizer

        viz = MatplotlibVisualizer()
        fig = viz.plot_trajectory_3d(trajectory)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150)

        import matplotlib.pyplot as plt

        plt.show()
    except ImportError:
        pass

    return trajectory


def _parse_and_subsample(trajectory_path, sheet_name, subsample):
    if not (trajectory_path is not None):
        raise ValueError("trajectory_path required")
    if not (sheet_name):
        raise ValueError("sheet_name required")
    if not (subsample > 0):
        raise ValueError("subsample must be positive")
    from motion_training.club_trajectory_parser import ClubTrajectoryParser

    parser = ClubTrajectoryParser(trajectory_path)
    trajectory = parser.parse(sheet_name=sheet_name)

    if subsample > 1:
        trajectory.frames = trajectory.frames[::subsample]

    return trajectory


def _init_and_solve_ik(urdf_path, trajectory):
    if not (urdf_path is not None):
        raise ValueError("urdf_path required")
    if not (trajectory is not None):
        raise ValueError("trajectory required")
    from motion_training.dual_hand_ik_solver import (
        IKSolverSettings,
        create_ik_solver,
    )

    settings = IKSolverSettings(
        dt=0.01,
        max_iterations=50,
        position_tolerance=0.005,  # 5mm tolerance
    )

    try:
        solver = create_ik_solver(
            urdf_path=urdf_path,
            settings=settings,
        )
    except Exception as e:  # noqa: BLE001, F841
        return None

    ik_result = solver.solve_trajectory(trajectory, verbose=True)

    return ik_result


def _export_results(ik_result, trajectory, output_dir):
    if not (ik_result is not None):
        raise ValueError("ik_result required")
    if not (trajectory is not None):
        raise ValueError("trajectory required")
    if not (output_dir is not None):
        raise ValueError("output_dir required")
    from motion_training.trajectory_exporter import TrajectoryExporter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = TrajectoryExporter(ik_result, trajectory)

    exporter.export(output_dir / "swing_trajectory", format="mujoco")

    exporter.export(output_dir / "swing_trajectory", format="csv")

    exporter.export(output_dir / "swing_trajectory", format="npz")

    try:
        from motion_training.motion_visualizer import MatplotlibVisualizer

        viz = MatplotlibVisualizer()

        fig = viz.plot_trajectory_3d(trajectory)
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150)

        fig = viz.plot_ik_errors(ik_result)
        fig.savefig(output_dir / "ik_errors.png", dpi=150)

        fig = viz.plot_joint_trajectories(ik_result)
        fig.savefig(output_dir / "joint_trajectories.png", dpi=150)

        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


def _run_visualization(urdf_path, trajectory, ik_result, visualize, playback):
    if visualize:
        try:
            from motion_training.motion_visualizer import MotionVisualizer

            motion_viz = MotionVisualizer(urdf_path=urdf_path)

            if playback:
                motion_viz.play_motion(trajectory, ik_result)
            else:
                motion_viz.show_static_trajectory(trajectory, ik_result)
                input("      Press Enter to exit...")

        except ImportError:
            pass
    else:
        pass


def run_ik_demo(
    trajectory_path: Path,
    sheet_name: str,
    urdf_path: Path,
    output_dir: Path,
    subsample: int = 10,
    visualize: bool = False,
    playback: bool = False,
):
    """Run the full IK demo."""
    if not (trajectory_path is not None):
        raise ValueError("trajectory_path required")
    if not (sheet_name):
        raise ValueError("sheet_name required")
    if not (urdf_path is not None):
        raise ValueError("urdf_path required")
    if not (output_dir is not None):
        raise ValueError("output_dir required")

    trajectory = _parse_and_subsample(trajectory_path, sheet_name, subsample)

    ik_result = _init_and_solve_ik(urdf_path, trajectory)
    if ik_result is None:
        return None

    _export_results(ik_result, trajectory, output_dir)
    _run_visualization(urdf_path, trajectory, ik_result, visualize, playback)

    return ik_result


def main():
    """Main entry point."""
    args = parse_args()

    trajectory_path = Path(args.trajectory)
    urdf_path = Path(args.urdf)
    output_dir = Path(args.output)

    if not trajectory_path.exists():
        sys.exit(1)

    if args.plot_only:
        run_trajectory_analysis(trajectory_path, args.sheet, output_dir)
    else:
        run_ik_demo(
            trajectory_path=trajectory_path,
            sheet_name=args.sheet,
            urdf_path=urdf_path,
            output_dir=output_dir,
            subsample=args.subsample,
            visualize=args.visualize,
            playback=args.playback,
        )


if __name__ == "__main__":
    main()
