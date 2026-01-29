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
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src/engines/physics_engines/pinocchio/python"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Motion Training Demo - Generate body motion from club trajectory"
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
            / "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf"
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
    from motion_training.club_trajectory_parser import ClubTrajectoryParser

    print("\n=== Trajectory Analysis ===")
    parser = ClubTrajectoryParser(trajectory_path)
    trajectory = parser.parse(sheet_name=sheet_name)

    print(f"Loaded trajectory: {trajectory.num_frames} frames")
    print(f"Duration: {trajectory.duration:.3f} seconds")
    print("Events:")
    print(f"  Address (A): frame {trajectory.events.address}")
    print(f"  Top (T): frame {trajectory.events.top}")
    print(f"  Impact (I): frame {trajectory.events.impact}")
    print(f"  Finish (F): frame {trajectory.events.finish}")
    print(f"  Club head speed: {trajectory.events.club_head_speed} mph")

    # Generate 3D plot
    try:
        from motion_training.motion_visualizer import MatplotlibVisualizer

        viz = MatplotlibVisualizer()
        fig = viz.plot_trajectory_3d(trajectory)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150)
        print(f"\nSaved trajectory plot to: {output_dir / 'trajectory_3d.png'}")

        import matplotlib.pyplot as plt

        plt.show()
    except ImportError as e:
        print(f"Matplotlib not available: {e}")

    return trajectory


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
    from motion_training.club_trajectory_parser import ClubTrajectoryParser
    from motion_training.dual_hand_ik_solver import (
        IKSolverSettings,
        create_ik_solver,
    )
    from motion_training.trajectory_exporter import TrajectoryExporter

    print("\n" + "=" * 60)
    print("Motion Training Demo")
    print("=" * 60)

    # Step 1: Parse trajectory
    print("\n[1/5] Parsing club trajectory...")
    parser = ClubTrajectoryParser(trajectory_path)
    trajectory = parser.parse(sheet_name=sheet_name)
    print(
        f"      Loaded {trajectory.num_frames} frames, {trajectory.duration:.2f}s duration"
    )

    # Subsample for faster processing
    if subsample > 1:
        trajectory.frames = trajectory.frames[::subsample]
        print(f"      Subsampled to {trajectory.num_frames} frames (1/{subsample})")

    # Step 2: Initialize IK solver
    print("\n[2/5] Initializing IK solver...")
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
        print(f"      Model DOF: {solver.model.nq}")
    except Exception as e:
        print(f"      Error loading model: {e}")
        print("      Try running with --plot-only to skip IK")
        return None

    # Step 3: Solve IK
    print("\n[3/5] Solving inverse kinematics...")
    print("      This may take a while for large trajectories...")

    ik_result = solver.solve_trajectory(trajectory, verbose=True)

    print(f"\n      Convergence rate: {ik_result.convergence_rate * 100:.1f}%")
    print(
        f"      Mean position errors: "
        f"L={sum(ik_result.left_hand_errors)/len(ik_result.left_hand_errors)*1000:.2f}mm, "
        f"R={sum(ik_result.right_hand_errors)/len(ik_result.right_hand_errors)*1000:.2f}mm"
    )

    # Step 4: Export results
    print("\n[4/5] Exporting results...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = TrajectoryExporter(ik_result, trajectory)

    # Export to MuJoCo format (primary)
    mujoco_path = exporter.export(output_dir / "swing_trajectory", format="mujoco")
    print(f"      MuJoCo: {mujoco_path}")

    # Export to other formats
    csv_path = exporter.export(output_dir / "swing_trajectory", format="csv")
    print(f"      CSV: {csv_path}")

    npz_path = exporter.export(output_dir / "swing_trajectory", format="npz")
    print(f"      NPZ: {npz_path}")

    # Generate plots
    try:
        from motion_training.motion_visualizer import MatplotlibVisualizer

        viz = MatplotlibVisualizer()

        fig = viz.plot_trajectory_3d(trajectory)
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150)

        fig = viz.plot_ik_errors(ik_result)
        fig.savefig(output_dir / "ik_errors.png", dpi=150)

        fig = viz.plot_joint_trajectories(ik_result)
        fig.savefig(output_dir / "joint_trajectories.png", dpi=150)

        print(f"      Plots: {output_dir}/*.png")

        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        print("      Matplotlib not available for plots")

    # Step 5: Visualization
    if visualize:
        print("\n[5/5] Launching visualization...")
        try:
            from motion_training.motion_visualizer import MotionVisualizer

            motion_viz = MotionVisualizer(urdf_path=urdf_path)
            print(f"      Open in browser: {motion_viz.viewer.url()}")

            if playback:
                print("      Press Ctrl+C to stop playback")
                motion_viz.play_motion(trajectory, ik_result)
            else:
                motion_viz.show_static_trajectory(trajectory, ik_result)
                input("      Press Enter to exit...")

        except ImportError as e:
            print(f"      Visualization not available: {e}")
    else:
        print("\n[5/5] Visualization skipped (use --visualize to enable)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return ik_result


def main():
    """Main entry point."""
    args = parse_args()

    trajectory_path = Path(args.trajectory)
    urdf_path = Path(args.urdf)
    output_dir = Path(args.output)

    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
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
