"""Main training pipeline for golf swing motion from club trajectory.

This module orchestrates the complete workflow:
1. Parse club trajectory from motion capture data
2. Solve inverse kinematics to generate body configurations
3. Visualize and validate the motion
4. Export trajectories for other physics engines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from motion_training.club_trajectory_parser import (
    ClubTrajectory,
    ClubTrajectoryParser,
)
from motion_training.dual_hand_ik_solver import (
    DualHandIKSolver,
    IKSolverSettings,
    TrajectoryIKResult,
    create_ik_solver,
)


@dataclass
class PipelineConfig:
    """Configuration for the motion training pipeline."""

    # Input data
    trajectory_file: str | Path = ""
    sheet_name: str = "TW_wiffle"

    # Model paths
    golfer_urdf: str | Path = ""

    # IK settings
    ik_settings: IKSolverSettings = field(default_factory=IKSolverSettings)

    # Output settings
    output_dir: str | Path = "output/motion_training"
    save_trajectory: bool = True
    save_plots: bool = True

    # Visualization
    visualize: bool = True
    playback: bool = False

    # Processing
    subsample_factor: int = 1  # Use every Nth frame
    start_frame: int = 0
    end_frame: int = -1  # -1 means all frames


@dataclass
class PipelineResult:
    """Result of the motion training pipeline."""

    trajectory: ClubTrajectory
    ik_result: TrajectoryIKResult
    success: bool
    message: str = ""

    @property
    def joint_trajectory(self) -> NDArray[np.float64]:
        """Return joint trajectory as numpy array."""
        return self.ik_result.q_trajectory

    @property
    def times(self) -> NDArray[np.float64]:
        """Return time array."""
        return np.array(self.ik_result.times)


class MotionTrainingPipeline:
    """Main pipeline for generating body motion from club trajectory.

    Workflow:
    1. Load and parse club trajectory data
    2. Initialize humanoid model and IK solver
    3. Solve IK for each frame
    4. Visualize and export results
    """

    DEFAULT_URDF = (
        "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf"
    )

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.trajectory: ClubTrajectory | None = None
        self.ik_solver: DualHandIKSolver | None = None
        self.ik_result: TrajectoryIKResult | None = None

    def run(self) -> PipelineResult:
        """Execute the full pipeline.

        Returns:
            PipelineResult with trajectory and IK results
        """
        logger.info("=" * 60)
        logger.info("Motion Training Pipeline")
        logger.info("=" * 60)

        # Step 1: Parse trajectory
        logger.info("\n[1/4] Parsing club trajectory...")
        self.trajectory = self._parse_trajectory()
        logger.info(f"      Loaded {self.trajectory.num_frames} frames")
        logger.info(f"      Duration: {self.trajectory.duration:.3f} seconds")
        logger.info(
            f"      Events: A={self.trajectory.events.address}, "
            f"T={self.trajectory.events.top}, "
            f"I={self.trajectory.events.impact}, "
            f"F={self.trajectory.events.finish}"
        )

        # Step 2: Initialize IK solver
        logger.info("\n[2/4] Initializing IK solver...")
        self._init_ik_solver()
        logger.info(f"      Model: {self.config.golfer_urdf}")
        logger.info(f"      DOF: {self.ik_solver.model.nq}")

        # Step 3: Solve IK
        logger.info("\n[3/4] Solving inverse kinematics...")
        self.ik_result = self._solve_ik()
        logger.info(
            f"      Convergence rate: {self.ik_result.convergence_rate * 100:.1f}%"
        )
        logger.info(
            f"      Mean left hand error: "
            f"{np.mean(self.ik_result.left_hand_errors) * 1000:.2f} mm"
        )
        logger.info(
            f"      Mean right hand error: "
            f"{np.mean(self.ik_result.right_hand_errors) * 1000:.2f} mm"
        )

        # Step 4: Save/visualize results
        logger.info("\n[4/4] Processing results...")
        if self.config.save_trajectory:
            self._save_results()
            logger.info(f"      Saved to: {self.config.output_dir}")

        if self.config.visualize:
            self._visualize()

        success = self.ik_result.convergence_rate > 0.5

        logger.info("\n" + "=" * 60)
        logger.info(f"Pipeline complete. Success: {success}")
        logger.info("=" * 60)

        return PipelineResult(
            trajectory=self.trajectory,
            ik_result=self.ik_result,
            success=success,
        )

    def _parse_trajectory(self) -> ClubTrajectory:
        """Parse and preprocess club trajectory."""
        parser = ClubTrajectoryParser(self.config.trajectory_file)
        trajectory = parser.parse(sheet_name=self.config.sheet_name)

        # Apply frame range
        start = self.config.start_frame
        end = (
            self.config.end_frame
            if self.config.end_frame > 0
            else len(trajectory.frames)
        )
        trajectory.frames = trajectory.frames[start:end]

        # Apply subsampling
        if self.config.subsample_factor > 1:
            trajectory.frames = trajectory.frames[:: self.config.subsample_factor]

        return trajectory

    def _init_ik_solver(self) -> None:
        """Initialize the IK solver."""
        urdf_path = self.config.golfer_urdf or self.DEFAULT_URDF
        self.ik_solver = create_ik_solver(
            urdf_path=urdf_path,
            settings=self.config.ik_settings,
        )

    def _solve_ik(self) -> TrajectoryIKResult:
        """Solve IK for the trajectory."""
        return self.ik_solver.solve_trajectory(
            self.trajectory,
            verbose=True,
        )

    def _save_results(self) -> None:
        """Save results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save joint trajectory as numpy file
        np.savez(
            output_dir / "joint_trajectory.npz",
            q=self.ik_result.q_trajectory,
            times=np.array(self.ik_result.times),
            left_errors=np.array(self.ik_result.left_hand_errors),
            right_errors=np.array(self.ik_result.right_hand_errors),
            convergence_rate=self.ik_result.convergence_rate,
        )

        # Save as CSV for easy inspection
        import csv

        with open(output_dir / "joint_trajectory.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["time"] + [
                f"q{i}" for i in range(self.ik_result.q_trajectory.shape[1])
            ]
            writer.writerow(header)
            for i, t in enumerate(self.ik_result.times):
                row = [t] + list(self.ik_result.q_trajectory[i])
                writer.writerow(row)

        if self.config.save_plots:
            self._save_plots(output_dir)

    def _save_plots(self, output_dir: Path) -> None:
        """Generate and save analysis plots."""
        try:
            from motion_training.motion_visualizer import MatplotlibVisualizer

            viz = MatplotlibVisualizer()

            # Trajectory plot
            fig = viz.plot_trajectory_3d(self.trajectory)
            fig.savefig(output_dir / "trajectory_3d.png", dpi=150)

            # IK error plot
            fig = viz.plot_ik_errors(self.ik_result)
            fig.savefig(output_dir / "ik_errors.png", dpi=150)

            # Joint trajectories
            fig = viz.plot_joint_trajectories(self.ik_result)
            fig.savefig(output_dir / "joint_trajectories.png", dpi=150)

            import matplotlib.pyplot as plt

            plt.close("all")

        except ImportError:
            logger.info("      Matplotlib not available, skipping plots")

    def _visualize(self) -> None:
        """Launch visualization."""
        try:
            from motion_training.motion_visualizer import MotionVisualizer

            viz = MotionVisualizer(urdf_path=self.config.golfer_urdf)

            if self.config.playback:
                viz.play_motion(self.trajectory, self.ik_result)
            else:
                viz.show_static_trajectory(self.trajectory, self.ik_result)
                logger.info(f"      Visualization URL: {viz.viewer.url()}")

        except ImportError as e:
            logger.info(f"      Visualization not available: {e}")


def run_motion_training(
    trajectory_file: str | Path,
    sheet_name: str = "TW_wiffle",
    output_dir: str | Path = "output/motion_training",
    golfer_urdf: str | Path = "",
    visualize: bool = True,
    playback: bool = False,
) -> PipelineResult:
    """Convenience function to run motion training.

    Args:
        trajectory_file: Path to Excel file with club trajectory
        sheet_name: Sheet name in Excel file
        output_dir: Directory for output files
        golfer_urdf: Path to golfer URDF (uses default if empty)
        visualize: Enable visualization
        playback: Enable playback animation

    Returns:
        PipelineResult with trajectory and IK results
    """
    config = PipelineConfig(
        trajectory_file=trajectory_file,
        sheet_name=sheet_name,
        output_dir=output_dir,
        golfer_urdf=golfer_urdf,
        visualize=visualize,
        playback=playback,
    )

    pipeline = MotionTrainingPipeline(config)
    return pipeline.run()


# CLI entry point
if __name__ == "__main__":
    import argparse
    import logging

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Train body motion from club trajectory using IK"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        required=True,
        help="Path to Excel file with club trajectory",
    )
    parser.add_argument(
        "--sheet",
        "-s",
        default="TW_wiffle",
        help="Sheet name in Excel file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/motion_training",
        help="Output directory",
    )
    parser.add_argument(
        "--urdf",
        "-u",
        default="",
        help="Path to golfer URDF",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Enable playback animation",
    )

    args = parser.parse_args()

    result = run_motion_training(
        trajectory_file=args.trajectory,
        sheet_name=args.sheet,
        output_dir=args.output,
        golfer_urdf=args.urdf,
        visualize=not args.no_visualize,
        playback=args.playback,
    )

    logger.info(f"\nResult: {'SUCCESS' if result.success else 'NEEDS REVIEW'}")
