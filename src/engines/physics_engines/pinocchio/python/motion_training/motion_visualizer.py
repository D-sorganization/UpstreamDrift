"""Visualization system for golf swing motion and club trajectory.

Provides visualization using:
1. Meshcat (3D web-based visualization)
2. Matplotlib (2D plots and static 3D views)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.engine_core.engine_availability import PINOCCHIO_AVAILABLE

if TYPE_CHECKING:
    from numpy.typing import NDArray

if PINOCCHIO_AVAILABLE:
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
else:
    pin = None  # type: ignore[assignment]
    MeshcatVisualizer = None  # type: ignore[misc, assignment]


try:
    import meshcat
    import meshcat.geometry as mcg
    import meshcat.transformations as mctf

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import logging

from motion_training.club_trajectory_parser import ClubTrajectory
from motion_training.dual_hand_ik_solver import TrajectoryIKResult

logger = logging.getLogger(__name__)


@dataclass
class VisualizerSettings:
    """Settings for motion visualization."""

    # Playback
    playback_speed: float = 1.0  # 1.0 = real-time
    loop: bool = True

    # Club trajectory visualization
    show_trajectory_path: bool = True
    trajectory_path_color: tuple = (0.2, 0.6, 1.0, 0.8)  # Blue
    trajectory_path_width: float = 0.005

    # Club visualization
    show_club: bool = True
    club_color: tuple = (0.3, 0.3, 0.3, 1.0)
    club_shaft_radius: float = 0.015
    club_shaft_length: float = 1.0
    club_head_size: tuple = (0.12, 0.08, 0.04)

    # Hand target visualization
    show_hand_targets: bool = True
    left_hand_color: tuple = (1.0, 0.0, 0.0, 0.8)  # Red
    right_hand_color: tuple = (0.0, 1.0, 0.0, 0.8)  # Green
    hand_target_radius: float = 0.02

    # Event markers
    show_event_markers: bool = True
    event_marker_radius: float = 0.03


class MotionVisualizer:
    """Visualizer for golf swing motion using Meshcat.

    Displays:
    - Club trajectory path as a line
    - Club model at each frame
    - Humanoid model following the motion
    - Hand target positions
    - Swing event markers (address, top, impact, finish)
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        settings: VisualizerSettings | None = None,
    ) -> None:
        """Initialize the visualizer.

        Args:
            urdf_path: Path to golfer URDF for humanoid visualization
            settings: Visualization settings
        """
        if not MESHCAT_AVAILABLE:
            raise ImportError("Meshcat required. Install with: pip install meshcat")

        self.settings = settings or VisualizerSettings()
        self.urdf_path = Path(urdf_path) if urdf_path else None

        # Initialize Meshcat viewer
        self.viewer = meshcat.Visualizer()
        logger.info("Meshcat viewer: %s", self.viewer.url())

        # Setup scene
        self._setup_scene()

        # Pinocchio visualizer for humanoid
        self.pin_viz: Any = None
        if self.urdf_path and PINOCCHIO_AVAILABLE:
            self._setup_humanoid()

    def _setup_scene(self) -> None:
        """Setup the basic scene (ground, lighting, etc.)."""
        # Clear previous scene
        self.viewer.delete()

        # Ground plane
        ground = mcg.Box([10.0, 10.0, 0.01])
        ground_material = mcg.MeshLambertMaterial(
            color=0x228B22,  # Forest green
            opacity=0.3,
            transparent=True,
        )
        self.viewer["ground"].set_object(ground, ground_material)
        self.viewer["ground"].set_transform(mctf.translation_matrix([0, 0, -0.005]))

        # Coordinate frame at origin
        self._add_coordinate_frame("origin", size=0.3)

    def _setup_humanoid(self) -> None:
        """Setup Pinocchio model and Meshcat visualizer for humanoid."""
        if not PINOCCHIO_AVAILABLE:
            return

        try:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
            self.data = self.model.createData()
            self.visual_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.urdf_path),
                pin.GeometryType.VISUAL,
            )
            self.collision_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.urdf_path),
                pin.GeometryType.COLLISION,
            )

            # Create Pinocchio Meshcat visualizer
            self.pin_viz = MeshcatVisualizer(
                self.model,
                self.collision_model,
                self.visual_model,
            )
            self.pin_viz.initViewer(viewer=self.viewer)
            self.pin_viz.loadViewerModel(rootNodeName="humanoid")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Warning: Could not load humanoid model: %s", e)
            self.pin_viz = None

    def _add_coordinate_frame(
        self,
        name: str,
        size: float = 0.1,
        transform: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a coordinate frame visualization."""
        # X axis (red)
        x_cyl = mcg.Cylinder(size, 0.002)
        x_mat = mcg.MeshBasicMaterial(color=0xFF0000)
        self.viewer[f"{name}/x"].set_object(x_cyl, x_mat)
        self.viewer[f"{name}/x"].set_transform(
            mctf.rotation_matrix(np.pi / 2, [0, 1, 0])
            @ mctf.translation_matrix([size / 2, 0, 0])
        )

        # Y axis (green)
        y_cyl = mcg.Cylinder(size, 0.002)
        y_mat = mcg.MeshBasicMaterial(color=0x00FF00)
        self.viewer[f"{name}/y"].set_object(y_cyl, y_mat)
        self.viewer[f"{name}/y"].set_transform(
            mctf.rotation_matrix(-np.pi / 2, [1, 0, 0])
            @ mctf.translation_matrix([0, size / 2, 0])
        )

        # Z axis (blue)
        z_cyl = mcg.Cylinder(size, 0.002)
        z_mat = mcg.MeshBasicMaterial(color=0x0000FF)
        self.viewer[f"{name}/z"].set_object(z_cyl, z_mat)
        self.viewer[f"{name}/z"].set_transform(
            mctf.translation_matrix([0, 0, size / 2])
        )

        if transform is not None:
            self.viewer[name].set_transform(transform)

    def add_club_trajectory_path(
        self,
        trajectory: ClubTrajectory,
    ) -> None:
        """Add the club trajectory as a path visualization."""
        if not self.settings.show_trajectory_path:
            return

        # Grip path
        grip_positions = trajectory.grip_positions
        if len(grip_positions) > 1:
            grip_line = mcg.Line(
                mcg.PointsGeometry(grip_positions.T),
                mcg.LineBasicMaterial(
                    color=self._color_to_hex(self.settings.trajectory_path_color[:3]),
                    linewidth=2,
                ),
            )
            self.viewer["trajectory/grip_path"].set_object(grip_line)

        # Club face path
        face_positions = trajectory.club_face_positions
        if len(face_positions) > 1:
            face_line = mcg.Line(
                mcg.PointsGeometry(face_positions.T),
                mcg.LineBasicMaterial(
                    color=0xFF6600,  # Orange
                    linewidth=2,
                ),
            )
            self.viewer["trajectory/face_path"].set_object(face_line)

        # Event markers
        if self.settings.show_event_markers:
            self._add_event_markers(trajectory)

    def _add_event_markers(self, trajectory: ClubTrajectory) -> None:
        """Add markers for swing events."""
        events = {
            "address": (0x00FF00, trajectory.events.address),  # Green
            "top": (0xFFFF00, trajectory.events.top),  # Yellow
            "impact": (0xFF0000, trajectory.events.impact),  # Red
            "finish": (0x0000FF, trajectory.events.finish),  # Blue
        }

        for name, (color, _frame_idx) in events.items():
            frame = trajectory.get_event_frame(name)
            if frame:
                sphere = mcg.Sphere(self.settings.event_marker_radius)
                mat = mcg.MeshBasicMaterial(color=color, transparent=True, opacity=0.8)
                self.viewer[f"events/{name}"].set_object(sphere, mat)
                self.viewer[f"events/{name}"].set_transform(
                    mctf.translation_matrix(frame.grip_position)
                )

    def add_club_at_frame(
        self,
        frame,
        name: str = "club",
    ) -> None:
        """Add club visualization at a specific frame."""
        s = self.settings

        # Club shaft
        shaft = mcg.Cylinder(s.club_shaft_length, s.club_shaft_radius)
        shaft_mat = mcg.MeshLambertMaterial(
            color=self._color_to_hex(s.club_color[:3]),
            opacity=s.club_color[3],
            transparent=True,
        )
        self.viewer[f"{name}/shaft"].set_object(shaft, shaft_mat)

        # Club head
        head = mcg.Box(s.club_head_size)
        head_mat = mcg.MeshLambertMaterial(color=0x111111)
        self.viewer[f"{name}/head"].set_object(head, head_mat)

        # Position the club based on grip frame
        self._update_club_transform(frame, name)

    def _update_club_transform(self, frame, name: str = "club") -> None:
        """Update club transform based on frame data."""
        s = self.settings

        # Compute transform from grip frame
        pos = frame.grip_position
        R = frame.grip_rotation

        # Club shaft is along Z-axis of grip frame, extending downward
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos

        # Shaft center is half the length below grip
        shaft_offset = np.eye(4)
        shaft_offset[2, 3] = -s.club_shaft_length / 2
        self.viewer[f"{name}/shaft"].set_transform(T @ shaft_offset)

        # Head is at the end of the shaft
        head_offset = np.eye(4)
        head_offset[2, 3] = -s.club_shaft_length
        self.viewer[f"{name}/head"].set_transform(T @ head_offset)

    def add_hand_targets(
        self,
        left_pos: NDArray[np.float64],
        right_pos: NDArray[np.float64],
    ) -> None:
        """Add hand target visualizations."""
        s = self.settings

        if s.show_hand_targets:
            # Left hand target
            left_sphere = mcg.Sphere(s.hand_target_radius)
            left_mat = mcg.MeshBasicMaterial(
                color=self._color_to_hex(s.left_hand_color[:3]),
                transparent=True,
                opacity=s.left_hand_color[3],
            )
            self.viewer["targets/left_hand"].set_object(left_sphere, left_mat)
            self.viewer["targets/left_hand"].set_transform(
                mctf.translation_matrix(left_pos)
            )

            # Right hand target
            right_sphere = mcg.Sphere(s.hand_target_radius)
            right_mat = mcg.MeshBasicMaterial(
                color=self._color_to_hex(s.right_hand_color[:3]),
                transparent=True,
                opacity=s.right_hand_color[3],
            )
            self.viewer["targets/right_hand"].set_object(right_sphere, right_mat)
            self.viewer["targets/right_hand"].set_transform(
                mctf.translation_matrix(right_pos)
            )

    def display_humanoid(self, q: NDArray[np.float64]) -> None:
        """Display humanoid at given configuration."""
        if self.pin_viz is not None:
            self.pin_viz.display(q)

    def play_motion(
        self,
        trajectory: ClubTrajectory,
        ik_result: TrajectoryIKResult | None = None,
    ) -> None:
        """Play back the motion animation.

        Args:
            trajectory: Club trajectory
            ik_result: Optional IK result with body configurations
        """
        s = self.settings

        # Add trajectory path
        self.add_club_trajectory_path(trajectory)

        # Add initial club
        if trajectory.frames:
            self.add_club_at_frame(trajectory.frames[0])

        logger.info(
            "Playing %s frames at %sx speed", trajectory.num_frames, s.playback_speed
        )
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                for i, frame in enumerate(trajectory.frames):
                    start_time = time.time()

                    # Update club
                    self._update_club_transform(frame)

                    # Update hand targets
                    from motion_training.club_trajectory_parser import (
                        compute_hand_positions,
                    )

                    left_pos, right_pos = compute_hand_positions(frame)
                    self.add_hand_targets(left_pos, right_pos)

                    # Update humanoid if IK result available
                    if ik_result and i < len(ik_result.configurations):
                        self.display_humanoid(ik_result.configurations[i])

                    # Calculate frame timing
                    if i < len(trajectory.frames) - 1:
                        dt = trajectory.frames[i + 1].time - frame.time
                        target_dt = abs(dt) / s.playback_speed
                        elapsed = time.time() - start_time
                        if target_dt > elapsed:
                            time.sleep(target_dt - elapsed)

                if not s.loop:
                    break

        except KeyboardInterrupt:
            logger.info("\nPlayback stopped")

    def show_static_trajectory(
        self,
        trajectory: ClubTrajectory,
        ik_result: TrajectoryIKResult | None = None,
        num_frames_to_show: int = 10,
    ) -> None:
        """Show static visualization with multiple frames overlaid.

        Args:
            trajectory: Club trajectory
            ik_result: Optional IK result
            num_frames_to_show: Number of frames to display
        """
        self.add_club_trajectory_path(trajectory)

        # Select frames to show
        if trajectory.num_frames <= num_frames_to_show:
            indices: list[int] | np.ndarray = list(range(trajectory.num_frames))
        else:
            indices = np.linspace(
                0, trajectory.num_frames - 1, num_frames_to_show
            ).astype(int)

        for i, idx in enumerate(indices):
            frame = trajectory.frames[idx]

            # Add semi-transparent club
            alpha = 0.3 + 0.5 * (i / len(indices))  # Fade from transparent to solid
            self._add_ghost_club(frame, f"ghost_club_{i}", alpha)

        # Show final frame with full opacity
        if trajectory.frames:
            self.add_club_at_frame(trajectory.frames[-1])

    def _add_ghost_club(
        self,
        frame,
        name: str,
        alpha: float,
    ) -> None:
        """Add a semi-transparent club visualization."""
        s = self.settings

        shaft = mcg.Cylinder(s.club_shaft_length, s.club_shaft_radius)
        shaft_mat = mcg.MeshLambertMaterial(
            color=0x666666,
            opacity=alpha,
            transparent=True,
        )
        self.viewer[f"{name}/shaft"].set_object(shaft, shaft_mat)

        head = mcg.Box(s.club_head_size)
        head_mat = mcg.MeshLambertMaterial(
            color=0x444444,
            opacity=alpha,
            transparent=True,
        )
        self.viewer[f"{name}/head"].set_object(head, head_mat)

        # Position
        pos = frame.grip_position
        R = frame.grip_rotation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos

        shaft_offset = np.eye(4)
        shaft_offset[2, 3] = -s.club_shaft_length / 2
        self.viewer[f"{name}/shaft"].set_transform(T @ shaft_offset)

        head_offset = np.eye(4)
        head_offset[2, 3] = -s.club_shaft_length
        self.viewer[f"{name}/head"].set_transform(T @ head_offset)

    @staticmethod
    def _color_to_hex(rgb: tuple) -> int:
        """Convert RGB tuple (0-1 range) to hex color."""
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        return (r << 16) + (g << 8) + b


class MatplotlibVisualizer:
    """2D/3D visualization using Matplotlib for static plots."""

    def __init__(self) -> None:
        """Initialize Matplotlib visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required")

    def plot_trajectory_3d(
        self,
        trajectory: ClubTrajectory,
        show_events: bool = True,
        figsize: tuple = (12, 10),
    ) -> plt.Figure:
        """Create 3D plot of club trajectory.

        Args:
            trajectory: Club trajectory
            show_events: Show event markers
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot grip path
        grip_pos = trajectory.grip_positions
        ax.plot(
            grip_pos[:, 0],
            grip_pos[:, 1],
            grip_pos[:, 2],
            "b-",
            linewidth=2,
            label="Grip Path",
        )

        # Plot club face path
        face_pos = trajectory.club_face_positions
        ax.plot(
            face_pos[:, 0],
            face_pos[:, 1],
            face_pos[:, 2],
            "r-",
            linewidth=2,
            label="Club Face Path",
        )

        # Event markers
        if show_events:
            events = [
                ("address", "go", "Address"),
                ("top", "y^", "Top"),
                ("impact", "rs", "Impact"),
                ("finish", "b*", "Finish"),
            ]
            for event_name, marker, label in events:
                frame = trajectory.get_event_frame(event_name)
                if frame:
                    pos = frame.grip_position
                    ax.plot(
                        [pos[0]], [pos[1]], [pos[2]], marker, markersize=15, label=label
                    )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("Golf Club Trajectory")

        # Equal aspect ratio
        max_range = np.max(
            [
                grip_pos[:, 0].max() - grip_pos[:, 0].min(),
                grip_pos[:, 1].max() - grip_pos[:, 1].min(),
                grip_pos[:, 2].max() - grip_pos[:, 2].min(),
            ]
        )
        mid = grip_pos.mean(axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        return fig

    def plot_ik_errors(
        self,
        ik_result: TrajectoryIKResult,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """Plot IK tracking errors over time.

        Args:
            ik_result: IK solving results
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        times = np.array(ik_result.times)

        # Position errors
        axes[0].plot(times, ik_result.left_hand_errors, "r-", label="Left Hand")
        axes[0].plot(times, ik_result.right_hand_errors, "g-", label="Right Hand")
        axes[0].axhline(y=0.001, color="k", linestyle="--", label="Tolerance")
        axes[0].set_ylabel("Position Error (m)")
        axes[0].legend()
        axes[0].set_title(
            f"IK Tracking Errors (Convergence: {ik_result.convergence_rate * 100:.1f}%)"
        )
        axes[0].grid(True)

        # Combined error
        combined = np.array(ik_result.left_hand_errors) + np.array(
            ik_result.right_hand_errors
        )
        axes[1].plot(times, combined, "b-", label="Total Error")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Combined Error (m)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        return fig

    def plot_joint_trajectories(
        self,
        ik_result: TrajectoryIKResult,
        joint_names: list[str] | None = None,
        figsize: tuple = (14, 8),
    ) -> plt.Figure:
        """Plot joint angle trajectories.

        Args:
            ik_result: IK solving results
            joint_names: Names of joints to plot (all if None)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        q_traj = ik_result.q_trajectory
        times = np.array(ik_result.times)
        n_joints = q_traj.shape[1]

        if joint_names is None:
            joint_names = [f"q{i}" for i in range(n_joints)]

        # Limit to reasonable number of subplots
        n_to_plot = min(n_joints, 12)

        n_rows = (n_to_plot + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize, sharex=True)
        axes = axes.flatten()

        for i in range(n_to_plot):
            axes[i].plot(times, np.rad2deg(q_traj[:, i]), "b-")
            axes[i].set_title(joint_names[i] if i < len(joint_names) else f"q{i}")
            axes[i].set_ylabel("Angle (deg)")
            axes[i].grid(True)

        for i in range(n_to_plot, len(axes)):
            axes[i].set_visible(False)

        axes[-3].set_xlabel("Time (s)")
        axes[-2].set_xlabel("Time (s)")
        axes[-1].set_xlabel("Time (s)")

        fig.suptitle("Joint Angle Trajectories")
        plt.tight_layout()
        return fig
