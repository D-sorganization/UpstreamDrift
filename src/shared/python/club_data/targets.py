"""Club target management for simulation visualization.

Manages target trajectories from professional player data for display
in Drake, Pinocchio, and MuJoCo visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    from .loader import ProPlayerData

logger = get_logger(__name__)


@dataclass
class TargetTrajectory:
    """Represents a target trajectory for visualization."""

    name: str
    time_series: np.ndarray  # (N,) time points
    positions: np.ndarray  # (N, 3) positions
    velocities: np.ndarray | None = None  # (N, 3) velocities (optional)

    # Phase markers (time indices)
    address_idx: int = 0
    top_idx: int = 0
    impact_idx: int = 0
    finish_idx: int = 0

    # Visual properties
    color: tuple[float, float, float] = (0.2, 0.8, 0.2)  # RGB, default green
    opacity: float = 0.7
    line_width: float = 2.0

    @property
    def duration(self) -> float:
        """Total duration of the trajectory."""
        return float(self.time_series[-1] - self.time_series[0])

    @property
    def num_frames(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.time_series)

    def get_position_at_time(self, t: float) -> np.ndarray:
        """Interpolate position at a specific time."""
        if t <= self.time_series[0]:
            return self.positions[0]
        if t >= self.time_series[-1]:
            return self.positions[-1]

        idx = np.searchsorted(self.time_series, t)
        t0, t1 = self.time_series[idx - 1], self.time_series[idx]
        alpha = (t - t0) / (t1 - t0)

        return self.positions[idx - 1] + alpha * (
            self.positions[idx] - self.positions[idx - 1]
        )

    def get_velocity_at_time(self, t: float) -> np.ndarray | None:
        """Interpolate velocity at a specific time."""
        if self.velocities is None:
            return None

        if t <= self.time_series[0]:
            return self.velocities[0]
        if t >= self.time_series[-1]:
            return self.velocities[-1]

        idx = np.searchsorted(self.time_series, t)
        t0, t1 = self.time_series[idx - 1], self.time_series[idx]
        alpha = (t - t0) / (t1 - t0)

        return self.velocities[idx - 1] + alpha * (
            self.velocities[idx] - self.velocities[idx - 1]
        )

    def get_phase_position(self, phase: str) -> np.ndarray | None:
        """Get position at a specific phase."""
        idx_map = {
            "address": self.address_idx,
            "top": self.top_idx,
            "impact": self.impact_idx,
            "finish": self.finish_idx,
        }

        idx = idx_map.get(phase, -1)
        if idx < 0 or idx >= len(self.positions):
            return None

        return self.positions[idx]

    def resample(self, num_points: int) -> TargetTrajectory:
        """Create a resampled version with specified number of points.

        Args:
            num_points: Number of points in resampled trajectory

        Returns:
            New TargetTrajectory with resampled data
        """
        t_new = np.linspace(self.time_series[0], self.time_series[-1], num_points)
        positions_new = np.zeros((num_points, 3))
        velocities_new = None if self.velocities is None else np.zeros((num_points, 3))

        for i, t in enumerate(t_new):
            positions_new[i] = self.get_position_at_time(t)
            if velocities_new is not None:
                v = self.get_velocity_at_time(t)
                if v is not None:
                    velocities_new[i] = v

        # Map phase indices to new sampling
        def map_idx(old_idx: int) -> int:
            if old_idx < 0 or old_idx >= len(self.time_series):
                return 0
            t = self.time_series[old_idx]
            return int(np.argmin(np.abs(t_new - t)))

        return TargetTrajectory(
            name=self.name,
            time_series=t_new,
            positions=positions_new,
            velocities=velocities_new,
            address_idx=map_idx(self.address_idx),
            top_idx=map_idx(self.top_idx),
            impact_idx=map_idx(self.impact_idx),
            finish_idx=map_idx(self.finish_idx),
            color=self.color,
            opacity=self.opacity,
            line_width=self.line_width,
        )


class ClubTargetManager:
    """Manages target trajectories for simulation visualization.

    This class provides a unified interface for displaying professional player
    swing data as targets in Drake, Pinocchio, and MuJoCo visualizations.
    """

    def __init__(self) -> None:
        """Initialize the target manager."""
        self._trajectories: dict[str, TargetTrajectory] = {}
        self._active_trajectory: str | None = None
        self._enabled = False
        self._update_callbacks: list[Any] = []

        # Display options
        self._show_path = True
        self._show_velocity_vectors = False
        self._show_phase_markers = True
        self._opacity = 0.7
        self._path_color = (0.2, 0.8, 0.2)  # Green

        # Phase marker colors
        self._phase_colors = {
            "address": (0.2, 0.2, 0.9, 0.8),  # Blue
            "top": (0.9, 0.9, 0.2, 0.8),  # Yellow
            "impact": (0.9, 0.2, 0.2, 0.8),  # Red
            "finish": (0.9, 0.2, 0.9, 0.8),  # Magenta
        }

    def add_trajectory_from_player(
        self,
        player: ProPlayerData,
        name: str | None = None,
    ) -> str:
        """Add a target trajectory from player data.

        Args:
            player: ProPlayerData with trajectory information
            name: Optional name (defaults to player name)

        Returns:
            Name of the added trajectory
        """
        if not player.has_trajectory_data():
            raise ValueError(f"Player {player.player_name} has no trajectory data")

        traj_name = name or player.player_name

        # Find phase indices
        ts = player.time_series
        if ts is None:
            raise ValueError("No time series data")

        def find_idx(t: float) -> int:
            if t <= 0:
                return 0
            return int(np.argmin(np.abs(ts - t)))

        trajectory = TargetTrajectory(
            name=traj_name,
            time_series=ts.copy(),
            positions=(
                player.club_head_positions.copy()
                if player.club_head_positions is not None
                else np.zeros((len(ts), 3))
            ),
            velocities=(
                player.club_head_velocities.copy()
                if player.club_head_velocities is not None
                else None
            ),
            address_idx=find_idx(player.address_time),
            top_idx=find_idx(player.top_of_backswing_time),
            impact_idx=find_idx(player.impact_time),
            finish_idx=find_idx(player.finish_time),
            color=self._path_color,
            opacity=self._opacity,
        )

        self._trajectories[traj_name] = trajectory
        logger.info(
            "Added trajectory '%s' with %d frames", traj_name, trajectory.num_frames
        )

        return traj_name

    def add_trajectory(self, trajectory: TargetTrajectory) -> None:
        """Add a target trajectory directly.

        Args:
            trajectory: TargetTrajectory to add
        """
        self._trajectories[trajectory.name] = trajectory
        logger.info("Added trajectory '%s'", trajectory.name)

    def remove_trajectory(self, name: str) -> bool:
        """Remove a trajectory by name.

        Args:
            name: Trajectory name

        Returns:
            True if removed, False if not found
        """
        if name in self._trajectories:
            del self._trajectories[name]
            if self._active_trajectory == name:
                self._active_trajectory = None
            return True
        return False

    def get_trajectory(self, name: str) -> TargetTrajectory | None:
        """Get a trajectory by name."""
        return self._trajectories.get(name)

    def get_active_trajectory(self) -> TargetTrajectory | None:
        """Get the currently active trajectory."""
        if self._active_trajectory is None:
            return None
        return self._trajectories.get(self._active_trajectory)

    def set_active_trajectory(self, name: str | None) -> bool:
        """Set the active trajectory.

        Args:
            name: Trajectory name or None to deactivate

        Returns:
            True if set successfully
        """
        if name is None:
            self._active_trajectory = None
            self._notify_update()
            return True

        if name in self._trajectories:
            self._active_trajectory = name
            self._notify_update()
            return True

        return False

    def list_trajectories(self) -> list[str]:
        """Get list of trajectory names."""
        return list(self._trajectories.keys())

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable target display."""
        self._enabled = enabled
        self._notify_update()

    def is_enabled(self) -> bool:
        """Check if target display is enabled."""
        return self._enabled

    def set_display_options(
        self,
        show_path: bool | None = None,
        show_velocity: bool | None = None,
        show_markers: bool | None = None,
        opacity: float | None = None,
        path_color: tuple[float, float, float] | None = None,
    ) -> None:
        """Update display options.

        Args:
            show_path: Show trajectory path
            show_velocity: Show velocity vectors
            show_markers: Show phase markers
            opacity: Path opacity (0-1)
            path_color: RGB color tuple
        """
        if show_path is not None:
            self._show_path = show_path
        if show_velocity is not None:
            self._show_velocity_vectors = show_velocity
        if show_markers is not None:
            self._show_phase_markers = show_markers
        if opacity is not None:
            self._opacity = opacity
            # Update all trajectories
            for traj in self._trajectories.values():
                traj.opacity = opacity
        if path_color is not None:
            self._path_color = path_color
            for traj in self._trajectories.values():
                traj.color = path_color

        self._notify_update()

    def get_display_options(self) -> dict[str, Any]:
        """Get current display options."""
        return {
            "show_path": self._show_path,
            "show_velocity": self._show_velocity_vectors,
            "show_markers": self._show_phase_markers,
            "opacity": self._opacity,
            "path_color": self._path_color,
        }

    def register_update_callback(self, callback: Any) -> None:
        """Register a callback to be called when targets change.

        Args:
            callback: Callable to invoke on changes
        """
        self._update_callbacks.append(callback)

    def _notify_update(self) -> None:
        """Notify all registered callbacks of an update."""
        for callback in self._update_callbacks:
            try:
                callback()
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Callback error: %s", e)

    # -------- Rendering Helpers --------

    def get_path_points(
        self,
        trajectory: TargetTrajectory | None = None,
        num_points: int = 100,
    ) -> np.ndarray | None:
        """Get interpolated path points for rendering.

        Args:
            trajectory: Trajectory to render (uses active if None)
            num_points: Number of points for path

        Returns:
            (N, 3) array of positions or None
        """
        if trajectory is None:
            trajectory = self.get_active_trajectory()

        if trajectory is None or not self._show_path:
            return None

        resampled = trajectory.resample(num_points)
        return resampled.positions

    def get_velocity_vectors(
        self,
        trajectory: TargetTrajectory | None = None,
        num_vectors: int = 20,
        scale: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get velocity vector data for rendering.

        Args:
            trajectory: Trajectory to render (uses active if None)
            num_vectors: Number of vectors to display
            scale: Scale factor for velocity magnitude

        Returns:
            Tuple of (origins, directions) or None
        """
        if trajectory is None:
            trajectory = self.get_active_trajectory()

        if (
            trajectory is None
            or trajectory.velocities is None
            or not self._show_velocity_vectors
        ):
            return None

        # Subsample for rendering
        indices = np.linspace(
            0, len(trajectory.time_series) - 1, num_vectors, dtype=int
        )

        origins = trajectory.positions[indices]
        directions = trajectory.velocities[indices] * scale

        return origins, directions

    def get_phase_markers(
        self,
        trajectory: TargetTrajectory | None = None,
    ) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
        """Get phase marker data for rendering.

        Args:
            trajectory: Trajectory to render (uses active if None)

        Returns:
            Dictionary mapping phase name to (position, color) tuple
        """
        if trajectory is None:
            trajectory = self.get_active_trajectory()

        if trajectory is None or not self._show_phase_markers:
            return {}

        markers = {}
        for phase in ["address", "top", "impact", "finish"]:
            pos = trajectory.get_phase_position(phase)
            if pos is not None:
                markers[phase] = (pos, self._phase_colors.get(phase, (1, 1, 1, 1)))

        return markers

    def compute_tracking_error(
        self,
        current_position: np.ndarray,
        current_time: float,
        trajectory: TargetTrajectory | None = None,
    ) -> dict[str, float]:
        """Compute tracking error between current state and target.

        Args:
            current_position: Current club head position (3,)
            current_time: Current simulation time
            trajectory: Target trajectory (uses active if None)

        Returns:
            Dictionary with error metrics
        """
        if trajectory is None:
            trajectory = self.get_active_trajectory()

        if trajectory is None:
            return {"position_error": float("nan"), "normalized_time": 0.0}

        target_pos = trajectory.get_position_at_time(current_time)
        position_error = float(np.linalg.norm(current_position - target_pos))

        # Normalized time (0-1 through swing)
        t_start = trajectory.time_series[0]
        t_end = trajectory.time_series[-1]
        normalized_time = (
            (current_time - t_start) / (t_end - t_start) if t_end > t_start else 0.0
        )
        normalized_time = np.clip(normalized_time, 0.0, 1.0)

        result = {
            "position_error": position_error,
            "normalized_time": float(normalized_time),
            "target_position": target_pos.tolist(),
        }

        # Add velocity error if available
        if trajectory.velocities is not None:
            target_vel = trajectory.get_velocity_at_time(current_time)
            if target_vel is not None:
                result["target_velocity"] = target_vel.tolist()

        return result
