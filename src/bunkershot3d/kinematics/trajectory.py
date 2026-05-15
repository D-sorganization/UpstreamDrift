"""
Trajectory parsing and prescription for clubhead kinematics.
"""

from pathlib import Path
import numpy as np
import pandas as pd


class SwingTrajectory:
    """Represents a prescribed swing trajectory for the clubhead."""

    def __init__(
        self,
        time: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
        lin_vel: np.ndarray,
        ang_vel: np.ndarray,
    ) -> None:
        """
        Initialize the trajectory.
        Args:
            time: (N,) array of time points
            positions: (N, 3) array of positions
            quaternions: (N, 4) array of quaternions (w, x, y, z)
            lin_vel: (N, 3) array of linear velocities
            ang_vel: (N, 3) array of angular velocities
        """
        self.time = time
        self.positions = positions
        self.quaternions = quaternions
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

    @classmethod
    def from_csv(cls, filepath: Path | str) -> "SwingTrajectory":
        """Load trajectory from a CSV file."""
        df = pd.read_csv(filepath)
        required_cols = [
            "time",
            "px",
            "py",
            "pz",
            "qw",
            "qx",
            "qy",
            "qz",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in trajectory CSV.")

        time = df["time"].values
        positions = df[["px", "py", "pz"]].values
        quaternions = df[["qw", "qx", "qy", "qz"]].values
        lin_vel = df[["vx", "vy", "vz"]].values
        ang_vel = df[["wx", "wy", "wz"]].values

        return cls(time, positions, quaternions, lin_vel, ang_vel)

    def interpolate(
        self, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate the trajectory at a given time t.
        Returns:
            position (3,), quaternion (4,), lin_vel (3,), ang_vel (3,)
        """
        # Linear interpolation for position, lin_vel, ang_vel
        # Spherical linear interpolation (SLERP) is ideal for quat, but linear is a first-draft approx

        # Clip time to bounds
        t = np.clip(t, self.time[0], self.time[-1])

        pos = np.zeros(3)
        for i in range(3):
            pos[i] = np.interp(t, self.time, self.positions[:, i])

        lvel = np.zeros(3)
        for i in range(3):
            lvel[i] = np.interp(t, self.time, self.lin_vel[:, i])

        avel = np.zeros(3)
        for i in range(3):
            avel[i] = np.interp(t, self.time, self.ang_vel[:, i])

        # Naive quaternion interpolation (should normalize)
        quat = np.zeros(4)
        for i in range(4):
            quat[i] = np.interp(t, self.time, self.quaternions[:, i])
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm

        return pos, quat, lvel, avel


def generate_reference_trajectory(filepath: Path | str) -> None:
    """Generate a mock reference trajectory (tour-pro bunker shot) and save to CSV."""
    # Clubhead speed ~25 m/s at impact
    # Attack angle -7 deg, Face open 8 deg

    t = np.linspace(0, 0.1, 100)  # 100ms swing segment

    # Simple straight line approach with constant velocity for draft
    velocity = np.array(
        [25.0 * np.cos(np.radians(-7)), 0.0, 25.0 * np.sin(np.radians(-7))]
    )

    positions = np.zeros((100, 3))
    for i, time_pt in enumerate(t):
        # start a bit before origin (impact at origin)
        positions[i] = velocity * (time_pt - 0.05)

    # Constant orientation (face open 8 deg)
    # y-axis rotation
    theta = np.radians(8.0)
    qw = np.cos(theta / 2)
    qx, qy, qz = 0.0, np.sin(theta / 2), 0.0
    quaternions = np.tile([qw, qx, qy, qz], (100, 1))

    lin_vel = np.tile(velocity, (100, 1))
    ang_vel = np.zeros((100, 3))

    df = pd.DataFrame(
        {
            "time": t,
            "px": positions[:, 0],
            "py": positions[:, 1],
            "pz": positions[:, 2],
            "qw": quaternions[:, 0],
            "qx": quaternions[:, 1],
            "qy": quaternions[:, 2],
            "qz": quaternions[:, 3],
            "vx": lin_vel[:, 0],
            "vy": lin_vel[:, 1],
            "vz": lin_vel[:, 2],
            "wx": ang_vel[:, 0],
            "wy": ang_vel[:, 1],
            "wz": ang_vel[:, 2],
        }
    )

    df.to_csv(filepath, index=False)
