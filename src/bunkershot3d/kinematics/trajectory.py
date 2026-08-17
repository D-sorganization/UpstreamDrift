"""
Trajectory parsing and prescription for clubhead kinematics.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.shared.python.core.contracts import require

#: Below this |dot| the two quaternions are effectively parallel and the SLERP
#: sines underflow; linear interpolation is then exact to machine precision.
_PARALLEL_DOT = 1.0 - 1.0e-9


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (w, x, y, z).

    Antipodal sign continuity is applied first: ``q`` and ``-q`` are the same
    rotation, so when ``q0 . q1 < 0`` the second quaternion is negated and the
    *short* arc is taken. Component-wise ``np.interp`` plus a renormalise
    (nlerp) does not do this — it interpolates through the long arc, and for a
    near-antipodal pair it produces a near-zero-norm quaternion whose
    normalisation is numerical noise (#8612, finding B8).

    Args:
        q0: Start quaternion, ``(4,)``, need not be unit.
        q1: End quaternion, ``(4,)``, need not be unit.
        fraction: Interpolation parameter, clamped to ``[0, 1]``.

    Returns:
        A unit quaternion, ``(4,)``.

    Raises:
        ValueError: Either quaternion has (near) zero norm.
    """
    q0_array = np.asarray(q0, dtype=float).reshape(4)
    q1_array = np.asarray(q1, dtype=float).reshape(4)

    norm_start = float(np.linalg.norm(q0_array))
    norm_end = float(np.linalg.norm(q1_array))
    if norm_start < 1.0e-12 or norm_end < 1.0e-12:
        raise ValueError("cannot interpolate a zero-norm quaternion")
    start = q0_array / norm_start
    end = q1_array / norm_end

    fraction = float(np.clip(fraction, 0.0, 1.0))

    dot = float(np.dot(start, end))
    if dot < 0.0:  # antipodal representation: take the short arc
        end = -end
        dot = -dot

    if dot > _PARALLEL_DOT:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    weight_start = np.sin((1.0 - fraction) * theta) / sin_theta
    weight_end = np.sin(fraction * theta) / sin_theta
    result = weight_start * start + weight_end * end
    return result / np.linalg.norm(result)


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
        time = np.asarray(time, dtype=float)
        require(time.ndim == 1 and time.size >= 1, "time must be a non-empty 1-D array")
        require(
            bool(np.all(np.diff(time) >= 0.0)),
            "trajectory time samples must be non-decreasing",
        )
        for name, array in (
            ("positions", positions),
            ("lin_vel", lin_vel),
            ("ang_vel", ang_vel),
        ):
            require(
                np.shape(array) == (time.size, 3),
                f"{name} must have shape (N, 3) matching time",
                np.shape(array),
            )
        require(
            np.shape(quaternions) == (time.size, 4),
            "quaternions must have shape (N, 4) matching time",
            np.shape(quaternions),
        )

        self.time = time
        self.positions = np.asarray(positions, dtype=float)
        self.quaternions = np.asarray(quaternions, dtype=float)
        self.lin_vel = np.asarray(lin_vel, dtype=float)
        self.ang_vel = np.asarray(ang_vel, dtype=float)

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

        time = df["time"].to_numpy(dtype=float)
        positions = df[["px", "py", "pz"]].to_numpy(dtype=float)
        quaternions = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
        lin_vel = df[["vx", "vy", "vz"]].to_numpy(dtype=float)
        ang_vel = df[["wx", "wy", "wz"]].to_numpy(dtype=float)

        return cls(time, positions, quaternions, lin_vel, ang_vel)

    @property
    def duration(self) -> float:
        """Span of the trajectory in seconds."""
        return float(self.time[-1] - self.time[0])

    def max_linear_speed(self) -> float:
        """Largest sampled linear speed (m/s).

        Used by the backends to derive the Courant timestep limit, so that the
        integration step follows the swing rather than the output sample rate.
        """
        return float(np.max(np.linalg.norm(self.lin_vel, axis=1)))

    def interpolate(
        self, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate the trajectory at time ``t`` (clamped to its range).

        Positions and velocities are interpolated linearly; the orientation
        uses SLERP with antipodal sign continuity (:func:`slerp`), because
        component-wise interpolation of a quaternion takes the wrong arc
        whenever successive samples are stored with opposite signs (#8612, B8).

        Returns:
            position (3,), quaternion (4,), lin_vel (3,), ang_vel (3,)
        """
        t = float(np.clip(t, self.time[0], self.time[-1]))

        pos = np.array(
            [np.interp(t, self.time, self.positions[:, i]) for i in range(3)]
        )
        lvel = np.array([np.interp(t, self.time, self.lin_vel[:, i]) for i in range(3)])
        avel = np.array([np.interp(t, self.time, self.ang_vel[:, i]) for i in range(3)])

        quat = self._interpolate_orientation(t)
        return pos, quat, lvel, avel

    def _interpolate_orientation(self, t: float) -> np.ndarray:
        """SLERP the stored quaternions at time ``t``."""
        if self.time.size == 1:
            quat = self.quaternions[0]
            return quat / np.linalg.norm(quat)

        index = int(np.searchsorted(self.time, t, side="right")) - 1
        index = int(np.clip(index, 0, self.time.size - 2))

        span = self.time[index + 1] - self.time[index]
        fraction = 0.0 if span <= 0.0 else (t - self.time[index]) / span
        return slerp(self.quaternions[index], self.quaternions[index + 1], fraction)


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
