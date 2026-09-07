"""A rigid 15-joint skeleton with fixed segment lengths and a swing-like motion.

The joint set is the intersection of the MediaPipe and BODY_25 layouts the
detectors report (plus ``mid_hip``/``neck``), so synthetic observations use
the same names ingest writes. Segment lengths are the only shape parameters;
poses are per-joint rotations applied to fixed parent offsets, which is the
model the geometric IK backend (``motion_pipeline/ik``) assumes. The motion is
a smooth, bounded caricature of a swing for evaluation only — nothing about
it claims to be biomechanically accurate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]

JOINT_NAMES: tuple[str, ...] = (
    "mid_hip",
    "neck",
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)
PARENTS: dict[str, str | None] = {
    "mid_hip": None,
    "neck": "mid_hip",
    "nose": "neck",
    "left_shoulder": "neck",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "neck",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "mid_hip",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "right_hip": "mid_hip",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
}
# Rest-pose offset direction of each joint from its parent (unit vectors in the
# world frame at T-pose: x toward target, y up, z to the golfer's right).
_REST_DIRECTION: dict[str, tuple[float, float, float]] = {
    "neck": (0, 1, 0),
    "nose": (0, 1, 0),
    "left_shoulder": (0, 0, -1),
    "left_elbow": (0, -1, 0),
    "left_wrist": (0, -1, 0),
    "right_shoulder": (0, 0, 1),
    "right_elbow": (0, -1, 0),
    "right_wrist": (0, -1, 0),
    "left_hip": (0, 0, -1),
    "left_knee": (0, -1, 0),
    "left_ankle": (0, -1, 0),
    "right_hip": (0, 0, 1),
    "right_knee": (0, -1, 0),
    "right_ankle": (0, -1, 0),
}
DEFAULT_LENGTHS_M: dict[str, float] = {
    "neck": 0.50,
    "nose": 0.18,
    "left_shoulder": 0.20,
    "left_elbow": 0.30,
    "left_wrist": 0.26,
    "right_shoulder": 0.20,
    "right_elbow": 0.30,
    "right_wrist": 0.26,
    "left_hip": 0.10,
    "left_knee": 0.44,
    "left_ankle": 0.42,
    "right_hip": 0.10,
    "right_knee": 0.44,
    "right_ankle": 0.42,
}
SYMMETRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


def rotation_from_axis_angle(axis: Array, angle_rad: float) -> Array:
    """Rodrigues rotation matrix (right-handed, radians)."""
    a = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(a))
    require(norm > 1e-9, "axis must be non-zero")
    a = a / norm
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * (k @ k)


@dataclass(frozen=True)
class RigidSkeleton:
    """Fixed segment lengths; poses are rotations about each joint's parent."""

    lengths_m: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_LENGTHS_M)
    )

    def __post_init__(self) -> None:
        missing = [n for n in JOINT_NAMES[1:] if n not in self.lengths_m]
        require(not missing, "lengths missing for joints", missing)
        bad = [n for n, v in self.lengths_m.items() if v <= 0]
        require(not bad, "segment lengths must be positive", bad)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return JOINT_NAMES

    def bone_lengths(self) -> dict[str, float]:
        return {n: float(self.lengths_m[n]) for n in JOINT_NAMES[1:]}

    def forward(
        self,
        root_position_m: Array,
        rotations: Mapping[str, Array],
    ) -> Array:
        """Joint positions ``(K, 3)`` for one pose.

        ``rotations[joint]`` is the 3x3 rotation applied to the rest direction
        of every segment *below* that joint (inclusive), composed down the tree;
        omitted joints are identity. Postcondition: every segment keeps its
        length to floating precision.
        """
        root = np.asarray(root_position_m, dtype=float).reshape(3)
        positions: dict[str, Array] = {"mid_hip": root}
        frames: dict[str, Array] = {
            "mid_hip": np.asarray(rotations.get("mid_hip", np.eye(3)))
        }
        for name in JOINT_NAMES[1:]:
            parent = PARENTS[name]
            assert parent is not None
            local = np.asarray(rotations.get(name, np.eye(3)), dtype=float)
            frame = frames[parent] @ local
            direction = frame @ np.asarray(_REST_DIRECTION[name], dtype=float)
            positions[name] = positions[parent] + self.lengths_m[name] * direction
            frames[name] = frame
        return np.stack([positions[n] for n in JOINT_NAMES])


def swing_trajectory(
    n_frames: int,
    fps: float,
    *,
    amplitude_rad: float = 1.2,
    period_s: float = 2.0,
) -> tuple[Array, list[dict[str, Array]]]:
    """Root positions and per-frame rotations of a smooth swing caricature.

    The pelvis and torso rotate about the world y axis, the arms swing about
    the golfer's z axis with a phase lead, and the knees flex slightly, all as
    sinusoids over ``period_s``. Precondition: positive frame count and rate.
    """
    require(n_frames > 0 and fps > 0, "n_frames and fps must be positive")
    require(period_s > 0, "period_s must be positive")
    t = np.arange(n_frames) / fps
    phase = 2 * np.pi * t / period_s
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    roots = np.column_stack([0.02 * np.sin(phase), np.full_like(t, 0.95), 0.0 * t])
    rotations: list[dict[str, Array]] = []
    for k in range(n_frames):
        torso = amplitude_rad * 0.5 * np.sin(phase[k])
        arm = amplitude_rad * np.sin(phase[k] + 0.4)
        knee = 0.25 + 0.15 * np.sin(2 * phase[k])
        rotations.append(
            {
                "mid_hip": rotation_from_axis_angle(y_axis, torso * 0.6),
                "neck": rotation_from_axis_angle(y_axis, torso * 0.4),
                "left_shoulder": rotation_from_axis_angle(z_axis, arm),
                "right_shoulder": rotation_from_axis_angle(z_axis, arm * 0.9),
                "left_elbow": rotation_from_axis_angle(x_axis, 0.3 * arm),
                "right_elbow": rotation_from_axis_angle(x_axis, -0.3 * arm),
                "left_knee": rotation_from_axis_angle(z_axis, knee),
                "right_knee": rotation_from_axis_angle(z_axis, -knee),
            }
        )
    return roots, rotations
