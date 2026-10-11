"""Explicit target-line frame for impact-parameter extraction (GCV-15)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_UNIT_TOL = 1e-9
HANDEDNESS = ("right", "left")

#: ADR-0041 / UD default: Z-up world, golfer faces -X, target line along -Y.
UD_DEFAULT_TARGET_DIR: tuple[float, float, float] = (0.0, -1.0, 0.0)
UD_DEFAULT_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)
UD_DEFAULT_FRAME_ID = "ud_world:z_up,target_-y(adr-0041)"


def _vec3(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class TargetFrame:
    """Target-line frame: x_t = target, z_t = up, y_t = z_t x x_t.

    ``y_t`` points left of the target for a right-handed golfer.  The frame
    (including the default) is recorded with every result.

    Attributes:
        target_dir: Horizontal unit vector toward the target (world frame).
        up: Unit up vector, orthogonal to ``target_dir``.
        ball_m: Ball position (world frame, metres).
        handedness: ``"right"`` or ``"left"``; left mirrors lateral signs only.
        ground_height_m: Ground height along ``up`` (metres).
        frame_id: Human-readable frame label recorded in results.
    """

    target_dir: tuple[float, float, float] = UD_DEFAULT_TARGET_DIR
    up: tuple[float, float, float] = UD_DEFAULT_UP
    ball_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    handedness: str = "right"
    ground_height_m: float = 0.0
    frame_id: str = field(default=UD_DEFAULT_FRAME_ID)

    def __post_init__(self) -> None:
        if self.handedness not in HANDEDNESS:
            raise ValueError(f"handedness must be one of {HANDEDNESS}")
        target = _vec3(self.target_dir, "target_dir")
        up = _vec3(self.up, "up")
        _vec3(self.ball_m, "ball_m")
        if abs(float(np.linalg.norm(target)) - 1.0) > _UNIT_TOL:
            raise ValueError("target_dir must be a unit vector")
        if abs(float(np.linalg.norm(up)) - 1.0) > _UNIT_TOL:
            raise ValueError("up must be a unit vector")
        if abs(float(target @ up)) > _UNIT_TOL:
            raise ValueError("target_dir must be horizontal (orthogonal to up)")
        if not np.isfinite(self.ground_height_m):
            raise ValueError("ground_height_m must be finite")

    @property
    def x_t(self) -> np.ndarray:
        """Target axis."""
        return np.asarray(self.target_dir, dtype=float)

    @property
    def z_t(self) -> np.ndarray:
        """Up axis."""
        return np.asarray(self.up, dtype=float)

    @property
    def y_t(self) -> np.ndarray:
        """Left-of-target axis for a right-handed golfer (z_t x x_t)."""
        return np.cross(self.z_t, self.x_t)

    @property
    def lateral_sign(self) -> float:
        """+1 for right-handed, -1 for left-handed (mirrors lateral signs)."""
        return 1.0 if self.handedness == "right" else -1.0

    def components(self, vector: object) -> tuple[float, float, float]:
        """Return ``(v.x_t, v.y_t, v.z_t)`` of a world-frame vector."""
        v = _vec3(vector, "vector")
        return float(v @ self.x_t), float(v @ self.y_t), float(v @ self.z_t)

    def to_record(self) -> dict[str, object]:
        """JSON-serialisable record stored with each result."""
        return {
            "frame_id": self.frame_id,
            "target_dir": list(self.target_dir),
            "up": list(self.up),
            "ball_m": list(self.ball_m),
            "handedness": self.handedness,
            "ground_height_m": self.ground_height_m,
        }
