"""Common force, couple, power, and constraint contracts for model tiers."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]


def _vector3(name: str, value: npt.ArrayLike) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class InteractionSample:
    """One reference-explicit interaction wrench and its power kinematics.

    The force and couple are the action on the declared distal subsystem. The
    linear and angular velocities belong to the same reference point and frame.
    This makes force power, couple power, and reference transport unambiguous.
    """

    model_tier: str
    time_s: float
    frame: str
    reference_point_m: Vector
    force_n: Vector
    couple_nm: Vector
    linear_velocity_m_s: Vector
    angular_velocity_rad_s: Vector

    def __post_init__(self) -> None:
        if not self.model_tier.strip():
            raise ValueError("model_tier must be non-empty")
        if not self.frame.strip():
            raise ValueError("frame must be non-empty")
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        for name in (
            "reference_point_m",
            "force_n",
            "couple_nm",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
        ):
            object.__setattr__(self, name, _vector3(name, getattr(self, name)))

    @property
    def force_power_w(self) -> float:
        """Return translational force power at the declared point."""
        return float(self.force_n @ self.linear_velocity_m_s)

    @property
    def couple_power_w(self) -> float:
        """Return rotational couple power at the declared point."""
        return float(self.couple_nm @ self.angular_velocity_rad_s)

    @property
    def total_power_w(self) -> float:
        """Return total wrench power."""
        return self.force_power_w + self.couple_power_w

    def transport(self, new_reference_point_m: npt.ArrayLike) -> InteractionSample:
        """Transport the wrench and point velocity without changing its power."""
        new_point = _vector3("new_reference_point_m", new_reference_point_m)
        offset = new_point - self.reference_point_m
        new_couple = self.couple_nm - np.cross(offset, self.force_n)
        new_velocity = self.linear_velocity_m_s + np.cross(
            self.angular_velocity_rad_s, offset
        )
        return replace(
            self,
            reference_point_m=new_point,
            couple_nm=new_couple,
            linear_velocity_m_s=new_velocity,
        )

    def rotate(self, transform: npt.ArrayLike, *, frame: str) -> InteractionSample:
        """Express the complete sample in a proper rotated Cartesian frame."""
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"transform must have shape (3, 3), got {matrix.shape}")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12):
            raise ValueError("transform must be orthogonal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12):
            raise ValueError("transform must be a proper rotation")
        return replace(
            self,
            frame=frame,
            reference_point_m=matrix @ self.reference_point_m,
            force_n=matrix @ self.force_n,
            couple_nm=matrix @ self.couple_nm,
            linear_velocity_m_s=matrix @ self.linear_velocity_m_s,
            angular_velocity_rad_s=matrix @ self.angular_velocity_rad_s,
        )

    def as_record(self) -> dict[str, object]:
        """Return a JSON-safe record with derived power fields."""
        return {
            "model_tier": self.model_tier,
            "time_s": self.time_s,
            "frame": self.frame,
            "reference_point_m": self.reference_point_m.tolist(),
            "force_n": self.force_n.tolist(),
            "couple_nm": self.couple_nm.tolist(),
            "linear_velocity_m_s": self.linear_velocity_m_s.tolist(),
            "angular_velocity_rad_s": self.angular_velocity_rad_s.tolist(),
            "force_power_w": self.force_power_w,
            "couple_power_w": self.couple_power_w,
            "total_power_w": self.total_power_w,
        }


def rotation_matrix(axis: npt.ArrayLike, angle_rad: float) -> npt.NDArray[np.float64]:
    """Return the right-handed Rodrigues rotation for a finite axis and angle."""
    unit = _vector3("axis", axis)
    norm = float(np.linalg.norm(unit))
    if norm <= 0.0:
        raise ValueError("axis must have nonzero length")
    if not np.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    unit = unit / norm
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return (
        identity + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )


def embed_planar_sample(
    *,
    model_tier: str,
    time_s: float,
    reference_point_xy_m: npt.ArrayLike,
    force_xy_n: npt.ArrayLike,
    couple_z_nm: float,
    linear_velocity_xy_m_s: npt.ArrayLike,
    angular_velocity_z_rad_s: float,
) -> InteractionSample:
    """Embed a planar wrench and its velocities into the common 3-D schema."""

    def xy(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        return vector

    point = xy("reference_point_xy_m", reference_point_xy_m)
    force = xy("force_xy_n", force_xy_n)
    velocity = xy("linear_velocity_xy_m_s", linear_velocity_xy_m_s)
    if not np.isfinite(couple_z_nm) or not np.isfinite(angular_velocity_z_rad_s):
        raise ValueError("planar couple and angular velocity must be finite")
    return InteractionSample(
        model_tier=model_tier,
        time_s=time_s,
        frame="planar-world",
        reference_point_m=np.array([point[0], point[1], 0.0]),
        force_n=np.array([force[0], force[1], 0.0]),
        couple_nm=np.array([0.0, 0.0, couple_z_nm]),
        linear_velocity_m_s=np.array([velocity[0], velocity[1], 0.0]),
        angular_velocity_rad_s=np.array([0.0, 0.0, angular_velocity_z_rad_s]),
    )


def mobile_hub_force_shift(
    supported_mass_kg: float, hub_acceleration_m_s2: npt.ArrayLike
) -> Vector:
    """Return the force increment required by a prescribed hub acceleration."""
    if not np.isfinite(supported_mass_kg) or supported_mass_kg <= 0.0:
        raise ValueError("supported_mass_kg must be positive and finite")
    return supported_mass_kg * _vector3("hub_acceleration_m_s2", hub_acceleration_m_s2)


def closed_loop_grip_jacobian(
    *,
    lead_angle_rad: float,
    trail_angle_rad: float,
    grip_angle_rad: float,
    lead_arm_length_m: float,
    trail_arm_length_m: float,
    grip_separation_m: float,
) -> npt.NDArray[np.float64]:
    """Return the planar two-hand loop-closure Jacobian.

    Coordinates are lead-arm angle, trail-arm angle, grip-center x/y, and grip
    angle. Four rows constrain both hand endpoints to their declared grip
    contacts, leaving one feasible closed-loop degree of freedom at regular
    configurations.
    """
    values = np.array(
        [
            lead_angle_rad,
            trail_angle_rad,
            grip_angle_rad,
            lead_arm_length_m,
            trail_arm_length_m,
            grip_separation_m,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("grip geometry must be finite")
    if lead_arm_length_m <= 0.0 or trail_arm_length_m <= 0.0:
        raise ValueError("arm lengths must be positive")
    if grip_separation_m <= 0.0:
        raise ValueError("grip_separation_m must be positive")

    half = 0.5 * grip_separation_m
    sin_grip = np.sin(grip_angle_rad)
    cos_grip = np.cos(grip_angle_rad)
    return np.array(
        [
            [
                lead_arm_length_m * np.cos(lead_angle_rad),
                0.0,
                -1.0,
                0.0,
                -half * sin_grip,
            ],
            [
                lead_arm_length_m * np.sin(lead_angle_rad),
                0.0,
                0.0,
                -1.0,
                half * cos_grip,
            ],
            [
                0.0,
                trail_arm_length_m * np.cos(trail_angle_rad),
                -1.0,
                0.0,
                half * sin_grip,
            ],
            [
                0.0,
                trail_arm_length_m * np.sin(trail_angle_rad),
                0.0,
                -1.0,
                -half * cos_grip,
            ],
        ]
    )
