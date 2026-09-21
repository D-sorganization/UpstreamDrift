"""Shared rigid-ground contact law for full-body engine variants (epic #10062).

One explicit law that every engine can apply as an external force, so that
MuJoCo, Drake and Pinocchio full-body models compare like for like: a compliant
normal force with Hunt-Crossley damping form, ``f_n = k * d * (1 + c * d_dot)``
clipped at zero, and a regularised Coulomb friction with viscous term,
``|f_t| = (mu(v) ) * f_n * tanh(|v_t| / v_transition)`` where
``mu(v) = mu_d + (mu_s - mu_d) * exp(-(|v_t| / v_transition)^2) + mu_v * |v_t|``.
Engines' stock contact solvers are not equivalent to this law by default; the
parity harness measures the difference on identical states.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

CONFORMANCE_VERSION: str = "1.0.0"
"""Version of the shared contact law and grip closure conformance specification (MS-72)."""


@dataclass(frozen=True)
class ContactParameters:
    """Positive stiffness and dissipation; friction coefficients ordered."""

    stiffness_n_m: float
    dissipation_s_m: float
    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity_m_s: float

    def __post_init__(self) -> None:
        values = (
            self.stiffness_n_m,
            self.dissipation_s_m,
            self.static_friction,
            self.dynamic_friction,
            self.viscous_friction,
            self.transition_velocity_m_s,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Contact parameters must be finite")
        if self.stiffness_n_m <= 0 or self.transition_velocity_m_s <= 0:
            raise ValueError("Stiffness and transition velocity must be positive")
        if min(self.dissipation_s_m, self.dynamic_friction, self.viscous_friction) < 0:
            raise ValueError(
                "Dissipation and friction coefficients must be nonnegative"
            )
        if self.static_friction < self.dynamic_friction:
            raise ValueError("Static friction must not be below dynamic friction")

    def as_document(self) -> dict[str, float]:
        return {
            "stiffness_n_m": self.stiffness_n_m,
            "dissipation_s_m": self.dissipation_s_m,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "viscous_friction": self.viscous_friction,
            "transition_velocity_m_s": self.transition_velocity_m_s,
        }


@dataclass(frozen=True)
class GroundPlane:
    """A plane with unit normal and signed height along that normal."""

    normal: tuple[float, float, float]
    height_m: float

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
            raise ValueError("Ground normal must be a finite nonzero 3-vector")
        if not math.isfinite(self.height_m):
            raise ValueError("Ground height must be finite")
        unit = n / np.linalg.norm(n)
        object.__setattr__(
            self, "normal", (float(unit[0]), float(unit[1]), float(unit[2]))
        )


class ContactSample(NamedTuple):
    """Per-sphere contact outcome in world coordinates."""

    penetration_m: float
    penetration_rate_m_s: float
    contact_point_m: Array
    normal_force_n: Array
    friction_force_n: Array


def _vector(value: Any, name: str) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return v


def sphere_ground_contact(
    center_m: Array,
    velocity_m_s: Array,
    radius_m: float,
    ground: GroundPlane,
    parameters: ContactParameters,
) -> ContactSample:
    """Evaluate the shared law for one sphere against the ground plane.

    Postconditions: normal force is along the plane normal and nonnegative;
    friction lies in the plane and opposes the tangential velocity; both are
    zero without penetration.
    """
    center = _vector(center_m, "center_m")
    velocity = _vector(velocity_m_s, "velocity_m_s")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Sphere radius must be positive")
    normal = np.asarray(ground.normal)
    signed_distance = float(normal @ center) - ground.height_m - radius_m
    penetration = max(0.0, -signed_distance)
    rate = float(-(normal @ velocity))  # positive when deepening
    point = center - normal * radius_m  # deepest point of the sphere
    zero = np.zeros(3)
    if penetration <= 0.0:
        return ContactSample(0.0, rate, point, zero, zero)
    magnitude = (
        parameters.stiffness_n_m
        * penetration
        * (1.0 + parameters.dissipation_s_m * rate)
    )
    magnitude = max(0.0, magnitude)
    normal_force = normal * magnitude
    tangential = velocity - normal * float(normal @ velocity)
    speed = float(np.linalg.norm(tangential))
    if speed <= 0.0 or magnitude <= 0.0:
        return ContactSample(penetration, rate, point, normal_force, zero)
    ratio = speed / parameters.transition_velocity_m_s
    mu = (
        parameters.dynamic_friction
        + (parameters.static_friction - parameters.dynamic_friction)
        * math.exp(-(ratio**2))
        + parameters.viscous_friction * speed
    )
    friction = -tangential / speed * mu * magnitude * math.tanh(ratio)
    return ContactSample(penetration, rate, point, normal_force, friction)


ContactAdapter = Callable[[Array, Array, float], ContactSample]


def random_contact_states(
    *, seed: int, count: int, radius: float
) -> list[tuple[Array, Array]]:
    """Deterministic (center, velocity) pairs straddling the ground plane."""
    if count <= 0 or radius <= 0:
        raise ValueError("count and radius must be positive")
    rng = np.random.default_rng(seed)
    centers = rng.uniform(
        [-0.5, -0.5, -radius], [0.5, 0.5, 2 * radius], size=(count, 3)
    )
    velocities = rng.normal(scale=1.0, size=(count, 3))
    return [(centers[i], velocities[i]) for i in range(count)]


def contact_parity_report(
    adapters: Mapping[str, ContactAdapter],
    states: Sequence[tuple[Array, Array]],
    *,
    radius: float,
) -> dict[str, Any]:
    """Compare adapters against the first-named one on identical states.

    Requires at least two adapters. Reports the maximum normal and friction
    force differences per adapter and how many states penetrated under the
    reference; differences are absolute newtons, never relative to zero.
    """
    names = list(adapters)
    if len(names) < 2 or not states:
        raise ValueError(
            "Parity needs a reference, at least one other adapter and states"
        )
    reference = names[0]
    normal_diff = dict.fromkeys(names[1:], 0.0)
    friction_diff = dict.fromkeys(names[1:], 0.0)
    penetrating = 0
    for center, velocity in states:
        base = adapters[reference](center, velocity, radius)
        penetrating += int(base.penetration_m > 0)
        for name in names[1:]:
            other = adapters[name](center, velocity, radius)
            normal_diff[name] = max(
                normal_diff[name],
                float(np.linalg.norm(other.normal_force_n - base.normal_force_n)),
            )
            friction_diff[name] = max(
                friction_diff[name],
                float(np.linalg.norm(other.friction_force_n - base.friction_force_n)),
            )
    return {
        "conformance_version": CONFORMANCE_VERSION,
        "reference": reference,
        "states": len(states),
        "penetrating_states": penetrating,
        "max_normal_force_difference_n": normal_diff,
        "max_friction_force_difference_n": friction_diff,
    }


def calibrate_ground_height_at_address(model: Any, address_q: Array) -> float:
    """Calibrate ground plane height to the lowest contact sphere surface at address."""
    q_dict = {
        name: float(address_q[i]) for i, name in enumerate(model.coordinate_order)
    }
    if hasattr(model, "_mj"):
        model.data.qpos[:] = model._vector(q_dict)
        model._mj.mj_fwdPosition(model.model, model.data)
        min_z = min(
            model.data.site_xpos[s_info["site_id"]][2] - s_info["radius"]
            for s_info in model._spheres.values()
        )
        return float(min_z)
    if hasattr(model, "_pin"):
        q = model.configuration(q_dict)
        model._pin.forwardKinematics(model.model, model.data, q)
        model._pin.updateFramePlacements(model.model, model.data)
        min_z = min(
            float(
                model.data.oMf[model._contact_frames[s.name]].translation[2]
                - s.radius_m
            )
            for s in model.contact_spheres
        )
        return float(min_z)
    if hasattr(model, "plant"):
        model.plant.SetPositions(model.context, model._vector(q_dict, model._q_indices))
        min_z = min(
            float(
                model.plant.CalcPointsPositions(
                    model.context,
                    s_info["frame"],
                    np.zeros(3),
                    model.plant.world_frame(),
                )[2, 0]
                - s_info["radius_m"]
            )
            for s_info in model._spheres.values()
        )
        return float(min_z)
    return 0.0
