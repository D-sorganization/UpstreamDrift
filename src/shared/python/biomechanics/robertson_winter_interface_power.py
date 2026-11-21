"""Engine-agnostic Robertson–Winter interface power helpers (#15390)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import precondition

__all__ = [
    "InterfacePowerFrame",
    "InterfacePowerJointSample",
    "InterfacePowerSegment",
    "endpoint_interface_power_w",
    "segment_energy_balance_residual_w",
]


@dataclass(frozen=True)
class InterfacePowerJointSample:
    """Endpoint power at one parent/child interface."""

    joint: str
    joint_index: int
    child_body: str
    child_body_index: int
    parent_body: str
    parent_body_index: int
    power_on_child_w: float
    power_on_parent_w: float
    force_power_on_child_w: float
    moment_power_on_child_w: float
    joint_coordinate_power_w: float


@dataclass(frozen=True)
class InterfacePowerSegment:
    """Sum of RW interface powers on one body plus energy-balance residual."""

    body: str
    body_index: int
    mass_kg: float
    total_interface_power_w: float
    external_power_w: float
    actuator_power_w: float
    mechanical_energy_rate_w: float
    balance_residual_w: float


@dataclass(frozen=True)
class InterfacePowerFrame:
    """Joint-level and per-body interface power at one simulation time."""

    schema_version: int
    time_s: float
    engine: str
    joints: tuple[InterfacePowerJointSample, ...]
    segments: tuple[InterfacePowerSegment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if not self.engine.strip():
            raise ValueError("engine must be a non-empty string")


@precondition(
    lambda f, m, lv, av: len(f) == len(m) == len(lv) == len(av) == 3,
    "force, moment and velocities must be 3-vectors",
)
def endpoint_interface_power_w(
    force_n: np.ndarray,
    moment_nm: np.ndarray,
    linear_velocity_m_s: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
) -> float:
    """Robertson–Winter endpoint power ``F · v + M · ω`` [W]."""
    return float(
        np.dot(force_n, linear_velocity_m_s) + np.dot(moment_nm, angular_velocity_rad_s)
    )


def segment_energy_balance_residual_w(
    *,
    mechanical_energy_rate_w: float,
    interface_power_w: float,
    external_power_w: float,
    actuator_power_w: float = 0.0,
) -> float:
    """``d(KE+PE)/dt − (interface + external + actuator)`` [W]."""
    accounted = interface_power_w + external_power_w + actuator_power_w
    return float(mechanical_energy_rate_w - accounted)
