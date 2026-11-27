"""Engine-agnostic segment kinetic-energy overlay contracts (#15389)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

MUJOCO_SYSTEM_KE_RTOL: float = 1e-9
MUJOCO_SYSTEM_KE_ATOL: float = 1e-12
PEAK_HIGHLIGHT_FRACTION: float = 0.98


@dataclass(frozen=True)
class SegmentKineticEnergySample:
    """Per-body kinetic energy at one instant."""

    body: str
    body_index: int
    com_m: tuple[float, float, float]
    translational_j: float
    rotational_j: float
    total_j: float

    def __post_init__(self) -> None:
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("body must be a non-empty string")
        if self.body_index < 0:
            raise ValueError("body_index must be non-negative")
        if len(self.com_m) != 3:
            raise ValueError("com_m must have length 3")
        for label, value in (
            ("translational_j", self.translational_j),
            ("rotational_j", self.rotational_j),
            ("total_j", self.total_j),
        ):
            if not isinstance(value, (int, float)) or value < -1e-15:
                raise ValueError(f"{label} must be a non-negative number")
        total = self.translational_j + self.rotational_j
        if abs(total - self.total_j) > max(1e-9, 1e-9 * max(total, self.total_j)):
            raise ValueError("translational_j + rotational_j must equal total_j")


@dataclass(frozen=True)
class SegmentKineticEnergyFrame:
    """All segment KE samples at one simulation time."""

    schema_version: int
    time_s: float
    engine: str
    segments: tuple[SegmentKineticEnergySample, ...]
    system_kinetic_j: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")


@dataclass(frozen=True)
class SegmentKineticEnergyPeak:
    """Peak total segment KE over a recorded swing."""

    body: str
    peak_total_j: float
    peak_time_s: float


@runtime_checkable
class SegmentKineticEnergyProvider(Protocol):
    """Sample per-segment kinetic energy from a loaded engine state."""

    ENGINE: ClassVar[str]

    def frame_at(
        self,
        qpos: object,
        qvel: object,
        *,
        time_s: float = 0.0,
    ) -> SegmentKineticEnergyFrame:
        """Return per-body translational, rotational and total KE in joules."""
        ...


def read_segment_kinetic_energy_frame(
    provider: SegmentKineticEnergyProvider,
    qpos: object,
    qvel: object,
    *,
    time_s: float = 0.0,
) -> SegmentKineticEnergyFrame:
    """Thin accessor mirroring :func:`force_overlay.read_force_torque_frame`."""
    return provider.frame_at(qpos, qvel, time_s=time_s)


__all__ = [
    "MUJOCO_SYSTEM_KE_ATOL",
    "MUJOCO_SYSTEM_KE_RTOL",
    "PEAK_HIGHLIGHT_FRACTION",
    "SegmentKineticEnergyFrame",
    "SegmentKineticEnergyPeak",
    "SegmentKineticEnergyProvider",
    "SegmentKineticEnergySample",
    "read_segment_kinetic_energy_frame",
]
