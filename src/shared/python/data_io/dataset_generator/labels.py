"""Native dataset label contracts for NM-02 (#10617).

Fail-closed evidence metadata for DatasetGenerator channels. Unavailable
optional fields must not be presented as zero measurements. Instantaneous
native acceleration is distinct from interval finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "LABEL_SCHEMA",
    "AccelerationKind",
    "ActuationKind",
    "ChannelAvailability",
    "ChannelEvidence",
    "ModelDoFLayout",
    "SampleProvenance",
    "dynamics_residual",
    "require_finite_array",
]

LABEL_SCHEMA = "native-dataset-labels/1.0.0"


class ChannelAvailability(str, Enum):
    """Whether a requested channel carries trustworthy values."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_REQUESTED = "not_requested"
    PARTIAL = "partial"


class AccelerationKind(str, Enum):
    """Semantic meaning of an acceleration buffer."""

    INSTANTANEOUS_NATIVE = "instantaneous_native"
    INTERVAL_FINITE_DIFFERENCE = "interval_finite_difference"
    UNKNOWN = "unknown"


class ActuationKind(str, Enum):
    """Whether recorded controls are requested or applied after saturation."""

    REQUESTED = "requested"
    APPLIED = "applied"
    APPLIED_SATURATED = "applied_saturated"
    MUSCLE_EXCITATION = "muscle_excitation"
    MUSCLE_ACTIVATION = "muscle_activation"


def require_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    """Reject non-finite arrays under python -O (no decorator reliance)."""
    if values is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} values must be finite")
    return arr


@dataclass(frozen=True)
class ModelDoFLayout:
    """Separate configuration, tangent, actuator and force dimensions.

    Design by Contract:
    - All counts are non-negative; ``n_q`` and ``n_v`` must be > 0.
    - ``n_q`` may differ from ``n_v`` (e.g. quaternion free joints).
    - ``n_u`` may differ from ``n_v`` (under-actuated or mapped actuators).
    """

    n_q: int
    n_v: int
    n_u: int
    n_force: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("n_q", self.n_q),
            ("n_v", self.n_v),
            ("n_u", self.n_u),
            ("n_force", self.n_force),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.n_q <= 0 or self.n_v <= 0:
            raise ValueError("n_q and n_v must be > 0")

    def ensure_physically_supervised(
        self,
        *,
        allow_unqualified_root_forces: bool,
        has_root_force_channels: bool,
    ) -> None:
        """Reject unqualified root-force channels in supervised corpora."""
        if has_root_force_channels and not allow_unqualified_root_forces:
            raise ValueError(
                "unqualified root-force channels are disallowed in "
                "physically supervised native dataset labels"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "n_q": self.n_q,
            "n_v": self.n_v,
            "n_u": self.n_u,
            "n_force": self.n_force,
        }


@dataclass(frozen=True)
class ChannelEvidence:
    """Availability and semantic metadata for one recorded channel."""

    name: str
    availability: ChannelAvailability
    semantic: str
    units: str
    shape: tuple[int, ...] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must be non-empty")
        if not self.semantic:
            raise ValueError("channel semantic must be non-empty")
        if self.availability is ChannelAvailability.AVAILABLE and self.shape is None:
            raise ValueError("AVAILABLE channels must declare shape")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "availability": self.availability.value,
            "semantic": self.semantic,
            "units": self.units,
            "shape": list(self.shape) if self.shape is not None else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SampleProvenance:
    """Settings, runtime and model identity attached to a sample."""

    model_name: str
    engine_name: str
    model_hash: str
    native_runtime: str
    settings_digest: str
    numerical_refinement: str = "none"

    def __post_init__(self) -> None:
        for name, value in (
            ("model_name", self.model_name),
            ("engine_name", self.engine_name),
            ("model_hash", self.model_hash),
            ("native_runtime", self.native_runtime),
            ("settings_digest", self.settings_digest),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "engine_name": self.engine_name,
            "model_hash": self.model_hash,
            "native_runtime": self.native_runtime,
            "settings_digest": self.settings_digest,
            "numerical_refinement": self.numerical_refinement,
        }


def dynamics_residual(
    *,
    mass: np.ndarray,
    acceleration: np.ndarray,
    bias: np.ndarray,
    applied: np.ndarray,
    contact_generalized: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``M a + h - u - J^T lambda`` (contact term optional).

    When ``contact_generalized`` is omitted the residual is ``M a + h - u``.
    All inputs must be finite and dimensionally compatible.
    """
    m = require_finite_array("mass", mass)
    a = require_finite_array("acceleration", acceleration)
    h = require_finite_array("bias", bias)
    u = require_finite_array("applied", applied)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"mass must be square, got shape {m.shape}")
    n = m.shape[0]
    if a.shape != (n,) or h.shape != (n,) or u.shape != (n,):
        raise ValueError(
            f"acceleration/bias/applied must have shape ({n},); "
            f"got {a.shape}, {h.shape}, {u.shape}"
        )
    residual = m @ a + h - u
    if contact_generalized is not None:
        lam = require_finite_array("contact_generalized", contact_generalized)
        if lam.shape != (n,):
            raise ValueError(
                f"contact_generalized must have shape ({n},); got {lam.shape}"
            )
        residual = residual - lam
    return residual
