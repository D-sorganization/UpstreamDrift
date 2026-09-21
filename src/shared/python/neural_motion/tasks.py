"""Learning-task contracts for neural motion matching (NM-01 #10616).

Three supervised tasks are distinct:

* **Forward dynamics** — ``(q, v, u, geometry, contact, dt?) -> a | next_state``
* **Inverse dynamics** — ``(q, v, a, geometry, contact) -> controls`` under a
  declared selection objective or multimodal target (never a physical
  uniqueness claim)
* **Masked trajectory-to-controls** — masked observation history plus
  conditioning → candidate state/control trajectories or coefficients

Conditioning always includes geometry, ``q0``/``v0`` dimensions, time/horizon,
constraints/contact regime and an observation mask. Checkpoint dimensions come
from the TB-00 roster via :mod:`roster`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from src.shared.python.tour_baselines.models import BackendType
from src.shared.python.tour_baselines.registry import get_golf_model

from .roster import CheckpointDimensions

__all__ = [
    "ForwardDynamicsTaskSpec",
    "InverseDynamicsTaskSpec",
    "InverseLabelMode",
    "LearningTaskKind",
    "MaskedTrajectoryTaskSpec",
    "ObservationMask",
    "TaskConditioning",
]


class LearningTaskKind(str, Enum):
    """Supervised learning task families for neural motion matching."""

    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    MASKED_TRAJECTORY_TO_CONTROLS = "masked_trajectory_to_controls"


class InverseLabelMode(str, Enum):
    """How nonunique inverse labels are represented.

    Inverse dynamics is underdetermined for contact/floating-base models.
    Training must declare either a selection objective that picks one feasible
    solution or a multimodal target. Physical uniqueness is never claimed.
    """

    SELECTION_OBJECTIVE = "selection_objective"
    MULTIMODAL_TARGET = "multimodal_target"


@dataclass(frozen=True, slots=True)
class ObservationMask:
    """Named observation channels that are visible to the learner."""

    channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.channel_ids:
            raise ValueError("observation mask must declare at least one channel")
        if any(not channel for channel in self.channel_ids):
            raise ValueError("observation mask channel ids must be non-empty")


@dataclass(frozen=True, slots=True)
class TaskConditioning:
    """Shared conditioning for every learning task.

    Design by Contract:
    - ``geometry_id`` non-empty (geometry must be declared, never implicit).
    - ``horizon_s`` > 0 and ``n_steps`` >= 1 (time must be declared).
    - ``observation_mask`` non-empty.
    - ``q0_dim`` / ``v0_dim`` positive.
    """

    geometry_id: str
    q0_dim: int
    v0_dim: int
    horizon_s: float
    n_steps: int
    contact_regime: str
    constraint_set_id: str
    observation_mask: ObservationMask

    def __post_init__(self) -> None:
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty (geometry is required)")
        if (
            isinstance(self.q0_dim, bool)
            or not isinstance(self.q0_dim, int)
            or self.q0_dim < 1
        ):
            raise ValueError(f"q0_dim must be a positive int, got {self.q0_dim!r}")
        if (
            isinstance(self.v0_dim, bool)
            or not isinstance(self.v0_dim, int)
            or self.v0_dim < 1
        ):
            raise ValueError(f"v0_dim must be a positive int, got {self.v0_dim!r}")
        if not isinstance(self.horizon_s, (int, float)) or self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be > 0, got {self.horizon_s!r}")
        if (
            isinstance(self.n_steps, bool)
            or not isinstance(self.n_steps, int)
            or self.n_steps < 1
        ):
            raise ValueError(f"n_steps must be a positive int, got {self.n_steps!r}")
        if not self.contact_regime:
            raise ValueError("contact_regime must be non-empty")
        if not self.constraint_set_id:
            raise ValueError("constraint_set_id must be non-empty")
        if not isinstance(self.observation_mask, ObservationMask):
            raise ValueError("observation_mask must be an ObservationMask")

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "q0_dim": self.q0_dim,
            "v0_dim": self.v0_dim,
            "horizon_s": float(self.horizon_s),
            "n_steps": self.n_steps,
            "contact_regime": self.contact_regime,
            "constraint_set_id": self.constraint_set_id,
            "observation_mask": list(self.observation_mask.channel_ids),
        }


def _validate_model_backend(model_id: str, backend: BackendType) -> None:
    identity = get_golf_model(model_id)
    if identity.backend is not backend:
        raise ValueError(
            f"backend mismatch for model_id={model_id!r}: "
            f"task declares {backend.value}, roster has {identity.backend.value}"
        )


def _validate_conditioning_dims(
    dimensions: CheckpointDimensions,
    conditioning: TaskConditioning,
) -> None:
    if conditioning.q0_dim != dimensions.n_q:
        raise ValueError(
            f"q0_dim ({conditioning.q0_dim}) must equal dimensions.n_q "
            f"({dimensions.n_q})"
        )
    if conditioning.v0_dim != dimensions.n_v:
        raise ValueError(
            f"v0_dim ({conditioning.v0_dim}) must equal dimensions.n_v "
            f"({dimensions.n_v})"
        )


@dataclass(frozen=True, slots=True)
class ForwardDynamicsTaskSpec:
    """Forward dynamics learning task: predict acceleration or next state."""

    model_id: str
    backend: BackendType
    dimensions: CheckpointDimensions
    conditioning: TaskConditioning
    output_kind: Literal["acceleration", "next_state"]
    include_dt: bool = True
    kind: LearningTaskKind = LearningTaskKind.FORWARD_DYNAMICS

    def __post_init__(self) -> None:
        if self.kind is not LearningTaskKind.FORWARD_DYNAMICS:
            raise ValueError("ForwardDynamicsTaskSpec.kind must be FORWARD_DYNAMICS")
        if self.dimensions.model_id != self.model_id:
            raise ValueError("dimensions.model_id must match task model_id")
        _validate_model_backend(self.model_id, self.backend)
        _validate_conditioning_dims(self.dimensions, self.conditioning)
        if self.output_kind not in {"acceleration", "next_state"}:
            raise ValueError(f"unsupported output_kind: {self.output_kind!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "model_id": self.model_id,
            "backend": self.backend.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "output_kind": self.output_kind,
            "include_dt": self.include_dt,
        }


@dataclass(frozen=True, slots=True)
class InverseDynamicsTaskSpec:
    """Inverse dynamics learning task with an explicit nonuniqueness policy."""

    model_id: str
    backend: BackendType
    dimensions: CheckpointDimensions
    conditioning: TaskConditioning
    label_mode: InverseLabelMode
    selection_objective: str | None = None
    claims_physical_uniqueness: bool = False
    kind: LearningTaskKind = LearningTaskKind.INVERSE_DYNAMICS

    def __post_init__(self) -> None:
        if self.kind is not LearningTaskKind.INVERSE_DYNAMICS:
            raise ValueError("InverseDynamicsTaskSpec.kind must be INVERSE_DYNAMICS")
        if self.dimensions.model_id != self.model_id:
            raise ValueError("dimensions.model_id must match task model_id")
        _validate_model_backend(self.model_id, self.backend)
        _validate_conditioning_dims(self.dimensions, self.conditioning)
        if self.claims_physical_uniqueness:
            raise ValueError(
                "inverse dynamics must not claim physical uniqueness; "
                "declare a selection_objective or multimodal target instead"
            )
        if self.label_mode is InverseLabelMode.SELECTION_OBJECTIVE:
            if not self.selection_objective:
                raise ValueError(
                    "selection_objective is required when "
                    "label_mode=SELECTION_OBJECTIVE"
                )
        elif self.label_mode is InverseLabelMode.MULTIMODAL_TARGET:
            if self.selection_objective:
                raise ValueError(
                    "selection_objective must be None for MULTIMODAL_TARGET"
                )
        else:
            raise ValueError(f"unsupported label_mode: {self.label_mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "model_id": self.model_id,
            "backend": self.backend.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "label_mode": self.label_mode.value,
            "selection_objective": self.selection_objective,
            "claims_physical_uniqueness": self.claims_physical_uniqueness,
        }


@dataclass(frozen=True, slots=True)
class MaskedTrajectoryTaskSpec:
    """Amortized masked-trajectory-to-controls proposal task."""

    model_id: str
    backend: BackendType
    dimensions: CheckpointDimensions
    conditioning: TaskConditioning
    proposal_kind: Literal["state_control_trajectory", "control_coefficients"]
    kind: LearningTaskKind = LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS

    def __post_init__(self) -> None:
        if self.kind is not LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS:
            raise ValueError(
                "MaskedTrajectoryTaskSpec.kind must be MASKED_TRAJECTORY_TO_CONTROLS"
            )
        if self.dimensions.model_id != self.model_id:
            raise ValueError("dimensions.model_id must match task model_id")
        _validate_model_backend(self.model_id, self.backend)
        _validate_conditioning_dims(self.dimensions, self.conditioning)
        if self.proposal_kind not in {
            "state_control_trajectory",
            "control_coefficients",
        }:
            raise ValueError(f"unsupported proposal_kind: {self.proposal_kind!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "model_id": self.model_id,
            "backend": self.backend.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "proposal_kind": self.proposal_kind,
        }
