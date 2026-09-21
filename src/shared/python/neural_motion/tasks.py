"""Learning-task contracts for neural motion matching (NM-01 #10616).

Defines distinct, typed contracts for forward dynamics, inverse dynamics and
masked-trajectory-to-controls proposals. Dimensions come from TB-00
``GolfModelIdentity`` (#10585), never a hard-coded 27x7 assumption. Inverse
labels require an explicit non-uniqueness policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.shared.python.tour_baselines.registry import get_golf_model
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provider import FitSwingProvider
from src.shared.python.training.config import TrainingConfig

_FIT_ANCHORS = (CanonicalFitResult, FitSwingProvider, TrainingConfig)

__all__ = [
    "ConditioningSpec",
    "ForwardDynamicsTask",
    "InverseDynamicsTask",
    "InverseLabelPolicy",
    "LearningTask",
    "LearningTaskKind",
    "MaskedTrajectoryTask",
    "TaskDimensions",
    "build_default_learning_tasks",
    "dimensions_from_model",
]


class LearningTaskKind(str, Enum):
    """Supervised learning task family."""

    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"
    MASKED_TRAJECTORY_TO_CONTROLS = "masked_trajectory_to_controls"


class InverseLabelPolicy(str, Enum):
    """How nonunique inverse labels are represented.

    Club-only / underdetermined inverse problems do not admit a unique
    physical control. Training must declare either a selection objective or a
    multimodal target distribution — never claim physical uniqueness.
    """

    SELECTION_OBJECTIVE = "selection_objective"
    MULTIMODAL_DISTRIBUTION = "multimodal_distribution"


def _require_finite_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        out.append(number)
    return tuple(out)


@dataclass(frozen=True)
class TaskDimensions:
    """Per-model tensor dimensions bound to a registered golf model.

    Design by Contract:
    - ``model_id`` must resolve in the TB-00 registry.
    - ``backend`` must equal the registered model's backend value.
    - All dims are non-negative; ``q_dim``, ``v_dim``, ``a_dim`` > 0.
    - No hard-coded 27x7 layout is implied.
    """

    model_id: str
    backend: str
    q_dim: int
    v_dim: int
    a_dim: int
    u_dim: int
    reaction_dim: int

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        identity = get_golf_model(self.model_id)
        if self.backend != identity.backend.value:
            raise ValueError(
                f"backend '{self.backend}' does not match registered "
                f"backend '{identity.backend.value}' for {self.model_id}"
            )
        for name, value in (
            ("q_dim", self.q_dim),
            ("v_dim", self.v_dim),
            ("a_dim", self.a_dim),
            ("u_dim", self.u_dim),
            ("reaction_dim", self.reaction_dim),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.q_dim <= 0 or self.v_dim <= 0 or self.a_dim <= 0:
            raise ValueError("q_dim, v_dim and a_dim must be > 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "q_dim": self.q_dim,
            "v_dim": self.v_dim,
            "a_dim": self.a_dim,
            "u_dim": self.u_dim,
            "reaction_dim": self.reaction_dim,
        }


def dimensions_from_model(model_id: str) -> TaskDimensions:
    """Build task dimensions from a registered ``GolfModelIdentity``."""
    identity = get_golf_model(model_id)
    return TaskDimensions(
        model_id=identity.model_id,
        backend=identity.backend.value,
        q_dim=identity.dof,
        v_dim=identity.dof,
        a_dim=identity.dof,
        u_dim=identity.independent_dof,
        reaction_dim=identity.constraint_count,
    )


@dataclass(frozen=True)
class ConditioningSpec:
    """Shared conditioning required by every learning task.

    Design by Contract:
    - ``geometry_id`` non-empty.
    - ``q0`` / ``v0`` non-empty and finite.
    - ``horizon_s`` and ``time_step_s`` positive and finite.
    - ``observation_mask`` non-empty (missing mask rejected).
    - constraint/contact profiles are non-empty strings.
    """

    geometry_id: str
    q0: tuple[float, ...]
    v0: tuple[float, ...]
    horizon_s: float
    time_step_s: float
    constraint_profile: str
    contact_profile: str
    observation_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty")
        object.__setattr__(self, "q0", _require_finite_sequence("q0", self.q0))
        object.__setattr__(self, "v0", _require_finite_sequence("v0", self.v0))
        if not math.isfinite(self.horizon_s) or self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be a positive finite time")
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be a positive finite time")
        if not self.constraint_profile:
            raise ValueError("constraint_profile must be non-empty")
        if not self.contact_profile:
            raise ValueError("contact_profile must be non-empty")
        if not self.observation_mask:
            raise ValueError("observation_mask must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "q0": list(self.q0),
            "v0": list(self.v0),
            "horizon_s": self.horizon_s,
            "time_step_s": self.time_step_s,
            "constraint_profile": self.constraint_profile,
            "contact_profile": self.contact_profile,
            "observation_mask": list(self.observation_mask),
        }


@dataclass(frozen=True)
class ForwardDynamicsTask:
    """(q, v, u, geometry, contact, dt) -> acceleration or next state."""

    task_id: str
    dimensions: TaskDimensions
    conditioning: ConditioningSpec
    output_kind: str
    kind: LearningTaskKind = LearningTaskKind.FORWARD_DYNAMICS

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.kind is not LearningTaskKind.FORWARD_DYNAMICS:
            raise ValueError("ForwardDynamicsTask kind must be FORWARD_DYNAMICS")
        if self.output_kind not in {"acceleration", "next_state"}:
            raise ValueError("output_kind must be 'acceleration' or 'next_state'")
        if self.dimensions.u_dim <= 0:
            raise ValueError("forward dynamics requires u_dim (control) > 0")
        if len(self.conditioning.q0) != self.dimensions.q_dim:
            raise ValueError("conditioning.q0 length must match q_dim")
        if len(self.conditioning.v0) != self.dimensions.v_dim:
            raise ValueError("conditioning.v0 length must match v_dim")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "output_kind": self.output_kind,
        }


@dataclass(frozen=True)
class InverseDynamicsTask:
    """(q, v, a, geometry, contact) -> feasible controls under a label policy."""

    task_id: str
    dimensions: TaskDimensions
    conditioning: ConditioningSpec
    label_policy: InverseLabelPolicy
    selection_objective: str | None
    claims_physical_uniqueness: bool
    kind: LearningTaskKind = LearningTaskKind.INVERSE_DYNAMICS

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.kind is not LearningTaskKind.INVERSE_DYNAMICS:
            raise ValueError("InverseDynamicsTask kind must be INVERSE_DYNAMICS")
        if not isinstance(self.label_policy, InverseLabelPolicy):
            raise ValueError(
                "label_policy must be InverseLabelPolicy "
                "(selection_objective or multimodal_distribution)"
            )
        if self.claims_physical_uniqueness:
            raise ValueError(
                "claims_physical_uniqueness must be False; "
                "inverse labels are nonunique without a declared policy"
            )
        if (
            self.label_policy is InverseLabelPolicy.SELECTION_OBJECTIVE
            and not self.selection_objective
        ):
            raise ValueError(
                "selection_objective is required for SELECTION_OBJECTIVE policy"
            )
        if (
            self.label_policy is InverseLabelPolicy.MULTIMODAL_DISTRIBUTION
            and self.selection_objective
        ):
            raise ValueError(
                "selection_objective must be None for MULTIMODAL_DISTRIBUTION"
            )
        if self.dimensions.u_dim <= 0:
            raise ValueError("inverse dynamics requires u_dim (control) > 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "label_policy": self.label_policy.value,
            "selection_objective": self.selection_objective,
            "claims_physical_uniqueness": self.claims_physical_uniqueness,
        }


@dataclass(frozen=True)
class MaskedTrajectoryTask:
    """Masked history + priors -> candidate state/control trajectories."""

    task_id: str
    dimensions: TaskDimensions
    conditioning: ConditioningSpec
    proposal_space: str
    kind: LearningTaskKind = LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.kind is not LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS:
            raise ValueError(
                "MaskedTrajectoryTask kind must be MASKED_TRAJECTORY_TO_CONTROLS"
            )
        if not self.proposal_space:
            raise ValueError("proposal_space must be non-empty")
        if not self.conditioning.observation_mask:
            raise ValueError("observation_mask is required for masked trajectory tasks")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dimensions": self.dimensions.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "proposal_space": self.proposal_space,
        }


LearningTask = ForwardDynamicsTask | InverseDynamicsTask | MaskedTrajectoryTask


def build_default_learning_tasks(model_id: str) -> tuple[LearningTask, ...]:
    """Return the three frozen learning-task contracts for one model."""
    dims = dimensions_from_model(model_id)
    mask = tuple(True for _ in range(max(dims.q_dim, 1)))
    # Pilot conditioning uses identity-sized zeros; real episodes supply data later.
    conditioning = ConditioningSpec(
        geometry_id=f"geom.{model_id}.pilot.v1",
        q0=tuple(0.0 for _ in range(dims.q_dim)),
        v0=tuple(0.0 for _ in range(dims.v_dim)),
        horizon_s=0.85,
        time_step_s=0.01,
        constraint_profile="model_native.v1",
        contact_profile="model_native.v1",
        observation_mask=mask,
    )
    return (
        ForwardDynamicsTask(
            task_id=f"fwd.{model_id}",
            dimensions=dims,
            conditioning=conditioning,
            output_kind="acceleration",
        ),
        InverseDynamicsTask(
            task_id=f"inv.{model_id}",
            dimensions=dims,
            conditioning=conditioning,
            label_policy=InverseLabelPolicy.SELECTION_OBJECTIVE,
            selection_objective="min_effort_with_contact_regularization",
            claims_physical_uniqueness=False,
        ),
        MaskedTrajectoryTask(
            task_id=f"mask.{model_id}",
            dimensions=dims,
            conditioning=conditioning,
            proposal_space="continuous_coefficients",
        ),
    )
