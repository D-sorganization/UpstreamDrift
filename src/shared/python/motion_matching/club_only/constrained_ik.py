"""Constrained IK starting guesses with explicit backend capabilities (CO-03 #10607).

Native Pink versus DLS fallback capabilities are recorded distinctly;
unsupported constraints cannot silently downgrade. Solutions are kinematic
previews pending CO-06 replay — never scientific acceptance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_only.hand_geometry import (
    ModelHandFrameOffsets,
)
from src.shared.python.motion_matching.club_only.observation import ClubObservation
from src.shared.python.motion_matching.club_only.profiles import ClubOnlyProfile
from src.shared.python.motion_matching.club_only.seeds import CandidateSeed

__all__ = [
    "ConstrainedIkRequest",
    "ConstrainedIkResult",
    "IkBackendCapabilities",
    "IkFailureReason",
    "IkSolverKind",
    "UnsupportedConstraintError",
    "assert_backend_supports",
    "capabilities_for",
    "generate_posture_branches",
    "run_constrained_ik_seeds",
]


class IkSolverKind(str, Enum):
    """Named IK backend identities with distinct capability sets."""

    PINK_NATIVE = "pink_native"
    DLS_FALLBACK = "dls_fallback"
    MISSING = "missing"


class IkFailureReason(str, Enum):
    """Fail-closed IK failure modes (no silent success)."""

    SINGULAR_JACOBIAN = "singular_jacobian"
    UNREACHABLE_ORIENTATION = "unreachable_orientation"
    MISSING_SOLVER = "missing_solver"
    UNSUPPORTED_CONSTRAINT = "unsupported_constraint"


class UnsupportedConstraintError(ValueError):
    """Raised when a backend cannot honour a required hard constraint."""


@dataclass(frozen=True)
class IkBackendCapabilities:
    """Declared capability surface for one IK backend."""

    kind: IkSolverKind
    supports_orientation_tasks: bool
    supports_hard_grip_closure: bool
    supports_stance_constraints: bool
    available: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "supports_orientation_tasks": self.supports_orientation_tasks,
            "supports_hard_grip_closure": self.supports_hard_grip_closure,
            "supports_stance_constraints": self.supports_stance_constraints,
            "available": self.available,
        }


@dataclass(frozen=True)
class ConstrainedIkRequest:
    """Hard constraints requested of an IK backend."""

    requires_orientation: bool
    requires_hard_grip: bool
    requires_stance: bool
    hand_offsets: ModelHandFrameOffsets

    def __post_init__(self) -> None:
        if not isinstance(self.hand_offsets, ModelHandFrameOffsets):
            raise TypeError("hand_offsets must be ModelHandFrameOffsets")


@dataclass(frozen=True)
class ConstrainedIkResult:
    """Bounded IK seeds or an explicit failure; always a kinematic preview."""

    seeds: tuple[CandidateSeed, ...]
    failure_reason: IkFailureReason | None
    is_kinematic_preview: bool
    backend: IkSolverKind

    def __post_init__(self) -> None:
        if not self.is_kinematic_preview:
            raise ValueError("IK solutions must remain kinematic previews (CO-06)")
        if self.failure_reason is not None and self.seeds:
            raise ValueError("failed IK cannot also return seeds")
        if self.failure_reason is None and not self.seeds:
            raise ValueError("successful IK must return at least one seed")


def capabilities_for(kind: IkSolverKind) -> IkBackendCapabilities:
    """Return the declared capability set for a named backend."""
    if not isinstance(kind, IkSolverKind):
        raise TypeError("kind must be IkSolverKind")
    if kind is IkSolverKind.PINK_NATIVE:
        return IkBackendCapabilities(
            kind=kind,
            supports_orientation_tasks=True,
            supports_hard_grip_closure=True,
            supports_stance_constraints=True,
            available=True,
        )
    if kind is IkSolverKind.DLS_FALLBACK:
        return IkBackendCapabilities(
            kind=kind,
            supports_orientation_tasks=False,
            supports_hard_grip_closure=False,
            supports_stance_constraints=False,
            available=True,
        )
    return IkBackendCapabilities(
        kind=IkSolverKind.MISSING,
        supports_orientation_tasks=False,
        supports_hard_grip_closure=False,
        supports_stance_constraints=False,
        available=False,
    )


@precondition(
    lambda caps, request: (
        isinstance(caps, IkBackendCapabilities)
        and isinstance(request, ConstrainedIkRequest)
    ),
    "caps and request required",
)
def assert_backend_supports(
    caps: IkBackendCapabilities, request: ConstrainedIkRequest
) -> None:
    """Fail closed when required constraints exceed backend capabilities."""
    if not caps.available or caps.kind is IkSolverKind.MISSING:
        raise UnsupportedConstraintError(
            f"IK backend missing; cannot satisfy constraints for "
            f"model={request.hand_offsets.model_id!r}"
        )
    if request.requires_orientation and not caps.supports_orientation_tasks:
        raise UnsupportedConstraintError(
            f"backend {caps.kind.value} does not support orientation tasks; "
            "silent downgrade refused"
        )
    if request.requires_hard_grip and not caps.supports_hard_grip_closure:
        raise UnsupportedConstraintError(
            f"backend {caps.kind.value} does not support hard grip closure; "
            "silent downgrade refused"
        )
    if request.requires_stance and not caps.supports_stance_constraints:
        raise UnsupportedConstraintError(
            f"backend {caps.kind.value} does not support stance constraints; "
            "silent downgrade refused"
        )


@precondition(
    lambda q0, n_branches=3, amplitude_rad=0.05: (
        isinstance(q0, np.ndarray)
        and q0.ndim == 1
        and q0.size >= 1
        and int(n_branches) >= 1
        and float(amplitude_rad) > 0.0
    ),
    "q0 1-D, n_branches>=1, amplitude_rad>0 required",
)
@postcondition(
    lambda result: (
        isinstance(result, tuple) and all(isinstance(q, np.ndarray) for q in result)
    ),
    "must return ndarray tuple",
)
def generate_posture_branches(
    q0: np.ndarray,
    *,
    n_branches: int = 3,
    amplitude_rad: float = 0.05,
) -> tuple[np.ndarray, ...]:
    """Generate a bounded set of alternative posture branches around ``q0``."""
    base = np.asarray(q0, dtype=np.float64)
    if not np.all(np.isfinite(base)):
        raise ValueError("q0 must be finite")
    if n_branches < 1:
        raise ValueError("n_branches must be >= 1")
    if not np.isfinite(amplitude_rad) or amplitude_rad <= 0.0:
        raise ValueError("amplitude_rad must be finite and > 0")

    branches: list[np.ndarray] = []
    for index in range(n_branches):
        offset = np.zeros_like(base)
        # Distinct, deterministic perturbations across joints.
        phase = (index + 1) * np.linspace(0.3, 1.0, base.size, dtype=np.float64)
        offset[:] = amplitude_rad * np.sin(phase + 0.17 * index)
        # Keep the first branch as the warm-start itself.
        candidate = base.copy() if index == 0 else base + offset
        branches.append(candidate)
    return tuple(branches)


def _body_hash(q: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def _warm_start_q(observation: ClubObservation, nq: int = 4) -> np.ndarray:
    """Event-aware bounded initialization from grip height (synthetic DOF)."""
    grip = np.asarray(observation.mid_hands_xyz, dtype=np.float64)
    if grip.ndim != 2 or grip.shape[1] != 3:
        raise ValueError("mid_hands_xyz must be (N, 3)")
    mean_z = float(np.nanmean(grip[:, 2]))
    if not np.isfinite(mean_z):
        raise ValueError("grip height must be finite")
    q = np.zeros(nq, dtype=np.float64)
    q[0] = 0.05 * mean_z
    q[1] = -0.02 * mean_z
    if observation.events:
        # Bounded event cue without time warp.
        q[2] = 0.01 * float(observation.events[0].sample_index) / max(grip.shape[0], 1)
    return q


@precondition(
    lambda observation, profile, hand_offsets, geometry_hash, profile_hash, backend, simulate_failure=None, n_branches=3: (  # noqa: E501
        isinstance(observation, ClubObservation)
        and isinstance(profile, ClubOnlyProfile)
        and isinstance(hand_offsets, ModelHandFrameOffsets)
        and bool(geometry_hash)
        and bool(profile_hash)
        and isinstance(backend, IkSolverKind)
    ),
    "observation, profile, hand_offsets, hashes, backend required",
)
@postcondition(
    lambda result: isinstance(result, ConstrainedIkResult),
    "must return ConstrainedIkResult",
)
def run_constrained_ik_seeds(
    *,
    observation: ClubObservation,
    profile: ClubOnlyProfile,
    hand_offsets: ModelHandFrameOffsets,
    geometry_hash: str,
    profile_hash: str,
    backend: IkSolverKind = IkSolverKind.PINK_NATIVE,
    simulate_failure: IkFailureReason | None = None,
    n_branches: int = 3,
) -> ConstrainedIkResult:
    """Produce constrained-IK seeds or an explicit failure reason.

    Hard grip/stance/orientation constraints are asserted against the backend
    capability surface. When ``simulate_failure`` is set (unit fixtures), the
    result records the failure without inventing a physical success.
    """
    request = ConstrainedIkRequest(
        requires_orientation=True,
        requires_hard_grip=True,
        requires_stance=True,
        hand_offsets=hand_offsets,
    )
    caps = capabilities_for(backend)
    try:
        assert_backend_supports(caps, request)
    except UnsupportedConstraintError:
        return ConstrainedIkResult(
            seeds=(),
            failure_reason=IkFailureReason.UNSUPPORTED_CONSTRAINT
            if backend is not IkSolverKind.MISSING
            else IkFailureReason.MISSING_SOLVER,
            is_kinematic_preview=True,
            backend=backend,
        )

    if simulate_failure is not None:
        return ConstrainedIkResult(
            seeds=(),
            failure_reason=simulate_failure,
            is_kinematic_preview=True,
            backend=backend,
        )

    # Fail closed: club_only has no integrated Pink/DLS solver.
    # No solve has run, so seeds cannot be returned with fabricated residuals
    # or unearned constraint claims (CO-03 #10607 / #10960 P1-1).
    return ConstrainedIkResult(
        seeds=(),
        failure_reason=IkFailureReason.MISSING_SOLVER,
        is_kinematic_preview=True,
        backend=backend,
    )
