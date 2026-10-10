"""A necessary native passive-load gate, never a muscle-matching certificate.

Equilibrium force balance does not establish acceptable passive loading. Limits
must be supplied with source provenance for the exact loaded model; this module
provides no universal physiological thresholds and changes no native force law.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from src.engines.physics_engines.opensim.python.tour_matching.native_constraint_state import (
    NativeConstraintStateAudit,
    audit_native_constraint_state,
    native_muscles,
)


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("passive evidence and limits must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("passive evidence and limits must be finite")
    return result


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be explicit nonempty text")


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(asdict(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MusclePassiveLimits:
    """Declared, individually sourced limits; force ratios use native Fmax."""

    path: str
    normalized_fiber_range: tuple[float, float]
    max_abs_passive_elastic_force_ratio: float
    max_abs_tendon_force_ratio: float
    expected_ignore_tendon_compliance: bool
    expected_ignore_activation_dynamics: bool
    provenance: str

    def __post_init__(self) -> None:
        _text(self.path, "muscle path")
        if not self.path.startswith("/"):
            raise ValueError("muscle path must be absolute")
        _text(self.provenance, "limit provenance")
        if any(
            not isinstance(option, bool)
            for option in (
                self.expected_ignore_tendon_compliance,
                self.expected_ignore_activation_dynamics,
            )
        ):
            raise TypeError("expected native muscle options must be explicit booleans")
        bounds = self.normalized_fiber_range
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError("normalized fiber range must be a two-value tuple")
        low, high = (_finite(value) for value in bounds)
        if low <= 0 or high < low:
            raise ValueError("normalized fiber range must be positive and ordered")
        if any(
            _finite(value) < 0
            for value in (
                self.max_abs_passive_elastic_force_ratio,
                self.max_abs_tendon_force_ratio,
            )
        ):
            raise ValueError("force ratio limits must be nonnegative")


@dataclass(frozen=True)
class PassiveReadinessPolicy:
    """Model-bound policy; provenance text is not independent scientific approval."""

    loaded_model_sha256: str
    preparation_scope: str
    limits: tuple[MusclePassiveLimits, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.loaded_model_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.loaded_model_sha256
        ):
            raise ValueError("policy requires the exact loaded model SHA-256")
        _text(self.preparation_scope, "preparation scope")
        if not isinstance(self.limits, tuple) or any(
            not isinstance(limit, MusclePassiveLimits) for limit in self.limits
        ):
            raise TypeError("policy limits must be a tuple of MusclePassiveLimits")
        if len({limit.path for limit in self.limits}) != len(self.limits):
            raise ValueError("duplicate muscle paths in policy")

    @property
    def identity_sha256(self) -> str:
        """Bind all declared limits, model identity and provenance."""
        return _identity(self)


@dataclass(frozen=True)
class NativePassiveMuscleObservation:
    """Actual native outputs, including the law and compliance/activation options."""

    path: str
    concrete_class: str
    path_length_m: float
    fiber_length_m: float
    normalized_fiber_length: float
    fiber_velocity_m_per_s: float
    maximum_isometric_force_n: float
    passive_fiber_force_n: float
    passive_elastic_fiber_force_n: float
    passive_damping_fiber_force_n: float
    passive_force_cancellation_n: float
    tendon_force_n: float
    fiber_along_tendon_force_n: float
    activation: float
    excitation: float
    ignore_tendon_compliance: bool
    ignore_activation_dynamics: bool

    @property
    def passive_elastic_force_ratio(self) -> float:
        return self.passive_elastic_fiber_force_n / self.maximum_isometric_force_n

    @property
    def tendon_force_ratio(self) -> float:
        return self.tendon_force_n / self.maximum_isometric_force_n


@dataclass(frozen=True)
class NativePassiveReadinessAudit:
    """Immutable necessary-gate evidence; complete state/anatomy stay unqualified."""

    native_state: NativeConstraintStateAudit
    muscles: tuple[NativePassiveMuscleObservation, ...]
    policy_sha256: str | None
    observer_sha256: str
    blockers: tuple[str, ...]
    qualification: str = field(default="not-qualified-for-muscle-matching", init=False)

    @property
    def within_declared_limits(self) -> bool:
        """Only means the native observations satisfy the supplied necessary gate."""
        return not self.blockers

    @property
    def observation_sha256(self) -> str:
        """Bind observations, not authenticate evidence or certify anatomy."""
        return _identity(self)


class PassiveReadinessError(ValueError):
    """Raised when the native source fails or lacks a declared passive-load policy."""


def _muscles(model: Any, osim: Any) -> list[Any]:
    muscles = list(native_muscles(model, osim))
    if not muscles:
        raise ValueError("native muscle registry must cover all recursive muscles")
    supported = {"Thelen2003Muscle", "Millard2012EquilibriumMuscle"}
    if any(muscle.getConcreteClassName() not in supported for muscle in muscles):
        raise ValueError("unsupported concrete native muscle law for passive audit")
    return muscles


def _observe_muscle(
    muscle: Any, state: Any, osim: Any
) -> NativePassiveMuscleObservation:
    maximum_force = _finite(muscle.getMaxIsometricForce())
    if maximum_force <= 0:
        raise ValueError("native maximum isometric force must be positive")
    total_passive = _finite(muscle.getPassiveFiberForce(state))
    if muscle.getConcreteClassName() == "Millard2012EquilibriumMuscle":
        concrete = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
        elastic = _finite(concrete.getPassiveFiberElasticForce(state))
        damping = _finite(concrete.getPassiveFiberDampingForce(state))
    else:
        # The exact admitted Thelen law has no parallel passive damping term.
        elastic, damping = total_passive, 0.0
    cancellation = (
        min(abs(elastic), abs(damping))
        if elastic > 0 > damping or damping > 0 > elastic
        else 0.0
    )
    return NativePassiveMuscleObservation(
        path=muscle.getAbsolutePathString(),
        concrete_class=muscle.getConcreteClassName(),
        path_length_m=_finite(muscle.getLength(state)),
        fiber_length_m=_finite(muscle.getFiberLength(state)),
        normalized_fiber_length=_finite(muscle.getNormalizedFiberLength(state)),
        fiber_velocity_m_per_s=_finite(muscle.getFiberVelocity(state)),
        maximum_isometric_force_n=maximum_force,
        passive_fiber_force_n=total_passive,
        passive_elastic_fiber_force_n=elastic,
        passive_damping_fiber_force_n=damping,
        passive_force_cancellation_n=cancellation,
        tendon_force_n=_finite(muscle.getTendonForce(state)),
        fiber_along_tendon_force_n=_finite(muscle.getFiberForceAlongTendon(state)),
        activation=_finite(muscle.getActivation(state)),
        excitation=_finite(muscle.getExcitation(state)),
        ignore_tendon_compliance=bool(muscle.getIgnoreTendonCompliance(state)),
        ignore_activation_dynamics=bool(muscle.getIgnoreActivationDynamics(state)),
    )


def _limit_blockers(
    muscles: tuple[NativePassiveMuscleObservation, ...],
    native: NativeConstraintStateAudit,
    policy: PassiveReadinessPolicy | None,
) -> tuple[str, ...]:
    blockers = []
    for actuator in native.actuators:
        if actuator.is_muscle:
            if not actuator.enabled:
                blockers.append(f"muscle-disabled:{actuator.path}")
            if actuator.overridden:
                blockers.append(f"muscle-overridden:{actuator.path}")
    if policy is None:
        return tuple(blockers + ["passive-policy-unavailable"])
    if policy.loaded_model_sha256 != native.loaded_model_sha256:
        blockers.append("passive-policy-model-mismatch")
    limits = {limit.path: limit for limit in policy.limits}
    if set(limits) != {muscle.path for muscle in muscles}:
        blockers.append("passive-policy-muscle-coverage-mismatch")
    for muscle in muscles:
        limit = limits.get(muscle.path)
        if limit is None:
            continue
        if (
            muscle.ignore_tendon_compliance != limit.expected_ignore_tendon_compliance
            or muscle.ignore_activation_dynamics
            != limit.expected_ignore_activation_dynamics
        ):
            blockers.append(f"muscle-policy-options:{muscle.path}")
        low, high = limit.normalized_fiber_range
        if not low <= muscle.normalized_fiber_length <= high:
            blockers.append(f"normalized-fiber-limit:{muscle.path}")
        if (
            abs(muscle.passive_elastic_force_ratio)
            > limit.max_abs_passive_elastic_force_ratio
        ):
            blockers.append(f"passive-elastic-force-limit:{muscle.path}")
        if abs(muscle.tendon_force_ratio) > limit.max_abs_tendon_force_ratio:
            blockers.append(f"tendon-force-limit:{muscle.path}")
    return tuple(blockers)


def audit_native_passive_readiness(
    model: Any, state: Any, policy: PassiveReadinessPolicy | None = None
) -> NativePassiveReadinessAudit:
    """Observe an owned native Model/State without preparing or integrating it.

    The caller must prevent concurrent model/state mutation. The State copy owns
    its realization caches, but model-owned mutable constraint targets can still
    be shared. The existing constraint audit retains that incomplete-state gate.
    This function never equilibrates, assembles, unlocks, disables or overrides.
    """
    import opensim as osim

    if not isinstance(model, osim.Model) or not isinstance(state, osim.State):
        raise TypeError("passive audit requires actual native Model and State")
    if policy is not None and not isinstance(policy, PassiveReadinessPolicy):
        raise TypeError("policy must be a PassiveReadinessPolicy or unavailable")
    muscles = _muscles(model, osim)
    working = osim.State(state)
    native = audit_native_constraint_state(model, working)
    model.realizeDynamics(working)
    observations = tuple(_observe_muscle(muscle, working, osim) for muscle in muscles)
    return NativePassiveReadinessAudit(
        native_state=native,
        muscles=observations,
        policy_sha256=policy.identity_sha256 if policy is not None else None,
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        blockers=_limit_blockers(observations, native, policy),
    )


def require_native_passive_readiness(
    model: Any, state: Any, policy: PassiveReadinessPolicy | None = None
) -> NativePassiveReadinessAudit:
    """Enforce a necessary matching-preparation gate using fresh native evidence.

    Passing returns only the declared-limit audit, never a full qualification.
    Missing source limits require further evidence, not inferred thresholds.
    """
    observed = audit_native_passive_readiness(model, state, policy)
    if not observed.within_declared_limits:
        raise PassiveReadinessError("; ".join(observed.blockers))
    return observed


__all__ = [
    "MusclePassiveLimits",
    "NativePassiveMuscleObservation",
    "NativePassiveReadinessAudit",
    "PassiveReadinessError",
    "PassiveReadinessPolicy",
    "audit_native_passive_readiness",
    "require_native_passive_readiness",
]
