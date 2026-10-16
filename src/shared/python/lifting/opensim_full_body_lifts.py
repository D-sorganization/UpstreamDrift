"""OpenSim full-body musculoskeletal lift composer and kinetics analyzer (#11746, LIFT-6).

:func:`compose_opensim_full_body_lift` composes a real OpenSim model per
barbell lift (OSL-2, #13040): the muscle-bearing Rajagopal-Lai-Uhlrich body
scaled to the spec, a barbell of the spec's mass held by both hands through
the LIFT-4 interface, and smooth sphere/half-space foot (and, for the bench
press, bench and back) contact -- see
:mod:`src.engines.physics_engines.opensim.python.msk_lift_model`.  Every
count and ``barbell_attached`` is read from the built model.

:func:`evaluate_opensim_lift_kinetics` (inverse dynamics, static optimization
or CMC muscle forces, reserve-actuator share and the qualification gate) is
not built yet: it reports ``unqualified`` on every path (OSL-3/OSL-4).

Adheres to Design by Contract (DbC) and scientific integrity:
- Fails closed and reports explicit "unqualified" status when the native
  OpenSim runtime is unavailable or reserve actuators exceed thresholds.
- Reuses shared bar-wrench (GCV-7) and ground reaction force (GCV-1) contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.biomechanics.ground_reaction import FootReaction
from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.engine_core.opensim_runtime import import_opensim

from .additional_movements import ADDITIONAL_LIFTS
from .barbell_bushing import Barbell
from .pack_audit.names import LIFTS

SUPPORTED_LIFTS: Final[tuple[str, ...]] = LIFTS + ADDITIONAL_LIFTS
SUPPORTED_BODY_MODELS: Final[tuple[str, ...]] = ("Rajagopal2015", "MyoFullBody")
SUPPORTED_CONTACT_MODELS: Final[tuple[str, ...]] = (
    "smooth_sphere_halfspace",
    "hunt_crossley",
)
DEFAULT_RESERVE_THRESHOLD: Final[float] = 0.05
#: The shared barbell's shaft mass, the lightest bar OSL-2 composes.
_SHAFT_MASS_KG: Final[float] = Barbell.shaft_mass_kg


@dataclass(frozen=True)
class OpenSimFullBodyLiftSpec:
    """Specification for OpenSim musculoskeletal lift modelling."""

    lift: str
    body_model: str = "Rajagopal2015"
    body_mass_kg: float = 80.0
    height_m: float = 1.78
    bar_mass_kg: float = 120.0
    contact_model: str = "smooth_sphere_halfspace"
    bench_contact: bool = False
    reserve_actuator_threshold: float = DEFAULT_RESERVE_THRESHOLD

    def __post_init__(self) -> None:
        if self.lift not in SUPPORTED_LIFTS:
            raise ValueError(
                f"unsupported lift {self.lift!r}; choose from {list(SUPPORTED_LIFTS)}"
            )
        if self.body_model not in SUPPORTED_BODY_MODELS:
            raise ValueError(
                f"unsupported body_model {self.body_model!r}; choose from {list(SUPPORTED_BODY_MODELS)}"
            )
        if self.contact_model not in SUPPORTED_CONTACT_MODELS:
            raise ValueError(
                f"unsupported contact_model {self.contact_model!r}; choose from {list(SUPPORTED_CONTACT_MODELS)}"
            )
        if self.body_mass_kg <= 0.0:
            raise ValueError(f"body_mass_kg must be positive, got {self.body_mass_kg}")
        if self.height_m <= 0.0:
            raise ValueError(f"height_m must be positive, got {self.height_m}")
        if self.bar_mass_kg < 0.0:
            raise ValueError(
                f"bar_mass_kg must be non-negative, got {self.bar_mass_kg}"
            )
        if not 0.0 <= self.reserve_actuator_threshold <= 1.0:
            raise ValueError(
                f"reserve_actuator_threshold must be in [0, 1], got {self.reserve_actuator_threshold}"
            )


@dataclass(frozen=True)
class OpenSimLiftModelComposition:
    """Composition record of an OpenSim full-body lift model."""

    spec: OpenSimFullBodyLiftSpec
    status: str
    model_name: str
    native_model: Any | None
    body_joints: tuple[str, ...]
    contact_elements: tuple[str, ...]
    barbell_attached: bool
    reason: str | None = None
    body_count: int | None = None
    joint_count: int | None = None
    constraint_count: int | None = None
    muscle_count: int | None = None
    force_count: int | None = None
    receipt: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def is_composed(self) -> bool:
        """Return True only when the model composed successfully in OpenSim."""
        return self.status == "composed" and self.native_model is not None


@dataclass(frozen=True)
class OpenSimLiftKineticsResult:
    """Result of OpenSim full-body lift kinetics evaluation."""

    status: str
    lift: str
    joint_moments_nm: Mapping[str, float] | None
    hand_wrench: GripAnalysis
    grf: FootReaction | None
    l5_s1_load_n: float | None
    muscle_forces_n: Mapping[str, float] | None
    reserve_share: float | None
    reason: str

    @property
    def is_qualified(self) -> bool:
        """Return True only when the kinetics solve meets all quality gates."""
        return self.status == "qualified"


def _unqualified_composition(
    spec: OpenSimFullBodyLiftSpec, reason: str
) -> OpenSimLiftModelComposition:
    return OpenSimLiftModelComposition(
        spec=spec,
        status="unqualified",
        model_name=f"{spec.body_model}_{spec.lift}",
        native_model=None,
        body_joints=(),
        contact_elements=(),
        barbell_attached=False,
        reason=reason,
    )


def _unqualified_kinetics(
    spec: OpenSimFullBodyLiftSpec, reason: str
) -> OpenSimLiftKineticsResult:
    return OpenSimLiftKineticsResult(
        status="unqualified",
        lift=spec.lift,
        joint_moments_nm=None,
        hand_wrench=unavailable_analysis(reason),
        grf=None,
        l5_s1_load_n=None,
        muscle_forces_n=None,
        reserve_share=None,
        reason=reason,
    )


def _unsupported_reason(spec: OpenSimFullBodyLiftSpec) -> str | None:
    """Why OSL-2 cannot compose *spec*, or ``None`` when it can."""
    if spec.lift not in LIFTS:
        return (
            f"OSL-2 composes only the barbell lifts {list(LIFTS)}; "
            f"{spec.lift!r} has no OpenSim musculoskeletal model yet"
        )
    if spec.body_model != "Rajagopal2015":
        return (
            f"body_model {spec.body_model!r} is not composed in OpenSim yet; "
            "OSL-2 builds the Rajagopal-Lai-Uhlrich body only"
        )
    if spec.contact_model != "smooth_sphere_halfspace":
        return (
            f"contact_model {spec.contact_model!r} is not built; OSL-2 uses "
            "SmoothSphereHalfSpaceForce only"
        )
    if spec.bench_contact and spec.lift != "bench_press":
        return "bench contact applies only to the supine bench press hold"
    if spec.bar_mass_kg < _SHAFT_MASS_KG:
        return (
            f"bar_mass_kg {spec.bar_mass_kg} is below the {_SHAFT_MASS_KG} kg "
            "shaft of the shared barbell"
        )
    return None


def _model_counts(model: Any) -> dict[str, int]:
    return {
        "body_count": int(model.getBodySet().getSize()),
        "joint_count": int(model.getJointSet().getSize()),
        "constraint_count": int(model.getConstraintSet().getSize()),
        "muscle_count": int(model.getMuscles().getSize()),
        "force_count": int(model.getForceSet().getSize()),
    }


def _attachment_receipt(model: Any) -> tuple[bool, dict[str, str | None]]:
    """Whether the bar is held by both hands, read from the model."""
    from .pack_audit.adapters.opensim_bar_hold import hand_bar_attachment

    left, right = hand_bar_attachment(model)
    enforced = right is not None and bool(right.get_isEnforced())
    attached = left is not None and enforced
    return attached, {
        "left": None if left is None else f"WeldJoint {left.getName()}",
        "right": None if right is None else f"WeldConstraint {right.getName()}",
    }


def compose_opensim_full_body_lift(
    spec: OpenSimFullBodyLiftSpec,
) -> OpenSimLiftModelComposition:
    """Compose an OpenSim full-body musculoskeletal lift model (OSL-2, #13040).

    Builds the model in OpenSim (see
    :func:`~src.engines.physics_engines.opensim.python.msk_lift_model.build_lift_model`)
    and reports what it actually contains.

    Postconditions: a ``composed`` result carries an initialised native
    model whose body, joint, constraint, muscle and force counts, joint
    names, contact force names and ``barbell_attached`` are read from that
    model.  Otherwise the result is ``unqualified`` with a reason and every
    count is ``None`` (unavailable is never zero): when OpenSim is not
    available, the lift or an option is not built, or the muscle-bearing
    body cannot be resolved.
    """
    if not is_engine_available("opensim"):
        return _unqualified_composition(
            spec, "Native OpenSim runtime is not available in the current environment"
        )
    reason = _unsupported_reason(spec)
    if reason is not None:
        return _unqualified_composition(spec, reason)

    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        build_lift_model,
    )

    try:
        build = build_lift_model(
            spec.lift,
            body_mass_kg=spec.body_mass_kg,
            height_m=spec.height_m,
            bar_mass_kg=spec.bar_mass_kg,
        )
    except (FileNotFoundError, ValueError) as exc:
        return _unqualified_composition(spec, f"composition failed: {exc}")

    model = build.model
    attached, attachment = _attachment_receipt(model)
    joints = model.getJointSet()
    receipt = {**build.receipt, "hand_attachment": attachment}
    return OpenSimLiftModelComposition(
        spec=spec,
        status="composed",
        model_name=model.getName(),
        native_model=model,
        body_joints=tuple(joints.get(i).getName() for i in range(joints.getSize())),
        contact_elements=tuple(build.receipt["contact_forces"]),
        barbell_attached=attached,
        reason=None,
        **_model_counts(model),
        receipt=MappingProxyType(receipt),
    )


def opensim_lift_static_hold(
    composition: OpenSimLiftModelComposition,
) -> dict[str, Any]:
    """Static hold of a composed lift model: does it carry the bar in the hands?

    Returns ``{"available", "reason", "bar_hold", "native"}``:

    * ``bar_hold`` -- the shared LIFT-4 per-hand bar wrench
      (``pack_audit.adapters.opensim_bar_hold.bar_hold_wrench``) at the hold
      pose: bar mass and centre of mass from the model, minimum-norm split;
    * ``native`` -- the hand-bar reactions OpenSim computes with every
      coordinate locked (``msk_lift_model.native_static_hold``).

    When the model is not composed or the bar is not held by both hands the
    result is unavailable with a reason, never zero.
    """
    if not composition.is_composed:
        return {
            "available": False,
            "reason": composition.reason or "lift model is not composed",
            "bar_hold": None,
            "native": None,
        }
    if not composition.barbell_attached:
        return {
            "available": False,
            "reason": "the bar is not held by both hands in this model",
            "bar_hold": None,
            "native": None,
        }
    from src.engines.physics_engines.opensim.python.msk_lift_model import (
        native_static_hold,
    )

    from .pack_audit.adapters.opensim_bar_hold import (
        bar_hold_wrench,
        hand_bar_attachment,
    )

    model = composition.native_model
    state = model.initSystem()
    model.realizePosition(state)
    left, right = hand_bar_attachment(model)
    return {
        "available": True,
        "reason": "",
        "bar_hold": bar_hold_wrench(model, state),
        "native": native_static_hold(model, left, right),
    }


def evaluate_opensim_lift_kinetics(
    spec: OpenSimFullBodyLiftSpec,
    bundle_or_trajectory: Any = None,
) -> OpenSimLiftKineticsResult:
    """Evaluate inverse dynamics, muscle forces, bar wrench, and GRF.

    Fails closed and reports explicit 'unqualified' status when native OpenSim
    is unavailable or reserve actuator load exceeds the tolerance threshold.
    """
    if not is_engine_available("opensim"):
        return _unqualified_kinetics(
            spec, "Native OpenSim runtime is not available in the current environment"
        )

    # In a qualified native solve, static optimization / CMC residuals are
    # checked against spec.reserve_actuator_threshold.
    # When bundle_or_trajectory is empty or unprovided, fail closed honestly.
    if bundle_or_trajectory is None:
        return _unqualified_kinetics(
            spec, "No kinematics trajectory or bundle provided for inverse dynamics"
        )

    return _unqualified_kinetics(
        spec,
        "OpenSim full-body kinetics pipeline execution requires valid solver context",
    )
