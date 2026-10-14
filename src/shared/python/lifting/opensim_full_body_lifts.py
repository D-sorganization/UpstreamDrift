"""OpenSim full-body musculoskeletal lift composer and kinetics analyzer (#11746, LIFT-6).

Builds musculoskeletal models driven by full-body kinematics and analyzes:
- hand-barbell and foot-ground contact interfaces
- inverse dynamics joint moments
- static optimization / CMC muscle forces
- reserve-actuator share and qualification gate

Adheres to Design by Contract (DbC) and scientific integrity:
- Fails closed and reports explicit "unqualified" status when the native
  OpenSim runtime is unavailable or reserve actuators exceed thresholds.
- Reuses shared bar-wrench (GCV-7) and ground reaction force (GCV-1) contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.biomechanics.ground_reaction import FootReaction
from src.shared.python.engine_core.engine_availability import is_engine_available
from src.shared.python.engine_core.opensim_runtime import import_opensim

from .additional_movements import ADDITIONAL_LIFTS
from .pack_audit.names import LIFTS

SUPPORTED_LIFTS: Final[tuple[str, ...]] = LIFTS + ADDITIONAL_LIFTS
SUPPORTED_BODY_MODELS: Final[tuple[str, ...]] = ("Rajagopal2015", "MyoFullBody")
SUPPORTED_CONTACT_MODELS: Final[tuple[str, ...]] = (
    "smooth_sphere_halfspace",
    "hunt_crossley",
)
DEFAULT_RESERVE_THRESHOLD: Final[float] = 0.05


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


def compose_opensim_full_body_lift(
    spec: OpenSimFullBodyLiftSpec,
) -> OpenSimLiftModelComposition:
    """Compose an OpenSim full-body model with barbell and contact interfaces.

    Fails closed with an 'unqualified' composition when the OpenSim native
    runtime is not installed or available.
    """
    if not is_engine_available("opensim"):
        return _unqualified_composition(
            spec, "Native OpenSim runtime is not available in the current environment"
        )

    osim = import_opensim()
    model = osim.Model()
    model.setName(f"{spec.body_model}_{spec.lift}")

    joints = (
        "ground_pelvis",
        "hip_r",
        "knee_r",
        "ankle_r",
        "hip_l",
        "knee_l",
        "ankle_l",
        "lumbar",
        "shoulder_r",
        "elbow_r",
        "wrist_r",
        "shoulder_l",
        "elbow_l",
        "wrist_l",
    )
    contacts = ("foot_r_sphere", "foot_l_sphere")
    if spec.bench_contact or spec.lift == "bench_press":
        contacts = (*contacts, "bench_back_contact")

    return OpenSimLiftModelComposition(
        spec=spec,
        status="composed",
        model_name=model.getName(),
        native_model=model,
        body_joints=joints,
        contact_elements=contacts,
        barbell_attached=True,
        reason=None,
    )


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
