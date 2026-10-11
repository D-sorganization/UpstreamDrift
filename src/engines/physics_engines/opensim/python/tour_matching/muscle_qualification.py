"""OpenSim muscle and tendon extension qualification (OG-08, #10402).

Part of OpenSim epic #10394 under matched-swing epic #10363.

Implements qualification gates for muscle/tendon extensions:
1. Anatomy Coverage & Claims:
   - Evaluates lower-extremity, torso/spine, shoulder/scapula, arm/forearm, wrist/hand, and head/neck.
   - Forbids lower-limb-only models (such as Rajagopal2015 80-muscle lower extremity) from claiming
     full-body golf swing actuation; raises UnsupportedAnatomyClaimError.
   - Mandates explicit declarations of omitted regions (e.g. rigid head/neck).
2. Parameter Provenance & Strength Scaling:
   - Explicit MuscleParameterProvenance records (F_max, l_opt, l_slack, pennation, citation, license, hash).
   - Validates physiological bounds and non-negativity.
   - Enforces disclaimer that OpenSim segment scaling does not qualify muscle strength.
3. Path & Wrapping Geometry:
   - MusclePathGeometry audits (minimum 2 points on distinct parent bodies, valid wrapping surfaces).
4. Moment Arm vs. Finite-Difference Derivative:
   - Compares moment arm against virtual work finite-difference derivative of path length:
     r_FD = -(l_MT(q + dq) - l_MT(q - dq)) / (2 * dq).
   - Raises MomentArmDerivativeMismatchError if discrepancy exceeds tolerance.
5. Activation Dynamics & Initial Tendon Equilibrium:
   - Audits activation bounds [a_min, 1.0].
   - Solves and audits static equilibrium (F_fiber * cos(alpha) = F_tendon) before forward simulation.
   - Raises UninitializedTendonStateError if state is uninitialized or non-equilibrated.
6. Evidence Boundaries:
   - Parameter-only orchestration runs anatomy and parameter checks only.
   - Native replay and unperformed model/state audits remain unqualified;
     no illustrative residuals or activations are reported as native evidence.
   - muscle_complete_status remains "IN_PROGRESS_QUALIFICATION" and
     independent_validation_status remains "PENDING_10375".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import enum
import hashlib
import logging
import math
from typing import Any

import numpy as np

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

# Constants
HALF_PI_RAD = 1.5707963267948966
DEFAULT_MIN_ACTIVATION = 0.01
EQUILIBRIUM_FORCE_TOLERANCE_N = 0.5


class UnsupportedAnatomyClaimError(ValueError):
    """Raised when a muscle model with partial anatomy claims full-body golf swing capability."""


class IncompleteAnatomyCoverageError(ValueError):
    """Raised when an anatomy set fails to cover required regions without explicit declared omissions."""


class InvalidMuscleParameterError(ValueError):
    """Raised when muscle parameters violate physiological bounds or lack valid provenance."""


class InvalidMusclePathError(ValueError):
    """Raised when muscle path points or wrapping geometry are invalid."""


class UninitializedTendonStateError(RuntimeError):
    """Raised when tendon dynamics are uninitialized or initial state violates static equilibrium."""


class MomentArmDerivativeMismatchError(AssertionError):
    """Raised when moment arm deviates from negative partial derivative of path length beyond tolerance."""


class AnatomicalRegion(enum.Enum):
    """Anatomical regions evaluated for golf swing biomechanics coverage."""

    LOWER_EXTREMITY = "lower_extremity"
    TORSO_SPINE = "torso_spine"
    SHOULDER_SCAPULA = "shoulder_scapula"
    ARM_FOREARM = "arm_forearm"
    WRIST_HAND = "wrist_hand"
    HEAD_NECK = "head_neck"


# Canonical muscle name substrings per anatomical region
_REGION_PATTERNS: dict[AnatomicalRegion, tuple[str, ...]] = {
    AnatomicalRegion.LOWER_EXTREMITY: (
        "gluteus",
        "femoris",
        "vastus",
        "gastrocnemius",
        "soleus",
        "tibialis",
        "semitendinosus",
        "semimembranosus",
        "adductor",
        "iliacus",
        "gracilis",
        "sartorius",
        "peroneus",
        "calcn",
        "tib",
    ),
    AnatomicalRegion.TORSO_SPINE: (
        "erector_spinae",
        "oblique",
        "abdominis",
        "psoas",
        "iliocostalis",
        "longissimus",
        "multifidus",
        "quadratus_lumborum",
    ),
    AnatomicalRegion.SHOULDER_SCAPULA: (
        "deltoid",
        "pectoralis",
        "latissimus",
        "trapezius",
        "infraspinatus",
        "supraspinatus",
        "subscapularis",
        "teres",
        "serratus",
        "rhomboid",
        "levator_scapulae",
    ),
    AnatomicalRegion.ARM_FOREARM: (
        "biceps_brachii",
        "triceps_brachii",
        "brachialis",
        "brachioradialis",
        "pronator",
        "supinator",
        "anconeus",
    ),
    AnatomicalRegion.WRIST_HAND: (
        "flexor_carpi",
        "extensor_carpi",
        "flexor_digitorum",
        "extensor_digitorum",
        "palmaris",
        "abductor_pollicis",
        "flexor_pollicis",
        "extensor_pollicis",
        "lumbrical",
        "interossei",
    ),
    AnatomicalRegion.HEAD_NECK: (
        "sternocleidomastoid",
        "splenius",
        "scalenus",
        "longus_capitis",
        "longus_colli",
    ),
}


@dataclass(frozen=True)
class AnatomyCoverageScope:
    """Truthful coverage audit of anatomical muscle groups and omissions."""

    covered_regions: tuple[AnatomicalRegion, ...]
    declared_omissions: tuple[AnatomicalRegion, ...]
    omission_notes: dict[str, str]
    is_full_golf_model: bool
    muscles_by_region: dict[AnatomicalRegion, tuple[str, ...]]


@dataclass(frozen=True)
class MuscleParameterProvenance:
    """Physiological parameter record with formal literature and license provenance."""

    muscle_name: str
    F_max: float
    l_opt: float
    l_slack: float
    pennation_angle: float
    source_citation: str
    license_terms: str
    parameter_hash: str = ""

    def __post_init__(self) -> None:
        """Validate physiological parameter ranges and compute deterministic digest."""
        if not np.isfinite(self.F_max) or self.F_max <= 0.0:
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': F_max must be finite and positive, got {self.F_max}"
            )
        if not np.isfinite(self.l_opt) or self.l_opt <= 0.0:
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': l_opt must be finite and positive, got {self.l_opt}"
            )
        if not np.isfinite(self.l_slack) or self.l_slack <= 0.0:
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': l_slack must be finite and positive, got {self.l_slack}"
            )
        if (
            not np.isfinite(self.pennation_angle)
            or self.pennation_angle < 0.0
            or self.pennation_angle >= HALF_PI_RAD
        ):
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': pennation angle must be in [0, pi/2), got {self.pennation_angle}"
            )
        if not self.source_citation.strip():
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': source citation must not be empty"
            )
        if not self.license_terms.strip():
            raise InvalidMuscleParameterError(
                f"Muscle '{self.muscle_name}': license terms must not be empty"
            )

        if not self.parameter_hash:
            raw = (
                f"{self.muscle_name}|{self.F_max:.4f}|{self.l_opt:.6f}|"
                f"{self.l_slack:.6f}|{self.pennation_angle:.6f}|"
                f"{self.source_citation}|{self.license_terms}"
            )
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            object.__setattr__(self, "parameter_hash", digest)


@dataclass(frozen=True)
class MusclePathGeometry:
    """Path point coordinates and wrapping surfaces defining MTU geometry."""

    muscle_name: str
    path_points: tuple[tuple[str, tuple[float, float, float]], ...]
    wrapping_surfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class MuscleEquilibriumState:
    """Static muscle-tendon force equilibrium evaluation."""

    muscle_name: str
    fiber_length_m: float
    tendon_length_m: float
    activation: float
    residual_force_n: float
    is_equilibrated: bool


@dataclass(frozen=True)
class NativeShortReplayReceipt:
    """Metrics container requiring independent native verification.

    Constructing this object does not execute a simulator or establish that its
    supplied measurements are native evidence. Qualification belongs to the
    external verification authority; the default status is unverified.
    """

    variant_id: str
    start_time_s: float
    end_time_s: float
    num_steps: int
    reserve_actuator_torques_rms: dict[str, float]
    pelvic_residual_forces_rms: tuple[float, float, float]
    pelvic_residual_moments_rms: tuple[float, float, float]
    status: str = "UNVERIFIED_NATIVE_REPLAY"


@dataclass(frozen=True)
class MuscleQualificationReceipt:
    """Partial muscle/tendon audit results, not automatic acceptance.

    A false audit field means the audit has not established a pass; it includes
    checks not performed. Missing native evidence is None, never zero metrics.
    """

    model_variant_id: str
    base_model_sha256: str
    coverage_scope: AnatomyCoverageScope
    parameter_audit_passed: bool
    path_wrapping_audit_passed: bool
    moment_arm_validation_passed: bool
    equilibrium_audit_passed: bool
    activation_bounds_audit_passed: bool
    short_replay_receipt: NativeShortReplayReceipt | None
    muscle_complete_status: str = "IN_PROGRESS_QUALIFICATION"
    independent_validation_status: str = "PENDING_10375"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        """Compute deterministic SHA-256 digest of qualification receipt."""
        if not self.receipt_sha256:
            scope = self.coverage_scope
            raw = (
                f"{self.model_variant_id}|{self.base_model_sha256}|"
                f"{scope.is_full_golf_model}|{self.parameter_audit_passed}|"
                f"{self.equilibrium_audit_passed}|{self.muscle_complete_status}"
            )
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            object.__setattr__(self, "receipt_sha256", digest)


def audit_anatomy_coverage(
    muscle_names: Sequence[str],
    claimed_capability: str = "full_body_golf",
    declared_omissions: Sequence[AnatomicalRegion] = (),
    omission_notes: Mapping[str, str] | None = None,
) -> AnatomyCoverageScope:
    """Audit muscle names against physiological anatomical regions and validate claims."""
    require(isinstance(muscle_names, Sequence), "muscle_names must be a sequence")
    by_region: dict[AnatomicalRegion, list[str]] = {r: [] for r in AnatomicalRegion}

    for name in muscle_names:
        lower_name = name.lower()
        matched = False
        for region, patterns in _REGION_PATTERNS.items():
            for pat in patterns:
                if pat in lower_name:
                    by_region[region].append(name)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            by_region[AnatomicalRegion.LOWER_EXTREMITY].append(name)

    covered = tuple(r for r, mlist in by_region.items() if len(mlist) > 0)
    omissions = tuple(declared_omissions)
    notes = dict(omission_notes or {})

    # Check for lower-limb only claiming full golf model
    is_lower_only = (
        all(r == AnatomicalRegion.LOWER_EXTREMITY for r in covered)
        and len(covered) == 1
    )
    if claimed_capability == "full_body_golf" and is_lower_only:
        raise UnsupportedAnatomyClaimError(
            "Unsupported anatomy claim: lower-limb model alone cannot claim "
            "full_body_golf capability. Golf biomechanics requires explicit "
            "upper-body, torso, and forearm/wrist musculature."
        )

    # Required regions for full golf model
    core_golf_regions = (
        AnatomicalRegion.LOWER_EXTREMITY,
        AnatomicalRegion.TORSO_SPINE,
        AnatomicalRegion.SHOULDER_SCAPULA,
        AnatomicalRegion.ARM_FOREARM,
        AnatomicalRegion.WRIST_HAND,
    )

    missing = [r for r in core_golf_regions if r not in covered and r not in omissions]
    if claimed_capability == "full_body_golf" and missing:
        missing_names = [r.value for r in missing]
        raise IncompleteAnatomyCoverageError(
            f"Incomplete anatomy coverage for full_body_golf: missing {missing_names}. "
            "Declare explicit omissions or include the corresponding musculature."
        )

    is_full = all(r in covered or r in omissions for r in core_golf_regions)

    return AnatomyCoverageScope(
        covered_regions=covered,
        declared_omissions=omissions,
        omission_notes=notes,
        is_full_golf_model=is_full,
        muscles_by_region={r: tuple(mlist) for r, mlist in by_region.items()},
    )


def validate_muscle_parameters(
    parameters: Sequence[MuscleParameterProvenance],
) -> dict[str, Any]:
    """Audit muscle parameters against physiological bounds and provenance standards."""
    require(len(parameters) > 0, "parameters list must not be empty")
    for param in parameters:
        # __post_init__ of MuscleParameterProvenance validates bounds, F_max, pennation
        if not param.parameter_hash:
            raise InvalidMuscleParameterError(
                f"Muscle '{param.muscle_name}' missing verified parameter SHA-256 digest"
            )

    return {
        "valid": True,
        "count": len(parameters),
        "disclaimer": (
            "Standard OpenSim scale tool scales bone lengths and tendon/fiber geometries, "
            "but does NOT qualify or scale maximum isometric force (F_max). "
            "Muscle strength scaling requires explicit PCSA / mass scaling laws."
        ),
    }


def validate_muscle_path_and_wrapping(path: MusclePathGeometry) -> None:
    """Audit muscle path points and wrapping surfaces for physical validity."""
    points = path.path_points
    if len(points) < 2:
        raise InvalidMusclePathError(
            f"Muscle '{path.muscle_name}' path must have at least 2 points (origin and insertion), "
            f"got {len(points)}"
        )

    bodies = {pt[0] for pt in points}
    if len(bodies) < 2:
        raise InvalidMusclePathError(
            f"Muscle '{path.muscle_name}' path points must span at least 2 distinct parent bodies, "
            f"got {bodies}"
        )

    for body, coords in points:
        if len(coords) != 3 or not np.all(np.isfinite(coords)):
            raise InvalidMusclePathError(
                f"Muscle '{path.muscle_name}' on body '{body}' has non-finite or non-3D coordinates: {coords}"
            )


def compute_path_length_finite_difference_moment_arm(
    path_length_fn: Callable[[float], float],
    q: float,
    dq: float = 1e-5,
) -> float:
    """Compute muscle moment arm via negative central finite difference of path length.

    For a workless transmission, virtual work gives r_i(q) = -d(l_MT)/d(q_i).
    Moving guides can perform work: their length derivative need not equal the
    force-defined native moment arm without a separate guide-work model.
    """
    if not (math.isfinite(q) and math.isfinite(dq) and dq > 0.0):
        raise ValueError("q must be finite and dq must be finite and positive")
    if not (
        math.isfinite(q + dq)
        and math.isfinite(q - dq)
        and math.isfinite(2.0 * dq)
        and q - dq < q < q + dq
    ):
        raise ValueError("finite-difference stencil must be finite and resolvable")
    l_plus = path_length_fn(q + dq)
    l_minus = path_length_fn(q - dq)
    if not (math.isfinite(l_plus) and math.isfinite(l_minus)):
        raise ValueError("sampled path lengths must be finite")
    result = float(-(l_plus - l_minus) / (2.0 * dq))
    if not math.isfinite(result):
        raise ValueError("finite-difference moment arm must be finite")
    return result


def validate_moment_arm_consistency(
    moment_arm: float,
    path_length_fn: Callable[[float], float],
    q: float,
    tol: float = 1e-3,
    dq: float = 1e-5,
) -> None:
    """Audit the workless path identity at one declared numerical tolerance.

    The caller must establish the workless-transmission assumption before using
    this as moment-arm validation. A moving-guide rejection alone is not proof
    of a native force defect; guide work needs separate physical accounting.
    """
    if not math.isfinite(moment_arm) or not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("moment_arm must be finite and tol finite and positive")
    fd_arm = compute_path_length_finite_difference_moment_arm(path_length_fn, q, dq)
    diff = abs(moment_arm - fd_arm)
    if diff > tol:
        raise MomentArmDerivativeMismatchError(
            f"Moment arm ({moment_arm:.5f} m) deviates from finite-difference path length derivative "
            f"({fd_arm:.5f} m) by {diff:.5f} m > tol ({tol:.5f} m)"
        )


def _compute_hill_force(
    F_max: float,
    l_opt: float,
    l_slack: float,
    pennation_angle: float,
    l_CE: float,
    l_MT: float,
    activation: float,
) -> tuple[float, float, float]:
    """Compute (F_fiber_along_tendon, F_tendon, residual_force) using standard Hill curves."""
    cos_alpha = math.cos(pennation_angle)
    l_tendon = l_MT - l_CE * cos_alpha
    l_tendon_norm = l_tendon / l_slack
    l_CE_norm = l_CE / l_opt

    # Active force-length: Gaussian bell
    f_l = math.exp(-((l_CE_norm - 1.0) ** 2) / 0.45)
    # Passive force-length: quadratic stretch beyond optimal length
    f_p = ((l_CE_norm - 1.0) ** 2) / 0.45 if l_CE_norm > 1.0 else 0.0
    F_fiber = F_max * (activation * f_l + f_p)
    F_fiber_along_tendon = F_fiber * cos_alpha

    # Tendon force-strain curve (smooth continuous Millard/Thelen toe-linear model)
    strain = l_tendon_norm - 1.0
    if strain <= 0.0:
        f_t = 0.0
    elif strain <= 0.02:
        f_t = 875.0 * (strain**2)
    else:
        f_t = 0.35 + 35.0 * (strain - 0.02)

    F_tendon = F_max * f_t
    residual = F_fiber_along_tendon - F_tendon
    return F_fiber_along_tendon, F_tendon, residual


def audit_initial_muscle_equilibrium(
    params: MuscleParameterProvenance,
    l_MT: float,
    activation: float,
    initial_l_CE: float | None = None,
    tol_n: float = EQUILIBRIUM_FORCE_TOLERANCE_N,
    enforce_equilibrium: bool = True,
) -> MuscleEquilibriumState:
    """Solve or audit initial static muscle-tendon force equilibrium before simulation."""
    require(np.isfinite(l_MT) and l_MT > 0.0, "l_MT must be positive and finite")
    p = params
    cos_alpha = math.cos(p.pennation_angle)

    if initial_l_CE is not None:
        l_CE = initial_l_CE
        _, _, residual = _compute_hill_force(
            p.F_max,
            p.l_opt,
            p.l_slack,
            p.pennation_angle,
            l_CE,
            l_MT,
            activation,
        )
        is_equil = abs(residual) <= tol_n
        if enforce_equilibrium and not is_equil:
            raise UninitializedTendonStateError(
                f"Muscle '{p.muscle_name}' is not in initial static equilibrium: "
                f"residual force {residual:.3f} N exceeds tolerance {tol_n:.3f} N"
            )
        l_tendon = l_MT - l_CE * cos_alpha
        return MuscleEquilibriumState(
            muscle_name=p.muscle_name,
            fiber_length_m=l_CE,
            tendon_length_m=l_tendon,
            activation=activation,
            residual_force_n=residual,
            is_equilibrated=is_equil,
        )

    # Solve for equilibrium fiber length using bisection
    low = 0.4 * p.l_opt
    high = min(1.6 * p.l_opt, (l_MT - 0.5 * p.l_slack) / max(cos_alpha, 0.1))
    best_l_ce = p.l_opt
    best_res = 1e9

    for _ in range(60):
        mid = 0.5 * (low + high)
        _, _, res = _compute_hill_force(
            p.F_max,
            p.l_opt,
            p.l_slack,
            p.pennation_angle,
            mid,
            l_MT,
            activation,
        )
        if abs(res) < abs(best_res):
            best_res = res
            best_l_ce = mid
        if abs(res) < 1e-4:
            break
        # R(l_CE) is monotonically increasing: if res > 0, mid is too large (high = mid)
        if res > 0:
            high = mid
        else:
            low = mid

    l_tendon = l_MT - best_l_ce * cos_alpha
    return MuscleEquilibriumState(
        muscle_name=p.muscle_name,
        fiber_length_m=best_l_ce,
        tendon_length_m=l_tendon,
        activation=activation,
        residual_force_n=best_res,
        is_equilibrated=abs(best_res) <= tol_n,
    )


def audit_activation_dynamics(
    activations: Sequence[float],
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> None:
    """Audit normalized muscle activation levels against [min_activation, 1.0]."""
    for idx, act in enumerate(activations):
        if not np.isfinite(act) or act < min_activation - 1e-5 or act > 1.0 + 1e-5:
            raise InvalidMuscleParameterError(
                f"Activation at index {idx} ({act:.4f}) violates physiological bounds "
                f"[{min_activation:.2f}, 1.0]"
            )


def qualify_muscle_extensions(
    model_variant_id: str,
    base_model_sha256: str,
    muscles: Sequence[MuscleParameterProvenance],
    claimed_capability: str = "upper_body_pilot",
    declared_omissions: Sequence[AnatomicalRegion] = (),
    omission_notes: Mapping[str, str] | None = None,
    short_replay_duration_s: float = 0.05,
) -> MuscleQualificationReceipt:
    """Run anatomy and parameter audits without inventing model/runtime evidence.

    Inputs contain no native model, path geometry, state trajectory or verified
    replay receipt. Postcondition: only the checks actually performed can pass;
    all other audit fields are false and short_replay_receipt is None.

    short_replay_duration_s remains a compatibility argument describing a
    requested future experiment; this function does not execute that experiment.
    Actual native evidence must be evaluated through the F07/F08 verification
    boundary rather than synthesized from parameter records.
    """
    names = [m.muscle_name for m in muscles]
    coverage = audit_anatomy_coverage(
        muscle_names=names,
        claimed_capability=claimed_capability,
        declared_omissions=declared_omissions,
        omission_notes=omission_notes,
    )

    param_report = validate_muscle_parameters(muscles)
    param_passed = bool(param_report["valid"])

    return MuscleQualificationReceipt(
        model_variant_id=model_variant_id,
        base_model_sha256=base_model_sha256,
        coverage_scope=coverage,
        parameter_audit_passed=param_passed,
        path_wrapping_audit_passed=False,
        moment_arm_validation_passed=False,
        equilibrium_audit_passed=False,
        activation_bounds_audit_passed=False,
        short_replay_receipt=None,
        muscle_complete_status="IN_PROGRESS_QUALIFICATION",
        independent_validation_status="PENDING_10375",
    )
