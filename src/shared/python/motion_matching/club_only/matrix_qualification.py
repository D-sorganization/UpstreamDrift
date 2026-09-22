"""Club-only matrix qualification and plausibility tradeoffs (CO-08 #10612).

Independently evaluate exported candidates over native observation times for
every workbook trial × #10585 roster cell. Freezes CO-02 numeric gates; visual
attractiveness cannot override physical failure. Software-contract fixtures
only — native qualification requires a separate receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.tour_baselines.registry import (
    init_default_registry,
    list_golf_models,
)
from src.shared.python.motion_matching.club_only.acceptance import (
    ClubOnlyResidualReport,
    evaluate_club_only_acceptance,
)
from src.shared.python.motion_matching.club_only.ambiguity import CandidateScore
from src.shared.python.motion_matching.club_only.observation import (
    ComponentStatus,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import (
    ClubOnlyProfile,
    build_roster_profiles,
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

MATRIX_SCHEMA = "club-matrix-qualification/1.0.0"
_GOVERNING_ISSUE = 10612
_DEFAULT_NATIVE_BLOCKERS: tuple[str, ...] = (
    "native_g1_qualification_requires_desk_native_receipt",
    "software_contract_matrix_is_not_native_evidence",
)

__all__ = [
    "MATRIX_SCHEMA",
    "ExportedCandidatePackage",
    "MatrixCellResult",
    "MatrixQualificationReport",
    "OrientationClaim",
    "PhaseCoverage",
    "WithheldBodyComparison",
    "WithheldBodyVerdict",
    "assert_honest_native_status",
    "assert_independent_evaluation",
    "assert_no_hidden_body_labels",
    "assert_no_target_resets",
    "assert_orientation_semantics",
    "assert_phase_coverage",
    "build_matrix_qualification_report",
    "compare_common_observables",
    "evaluate_withheld_body_experiment",
    "matrix_qualification_evidence_payload",
    "physical_overrides_visual",
    "validate_package_integrity",
]


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OrientationClaim:
    """Declared orientation component semantics for independent review."""

    component: str
    declared_status: ComponentStatus
    scored_as_measured: bool

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("component must be non-empty")
        if not isinstance(self.declared_status, ComponentStatus):
            raise TypeError("declared_status must be ComponentStatus")

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "declared_status": self.declared_status.value,
            "scored_as_measured": self.scored_as_measured,
        }


@dataclass(frozen=True)
class PhaseCoverage:
    """Required swing-phase presence for matrix scoring."""

    address_present: bool
    top_present: bool
    impact_present: bool
    finish_present: bool
    phase_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.phase_labels:
            raise ValueError("phase_labels must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "address_present": self.address_present,
            "top_present": self.top_present,
            "impact_present": self.impact_present,
            "finish_present": self.finish_present,
            "phase_labels": list(self.phase_labels),
        }


@dataclass(frozen=True)
class WithheldBodyComparison:
    """Optional post-fit body comparison for the withheld-body experiment."""

    body_markers_used_in_fit: bool
    optional_body_rmse_m: float
    club_only_feasible: bool
    claims_true_body_identified: bool

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.optional_body_rmse_m)
            or self.optional_body_rmse_m < 0
        ):
            raise ValueError("optional_body_rmse_m must be finite and >= 0")


@dataclass(frozen=True)
class WithheldBodyVerdict:
    """Outcome of the withheld-body experiment."""

    club_only_accepted: bool
    body_disagreement_invalidates: bool
    identifies_true_body: bool


@dataclass(frozen=True)
class ExportedCandidatePackage:
    """Candidate package shaped for fresh-process independent evaluation."""

    candidate_id: str
    trial_id: str
    model_id: str
    geometry_hash: str
    profile_hash: str
    timestamps_s: NDArray[np.floating]
    q0: NDArray[np.floating]
    v0: NDArray[np.floating]
    grip_position_rmse_m: float
    face_position_rmse_m: float
    original_3d_rmse_m: float
    in_plane_rmse_m: float
    out_of_plane_rmse_m: float
    grip_orientation_rmse_rad: float | None
    face_orientation_rmse_rad: float | None
    native_coverage_fraction: float
    phase_error_s: float | None
    closure_residual_m: float
    contact_feasible: bool
    used_measured_state_reset: bool
    body_labels_hidden: bool
    body_marker_status: str
    orientation_claims: tuple[OrientationClaim, ...]
    phases: PhaseCoverage
    fitting_prior_trial_ids: tuple[str, ...]
    unsupported_components: frozenset[str]
    supported_observables: frozenset[str]
    native_g1_pass: bool
    claims_native_qualification: bool
    qualification_blockers: tuple[str, ...]
    visual_attractiveness: float
    physical_failed: bool
    package_content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.trial_id or not self.model_id:
            raise ValueError("candidate_id, trial_id, and model_id required")
        times = np.asarray(self.timestamps_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("timestamps_s must be 1-D with >= 2 samples")
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
            raise ValueError("timestamps_s must be finite and strictly increasing")
        object.__setattr__(self, "timestamps_s", times)
        object.__setattr__(self, "q0", np.asarray(self.q0, dtype=np.float64))
        object.__setattr__(self, "v0", np.asarray(self.v0, dtype=np.float64))
        for name, value in (
            ("grip_position_rmse_m", self.grip_position_rmse_m),
            ("face_position_rmse_m", self.face_position_rmse_m),
            ("original_3d_rmse_m", self.original_3d_rmse_m),
            ("in_plane_rmse_m", self.in_plane_rmse_m),
            ("out_of_plane_rmse_m", self.out_of_plane_rmse_m),
            ("native_coverage_fraction", self.native_coverage_fraction),
            ("closure_residual_m", self.closure_residual_m),
            ("visual_attractiveness", self.visual_attractiveness),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0.0 <= self.native_coverage_fraction <= 1.0:
            raise ValueError("native_coverage_fraction must be in [0, 1]")
        if not 0.0 <= self.visual_attractiveness <= 1.0:
            raise ValueError("visual_attractiveness must be in [0, 1]")
        if self.body_marker_status not in {"withheld", "present", "absent"}:
            raise ValueError("body_marker_status must be withheld, present, or absent")
        if not self.package_content_hash:
            object.__setattr__(
                self, "package_content_hash", self.compute_content_hash()
            )

    def integrity_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "geometry_hash": self.geometry_hash,
            "profile_hash": self.profile_hash,
            "timestamps_s": self.timestamps_s.tolist(),
            "q0": self.q0.tolist(),
            "v0": self.v0.tolist(),
            "grip_position_rmse_m": self.grip_position_rmse_m,
            "face_position_rmse_m": self.face_position_rmse_m,
            "original_3d_rmse_m": self.original_3d_rmse_m,
            "in_plane_rmse_m": self.in_plane_rmse_m,
            "out_of_plane_rmse_m": self.out_of_plane_rmse_m,
            "grip_orientation_rmse_rad": self.grip_orientation_rmse_rad,
            "face_orientation_rmse_rad": self.face_orientation_rmse_rad,
            "native_coverage_fraction": self.native_coverage_fraction,
            "phase_error_s": self.phase_error_s,
            "closure_residual_m": self.closure_residual_m,
            "contact_feasible": self.contact_feasible,
            "used_measured_state_reset": self.used_measured_state_reset,
            "body_labels_hidden": self.body_labels_hidden,
            "body_marker_status": self.body_marker_status,
            "orientation_claims": [c.as_dict() for c in self.orientation_claims],
            "phases": self.phases.as_dict(),
            "fitting_prior_trial_ids": list(self.fitting_prior_trial_ids),
            "unsupported_components": sorted(self.unsupported_components),
            "supported_observables": sorted(self.supported_observables),
            "native_g1_pass": self.native_g1_pass,
            "claims_native_qualification": self.claims_native_qualification,
            "qualification_blockers": list(self.qualification_blockers),
            "visual_attractiveness": self.visual_attractiveness,
            "physical_failed": self.physical_failed,
        }

    def compute_content_hash(self) -> str:
        return _sha256_payload(self.integrity_payload())


@dataclass(frozen=True)
class MatrixCellResult:
    """One model × trial matrix cell."""

    model_id: str
    trial_id: str
    status: str
    blocker: str | None = None
    original_3d_rmse_m: float | None = None
    unsupported_components: tuple[str, ...] | None = None
    gate_failures: tuple[str, ...] | None = None
    common_observables: Mapping[str, float] = field(default_factory=dict)
    physical_failed: bool = False
    visual_attractiveness: float = 0.0

    def __post_init__(self) -> None:
        allowed = {
            "scored",
            "rejected",
            "unsupported",
            "missing_runtime",
            "unqualified",
        }
        if self.status not in allowed:
            raise ValueError(f"invalid status={self.status!r}")
        if self.status != "scored" and not self.blocker:
            raise ValueError("non-scored cells require an explicit blocker")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "status": self.status,
            "blocker": self.blocker,
            "original_3d_rmse_m": self.original_3d_rmse_m,
            "unsupported_components": (
                list(self.unsupported_components)
                if self.unsupported_components is not None
                else None
            ),
            "gate_failures": (
                list(self.gate_failures) if self.gate_failures is not None else None
            ),
            "common_observables": dict(self.common_observables),
            "physical_failed": self.physical_failed,
            "visual_attractiveness": self.visual_attractiveness,
        }


@dataclass(frozen=True)
class MatrixQualificationReport:
    """Full trial × roster matrix qualification receipt."""

    schema: str
    governing_issue: int
    cells: tuple[MatrixCellResult, ...]
    limitations: tuple[str, ...]
    profile_gate_freeze_hash: str
    qualification_blockers: tuple[str, ...]
    claims_native_qualification: bool = False
    native_g1_pass: bool = False
    comparison_rule: str = "common_observables_not_weighted_objectives"

    def __post_init__(self) -> None:
        if self.schema != MATRIX_SCHEMA:
            raise ValueError(f"schema must be {MATRIX_SCHEMA}")
        if self.governing_issue != _GOVERNING_ISSUE:
            raise ValueError(f"governing_issue must be {_GOVERNING_ISSUE}")
        if not self.cells:
            raise ValueError("cells must be non-empty")
        if not self.qualification_blockers:
            raise ValueError("qualification_blockers required for software matrix")


def validate_package_integrity(
    package: ExportedCandidatePackage,
    *,
    profile: ClubOnlyProfile,
    geometry_hash: str,
) -> None:
    """Reject tampered packages, wrong profiles, or mismatched geometry."""
    if not isinstance(package, ExportedCandidatePackage):
        raise TypeError("package must be ExportedCandidatePackage")
    if package.model_id != profile.model_id:
        raise ValueError(
            f"profile model_id={profile.model_id!r} does not match "
            f"package model_id={package.model_id!r}"
        )
    expected_profile = profile_content_hash(profile)
    if package.profile_hash != expected_profile:
        raise ValueError("profile hash mismatch: tampered profile rejected")
    if package.geometry_hash != geometry_hash:
        raise ValueError(
            f"geometry hash mismatch: package={package.geometry_hash!r} "
            f"expected={geometry_hash!r}"
        )
    expected_pkg = package.compute_content_hash()
    if package.package_content_hash != expected_pkg:
        raise ValueError(
            "package_content_hash integrity failure: tampered package rejected"
        )


def assert_no_hidden_body_labels(package: ExportedCandidatePackage) -> None:
    if package.body_labels_hidden:
        raise ValueError("hidden body labels are not allowed in matrix evaluation")


def assert_independent_evaluation(package: ExportedCandidatePackage) -> None:
    if package.trial_id in package.fitting_prior_trial_ids:
        raise ValueError(
            "trial leakage: evaluation trial appears in fitting priors; "
            "independent evaluation rejected"
        )


def assert_phase_coverage(package: ExportedCandidatePackage) -> None:
    phases = package.phases
    missing: list[str] = []
    if not phases.address_present:
        missing.append("address")
    if not phases.top_present:
        missing.append("top")
    if not phases.impact_present:
        missing.append("impact")
    if not phases.finish_present:
        missing.append("finish")
    if missing:
        raise ValueError(f"missing required phase coverage: {', '.join(missing)}")


def assert_orientation_semantics(package: ExportedCandidatePackage) -> None:
    for claim in package.orientation_claims:
        if (
            claim.declared_status
            in {ComponentStatus.DERIVED, ComponentStatus.UNOBSERVED}
            and claim.scored_as_measured
        ):
            raise ValueError(
                f"inferred/derived orientation scored as measured for "
                f"{claim.component!r}"
            )


def assert_no_target_resets(package: ExportedCandidatePackage) -> None:
    if package.used_measured_state_reset:
        raise ValueError("target reset / measured-state injection rejected")


def assert_honest_native_status(package: ExportedCandidatePackage) -> None:
    if package.claims_native_qualification and not package.qualification_blockers:
        raise ValueError(
            "native claim without qualification blockers rejected for software matrix"
        )
    if package.native_g1_pass and package.claims_native_qualification:
        raise ValueError("false native status: software fixtures cannot pass native G1")
    if package.native_g1_pass:
        raise ValueError(
            "native G1 pass requires an empty-blocker desk receipt; "
            "software blockers present"
        )


def physical_overrides_visual(package: ExportedCandidatePackage) -> bool:
    """Visual attractiveness cannot override physical failure."""
    if package.physical_failed:
        raise ValueError(
            "visual attractiveness cannot override physical failure "
            f"(visual={package.visual_attractiveness})"
        )
    return True


def evaluate_withheld_body_experiment(
    package: ExportedCandidatePackage,
    comparison: WithheldBodyComparison,
) -> WithheldBodyVerdict:
    """Withheld-body disagreement must not invent true-body identity."""
    if comparison.body_markers_used_in_fit:
        raise ValueError(
            "body markers used in fit violate withheld-body experiment contract"
        )
    if comparison.claims_true_body_identified:
        raise ValueError(
            "cannot identify true body from withheld-body disagreement alone"
        )
    if package.body_marker_status != "withheld":
        raise ValueError("package body_marker_status must be withheld for this probe")
    return WithheldBodyVerdict(
        club_only_accepted=comparison.club_only_feasible,
        body_disagreement_invalidates=False,
        identifies_true_body=False,
    )


def compare_common_observables(
    packages: Sequence[ExportedCandidatePackage],
) -> frozenset[str]:
    """Return observable names supported by every package (not weighted objectives)."""
    if len(packages) < 2:
        raise ValueError("need at least two packages for common-observable comparison")
    common = set(packages[0].supported_observables)
    for package in packages[1:]:
        common &= set(package.supported_observables)
    return frozenset(common)


def _gate_failures_for(
    package: ExportedCandidatePackage, profile: ClubOnlyProfile
) -> tuple[str, ...]:
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=package.grip_position_rmse_m,
        face_position_rmse_m=package.face_position_rmse_m,
        grip_orientation_rmse_rad=package.grip_orientation_rmse_rad,
        face_orientation_rmse_rad=package.face_orientation_rmse_rad,
        native_coverage_fraction=package.native_coverage_fraction,
        speed_error_m_s=None,
        phase_error_s=package.phase_error_s,
        unweighted_physical={
            "closure_residual_m": package.closure_residual_m,
        },
    )
    candidate = CandidateScore(
        candidate_id=package.candidate_id,
        measured_residual_m=package.original_3d_rmse_m,
        prior_score=0.0,
        body_configuration_hash=_sha256_payload({"q0": package.q0.tolist()}),
        closure_residual_m=package.closure_residual_m,
        contact_feasible=package.contact_feasible,
        claims_force_measurement=False,
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        (candidate,),
        profile,
        body_markers_present=package.body_marker_status == "present",
        force_labels_synthetic=True,
        torque_replay_validated=False,
        kinematic_preview_ok=not package.used_measured_state_reset,
    )
    failures = [g.name for g in verdict.gates if not g.passed]
    if package.physical_failed:
        failures.append("physical_failed_flag")
    return tuple(failures)


def _common_observable_scores(package: ExportedCandidatePackage) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "grip_position" in package.supported_observables:
        scores["grip_position"] = package.grip_position_rmse_m
    if "face_position" in package.supported_observables:
        scores["face_position"] = package.face_position_rmse_m
    if "native_coverage" in package.supported_observables:
        scores["native_coverage"] = package.native_coverage_fraction
    if (
        "grip_orientation" in package.supported_observables
        and package.grip_orientation_rmse_rad is not None
    ):
        scores["grip_orientation"] = package.grip_orientation_rmse_rad
    if (
        "face_orientation" in package.supported_observables
        and package.face_orientation_rmse_rad is not None
    ):
        scores["face_orientation"] = package.face_orientation_rmse_rad
    scores["original_3d"] = package.original_3d_rmse_m
    scores["in_plane"] = package.in_plane_rmse_m
    scores["out_of_plane"] = package.out_of_plane_rmse_m
    return scores


_MISSING_RUNTIME: frozenset[str] = frozenset(
    {
        "full_body_opensim",
        "full_body_simscape",
        "full_body_myosuite",
        "opensim_golfer",
        "myosuite_body",
    }
)


def _cell_blocker_for(model_id: str, topology: str) -> tuple[str, str] | None:
    if model_id in _MISSING_RUNTIME:
        return (
            "missing_runtime",
            f"native runtime for {model_id} not available on agent host; "
            "unqualified pending desk native receipt",
        )
    if topology == "kinematic_reconstruction":
        return (
            "unsupported",
            "kinematic reconstruction is not club-only dynamics evidence",
        )
    if topology == "reference_catalog_urdf":
        return (
            "unsupported",
            "reference catalog URDF/MJCF requires a native adapter receipt",
        )
    if topology == "full_body_multibody":
        return (
            "unqualified",
            "full-body native G1 matrix cell requires desk-native receipt",
        )
    return None


def _synthetic_package(
    trial_id: str, profile: ClubOnlyProfile
) -> ExportedCandidatePackage:
    obs = build_calibrated_observation_fixture(trial_id)
    geo = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )
    return ExportedCandidatePackage(
        candidate_id=f"co08:{trial_id}:{profile.model_id}",
        trial_id=trial_id,
        model_id=profile.model_id,
        geometry_hash=geo,
        profile_hash=profile_content_hash(profile),
        timestamps_s=obs.native_time_s.copy(),
        q0=np.zeros(4, dtype=np.float64),
        v0=np.zeros(4, dtype=np.float64),
        grip_position_rmse_m=min(
            profile.observation.max_grip_position_rmse_m * 0.25, 0.02
        ),
        face_position_rmse_m=min(
            profile.observation.max_face_position_rmse_m * 0.25, 0.025
        ),
        original_3d_rmse_m=0.03,
        in_plane_rmse_m=0.02,
        out_of_plane_rmse_m=0.015,
        grip_orientation_rmse_rad=(
            0.05
            if profile.observation.max_grip_orientation_rmse_rad is not None
            else None
        ),
        face_orientation_rmse_rad=(
            0.05
            if profile.observation.max_face_orientation_rmse_rad is not None
            else None
        ),
        native_coverage_fraction=max(
            profile.observation.min_native_coverage_fraction, 0.95
        ),
        phase_error_s=0.01,
        closure_residual_m=min(profile.physical.max_closure_residual_m * 0.2, 0.001),
        contact_feasible=True,
        used_measured_state_reset=False,
        body_labels_hidden=False,
        body_marker_status="withheld",
        orientation_claims=(
            OrientationClaim(
                component="mid_hands_orientation",
                declared_status=ComponentStatus.MEASURED,
                scored_as_measured=True,
            ),
            OrientationClaim(
                component="face_orientation",
                declared_status=ComponentStatus.DERIVED,
                scored_as_measured=False,
            ),
        ),
        phases=PhaseCoverage(
            address_present=True,
            top_present=True,
            impact_present=True,
            finish_present=True,
            phase_labels=("A", "T", "I", "F"),
        ),
        fitting_prior_trial_ids=(),
        unsupported_components=frozenset(profile.observation.unsupported_components),
        supported_observables=frozenset(profile.observation.supported_observables),
        native_g1_pass=False,
        claims_native_qualification=False,
        qualification_blockers=_DEFAULT_NATIVE_BLOCKERS,
        visual_attractiveness=0.5,
        physical_failed=False,
    )


def _assert_package_contracts(
    package: ExportedCandidatePackage, *, profile: ClubOnlyProfile
) -> None:
    """Fail closed on integrity, independence, and honesty gates before scoring."""
    assert_independent_evaluation(package)
    assert_no_hidden_body_labels(package)
    assert_phase_coverage(package)
    assert_orientation_semantics(package)
    assert_no_target_resets(package)
    assert_honest_native_status(package)
    validate_package_integrity(
        package, profile=profile, geometry_hash=package.geometry_hash
    )
    physical_overrides_visual(package)


def _score_matrix_cell(
    *,
    model_id: str,
    trial_id: str,
    profile: ClubOnlyProfile,
) -> MatrixCellResult:
    """Score one model×trial cell under frozen CO-02 gates."""
    package = _synthetic_package(trial_id, profile)
    _assert_package_contracts(package, profile=profile)
    failures = _gate_failures_for(package, profile)
    unsupported = tuple(sorted(package.unsupported_components))
    observables = _common_observable_scores(package)
    if failures or package.physical_failed:
        return MatrixCellResult(
            model_id=model_id,
            trial_id=trial_id,
            status="rejected",
            blocker="; ".join(failures) or "physical_failed",
            original_3d_rmse_m=package.original_3d_rmse_m,
            unsupported_components=unsupported,
            gate_failures=failures,
            common_observables=observables,
            physical_failed=True,
            visual_attractiveness=package.visual_attractiveness,
        )
    return MatrixCellResult(
        model_id=model_id,
        trial_id=trial_id,
        status="scored",
        original_3d_rmse_m=package.original_3d_rmse_m,
        unsupported_components=unsupported,
        gate_failures=(),
        common_observables=observables,
        physical_failed=False,
        visual_attractiveness=package.visual_attractiveness,
    )


def build_matrix_qualification_report(
    *,
    model_ids: Sequence[str] | None = None,
    trial_ids: Sequence[str] | None = None,
) -> MatrixQualificationReport:
    """Build the trial × roster matrix with frozen CO-02 gates."""
    init_default_registry()
    roster = build_roster_profiles()
    models = (
        list(model_ids)
        if model_ids is not None
        else [m.model_id for m in list_golf_models()]
    )
    trials = list(trial_ids) if trial_ids is not None else list(CANONICAL_TRIAL_SHEETS)
    freeze_payload = {
        model_id: roster[model_id].as_dict()
        for model_id in models
        if model_id in roster
    }
    freeze_hash = _sha256_payload(freeze_payload)
    cells: list[MatrixCellResult] = []
    for model_id in models:
        profile = (
            roster[model_id] if model_id in roster else get_club_only_profile(model_id)
        )
        blocked = _cell_blocker_for(model_id, profile.topology)
        for trial_id in trials:
            if blocked is not None:
                status, blocker = blocked
                cells.append(
                    MatrixCellResult(
                        model_id=model_id,
                        trial_id=trial_id,
                        status=status,
                        blocker=blocker,
                    )
                )
                continue
            cells.append(
                _score_matrix_cell(
                    model_id=model_id, trial_id=trial_id, profile=profile
                )
            )
    return MatrixQualificationReport(
        schema=MATRIX_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        cells=tuple(cells),
        limitations=(
            "Software-contract matrix only; native G1 remains blocked without receipt.",
            "Withheld-body disagreement does not invalidate feasible club-only cells.",
            "Visual plausibility never overrides physical or measured gate failure.",
            "CO-02 profile numeric gates are frozen via profile_gate_freeze_hash.",
            "Common observables are compared across complexities; not weighted objectives.",
        ),
        profile_gate_freeze_hash=freeze_hash,
        qualification_blockers=_DEFAULT_NATIVE_BLOCKERS,
        claims_native_qualification=False,
        native_g1_pass=False,
    )


def matrix_qualification_evidence_payload(
    report: MatrixQualificationReport | None = None,
) -> dict[str, Any]:
    """Serialize the CO-08 evidence receipt."""
    built = report if report is not None else build_matrix_qualification_report()
    models: dict[str, Any] = {}
    for cell in built.cells:
        models.setdefault(cell.model_id, {"cells": []})
        models[cell.model_id]["cells"].append(cell.as_dict())
    return {
        "schema": built.schema,
        "governing_issue": built.governing_issue,
        "claims_native_qualification": built.claims_native_qualification,
        "native_g1_pass": built.native_g1_pass,
        "qualification_blockers": list(built.qualification_blockers),
        "profile_gate_freeze_hash": built.profile_gate_freeze_hash,
        "comparison_rule": built.comparison_rule,
        "cell_count": len(built.cells),
        "scored_count": sum(1 for c in built.cells if c.status == "scored"),
        "cells": [cell.as_dict() for cell in built.cells],
        "models": {
            model_id: {
                "model_id": model_id,
                "cell_count": len(payload["cells"]),
                "statuses": sorted({c["status"] for c in payload["cells"]}),
            }
            for model_id, payload in models.items()
        },
        "limitations": list(built.limitations),
        "notes": [
            "Independent evaluation rejects tampered profiles, geometry mismatch, "
            "trial leakage, inferred-as-measured orientation, target resets, and "
            "false native claims before scoring.",
            "Synthetic fixtures validate software contracts only.",
        ],
    }
