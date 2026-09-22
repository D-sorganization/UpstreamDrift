"""Per-model club-only observation, physical and plausibility profiles (CO-02 #10606)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from src.shared.python.motion_matching.acceptance import AcceptanceGates
from src.shared.python.motion_matching.club_only.priors import GolfPlausibilityPriors
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.tour_baselines.models import ModelTopology
from src.shared.python.tour_baselines.registry import (
    init_default_registry,
    list_golf_models,
)

PROFILE_SCHEMA = "club-plausibility-acceptance/1.0.0"


class ObjectiveRule(str, Enum):
    """Frozen ranking rule for measured residual vs prior vs runtime."""

    LEXICOGRAPHIC_MEASURED_THEN_PRIOR = "lexicographic_measured_then_prior"
    PARETO_MEASURED_PRIOR_RUNTIME = "pareto_measured_prior_runtime"


@dataclass(frozen=True)
class ObservationProfile:
    """Numeric observation gates and unsupported component declarations."""

    supported_observables: frozenset[str]
    unsupported_components: frozenset[str]
    max_grip_position_rmse_m: float
    max_face_position_rmse_m: float
    max_grip_orientation_rmse_rad: float | None
    max_face_orientation_rmse_rad: float | None
    min_native_coverage_fraction: float
    max_speed_error_m_s: float | None
    max_phase_error_s: float | None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_grip_position_rmse_m", self.max_grip_position_rmse_m),
            ("max_face_position_rmse_m", self.max_face_position_rmse_m),
            ("min_native_coverage_fraction", self.min_native_coverage_fraction),
        ):
            if value != value or value < 0.0:  # NaN check without math import
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0.0 <= self.min_native_coverage_fraction <= 1.0:
            raise ValueError("min_native_coverage_fraction must be in [0, 1]")
        for name, optional in (
            ("max_grip_orientation_rmse_rad", self.max_grip_orientation_rmse_rad),
            ("max_face_orientation_rmse_rad", self.max_face_orientation_rmse_rad),
            ("max_speed_error_m_s", self.max_speed_error_m_s),
            ("max_phase_error_s", self.max_phase_error_s),
        ):
            if optional is not None and (optional != optional or optional < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 when set")
        if "face_orientation" in self.unsupported_components:
            if self.max_face_orientation_rmse_rad is not None:
                raise ValueError(
                    "unsupported face_orientation cannot declare an orientation gate"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported_observables": sorted(self.supported_observables),
            "unsupported_components": sorted(self.unsupported_components),
            "max_grip_position_rmse_m": self.max_grip_position_rmse_m,
            "max_face_position_rmse_m": self.max_face_position_rmse_m,
            "max_grip_orientation_rmse_rad": self.max_grip_orientation_rmse_rad,
            "max_face_orientation_rmse_rad": self.max_face_orientation_rmse_rad,
            "min_native_coverage_fraction": self.min_native_coverage_fraction,
            "max_speed_error_m_s": self.max_speed_error_m_s,
            "max_phase_error_s": self.max_phase_error_s,
        }


@dataclass(frozen=True)
class PhysicalProfile:
    """Unweighted physical feasibility limits (closure, contact, penetration)."""

    max_closure_residual_m: float
    max_penetration_m: float
    requires_contact_feasibility: bool
    requires_body_markers: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("max_closure_residual_m", self.max_closure_residual_m),
            ("max_penetration_m", self.max_penetration_m),
        ):
            if value != value or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_closure_residual_m": self.max_closure_residual_m,
            "max_penetration_m": self.max_penetration_m,
            "requires_contact_feasibility": self.requires_contact_feasibility,
            "requires_body_markers": self.requires_body_markers,
        }


@dataclass(frozen=True)
class PlausibilityProfile:
    """Named prior envelope attached to a model class."""

    priors: GolfPlausibilityPriors
    ambiguity_residual_tol_m: float
    min_diverse_candidates: int

    def __post_init__(self) -> None:
        if (
            self.ambiguity_residual_tol_m != self.ambiguity_residual_tol_m
            or self.ambiguity_residual_tol_m < 0.0
        ):
            raise ValueError("ambiguity_residual_tol_m must be finite and >= 0")
        if self.min_diverse_candidates < 1:
            raise ValueError("min_diverse_candidates must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "priors": self.priors.as_dict(),
            "ambiguity_residual_tol_m": self.ambiguity_residual_tol_m,
            "min_diverse_candidates": self.min_diverse_candidates,
        }


@dataclass(frozen=True)
class ClubOnlyProfile:
    """Frozen per-model club-only acceptance profile."""

    model_id: str
    topology: str
    schema: str
    observation: ObservationProfile
    physical: PhysicalProfile
    plausibility: PlausibilityProfile
    objective_rule: ObjectiveRule
    limitations: tuple[str, ...]
    preserves_full_body_g3: bool

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.schema != PROFILE_SCHEMA:
            raise ValueError(f"schema must be {PROFILE_SCHEMA!r}")
        if self.preserves_full_body_g3 and self.topology != (
            ModelTopology.FULL_BODY_MULTIBODY.value
        ):
            raise ValueError("only full-body multibody profiles preserve G3 gates")

    def max_closure_residual_m(self) -> float:
        """LoD facade for physical closure residual (avoids profile.physical.*)."""
        return float(self.physical.max_closure_residual_m)

    def golf_priors(self) -> GolfPlausibilityPriors:
        """LoD facade for attached plausibility priors."""
        return self.plausibility.priors

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "topology": self.topology,
            "schema": self.schema,
            "objective_rule": self.objective_rule.value,
            "preserves_full_body_g3": self.preserves_full_body_g3,
            "supported_observables": sorted(self.observation.supported_observables),
            "unsupported_components": sorted(self.observation.unsupported_components),
            "observation": self.observation.as_dict(),
            "physical": self.physical.as_dict(),
            "plausibility": self.plausibility.as_dict(),
            "limitations": list(self.limitations),
        }


def _acceptance_physical(
    *, requires_body: bool, requires_contact: bool
) -> PhysicalProfile:
    gates = AcceptanceGates()
    return PhysicalProfile(
        max_closure_residual_m=gates.max_closure_residual_m,
        max_penetration_m=gates.max_penetration_m,
        requires_contact_feasibility=requires_contact,
        requires_body_markers=requires_body,
    )


def _planar_observation() -> ObservationProfile:
    return ObservationProfile(
        supported_observables=frozenset(
            {"grip_position", "face_position", "native_coverage"}
        ),
        unsupported_components=frozenset({"face_orientation", "grip_orientation"}),
        max_grip_position_rmse_m=0.150,
        max_face_position_rmse_m=0.150,
        max_grip_orientation_rmse_rad=None,
        max_face_orientation_rmse_rad=None,
        min_native_coverage_fraction=0.80,
        max_speed_error_m_s=5.0,
        max_phase_error_s=0.05,
    )


def _upper_body_observation() -> ObservationProfile:
    return ObservationProfile(
        supported_observables=frozenset(
            {
                "grip_position",
                "face_position",
                "grip_orientation",
                "native_coverage",
            }
        ),
        unsupported_components=frozenset({"face_orientation"}),
        max_grip_position_rmse_m=0.080,
        max_face_position_rmse_m=0.080,
        max_grip_orientation_rmse_rad=0.12,
        max_face_orientation_rmse_rad=None,
        min_native_coverage_fraction=0.80,
        max_speed_error_m_s=3.0,
        max_phase_error_s=0.03,
    )


def _full_body_observation() -> ObservationProfile:
    gates = AcceptanceGates()
    return ObservationProfile(
        supported_observables=frozenset(
            {
                "grip_position",
                "face_position",
                "grip_orientation",
                "face_orientation",
                "body_markers",
                "native_coverage",
                "speed",
                "phase",
            }
        ),
        unsupported_components=frozenset(),
        max_grip_position_rmse_m=gates.g3_club_rmse_m,
        max_face_position_rmse_m=gates.g3_club_rmse_m,
        max_grip_orientation_rmse_rad=0.10,
        max_face_orientation_rmse_rad=0.10,
        min_native_coverage_fraction=gates.min_club_marker_coverage_fraction,
        max_speed_error_m_s=2.0,
        max_phase_error_s=0.02,
    )


def _reconstruction_observation() -> ObservationProfile:
    return ObservationProfile(
        supported_observables=frozenset(
            {"grip_position", "face_position", "native_coverage"}
        ),
        unsupported_components=frozenset({"face_orientation", "grip_orientation"}),
        max_grip_position_rmse_m=0.120,
        max_face_position_rmse_m=0.120,
        max_grip_orientation_rmse_rad=None,
        max_face_orientation_rmse_rad=None,
        min_native_coverage_fraction=0.80,
        max_speed_error_m_s=4.0,
        max_phase_error_s=0.04,
    )


def _reference_observation() -> ObservationProfile:
    return ObservationProfile(
        supported_observables=frozenset(
            {"grip_position", "face_position", "native_coverage"}
        ),
        unsupported_components=frozenset({"face_orientation"}),
        max_grip_position_rmse_m=0.120,
        max_face_position_rmse_m=0.120,
        max_grip_orientation_rmse_rad=0.15,
        max_face_orientation_rmse_rad=None,
        min_native_coverage_fraction=0.75,
        max_speed_error_m_s=None,
        max_phase_error_s=None,
    )


def _limitations_for(topology: ModelTopology) -> tuple[str, ...]:
    if topology is ModelTopology.PLANAR_DRIVEN_PENDULUM:
        return (
            "Planar topology cannot score out-of-plane face orientation; "
            "orientation residuals are recorded as limitations, not gates.",
            "Reduced-model club-only profiles do not satisfy full-body G3.",
        )
    if topology is ModelTopology.CONSTRAINED_UPPER_BODY:
        return (
            "Face orientation is unsupported without a calibrated face frame; "
            "orientation limitation is declared rather than a distorted target.",
            "Lower-extremity body markers are out of scope for this model class.",
            "Reduced-model club-only profiles do not satisfy full-body G3.",
        )
    if topology is ModelTopology.KINEMATIC_RECONSTRUCTION:
        return (
            "Kinematic reconstruction omits simulated club dynamics; "
            "orientation and force claims remain unsupported.",
            "Reduced-model club-only profiles do not satisfy full-body G3.",
        )
    if topology is ModelTopology.REFERENCE_CATALOG_URDF:
        return (
            "Catalog URDF/MJCF baselines are exploratory for club-only matching.",
            "Orientation/force qualification requires a native adapter receipt.",
        )
    return (
        "Full-body G3 gates remain binding and separate from club-only priors.",
        "Scientific qualification still requires native continuous replay evidence.",
    )


def _profile_for_model(model_id: str, topology: ModelTopology) -> ClubOnlyProfile:
    priors = GolfPlausibilityPriors.default()
    plausibility = PlausibilityProfile(
        priors=priors,
        ambiguity_residual_tol_m=0.005,
        min_diverse_candidates=2,
    )
    if topology is ModelTopology.PLANAR_DRIVEN_PENDULUM:
        observation = _planar_observation()
        physical = _acceptance_physical(requires_body=False, requires_contact=True)
        rule = ObjectiveRule.LEXICOGRAPHIC_MEASURED_THEN_PRIOR
        preserves_g3 = False
    elif topology is ModelTopology.CONSTRAINED_UPPER_BODY:
        observation = _upper_body_observation()
        physical = _acceptance_physical(requires_body=False, requires_contact=True)
        rule = ObjectiveRule.PARETO_MEASURED_PRIOR_RUNTIME
        preserves_g3 = False
    elif topology is ModelTopology.FULL_BODY_MULTIBODY:
        observation = _full_body_observation()
        physical = _acceptance_physical(requires_body=True, requires_contact=True)
        rule = ObjectiveRule.LEXICOGRAPHIC_MEASURED_THEN_PRIOR
        preserves_g3 = True
    elif topology is ModelTopology.KINEMATIC_RECONSTRUCTION:
        observation = _reconstruction_observation()
        physical = _acceptance_physical(requires_body=False, requires_contact=False)
        rule = ObjectiveRule.LEXICOGRAPHIC_MEASURED_THEN_PRIOR
        preserves_g3 = False
    else:
        observation = _reference_observation()
        physical = _acceptance_physical(requires_body=False, requires_contact=False)
        rule = ObjectiveRule.PARETO_MEASURED_PRIOR_RUNTIME
        preserves_g3 = False

    return ClubOnlyProfile(
        model_id=model_id,
        topology=topology.value,
        schema=PROFILE_SCHEMA,
        observation=observation,
        physical=physical,
        plausibility=plausibility,
        objective_rule=rule,
        limitations=_limitations_for(topology),
        preserves_full_body_g3=preserves_g3,
    )


def build_roster_profiles() -> dict[str, ClubOnlyProfile]:
    """Return a profile for every registered golf model identity."""
    init_default_registry()
    return {
        model.model_id: _profile_for_model(model.model_id, model.topology)
        for model in list_golf_models()
    }


def resolve_roster_matrix_scope(
    *,
    model_ids: Sequence[str] | None = None,
    trial_ids: Sequence[str] | None = None,
) -> tuple[dict[str, ClubOnlyProfile], list[str], list[str]]:
    """Build the roster and resolve model/trial ids for matrix reports.

    Preconditions:
        ``model_ids`` / ``trial_ids``, when provided, must be non-string sequences
        of strings (empty sequences are allowed and preserved).

    Returns:
        ``(roster, models, trials)`` where omitted selectors default to the full
        registered golf-model roster and ``CANONICAL_TRIAL_SHEETS``.
    """
    if isinstance(model_ids, (str, bytes)):
        raise TypeError("model_ids must be a sequence of strings, not a bare string")
    if isinstance(trial_ids, (str, bytes)):
        raise TypeError("trial_ids must be a sequence of strings, not a bare string")
    init_default_registry()
    roster = build_roster_profiles()
    models = (
        list(model_ids)
        if model_ids is not None
        else [m.model_id for m in list_golf_models()]
    )
    trials = list(trial_ids) if trial_ids is not None else list(CANONICAL_TRIAL_SHEETS)
    return roster, models, trials


def profile_from_roster(
    roster: dict[str, ClubOnlyProfile], model_id: str
) -> ClubOnlyProfile:
    """Resolve a profile from a prebuilt roster, falling back to registry lookup."""
    if not model_id:
        raise ValueError("model_id must be non-empty")
    if model_id in roster:
        return roster[model_id]
    return get_club_only_profile(model_id)


def get_club_only_profile(model_id: str) -> ClubOnlyProfile:
    """Lookup a frozen club-only profile by registered model id."""
    roster = build_roster_profiles()
    if model_id not in roster:
        raise ValueError(f"unknown club-only model_id={model_id!r}")
    return roster[model_id]


def evidence_payload(
    roster: dict[str, ClubOnlyProfile] | None = None,
) -> dict[str, Any]:
    """Serialize roster profiles into the CO-02 evidence receipt shape."""
    profiles = roster if roster is not None else build_roster_profiles()
    return {
        "schema": PROFILE_SCHEMA,
        "governing_issue": 10606,
        "objective_rules": [rule.value for rule in ObjectiveRule],
        "models": {
            model_id: profile.as_dict() for model_id, profile in profiles.items()
        },
        "notes": [
            "Profiles freeze supported observables and numeric gates before campaigns.",
            "Priors are named assumptions; synthetic labels are not force measurements.",
            "Full-body G3 gates stay in AcceptanceGates and are not relaxed here.",
        ],
    }
