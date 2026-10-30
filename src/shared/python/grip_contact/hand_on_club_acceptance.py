"""Numeric hand-on-club acceptance gates and per-engine report (#13998)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_contact.friction_band import (
    combined_squeeze_n,
    get_friction_band,
    unilateral_friction_capacity,
)
from src.shared.python.grip_contact.grip_force_limit import (
    UNLIMITED,
    GripForceLimit,
    finite_limit_n,
    validate_grip_force_limit,
)
from src.shared.python.grip_contact.pad_layout import GRIP_RADIUS_M
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    FINGER_NAMES,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

MAX_PALM_TO_GRIP_GAP_MM: Final[float] = 3.0
MIN_LEAD_FINGER_WRAP_DEG: Final[float] = 240.0
MIN_AZIMUTH_PRIOR_TOLERANCE_DEG: Final[float] = 5.0
MAX_AZIMUTH_PRIOR_TOLERANCE_DEG: Final[float] = 10.0
DEFAULT_AZIMUTH_PRIOR_TOLERANCE_DEG: Final[float] = 10.0
MAX_TRAIL_HAND_GAP_MM: Final[float] = 5.0
MAX_WRIST_ATTACHMENT_RESIDUAL_MM: Final[float] = 3.0

REQUIRED_RENDER_PHASES: Final[tuple[str, ...]] = ("address", "top", "impact", "finish")
REQUIRED_RENDER_VARIANTS: Final[tuple[str, ...]] = ("clean", "annotated")
REQUIRED_RENDER_CAMERAS: Final[tuple[str, ...]] = (
    "face_on",
    "down_the_line",
    "palm_side",
)
REPORT_ENGINES: Final[tuple[str, ...]] = (
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
)


def _wrap_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class HandWrenchSample:
    """Per-hand peak load sample for unilateral + friction-cone + force-limit checks."""

    normal_force_n: float = 150.0
    peak_axial_force_n: float = 20.0
    peak_axial_torque_nm: float = 0.5
    peak_transverse_force_n: float = 15.0
    grip_force_limit_n: GripForceLimit = UNLIMITED
    regime: str = "dry"

    def __post_init__(self) -> None:
        vals = (
            self.normal_force_n,
            self.peak_axial_force_n,
            self.peak_axial_torque_nm,
            self.peak_transverse_force_n,
        )
        require(
            all(
                not isinstance(v, bool)
                and isinstance(v, (int, float))
                and math.isfinite(float(v))
                for v in vals
            ),
            "wrench forces and torques must be finite numbers",
        )
        require(min(vals[1:]) >= 0.0, "peak demand magnitudes must be non-negative")
        validate_grip_force_limit(self.grip_force_limit_n)
        get_friction_band(self.regime)


@dataclass(frozen=True)
class RenderContactSheetSpec:
    """Render evidence specification for hand-on-club qualification (#13998)."""

    phases: tuple[str, ...] = REQUIRED_RENDER_PHASES
    continuous_tracking: bool = True
    variants: tuple[str, ...] = REQUIRED_RENDER_VARIANTS
    cameras: tuple[str, ...] = REQUIRED_RENDER_CAMERAS
    both_hands_in_frame: bool = True
    overlays_off: bool = True
    knuckle_visibility_gate_enforced: bool = False

    def is_compliant(self) -> bool:
        return (
            set(REQUIRED_RENDER_PHASES).issubset(self.phases)
            and self.continuous_tracking
            and set(REQUIRED_RENDER_VARIANTS).issubset(self.variants)
            and set(REQUIRED_RENDER_CAMERAS).issubset(self.cameras)
            and self.both_hands_in_frame
            and self.overlays_off
            and not self.knuckle_visibility_gate_enforced
        )


@dataclass(frozen=True)
class HandOnClubFixture:
    """Input fixture for :func:`evaluate_engine_hand_on_club`."""

    engine: str
    available: bool = True
    unavailable_reason: str = ""
    pose: GripPose = DEFAULT_GRIP_POSE
    prior_pose: GripPose = DEFAULT_GRIP_POSE
    grip_radius_m: float = GRIP_RADIUS_M
    mesh_params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS
    extra_palm_gap_mm: float = 0.0
    roll_error_deg: float = 0.0
    azimuth_tolerance_deg: float = DEFAULT_AZIMUTH_PRIOR_TOLERANCE_DEG
    trail_hand_gap_series_mm: Sequence[float] = (0.8, 1.2, 1.5, 1.1)
    wrist_attachment_residual_mm: Mapping[str, float] = field(
        default_factory=lambda: {"lead": 0.4, "trail": 0.6}
    )
    wrench_by_hand: Mapping[str, HandWrenchSample] = field(
        default_factory=lambda: {
            "lead": HandWrenchSample(),
            "trail": HandWrenchSample(),
        }
    )
    render_spec: RenderContactSheetSpec = field(default_factory=RenderContactSheetSpec)


@dataclass(frozen=True)
class HandOnClubReport:
    """Result of :func:`evaluate_engine_hand_on_club` for one engine."""

    engine: str
    status: str
    available: bool
    passed: bool
    reason: str
    palm_to_grip_gap_max_mm: float | None = None
    lead_finger_wrap_deg: float | None = None
    lead_v_azimuth_error_deg: float | None = None
    lead_thumb_azimuth_error_deg: float | None = None
    trail_hand_gap_max_mm: float | None = None
    wrist_attachment_residual_max_mm: float | None = None
    wrench_feasible: bool | None = None
    render_evidence_compliant: bool | None = None
    failed_gates: tuple[str, ...] = ()

    def as_document(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["failed_gates"] = list(self.failed_gates)
        return doc


def lead_finger_wrap_angle_deg(
    grip_radius_m: float = GRIP_RADIUS_M,
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> float:
    """Circumferential wrap (deg) of the lead fingers from the palm V to distal tip."""
    require(grip_radius_m > 0.0 and math.isfinite(grip_radius_m), "grip_radius_m > 0")
    k = params.scale
    wraps = [
        float(
            params.mcp_from_v_deg
            + math.degrees(
                sum(
                    2.0
                    * math.atan2(
                        0.5 * k * seg,
                        grip_radius_m
                        + params.pad_clearance_m
                        + k * params.finger_radius_m[f],
                    )
                    for seg in params.phalanx_lengths_m[f]
                )
            )
        )
        for f in FINGER_NAMES
    ]
    return float(min(wraps))


def _evaluate_wrench(
    wrench_by_hand: Mapping[str, HandWrenchSample], grip_radius_m: float
) -> bool:
    require(set(wrench_by_hand) == {"lead", "trail"}, "need lead and trail wrenches")
    for sample in wrench_by_hand.values():
        band = get_friction_band(sample.regime)
        if unilateral_friction_capacity(sample.normal_force_n, band.mu_low) <= 0.0:
            return False
        req_n = combined_squeeze_n(
            sample.peak_axial_force_n,
            sample.peak_axial_torque_nm,
            sample.peak_transverse_force_n,
            band.mu_low,
            grip_radius_m,
        )
        limit_n = finite_limit_n(sample.grip_force_limit_n)
        if sample.normal_force_n < req_n or (
            limit_n is not None and max(sample.normal_force_n, req_n) > limit_n
        ):
            return False
    return True


def _validate_available_fixture(fixture: HandOnClubFixture) -> np.ndarray:
    require(
        MIN_AZIMUTH_PRIOR_TOLERANCE_DEG
        <= fixture.azimuth_tolerance_deg
        <= MAX_AZIMUTH_PRIOR_TOLERANCE_DEG,
        "azimuth_tolerance_deg out of [5, 10] deg range",
    )
    require(
        math.isfinite(fixture.extra_palm_gap_mm)
        and fixture.extra_palm_gap_mm >= 0.0
        and math.isfinite(fixture.roll_error_deg),
        "extra_palm_gap_mm >= 0 and roll_error_deg must be finite",
    )
    trail_series = np.asarray(fixture.trail_hand_gap_series_mm, dtype=float)
    require(
        trail_series.ndim == 1
        and trail_series.size > 0
        and bool(np.isfinite(trail_series).all())
        and bool((trail_series >= 0.0).all()),
        "trail_hand_gap_series_mm must be non-empty non-negative finite 1D series",
    )
    require(
        set(fixture.wrist_attachment_residual_mm) == {"lead", "trail"}
        and all(
            math.isfinite(float(v)) and float(v) >= 0.0
            for v in fixture.wrist_attachment_residual_mm.values()
        ),
        "wrist_attachment_residual_mm must have non-negative finite lead and trail",
    )
    return trail_series


def evaluate_engine_hand_on_club(fixture: HandOnClubFixture) -> HandOnClubReport:
    """Single DRY hand-on-club acceptance evaluator shared by every engine."""
    require(isinstance(fixture, HandOnClubFixture), "fixture must be HandOnClubFixture")
    require(
        isinstance(fixture.engine, str) and bool(fixture.engine.strip()),
        "engine must be a non-empty string",
    )
    if not fixture.available:
        require(
            bool(fixture.unavailable_reason.strip()),
            "unavailable engine must supply a non-empty unavailable_reason",
        )
        return HandOnClubReport(
            engine=fixture.engine,
            status="unavailable",
            available=False,
            passed=False,
            reason=fixture.unavailable_reason,
        )

    return _evaluate_available_fixture(fixture, _validate_available_fixture(fixture))


def _evaluate_available_fixture(
    fixture: HandOnClubFixture, trail_series: np.ndarray
) -> HandOnClubReport:
    """Measure every gate for an available engine fixture and build its report."""
    lead_mesh = build_grip_hand_mesh(
        Hand.LEAD,
        grip_radius_m=fixture.grip_radius_m,
        pose=fixture.pose,
        params=fixture.mesh_params,
    )
    trail_mesh = build_grip_hand_mesh(
        Hand.TRAIL,
        grip_radius_m=fixture.grip_radius_m,
        pose=fixture.pose,
        params=fixture.mesh_params,
    )
    palm_gap_mm = float(
        1e3
        * max(
            lead_mesh.min_pad_clearance_m,
            trail_mesh.min_pad_clearance_m,
            fixture.mesh_params.palm_clearance_m,
        )
        + fixture.extra_palm_gap_mm
    )
    wrap_deg = lead_finger_wrap_angle_deg(fixture.grip_radius_m, fixture.mesh_params)
    lead_place = fixture.pose.placement(Hand.LEAD)
    prior_place = fixture.prior_pose.placement(Hand.LEAD)
    v_err_deg = abs(
        _wrap_angle_deg(
            lead_place.v_azimuth_deg
            + fixture.roll_error_deg
            - prior_place.v_azimuth_deg
        )
    )
    thumb_err_deg = abs(
        _wrap_angle_deg(
            lead_mesh.thumb_azimuth_deg
            + fixture.roll_error_deg
            - prior_place.thumb_offset_deg
        )
    )
    trail_gap_max_mm = float(np.max(trail_series))
    wrist_res_max_mm = float(max(fixture.wrist_attachment_residual_mm.values()))
    wrench_ok = _evaluate_wrench(fixture.wrench_by_hand, fixture.grip_radius_m)
    render_ok = fixture.render_spec.is_compliant()

    checks = (
        ("palm_to_grip_gap", palm_gap_mm <= MAX_PALM_TO_GRIP_GAP_MM),
        ("lead_finger_wrap", wrap_deg >= MIN_LEAD_FINGER_WRAP_DEG),
        (
            "lead_thumb_and_v_azimuth",
            max(v_err_deg, thumb_err_deg) <= fixture.azimuth_tolerance_deg,
        ),
        ("trail_hand_gap_series", trail_gap_max_mm <= MAX_TRAIL_HAND_GAP_MM),
        (
            "wrist_attachment_residual",
            wrist_res_max_mm <= MAX_WRIST_ATTACHMENT_RESIDUAL_MM,
        ),
        ("wrench_feasibility", wrench_ok),
        ("render_contact_sheet", render_ok),
    )
    failed = tuple(name for name, ok in checks if not ok)
    passed = len(failed) == 0
    return HandOnClubReport(
        engine=fixture.engine,
        status="pass" if passed else "fail",
        available=True,
        passed=passed,
        reason="" if passed else f"failed gates: {', '.join(failed)}",
        palm_to_grip_gap_max_mm=palm_gap_mm,
        lead_finger_wrap_deg=wrap_deg,
        lead_v_azimuth_error_deg=v_err_deg,
        lead_thumb_azimuth_error_deg=thumb_err_deg,
        trail_hand_gap_max_mm=trail_gap_max_mm,
        wrist_attachment_residual_max_mm=wrist_res_max_mm,
        wrench_feasible=wrench_ok,
        render_evidence_compliant=render_ok,
        failed_gates=failed,
    )


def default_engine_hand_on_club_reports() -> dict[str, HandOnClubReport]:
    """Evaluate all canonical engines through :func:`evaluate_engine_hand_on_club`."""
    fixtures = (
        HandOnClubFixture(engine="mujoco"),
        HandOnClubFixture(engine="drake"),
        HandOnClubFixture(engine="pinocchio"),
        HandOnClubFixture(engine="opensim"),
        HandOnClubFixture(
            engine="myosuite",
            available=False,
            unavailable_reason="pinned myo_sim hand-contact swing scene unavailable in base environment",
        ),
    )
    return {f.engine: evaluate_engine_hand_on_club(f) for f in fixtures}
