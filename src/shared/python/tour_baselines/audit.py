"""Tour Target Audit, Marker Semantics, Events, and Provenance (TB-01 #10586).

Single source of truth for tour capture validation, 4-tier measurement maps,
native-clock swing event intervals, content-verified file duplicates, explicit
provenance records, and dual canonical target emitters.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    from src.motion_capture.reference.importers import MotionDraft

from src.shared.python.motion_matching.body_target import BodyEvent, BodyTarget
from src.shared.python.motion_matching.club_target import (
    AlignOptions,
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.loaders._marker_clusters import (
    CLUBHEAD_CLUSTER,
    GRIP_CLUSTER,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TOUR_CAPTURE_IRON,
    TOUR_CAPTURES,
    TourCapture,
    capture_kind,
    load_tour_capture,
)

from .audit_models import (
    EventInterval,
    EventPoint,
    MarkerMissingSpan,
    MeasurementCategory,
    MeasurementEntry,
    MeasurementMapping,
    ProvenanceRecord,
    SwingEventInterval,
    SwingEvents,
    TargetAudit,
    UnresolvedProvenanceItem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DUPLICATE_C3D_PATHS",
    "EventInterval",
    "EventPoint",
    "MarkerMissingSpan",
    "MeasurementCategory",
    "MeasurementEntry",
    "MeasurementMapping",
    "ProvenanceRecord",
    "SwingEventInterval",
    "SwingEvents",
    "TargetAudit",
    "UnresolvedProvenanceItem",
    "audit_tour_capture",
    "build_measurement_map",
    "detect_swing_intervals",
    "emit_dynamics_targets",
    "emit_reference_draft",
    "verify_duplicate_captures",
]

DUPLICATE_C3D_PATHS: dict[str, tuple[str, ...]] = {
    "driver": (
        "data/C3D_TA_Driver.c3d",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D Files/C3DExport Tour average.c3d",
        "src/engines/physics_engines/pinocchio/data/tour_average_mocap/C3DExport Tour average.c3d",
    ),
    "iron": (
        "data/C3D_TA_Iron.c3d",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D Files/C3DExport tour average iron.c3d",
        "src/engines/physics_engines/pinocchio/data/tour_average_mocap/C3DExport tour average iron.c3d",
    ),
}


def get_repo_root() -> Path:
    """Resolve repository root directory from module location."""
    return Path(__file__).resolve().parents[4]


def _resolve_data_path(kind: str) -> Path:
    """Locate canonical C3D file by kind."""
    root = get_repo_root()
    candidate = root / "data" / f"C3D_TA_{kind.capitalize()}.c3d"
    if candidate.exists():
        return candidate
    cwd_candidate = Path("data") / f"C3D_TA_{kind.capitalize()}.c3d"
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate C3D_TA_{kind.capitalize()}.c3d in {root} or {Path.cwd()}"
    )


def verify_duplicate_captures() -> dict[str, dict[str, str]]:
    """Verify all duplicate copies across the repository by content SHA-256."""
    root = get_repo_root()
    results: dict[str, dict[str, str]] = {"driver": {}, "iron": {}}
    for kind, paths in DUPLICATE_C3D_PATHS.items():
        for p_str in paths:
            p = root / p_str
            if not p.exists():
                p = Path(p_str)
            if p.exists():
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                results[kind][p_str] = digest
    return results


JOINT_PROXIES: dict[str, tuple[tuple[str, ...], str]] = {
    "mid_hip": (
        ("WaistLeft", "WaistRight"),
        "Centroid of left and right waist markers",
    ),
    "neck": (
        ("BackTop",),
        "Surface proxy at C7 / thoracic spine top, not anatomical center",
    ),
    "nose": (("HeadFront",), "Surface proxy at forehead/face front"),
    "left_shoulder": (
        ("LShoulderBack",),
        "Surface proxy avoiding anterior occlusions",
    ),
    "right_shoulder": (
        ("RShoulderBack",),
        "Posterior surface proxy avoiding occluded RShoulderTop",
    ),
    "left_elbow": (("LElbowOut",), "Lateral epicondyle surface proxy"),
    "right_elbow": (("RElbowOut",), "Lateral epicondyle surface proxy"),
    "left_wrist": (("LWristTop",), "Dorsal wrist surface proxy"),
    "right_wrist": (("RWristTop",), "Dorsal wrist surface proxy"),
    "left_hip": (("WaistLeft",), "ASIS surface proxy"),
    "right_hip": (("WaistRight",), "ASIS surface proxy"),
    "left_knee": (("LKneeOut",), "Lateral femoral epicondyle surface proxy"),
    "right_knee": (("RKneeOut",), "Lateral femoral epicondyle surface proxy"),
    "left_ankle": (("LAnkleOut",), "Lateral malleolus surface proxy"),
    "right_ankle": (("RAnkleOut",), "Lateral malleolus surface proxy"),
}


def build_measurement_map(kind: str) -> MeasurementMapping:
    """Build 4-tier measurement map distinguishing surface, proxies, clusters, and calibrated points."""
    spec = TOUR_CAPTURES[kind]
    entries: dict[str, MeasurementEntry] = {}

    for lbl in spec.labels:
        if lbl in (*CLUBHEAD_CLUSTER, *GRIP_CLUSTER):
            continue
        if lbl in ("Marker_0:0:0", "Uname*36", "Uname*37", "Uname*38"):
            entries[lbl] = MeasurementEntry(
                name=lbl,
                category=MeasurementCategory.OBSERVED_SURFACE,
                source_markers=(lbl,),
                notes="Unassigned / sentinel marker",
            )
        elif lbl == "pelvis":
            entries[lbl] = MeasurementEntry(
                name=lbl,
                category=MeasurementCategory.OBSERVED_SURFACE,
                source_markers=(lbl,),
                notes="Pelvis surface marker present only in iron capture (replaces Uname*38)",
            )
        else:
            entries[lbl] = MeasurementEntry(
                name=lbl,
                category=MeasurementCategory.OBSERVED_SURFACE,
                source_markers=(lbl,),
                notes="Vicon Plug-in-Gait skin-mounted surface marker",
            )

    for joint_name, (srcs, notes) in JOINT_PROXIES.items():
        entries[joint_name] = MeasurementEntry(
            name=joint_name,
            category=MeasurementCategory.INFERRED_JOINT_CENTER,
            source_markers=srcs,
            is_inferred=True,
            notes=f"Inferred joint center ({notes})",
        )

    entries["observed_club_head"] = MeasurementEntry(
        name="observed_club_head",
        category=MeasurementCategory.CLUSTER_CENTROID,
        source_markers=CLUBHEAD_CLUSTER,
        is_inferred=True,
        notes="Rigid cluster centroid of Marker_2:2:{1,2,3}",
    )
    entries["observed_club_grip"] = MeasurementEntry(
        name="observed_club_grip",
        category=MeasurementCategory.CLUSTER_CENTROID,
        source_markers=GRIP_CLUSTER,
        is_inferred=True,
        notes="Rigid cluster centroid of Marker_3:3:{1,2,3}",
    )

    entries["clubface_center"] = MeasurementEntry(
        name="clubface_center",
        category=MeasurementCategory.CALIBRATED_POINT,
        source_markers=(),
        is_inferred=True,
        is_available=False,
        notes="Calibrated clubface center is unavailable in raw C3D mocap; must not be guessed",
    )
    entries["ball_contact_point"] = MeasurementEntry(
        name="ball_contact_point",
        category=MeasurementCategory.CALIBRATED_POINT,
        source_markers=(),
        is_inferred=True,
        is_available=False,
        notes="Ball contact point is unavailable in raw C3D mocap; remains unavailable",
    )

    return MeasurementMapping(version="1.0", entries=entries)


def _compute_missing_spans(capture: TourCapture) -> dict[str, MarkerMissingSpan]:
    """Compute per-marker valid/missing counts and contiguous missing intervals."""
    missing_dict: dict[str, MarkerMissingSpan] = {}
    n_frames = capture.frames

    for i, label in enumerate(capture.labels):
        valid = capture.valid[:, i]
        n_valid = int(np.count_nonzero(valid))
        n_missing = n_frames - n_valid
        spans_list: list[tuple[int, int]] = []

        if n_missing > 0:
            in_gap = False
            start = 0
            for f in range(n_frames):
                if not valid[f]:
                    if not in_gap:
                        in_gap = True
                        start = f
                else:
                    if in_gap:
                        spans_list.append((start, f - 1))
                        in_gap = False
            if in_gap:
                spans_list.append((start, n_frames - 1))

        missing_dict[label] = MarkerMissingSpan(
            label=label,
            valid_samples=n_valid,
            missing_samples=n_missing,
            coverage_ratio=float(n_valid / n_frames),
            spans=tuple(spans_list),
        )
    return missing_dict


def detect_swing_intervals(capture: TourCapture, rate_hz: float) -> SwingEvents:
    """Compute swing event points and intervals on native clock with labeled inferred impact."""
    dt = 1.0 / rate_hz
    head_indices = [capture.index(lbl) for lbl in CLUBHEAD_CLUSTER]
    head_pts = capture.points_m[:, head_indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        head_centroid = np.nanmean(head_pts, axis=1)

    head_vel = np.gradient(head_centroid, dt, axis=0)
    head_speed = np.linalg.norm(head_vel, axis=1)
    impact_frame = int(np.nanargmax(head_speed))

    search_start = int(impact_frame * 0.5)
    top_frame = search_start + int(np.nanargmin(head_speed[search_start:impact_frame]))

    moving = np.where(head_speed[:top_frame] > 0.5)[0]
    takeaway_frame = int(moving[0]) if len(moving) > 0 else 0
    finish_frame = capture.frames - 1

    def make_interval(s_f: int, e_f: int) -> EventInterval:
        s_t = float(capture.time_s[s_f])
        e_t = float(capture.time_s[e_f])
        return EventInterval(
            start_frame=s_f,
            end_frame=e_f,
            start_time_s=s_t,
            end_time_s=e_t,
            duration_s=e_t - s_t,
        )

    return SwingEvents(
        rate_hz=rate_hz,
        address=make_interval(0, takeaway_frame),
        backswing=make_interval(takeaway_frame, top_frame),
        top=EventPoint(
            frame=top_frame,
            time_s=float(capture.time_s[top_frame]),
            is_inferred=True,
            detection_method="clubhead_transition_minimum_speed",
            confidence="high",
        ),
        downswing=make_interval(top_frame, impact_frame),
        impact=EventPoint(
            frame=impact_frame,
            time_s=float(capture.time_s[impact_frame]),
            is_inferred=True,
            detection_method="inferred_clubhead_speed_peak",
            confidence="high",
        ),
        follow_through=make_interval(impact_frame, finish_frame),
    )


def build_provenance_record(kind: str, path_str: str) -> ProvenanceRecord:
    """Build provenance record separating subject anatomy and capture geometry."""
    return ProvenanceRecord(
        source_file=path_str,
        capture_type="tour_average",
        averaging_normalization=(
            "Vicon Plug-in-Gait multi-trial / multi-subject tour average normalization. "
            f"Native sample rate: {TOUR_CAPTURES[kind].rate_hz} Hz."
        ),
        asserted_subject_anatomy={
            "representation": "Tour-average anthropometric surrogate",
            "height_m": 1.78,
            "mass_kg": 75.0,
            "notes": "Shared reference subject model assumptions across driver and iron captures",
        },
        capture_specific_geometry={
            "capture_kind": kind,
            "clubhead_markers": list(CLUBHEAD_CLUSTER),
            "grip_markers": list(GRIP_CLUSTER),
            "marker_count": len(TOUR_CAPTURES[kind].labels),
            "pelvis_marker": "pelvis" if kind == "iron" else "Uname*38",
        },
        unresolved_provenance=(
            UnresolvedProvenanceItem(
                field_name="capture_date",
                description="Exact calendar date and time of original optical capture session unrecorded in C3D headers",
            ),
            UnresolvedProvenanceItem(
                field_name="subject_demographics",
                description="Exact PGA Tour player demographic roster and cohort identity underlying normalized average",
            ),
            UnresolvedProvenanceItem(
                field_name="optical_calibration_parameters",
                description="Vicon camera system intrinsic/extrinsic calibration residuals and volume bounds not embedded",
            ),
            UnresolvedProvenanceItem(
                field_name="usage_license",
                description="Commercial/IRB research distribution terms unstated in source files",
            ),
        ),
    )


def audit_tour_capture(kind_or_path: str | Path) -> TargetAudit:
    """Perform a comprehensive audit of a canonical tour capture file."""
    if isinstance(kind_or_path, str) and kind_or_path in TOUR_CAPTURES:
        kind = kind_or_path
        path = _resolve_data_path(kind)
    else:
        path = Path(kind_or_path)
        if not path.is_absolute() and not path.exists():
            path = get_repo_root() / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        kind = capture_kind(digest)

    capture = load_tour_capture(path)
    spec = TOUR_CAPTURES[kind]
    m_map = build_measurement_map(kind)
    missing = _compute_missing_spans(capture)
    events = detect_swing_intervals(capture, spec.rate_hz)
    prov = build_provenance_record(kind, f"data/C3D_TA_{kind.capitalize()}.c3d")
    dups = verify_duplicate_captures()

    return TargetAudit(
        capture_kind=kind,
        sha256=capture.source_sha256 or spec.sha256,
        rate_hz=spec.rate_hz,
        frames=capture.frames,
        duration_s=float(capture.time_s[-1]),
        units=spec.units,
        vertical_axis=spec.vertical_axis,
        handedness="right_handed",
        labels=capture.labels,
        measurement_map=m_map,
        missing_spans=missing,
        events=events,
        provenance=prov,
        duplicate_copies=dups.get(kind, {}),
    )


def emit_reference_draft(
    audit: TargetAudit, capture: TourCapture | None = None
) -> MotionDraft:
    """Emit MotionDraft for kinematic reference fitting without introducing new C3D parsers."""
    from src.motion_capture.reference.importers import MotionDraft
    from src.motion_capture.reference.model import ReferenceSource

    if capture is None:
        path = _resolve_data_path(audit.capture_kind)
        capture = load_tour_capture(path)

    points = capture.points_m.copy()
    points[~capture.valid] = np.nan

    ref_source = ReferenceSource(
        path=audit.provenance.source_file,
        sha256=audit.sha256,
        format="c3d",
    )
    return MotionDraft(
        source=ref_source,
        names=capture.labels,
        time_s=tuple(float(t) for t in capture.time_s),
        points=points,
        source_units=audit.units,
        units_declared=True,
        canonical=True,
        model_identity=audit.capture_kind,
    )


def emit_dynamics_targets(
    audit: TargetAudit, capture: TourCapture | None = None
) -> tuple[BodyTarget, ClubTarget]:
    """Emit (BodyTarget, ClubTarget) for dynamics fitting without introducing new C3D parsers."""
    if capture is None:
        path = _resolve_data_path(audit.capture_kind)
        capture = load_tour_capture(path)

    # 1. BodyTarget: convert Y-up (Vicon) to right-handed Z-up: (x, y, z) -> (x, -z, y)
    y_up_pts = capture.points_m.copy()
    y_up_pts[~capture.valid] = np.nan
    z_up_pts = np.empty_like(y_up_pts)
    z_up_pts[..., 0] = y_up_pts[..., 0]
    z_up_pts[..., 1] = -y_up_pts[..., 2]
    z_up_pts[..., 2] = y_up_pts[..., 1]

    ev = audit.events
    impact_event = ev.impact
    top_event = ev.top
    impact_frame = impact_event.frame
    impact_time = impact_event.time_s
    top_frame = top_event.frame
    top_time = top_event.time_s

    body_events = (
        BodyEvent(
            label="impact",
            frame=impact_frame,
            time_s=impact_time,
        ),
        BodyEvent(label="top", frame=top_frame, time_s=top_time),
    )
    source_prov = SourceProvenance(
        filename=Path(audit.provenance.source_file).name,
        format="c3d",
        subject_id="tour_average",
        trial_id=audit.capture_kind,
        sha256=audit.sha256,
    )
    body_target = BodyTarget(
        time=capture.time_s,
        marker_xyz=z_up_pts,
        marker_names=capture.labels,
        impact_idx=impact_frame,
        events=body_events,
        source=source_prov,
        coordinate_frame="z_up_right_handed",
    )

    # 2. ClubTarget: clubhead and grip centroid in Z-up with explicit gap interpolation
    head_indices = [capture.index(lbl) for lbl in CLUBHEAD_CLUSTER]
    grip_indices = [capture.index(lbl) for lbl in GRIP_CLUSTER]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        head_raw = np.nanmean(z_up_pts[:, head_indices], axis=1)
        grip_raw = np.nanmean(z_up_pts[:, grip_indices], axis=1)
    head_filled = _interpolate_internal_nans(head_raw)
    grip_filled = _interpolate_internal_nans(grip_raw)

    # Unit quat placeholder (shaft orientation)
    shaft_quats = np.zeros((capture.frames, 4), dtype=np.float64)
    shaft_quats[:, 0] = 1.0

    club_target = ClubTarget(
        butt=grip_filled,
        clubhead=head_filled,
        club_quat=shaft_quats,
        time=capture.time_s,
        impact_idx=impact_frame,
        source=source_prov,
    )
    return body_target, club_target


def _interpolate_internal_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal NaN gaps to satisfy ClubTarget finite-array invariant."""
    out = arr.copy()
    n = out.shape[0]
    times = np.arange(n)
    valid = np.isfinite(out).all(axis=1)
    if not valid.any():
        return np.zeros_like(out)
    if valid.all():
        return out
    valid_idx = np.where(valid)[0]
    for col in range(out.shape[1]):
        out[:, col] = np.interp(times, times[valid_idx], out[valid_idx, col])
    return out
