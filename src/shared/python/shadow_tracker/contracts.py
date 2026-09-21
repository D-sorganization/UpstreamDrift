"""Immutable evidence contracts and service protocols for Shadow Tracker (ST-02)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any, Final, Literal, Protocol, runtime_checkable

from ._validation import (
    CANDIDATE_RESULT_SCHEMA_VERSION,
    CAMERA_TRACK_SCHEMA_VERSION,
    FIT_REQUEST_SCHEMA_VERSION,
    FRAME_OBSERVATION_SCHEMA_VERSION,
    REPLAY_AUDIT_SCHEMA_VERSION,
    RESULT_BUNDLE_SCHEMA_VERSION,
    SHOT_SCHEMA_VERSION,
    SUBJECT_BINDING_SCHEMA_VERSION,
    check_bool,
    check_id,
    check_int,
    check_nonneg_float,
    check_optional_float,
    check_payload_keys,
    check_pos_float,
    check_pos_int,
    check_schema_version,
    check_sha256,
    check_str,
    check_strict_float,
)


# ---------------------------------------------------------------------------
# Type Aliases & Allowed Keys
# ---------------------------------------------------------------------------

ExecutionStatus = Literal["completed", "cancelled", "budget_exhausted", "failed"]
EvidenceQuality = Literal[
    "unreviewed",
    "kinematic_only",
    "dynamic_candidate",
    "validated_profile",
    "insufficient_evidence",
]
CameraTrackStatus = Literal["measured", "estimated", "unknown"]
Handedness = Literal["right", "left"]

_VALID_EXECUTION_STATUSES = frozenset(
    ("completed", "cancelled", "budget_exhausted", "failed")
)
_VALID_EVIDENCE_QUALITIES = frozenset(
    (
        "unreviewed",
        "kinematic_only",
        "dynamic_candidate",
        "validated_profile",
        "insufficient_evidence",
    )
)
_VALID_CAMERA_TRACK_STATUSES = frozenset(("measured", "estimated", "unknown"))
_VALID_HANDEDNESS = frozenset(("right", "left"))

_SHOT_KEYS = frozenset(
    (
        "schema_version",
        "asset_id",
        "shot_id",
        "start_pts",
        "end_pts",
        "start_frame_id",
        "end_frame_id",
        "subject_id",
        "swing_id",
        "camera_id",
        "cuts",
        "transforms",
    )
)

_FRAME_OBSERVATION_KEYS = frozenset(
    (
        "schema_version",
        "shot_id",
        "camera_id",
        "frame_id",
        "pts_ticks",
        "timebase_numerator",
        "timebase_denominator",
        "physical_time_s",
        "physical_time_reason",
        "body_mask_ref",
        "club_mask_ref",
        "valid_mask_ref",
        "confidence_provenance",
        "timing_mode",
        "is_timing_exact",
        "clock_evidence",
        "decoder_name",
        "decoder_version",
        "pixel_format",
    )
)

_LEGACY_FRAME_OBSERVATION_KEYS = frozenset(
    (
        "schema_version",
        "shot_id",
        "camera_id",
        "frame_id",
        "pts_ticks",
        "timebase_numerator",
        "timebase_denominator",
        "physical_time_s",
        "physical_time_reason",
        "body_mask_ref",
        "club_mask_ref",
        "valid_mask_ref",
        "confidence_provenance",
    )
)

_FRAME_OBSERVATION_SCHEMA_VERSIONS = frozenset(
    (
        FRAME_OBSERVATION_SCHEMA_VERSION,
        "shadow-tracker/frame-observation/1.1.0",
    )
)

_CAMERA_TRACK_KEYS = frozenset(
    (
        "schema_version",
        "camera_id",
        "shot_id",
        "frame_convention",
        "track_times",
        "status",
        "uncertainty",
        "image_transforms",
    )
)

_SUBJECT_BINDING_KEYS = frozenset(
    (
        "schema_version",
        "subject_id",
        "model_hash",
        "joint_ids",
        "body_ids",
        "visual_envelope",
        "mass_kg",
        "scale_evidence",
        "handedness",
    )
)

_FIT_REQUEST_KEYS = frozenset(
    (
        "schema_version",
        "request_id",
        "shot_id",
        "model_hash",
        "candidate_count",
        "objective_profile",
        "time_window_start_pts",
        "time_window_end_pts",
        "budget_seconds",
        "engine_capability_requirement",
    )
)

_REPLAY_AUDIT_KEYS = frozenset(
    (
        "schema_version",
        "candidate_id",
        "reset_count",
        "integrator_name",
        "integrator_version",
        "coverage_start_s",
        "coverage_end_s",
        "max_grip_translation_error_m",
        "max_grip_rotation_error_rad",
        "is_physically_accepted",
    )
)

_CANDIDATE_RESULT_KEYS = frozenset(
    (
        "schema_version",
        "candidate_id",
        "request_id",
        "initial_state",
        "trajectory",
        "diagnostics",
        "uncertainty_method",
        "replay_audit",
        "is_accepted",
    )
)

_RESULT_BUNDLE_KEYS = frozenset(
    (
        "schema_version",
        "bundle_id",
        "request",
        "candidates",
        "replay_audits",
        "execution_status",
        "evidence_quality",
        "metrics",
        "hashes",
    )
)


# ---------------------------------------------------------------------------
# Data Records (Slotted, Frozen Dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Shot:
    """Bounded continuous interval in a source asset."""

    schema_version: str
    asset_id: str
    shot_id: str
    start_pts: int
    end_pts: int
    start_frame_id: str
    end_frame_id: str
    subject_id: str
    swing_id: str
    camera_id: str
    cuts: tuple[tuple[int, int], ...]
    transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, SHOT_SCHEMA_VERSION)
        check_id(self.asset_id, "asset_id")
        check_id(self.shot_id, "shot_id")
        check_int(self.start_pts, "start_pts")
        check_int(self.end_pts, "end_pts")
        if self.start_pts > self.end_pts:
            raise ValueError(
                f"start_pts ({self.start_pts}) cannot exceed end_pts ({self.end_pts})"
            )
        check_id(self.start_frame_id, "start_frame_id")
        check_id(self.end_frame_id, "end_frame_id")
        check_id(self.subject_id, "subject_id")
        check_id(self.swing_id, "swing_id")
        check_id(self.camera_id, "camera_id")

        cuts_copy = []
        for c in self.cuts:
            if not isinstance(c, (tuple, list)) or len(c) != 2:
                raise TypeError(f"cut interval must be a 2-tuple of ints, got {c!r}")
            c0, c1 = check_int(c[0], "cut_start"), check_int(c[1], "cut_end")
            if c0 > c1:
                raise ValueError(f"cut interval start {c0} exceeds end {c1}")
            if c0 < self.start_pts or c1 > self.end_pts:
                raise ValueError(
                    f"cuts interval ({c0}, {c1}) outside shot range [{self.start_pts}, {self.end_pts}]"
                )
            cuts_copy.append((c0, c1))
        object.__setattr__(self, "cuts", tuple(cuts_copy))

        transforms_copy = []
        for t in self.transforms:
            transforms_copy.append(check_str(t, "transform"))
        object.__setattr__(self, "transforms", tuple(transforms_copy))

    @property
    def duration_pts(self) -> int:
        return self.end_pts - self.start_pts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_id": self.asset_id,
            "shot_id": self.shot_id,
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "subject_id": self.subject_id,
            "swing_id": self.swing_id,
            "camera_id": self.camera_id,
            "cuts": [list(c) for c in self.cuts],
            "transforms": list(self.transforms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Shot:
        check_payload_keys(payload, _SHOT_KEYS)
        raw_cuts = payload["cuts"]
        if not isinstance(raw_cuts, Sequence) or isinstance(raw_cuts, (str, bytes)):
            raise TypeError("cuts must be a sequence")
        cuts_tuple = tuple((int(c[0]), int(c[1])) for c in raw_cuts)
        raw_transforms = payload["transforms"]
        if not isinstance(raw_transforms, Sequence) or isinstance(
            raw_transforms, (str, bytes)
        ):
            raise TypeError("transforms must be a sequence")
        transforms_tuple = tuple(str(t) for t in raw_transforms)
        return cls(
            schema_version=payload["schema_version"],
            asset_id=payload["asset_id"],
            shot_id=payload["shot_id"],
            start_pts=payload["start_pts"],
            end_pts=payload["end_pts"],
            start_frame_id=payload["start_frame_id"],
            end_frame_id=payload["end_frame_id"],
            subject_id=payload["subject_id"],
            swing_id=payload["swing_id"],
            camera_id=payload["camera_id"],
            cuts=cuts_tuple,
            transforms=transforms_tuple,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameObservation:
    """Observation authority linking raw PTS to masks and physical time."""

    schema_version: str
    shot_id: str
    camera_id: str
    frame_id: str
    pts_ticks: int
    timebase_numerator: int
    timebase_denominator: int
    physical_time_s: float | None
    physical_time_reason: str
    body_mask_ref: str
    club_mask_ref: str
    valid_mask_ref: str
    confidence_provenance: str
    timing_mode: str = "estimated_cfr"
    is_timing_exact: bool = False
    clock_evidence: str = "unverified_legacy_record"
    decoder_name: str = "opencv"
    decoder_version: str = "legacy"
    pixel_format: str = "bgr24"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise TypeError(
                f"schema_version must be a str, got {type(self.schema_version).__name__}"
            )
        if self.schema_version not in _FRAME_OBSERVATION_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version must be one of {sorted(_FRAME_OBSERVATION_SCHEMA_VERSIONS)}, got {self.schema_version!r}"
            )
        check_id(self.shot_id, "shot_id")
        check_id(self.camera_id, "camera_id")
        check_id(self.frame_id, "frame_id")
        check_int(self.pts_ticks, "pts_ticks")
        num = check_pos_int(self.timebase_numerator, "timebase_numerator")
        den = check_pos_int(self.timebase_denominator, "timebase_denominator")
        if math.gcd(num, den) != 1:
            raise ValueError(
                f"timebase fraction {num}/{den} is not reduced (gcd={math.gcd(num, den)})"
            )
        check_optional_float(self.physical_time_s, "physical_time_s")
        if self.physical_time_s is None:
            if (
                not self.physical_time_reason
                or self.physical_time_reason.strip() != self.physical_time_reason
            ):
                raise ValueError(
                    "physical_time_reason must be nonempty and trimmed when physical_time_s is None"
                )
        else:
            check_str(self.physical_time_reason, "physical_time_reason")

        check_id(self.body_mask_ref, "body_mask_ref")
        check_id(self.club_mask_ref, "club_mask_ref")
        check_id(self.valid_mask_ref, "valid_mask_ref")
        check_str(self.confidence_provenance, "confidence_provenance")
        check_str(self.timing_mode, "timing_mode")
        check_bool(self.is_timing_exact, "is_timing_exact")
        check_str(self.clock_evidence, "clock_evidence")
        check_str(self.decoder_name, "decoder_name")
        check_str(self.decoder_version, "decoder_version")
        check_str(self.pixel_format, "pixel_format")

    @property
    def presentation_time(self) -> Fraction:
        return Fraction(
            self.pts_ticks * self.timebase_numerator, self.timebase_denominator
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shot_id": self.shot_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "pts_ticks": self.pts_ticks,
            "timebase_numerator": self.timebase_numerator,
            "timebase_denominator": self.timebase_denominator,
            "physical_time_s": self.physical_time_s,
            "physical_time_reason": self.physical_time_reason,
            "body_mask_ref": self.body_mask_ref,
            "club_mask_ref": self.club_mask_ref,
            "valid_mask_ref": self.valid_mask_ref,
            "confidence_provenance": self.confidence_provenance,
            "timing_mode": self.timing_mode,
            "is_timing_exact": self.is_timing_exact,
            "clock_evidence": self.clock_evidence,
            "decoder_name": self.decoder_name,
            "decoder_version": self.decoder_version,
            "pixel_format": self.pixel_format,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameObservation:
        if not isinstance(payload, dict):
            raise TypeError(f"Payload must be a dict, got {type(payload).__name__}")
        extra = set(payload.keys()) - _FRAME_OBSERVATION_KEYS
        if extra:
            raise ValueError(f"Unknown fields rejected: {sorted(extra)}")
        missing = _LEGACY_FRAME_OBSERVATION_KEYS - set(payload.keys())
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        return cls(
            schema_version=payload["schema_version"],
            shot_id=payload["shot_id"],
            camera_id=payload["camera_id"],
            frame_id=payload["frame_id"],
            pts_ticks=payload["pts_ticks"],
            timebase_numerator=payload["timebase_numerator"],
            timebase_denominator=payload["timebase_denominator"],
            physical_time_s=payload["physical_time_s"],
            physical_time_reason=payload["physical_time_reason"],
            body_mask_ref=payload["body_mask_ref"],
            club_mask_ref=payload["club_mask_ref"],
            valid_mask_ref=payload["valid_mask_ref"],
            confidence_provenance=payload["confidence_provenance"],
            timing_mode=payload.get("timing_mode", "estimated_cfr"),
            is_timing_exact=payload.get("is_timing_exact", False),
            clock_evidence=payload.get("clock_evidence", "unverified_legacy_record"),
            decoder_name=payload.get("decoder_name", "opencv"),
            decoder_version=payload.get("decoder_version", "legacy"),
            pixel_format=payload.get("pixel_format", "bgr24"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraTrack:
    """Camera hypothesis track over a shot interval."""

    schema_version: str
    camera_id: str
    shot_id: str
    frame_convention: str
    track_times: tuple[int, ...]
    status: CameraTrackStatus
    uncertainty: float | None
    image_transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, CAMERA_TRACK_SCHEMA_VERSION)
        check_id(self.camera_id, "camera_id")
        check_id(self.shot_id, "shot_id")
        check_str(self.frame_convention, "frame_convention")
        times_copy = []
        for t in self.track_times:
            times_copy.append(check_int(t, "track_time"))
        object.__setattr__(self, "track_times", tuple(times_copy))
        if self.status not in _VALID_CAMERA_TRACK_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_CAMERA_TRACK_STATUSES)}, got {self.status!r}"
            )
        if self.uncertainty is not None:
            check_nonneg_float(self.uncertainty, "uncertainty")
        transforms_copy = []
        for tr in self.image_transforms:
            transforms_copy.append(check_str(tr, "image_transform"))
        object.__setattr__(self, "image_transforms", tuple(transforms_copy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "shot_id": self.shot_id,
            "frame_convention": self.frame_convention,
            "track_times": list(self.track_times),
            "status": self.status,
            "uncertainty": self.uncertainty,
            "image_transforms": list(self.image_transforms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraTrack:
        check_payload_keys(payload, _CAMERA_TRACK_KEYS)
        raw_times = payload["track_times"]
        raw_transforms = payload["image_transforms"]
        return cls(
            schema_version=payload["schema_version"],
            camera_id=payload["camera_id"],
            shot_id=payload["shot_id"],
            frame_convention=payload["frame_convention"],
            track_times=tuple(int(t) for t in raw_times),
            status=payload["status"],
            uncertainty=payload["uncertainty"],
            image_transforms=tuple(str(tr) for tr in raw_transforms),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubjectModelBinding:
    """Subject morphology and visual envelope binding."""

    schema_version: str
    subject_id: str
    model_hash: str
    joint_ids: tuple[str, ...]
    body_ids: tuple[str, ...]
    visual_envelope: dict[str, float]
    mass_kg: float
    scale_evidence: str
    handedness: Handedness

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, SUBJECT_BINDING_SCHEMA_VERSION)
        check_id(self.subject_id, "subject_id")
        check_sha256(self.model_hash, "model_hash")
        for j in self.joint_ids:
            check_id(j, "joint_id")
        for b in self.body_ids:
            check_id(b, "body_id")
        envelope_copy = {}
        for k, v in self.visual_envelope.items():
            check_str(k, "envelope_key")
            envelope_copy[k] = check_pos_float(v, f"visual_envelope[{k}]")
        object.__setattr__(self, "visual_envelope", envelope_copy)
        object.__setattr__(self, "joint_ids", tuple(self.joint_ids))
        object.__setattr__(self, "body_ids", tuple(self.body_ids))
        check_pos_float(self.mass_kg, "mass_kg")
        check_str(self.scale_evidence, "scale_evidence")
        if self.handedness not in _VALID_HANDEDNESS:
            raise ValueError(
                f"handedness must be one of {sorted(_VALID_HANDEDNESS)}, got {self.handedness!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "model_hash": self.model_hash,
            "joint_ids": list(self.joint_ids),
            "body_ids": list(self.body_ids),
            "visual_envelope": dict(self.visual_envelope),
            "mass_kg": self.mass_kg,
            "scale_evidence": self.scale_evidence,
            "handedness": self.handedness,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SubjectModelBinding:
        check_payload_keys(payload, _SUBJECT_BINDING_KEYS)
        raw_joints = payload["joint_ids"]
        raw_bodies = payload["body_ids"]
        raw_envelope = payload["visual_envelope"]
        return cls(
            schema_version=payload["schema_version"],
            subject_id=payload["subject_id"],
            model_hash=payload["model_hash"],
            joint_ids=tuple(str(j) for j in raw_joints),
            body_ids=tuple(str(b) for b in raw_bodies),
            visual_envelope={str(k): float(v) for k, v in raw_envelope.items()},
            mass_kg=payload["mass_kg"],
            scale_evidence=payload["scale_evidence"],
            handedness=payload["handedness"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FitRequest:
    """Optimization request specification."""

    schema_version: str
    request_id: str
    shot_id: str
    model_hash: str
    candidate_count: int
    objective_profile: str
    time_window_start_pts: int
    time_window_end_pts: int
    budget_seconds: float
    engine_capability_requirement: tuple[str, ...]

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, FIT_REQUEST_SCHEMA_VERSION)
        check_id(self.request_id, "request_id")
        check_id(self.shot_id, "shot_id")
        check_sha256(self.model_hash, "model_hash")
        c = check_pos_int(self.candidate_count, "candidate_count")
        if c < 1:
            raise ValueError(f"candidate_count must be >= 1, got {c}")
        check_str(self.objective_profile, "objective_profile")
        check_int(self.time_window_start_pts, "time_window_start_pts")
        check_int(self.time_window_end_pts, "time_window_end_pts")
        if self.time_window_start_pts > self.time_window_end_pts:
            raise ValueError(
                f"time_window_start_pts ({self.time_window_start_pts}) cannot exceed "
                f"time_window_end_pts ({self.time_window_end_pts})"
            )
        check_pos_float(self.budget_seconds, "budget_seconds")
        req_copy = []
        for req in self.engine_capability_requirement:
            req_copy.append(check_str(req, "engine_capability_requirement"))
        object.__setattr__(self, "engine_capability_requirement", tuple(req_copy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "shot_id": self.shot_id,
            "model_hash": self.model_hash,
            "candidate_count": self.candidate_count,
            "objective_profile": self.objective_profile,
            "time_window_start_pts": self.time_window_start_pts,
            "time_window_end_pts": self.time_window_end_pts,
            "budget_seconds": self.budget_seconds,
            "engine_capability_requirement": list(self.engine_capability_requirement),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FitRequest:
        check_payload_keys(payload, _FIT_REQUEST_KEYS)
        raw_caps = payload["engine_capability_requirement"]
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            shot_id=payload["shot_id"],
            model_hash=payload["model_hash"],
            candidate_count=payload["candidate_count"],
            objective_profile=payload["objective_profile"],
            time_window_start_pts=payload["time_window_start_pts"],
            time_window_end_pts=payload["time_window_end_pts"],
            budget_seconds=payload["budget_seconds"],
            engine_capability_requirement=tuple(str(r) for r in raw_caps),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayAudit:
    """Independent simulation replay audit metrics."""

    schema_version: str
    candidate_id: str
    reset_count: int
    integrator_name: str
    integrator_version: str
    coverage_start_s: float
    coverage_end_s: float
    max_grip_translation_error_m: float
    max_grip_rotation_error_rad: float
    is_physically_accepted: bool

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, REPLAY_AUDIT_SCHEMA_VERSION)
        check_id(self.candidate_id, "candidate_id")
        check_pos_int(self.reset_count, "reset_count")
        check_str(self.integrator_name, "integrator_name")
        check_str(self.integrator_version, "integrator_version")
        c_start = check_nonneg_float(self.coverage_start_s, "coverage_start_s")
        c_end = check_nonneg_float(self.coverage_end_s, "coverage_end_s")
        if c_start > c_end:
            raise ValueError(
                f"coverage_start_s ({c_start}) cannot exceed coverage_end_s ({c_end})"
            )
        check_nonneg_float(
            self.max_grip_translation_error_m, "max_grip_translation_error_m"
        )
        check_nonneg_float(
            self.max_grip_rotation_error_rad, "max_grip_rotation_error_rad"
        )
        check_bool(self.is_physically_accepted, "is_physically_accepted")

        # Invariant: continuous reference-free execution requires exactly 1 reset
        if self.reset_count != 1 and self.is_physically_accepted:
            raise ValueError(
                f"is_physically_accepted cannot be true when reset_count is {self.reset_count} (!= 1)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "reset_count": self.reset_count,
            "integrator_name": self.integrator_name,
            "integrator_version": self.integrator_version,
            "coverage_start_s": self.coverage_start_s,
            "coverage_end_s": self.coverage_end_s,
            "max_grip_translation_error_m": self.max_grip_translation_error_m,
            "max_grip_rotation_error_rad": self.max_grip_rotation_error_rad,
            "is_physically_accepted": self.is_physically_accepted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayAudit:
        check_payload_keys(payload, _REPLAY_AUDIT_KEYS)
        return cls(**payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateResult:
    """Candidate trajectory result with physical replay audit."""

    schema_version: str
    candidate_id: str
    request_id: str
    initial_state: tuple[float, ...]
    trajectory: tuple[tuple[float, ...], ...]
    diagnostics: dict[str, Any]
    uncertainty_method: str
    replay_audit: ReplayAudit | None
    is_accepted: bool

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, CANDIDATE_RESULT_SCHEMA_VERSION)
        check_id(self.candidate_id, "candidate_id")
        check_id(self.request_id, "request_id")
        init_copy = []
        for s in self.initial_state:
            check_nonneg_float(abs(s), "initial_state_element")
            init_copy.append(float(s))
        object.__setattr__(self, "initial_state", tuple(init_copy))
        traj_copy = []
        for row in self.trajectory:
            traj_copy.append(tuple(float(y) for y in row))
        object.__setattr__(self, "trajectory", tuple(traj_copy))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        check_str(self.uncertainty_method, "uncertainty_method")
        check_bool(self.is_accepted, "is_accepted")

        # DbC Invariant: is_accepted requires fresh passing replay audit
        if self.is_accepted:
            if self.replay_audit is None:
                raise ValueError(
                    "is_accepted cannot be true without a fresh replay audit"
                )
            if not self.replay_audit.is_physically_accepted:
                raise ValueError(
                    "is_accepted cannot be true when replay audit is not physically accepted"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "request_id": self.request_id,
            "initial_state": list(self.initial_state),
            "trajectory": [list(row) for row in self.trajectory],
            "diagnostics": dict(self.diagnostics),
            "uncertainty_method": self.uncertainty_method,
            "replay_audit": self.replay_audit.to_dict()
            if self.replay_audit is not None
            else None,
            "is_accepted": self.is_accepted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateResult:
        check_payload_keys(payload, _CANDIDATE_RESULT_KEYS)
        raw_audit = payload["replay_audit"]
        audit = (
            ReplayAudit.from_dict(raw_audit) if isinstance(raw_audit, dict) else None
        )
        return cls(
            schema_version=payload["schema_version"],
            candidate_id=payload["candidate_id"],
            request_id=payload["request_id"],
            initial_state=tuple(float(x) for x in payload["initial_state"]),
            trajectory=tuple(
                tuple(float(y) for y in row) for row in payload["trajectory"]
            ),
            diagnostics=dict(payload["diagnostics"]),
            uncertainty_method=payload["uncertainty_method"],
            replay_audit=audit,
            is_accepted=payload["is_accepted"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultBundle:
    """Atomic bundle of fit results, audits, and provenance."""

    schema_version: str
    bundle_id: str
    request: FitRequest
    candidates: tuple[CandidateResult, ...]
    replay_audits: tuple[ReplayAudit, ...]
    execution_status: ExecutionStatus
    evidence_quality: EvidenceQuality
    metrics: dict[str, Any]
    hashes: dict[str, str]

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version, RESULT_BUNDLE_SCHEMA_VERSION)
        check_id(self.bundle_id, "bundle_id")
        if self.execution_status not in _VALID_EXECUTION_STATUSES:
            raise ValueError(
                f"execution_status must be one of {sorted(_VALID_EXECUTION_STATUSES)}, got {self.execution_status!r}"
            )
        if self.evidence_quality not in _VALID_EVIDENCE_QUALITIES:
            raise ValueError(
                f"evidence_quality must be one of {sorted(_VALID_EVIDENCE_QUALITIES)}, got {self.evidence_quality!r}"
            )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "replay_audits", tuple(self.replay_audits))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "hashes", dict(self.hashes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "request": self.request.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "replay_audits": [a.to_dict() for a in self.replay_audits],
            "execution_status": self.execution_status,
            "evidence_quality": self.evidence_quality,
            "metrics": dict(self.metrics),
            "hashes": dict(self.hashes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResultBundle:
        check_payload_keys(payload, _RESULT_BUNDLE_KEYS)
        req = FitRequest.from_dict(payload["request"])
        candidates = tuple(CandidateResult.from_dict(c) for c in payload["candidates"])
        audits = tuple(ReplayAudit.from_dict(a) for a in payload["replay_audits"])
        return cls(
            schema_version=payload["schema_version"],
            bundle_id=payload["bundle_id"],
            request=req,
            candidates=candidates,
            replay_audits=audits,
            execution_status=payload["execution_status"],
            evidence_quality=payload["evidence_quality"],
            metrics=dict(payload["metrics"]),
            hashes=dict(payload["hashes"]),
        )


# ---------------------------------------------------------------------------
# Auxiliary Protocol DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentationRequest:
    shot_id: str
    frame_ids: tuple[str, ...]
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_id(self.shot_id, "shot_id")
        if not isinstance(self.frame_ids, tuple) or not self.frame_ids:
            raise ValueError(
                f"frame_ids must be a non-empty tuple of frame IDs, got {self.frame_ids!r}"
            )
        for fid in self.frame_ids:
            check_id(fid, "frame_id")
        if not isinstance(self.options, dict):
            raise TypeError(
                f"options must be a dict, got {type(self.options).__name__}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentationResult:
    shot_id: str
    mask_count: int
    provenance: str

    def __post_init__(self) -> None:
        check_id(self.shot_id, "shot_id")
        check_int(self.mask_count, "mask_count")
        if self.mask_count < 0:
            raise ValueError(f"mask_count must be non-negative, got {self.mask_count}")
        check_str(self.provenance, "provenance")


POINT_LANDMARKS_CONVENTION: Final[str] = "point_landmarks"
CANONICAL_ARTICULATED_CONVENTION: Final[str] = "canonical_articulated_v1"
_VALID_RENDER_STATE_CONVENTIONS: frozenset[str] = frozenset(
    (POINT_LANDMARKS_CONVENTION, CANONICAL_ARTICULATED_CONVENTION)
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderRequest:
    camera_id: str
    state: tuple[float, ...]
    image_size_px: tuple[int, int]
    state_convention: str = POINT_LANDMARKS_CONVENTION

    def __post_init__(self) -> None:
        check_id(self.camera_id, "camera_id")
        if not isinstance(self.state, tuple):
            raise TypeError(f"state must be a tuple, got {type(self.state).__name__}")
        for s in self.state:
            check_strict_float(s, "state element")
        if not isinstance(self.image_size_px, tuple) or len(self.image_size_px) != 2:
            raise ValueError(
                f"image_size_px must be a 2-element tuple (width, height), got {self.image_size_px!r}"
            )
        check_pos_int(self.image_size_px[0], "width_px")
        check_pos_int(self.image_size_px[1], "height_px")
        check_str(self.state_convention, "state_convention")
        if self.state_convention not in _VALID_RENDER_STATE_CONVENTIONS:
            raise ValueError(
                f"state_convention must be one of {sorted(_VALID_RENDER_STATE_CONVENTIONS)}, got {self.state_convention!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderResult:
    body_mask: tuple[int, ...]

    club_mask: tuple[int, ...]
    visibility_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.body_mask, tuple):
            raise TypeError(
                f"body_mask must be a tuple, got {type(self.body_mask).__name__}"
            )
        if not isinstance(self.club_mask, tuple):
            raise TypeError(
                f"club_mask must be a tuple, got {type(self.club_mask).__name__}"
            )
        if not isinstance(self.visibility_mask, tuple):
            raise TypeError(
                f"visibility_mask must be a tuple, got {type(self.visibility_mask).__name__}"
            )
        if len(self.body_mask) != len(self.club_mask) or len(self.body_mask) != len(
            self.visibility_mask
        ):
            raise ValueError(
                f"Mask sizes must match: body={len(self.body_mask)}, club={len(self.club_mask)}, vis={len(self.visibility_mask)}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilities:
    supported_bodies: tuple[str, ...]
    state_convention: str
    actuator_modes: tuple[str, ...]
    contact_modes: tuple[str, ...]
    is_available: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutRequest:
    initial_state: tuple[float, ...]
    controls: tuple[tuple[float, ...], ...]
    time_points_s: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutResult:
    trajectory: tuple[tuple[float, ...], ...]
    realized_controls: tuple[tuple[float, ...], ...]
    time_points_s: tuple[float, ...]
    audit: ReplayAudit


# ---------------------------------------------------------------------------
# Service Protocols (Runtime Checkable)
# ---------------------------------------------------------------------------


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for segmentation providers."""

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...


@runtime_checkable
class SilhouetteRenderer(Protocol):
    """Protocol for silhouette renderers."""

    def render(self, request: RenderRequest) -> RenderResult: ...


@runtime_checkable
class ForwardModel(Protocol):
    """Protocol for dynamic forward models."""

    def capabilities(self) -> ModelCapabilities: ...

    def rollout(self, request: RolloutRequest) -> RolloutResult: ...


@runtime_checkable
class ShadowTrackerService(Protocol):
    """Protocol for Shadow Tracker fit and evidence orchestration."""

    def fit(self, request: FitRequest) -> ResultBundle: ...
