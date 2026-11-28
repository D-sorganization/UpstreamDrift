"""Frozen independent held-out phase annotations for the Tiger 2000 scorecard (#15837).

The scorecard in :mod:`.markerless_scorecard` needs ground truth that was not
produced by the tracker under test or by the null template itself. This module
freezes that ground truth as a versioned, hash-pinned artifact whose records are
bound to decoded image digests from the ``#15844`` frame map, so an annotation set
cannot be silently regenerated against a different rendition or copied from the
baseline it is meant to referee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from . import _validation as _v
from .markerless_scorecard import FramePoseRecord, PhaseAlignedNullTemplate

HELDOUT_SCHEMA_VERSION: Final[str] = "heldout-phase-annotations/1.0.0"
DEFAULT_HELDOUT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "tiger_2000_heldout_phase_annotations.json"
)
#: Gate pins that the held-out set must agree with before any scoring is allowed.
SHARED_PIN_KEYS: Final[tuple[str, ...]] = (
    "epic_issue",
    "scorecard_issue",
    "video_id",
    "source_url",
    "source_sha256",
    "frame_map_sha256",
    "dimensions_px",
    "timebase",
    "candidate_window_source_indices",
    "timebase_status",
    "physical_time_verified",
    "qualification_floor",
)
AnnotationStatus = Literal["reviewed", "pending"]
NoiseFloorStatus = Literal["measured", "assumed"]
AnnotatorKind = Literal["human_reviewed", "model_assisted"]


@dataclass(frozen=True, slots=True)
class IndependenceDeclaration:
    """Attestation of how the held-out annotations were produced."""

    annotator_kind: AnnotatorKind
    derived_from_null_template: bool
    derived_from_candidate_tracker: bool

    def __post_init__(self) -> None:
        """Reject unknown annotator kinds and non-boolean derivation flags."""
        _v.check_str(self.annotator_kind, "annotator_kind")
        if self.annotator_kind not in ("human_reviewed", "model_assisted"):
            raise ValueError("annotator_kind must be human_reviewed or model_assisted")
        for flag in (
            self.derived_from_null_template,
            self.derived_from_candidate_tracker,
        ):
            _v.check_bool(flag, "derivation flag")

    @property
    def is_independent(self) -> bool:
        """Return True when the declaration cites no forbidden derivation source."""
        return not (
            self.derived_from_null_template or self.derived_from_candidate_tracker
        )


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """Annotator repeatability floor, either measured from a repeat pass or assumed."""

    status: NoiseFloorStatus
    keypoint_rmse_px: float | None
    shaft_angle_deg: float | None
    repeat_pair_count: int

    def __post_init__(self) -> None:
        """Require measured values and repeat coverage whenever status is measured."""
        if self.status not in ("measured", "assumed"):
            raise ValueError("noise floor status must be measured or assumed")
        count = _v.check_int(self.repeat_pair_count, "repeat_pair_count")
        object.__setattr__(self, "repeat_pair_count", count)
        if count < 0:
            raise ValueError("repeat_pair_count must be non-negative")
        for key in ("keypoint_rmse_px", "shaft_angle_deg"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, _v.check_pos_float(value, key))
        if self.status == "measured":
            if count < 2:
                raise ValueError(
                    "a measured noise floor needs at least two repeat frames"
                )
            if self.keypoint_rmse_px is None or self.shaft_angle_deg is None:
                raise ValueError("measured noise floor requires both error channels")


@dataclass(frozen=True, slots=True)
class AnnotationReview:
    """Review trail naming the annotation method and, once reviewed, its owner."""

    method: str
    second_pass_required: bool
    reviewer: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        """Validate the method label and the optional completion trail."""
        _v.check_str(self.method, "method")
        _v.check_bool(self.second_pass_required, "second_pass_required")
        for field_name in ("reviewer", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _v.check_str(value, field_name))


@dataclass(frozen=True, slots=True)
class HeldoutFrameAnnotation:
    """One phase label bound to the digest of the decoded frame it was read from."""

    frame_index: int
    phase: str
    frame_sha256: str
    keypoints_px: dict[str, tuple[float, float]]
    shaft_angle_deg: float

    def __post_init__(self) -> None:
        """Validate the frame identity, image digest, and canonical 2D pose."""
        if _v.check_int(self.frame_index, "frame_index") < 0:
            raise ValueError("frame_index must be >= 0")
        _v.check_str(self.phase, "phase")
        _v.check_sha256(self.frame_sha256, "frame_sha256")
        _v.check_float(self.shaft_angle_deg, "shaft_angle_deg")
        if not self.keypoints_px:
            raise ValueError("keypoints_px must be a non-empty mapping")
        cleaned = {}
        for name, xy in self.keypoints_px.items():
            if not isinstance(xy, (tuple, list)) or len(xy) != 2:
                raise ValueError(f"keypoints_px[{name}] must be a (x, y) pair")
            cleaned[_v.check_str(name, "kp")] = (
                _v.check_float(xy[0], f"{name}.x"),
                _v.check_float(xy[1], f"{name}.y"),
            )
        object.__setattr__(self, "keypoints_px", cleaned)
        object.__setattr__(self, "shaft_angle_deg", float(self.shaft_angle_deg))

    @property
    def pose(self) -> FramePoseRecord:
        """Scorecard pose view, so keypoint validation lives in one place."""
        return FramePoseRecord(
            self.frame_index, self.phase, self.keypoints_px, self.shaft_angle_deg
        )


@dataclass(frozen=True, slots=True)
class HeldoutPhaseAnnotations:
    """Immutable held-out annotation set with a deterministic SHA-256 content hash."""

    schema_version: str
    annotation_set_id: str
    status: AnnotationStatus
    annotator: str
    pinned_reference: dict[str, Any]
    required_phases: tuple[str, ...]
    frames: tuple[HeldoutFrameAnnotation, ...]
    independence: IndependenceDeclaration
    noise_floor: NoiseFloor
    review: AnnotationReview
    content_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate schema, phase vocabulary, pins, bounds and the content hash."""
        _v.check_schema_version(self.schema_version, HELDOUT_SCHEMA_VERSION)
        _v.check_id(self.annotation_set_id, "annotation_set_id")
        _v.check_str(self.annotator, "annotator")
        if self.status not in ("reviewed", "pending"):
            raise ValueError("status must be reviewed or pending")
        phases = tuple(_v.check_str(p, "phase") for p in self.required_phases)
        if not phases or len(set(phases)) != len(phases):
            raise ValueError("required_phases must be non-empty and unique")
        object.__setattr__(self, "required_phases", phases)
        object.__setattr__(self, "frames", tuple(self.frames))
        keys = [(r.frame_index, r.phase) for r in self.frames]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate held-out frame/phase annotation")
        for record in self.frames:
            if record.phase not in phases:
                raise ValueError(f"phase {record.phase!r} is outside required_phases")
        if self.status == "reviewed":
            if not self.review.reviewer or not self.review.completed_at:
                raise ValueError("a reviewed set must name reviewer and completion")
        self._check_within_frame()
        digest = _content_hash(self.to_payload_dict())
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", digest)
        elif self.content_sha256 != digest:
            raise ValueError("content_sha256 mismatch for the held-out annotation set")

    def _check_within_frame(self) -> None:
        """Require every annotated coordinate to lie inside the pinned frame."""
        dims = self.pinned_reference.get("dimensions_px")
        if not isinstance(dims, (list, tuple)) or len(dims) != 2:
            raise ValueError("pinned_reference.dimensions_px must be [width, height]")
        width, height = float(dims[0]), float(dims[1])
        for record in self.frames:
            for name, (x, y) in record.keypoints_px.items():
                if not (0.0 <= x <= width and 0.0 <= y <= height):
                    raise ValueError(
                        f"keypoint {name!r} at ({x}, {y}) is outside pinned frame "
                        f"{int(width)}x{int(height)}"
                    )

    @property
    def window(self) -> tuple[int, int]:
        """Pinned candidate source-index window the annotations must live inside."""
        window = self.pinned_reference.get("candidate_window_source_indices")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(
                "pinned_reference.candidate_window_source_indices required"
            )
        return int(window[0]), int(window[1])

    def to_records(self) -> tuple[FramePoseRecord, ...]:
        """Expose the annotation set as scorecard pose records."""
        return tuple(record.pose for record in self.frames)

    def scoreability_blockers(self) -> tuple[str, ...]:
        """Return fail-closed reasons why this set may not referee a pass verdict."""
        blockers: list[str] = []
        if self.status != "reviewed":
            blockers.append("heldout_annotation_pass_pending")
        if not self.frames:
            blockers.append("heldout_frames_missing")
        if self.noise_floor.status != "measured":
            blockers.append("annotator_noise_floor_unmeasured")
        if not self.independence.is_independent:
            blockers.append("heldout_annotations_not_independent")
        if self.frames and set(self.required_phases) - {r.phase for r in self.frames}:
            blockers.append("heldout_phase_coverage_incomplete")
        if self.review.reviewer and self.review.reviewer == self.annotator:
            blockers.append("heldout_review_not_independent")
        return tuple(blockers)

    def verify_frame_map(self, frame_map: dict[int, str]) -> None:
        """Check each cited frame digest against the pinned source-to-image map."""
        low, high = self.window
        for record in self.frames:
            if not low <= record.frame_index <= high:
                raise ValueError(
                    f"frame {record.frame_index} is outside the pinned candidate window"
                )
            cited = frame_map.get(record.frame_index)
            if cited is None:
                raise ValueError(
                    f"frame {record.frame_index} absent from pinned frame map"
                )
            if cited != record.frame_sha256:
                raise ValueError(
                    f"frame_sha256 for {record.frame_index} does not match the pinned map"
                )

    def assert_carries_signal(self, null_template: PhaseAlignedNullTemplate) -> None:
        """Reject an annotation set that merely copies the null template poses."""
        empty = [
            record
            for record in self.frames
            if _template_gap(record, null_template) == 0.0
        ]
        if len(empty) == len(self.frames):
            raise ValueError("held-out annotations identical to the null template")

    def assert_pins_match(self, gate_pins: dict[str, Any]) -> None:
        """Require agreement with the gate config on every shared provenance pin."""
        assert_pins_match(gate_pins, self.pinned_reference)

    def to_payload_dict(self) -> dict[str, Any]:
        """Serialize canonical held-out fields excluding content_sha256."""
        return {k: v for k, v in asdict(self).items() if k != "content_sha256"}

    def to_json(self) -> str:
        """Serialize the held-out set as deterministic indented JSON."""
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False, indent=2)


def _template_gap(
    record: HeldoutFrameAnnotation, null_template: PhaseAlignedNullTemplate
) -> float:
    """Return the largest keypoint/angle difference between a record and its phase pose."""
    pose = null_template.phase_poses.get(record.phase)
    if pose is None:
        raise ValueError(f"Null template missing phase {record.phase!r}")
    gap = abs(pose.shaft_angle_deg - record.shaft_angle_deg)
    for name, (x, y) in record.keypoints_px.items():
        if name not in pose.keypoints_px:
            return math.inf
        px, py = pose.keypoints_px[name]
        gap = max(gap, abs(px - x), abs(py - y))
    return gap


def _content_hash(payload: dict[str, Any]) -> str:
    """Return the SHA-256 of the canonical JSON encoding of a held-out payload."""
    raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assert_pins_match(gate_pins: dict[str, Any], heldout_pins: dict[str, Any]) -> None:
    """Fail closed when the gate config and held-out set disagree on any shared pin."""
    for key in SHARED_PIN_KEYS:
        for side, pins in (("gate", gate_pins), ("held-out", heldout_pins)):
            if key not in pins:
                raise ValueError(f"{side} pinned_reference is missing {key}")
        if gate_pins[key] != heldout_pins[key]:
            raise ValueError(
                f"held-out pinned_reference.{key} disagrees with the gate config"
            )


def measure_repeat_noise_floor(
    first_pass: Sequence[HeldoutFrameAnnotation],
    second_pass: Sequence[HeldoutFrameAnnotation],
) -> NoiseFloor:
    """Derive the annotator noise floor from two independent passes over one clip."""
    first = {(r.frame_index, r.phase): r for r in first_pass}
    second = {(r.frame_index, r.phase): r for r in second_pass}
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("repeat passes have no overlapping frames to measure")
    kp_sq: list[float] = []
    ang_sq: list[float] = []
    for key in shared:
        a, b = first[key], second[key]
        names_a, names_b = set(a.keypoints_px), set(b.keypoints_px)
        if names_a != names_b:
            raise ValueError(
                f"repeat passes disagree on keypoint names for frame {key[0]}"
            )
        for name in sorted(names_a):
            (ax, ay), (bx, by) = a.keypoints_px[name], b.keypoints_px[name]
            kp_sq.append((bx - ax) ** 2 + (by - ay) ** 2)
        delta = (b.shaft_angle_deg - a.shaft_angle_deg + 180.0) % 360.0 - 180.0
        ang_sq.append(delta * delta)
    return NoiseFloor(
        status="measured",
        keypoint_rmse_px=math.sqrt(sum(kp_sq) / len(kp_sq)),
        shaft_angle_deg=math.sqrt(sum(ang_sq) / len(ang_sq)),
        repeat_pair_count=len(shared),
    )


def load_heldout_annotations(path: Path | None = None) -> HeldoutPhaseAnnotations:
    """Load and validate a frozen held-out annotation set."""
    raw = json.loads((path or DEFAULT_HELDOUT_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw.get("frames"), list):
        raise ValueError("held-out frames must be a list")
    return HeldoutPhaseAnnotations(
        schema_version=raw["schema_version"],
        annotation_set_id=raw["annotation_set_id"],
        status=raw["status"],
        annotator=raw["annotator"],
        pinned_reference=raw["pinned_reference"],
        required_phases=tuple(raw["required_phases"]),
        frames=tuple(HeldoutFrameAnnotation(**row) for row in raw["frames"]),
        independence=IndependenceDeclaration(**raw["independence"]),
        noise_floor=NoiseFloor(**raw["noise_floor"]),
        review=AnnotationReview(**raw["review"]),
        content_sha256=raw["content_sha256"],
    )
