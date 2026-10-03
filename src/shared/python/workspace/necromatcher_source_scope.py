"""Reviewed source-domain declarations and authenticated scope bindings (#11414).

Neither reviewed boundaries nor exact container PTS establish measured hand
contact or physical time. Callers obtain CaptureIdentity through its public
canonical loader before binding; a constructed identity is not library proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
)
from src.shared.python.shadow_tracker.source_records import FrameIdentity
from .artifact_handoff import ArtifactKind, ArtifactReference

if TYPE_CHECKING:
    from .necromatcher_capture_identity import CaptureIdentity

SOURCE_SCOPE_REVIEW_SCHEMA = "necromatcher/source-fit-scope-review/1"
SOURCE_FIT_SCOPE_SCHEMA = "necromatcher/source-fit-scope/1"
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"Scope {name} requires a canonical sha256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Scope {name} requires nonempty trimmed text")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Scope {name} requires an integer >= {minimum}")
    return value


def _fields(record: object, names: set[str]) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != names:
        raise ValueError("Malformed source scope record fields")
    return record


@dataclass(frozen=True)
class SourceScopeReview:
    """Immutable uncalibrated declaration; binding verifies actual receipt bytes."""

    artifact: ArtifactReference
    receipt_bytes: int
    first_identity: FrameIdentity
    excluded_identity: FrameIdentity | None
    reason: str
    uncertainty_policy: str
    contact_calibrated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactReference):
            raise ValueError("Scope review requires a typed artifact reference")
        if (
            self.artifact.kind != ArtifactKind.RECEIPT
            or self.artifact.schema != SOURCE_SCOPE_REVIEW_SCHEMA
        ):
            raise ValueError(
                "Scope review requires the exact source-fit-scope-review/1 receipt schema"
            )
        _digest(self.artifact.hash, "review artifact hash")
        _integer(self.receipt_bytes, "receipt_bytes", 1)
        if not isinstance(self.first_identity, FrameIdentity) or (
            self.excluded_identity is not None
            and not isinstance(self.excluded_identity, FrameIdentity)
        ):
            raise ValueError("Scope review requires canonical frame identities")
        for frame in (self.first_identity, self.excluded_identity):
            if frame is not None and (
                frame.timing_mode != "container_pts"
                or not frame.is_timing_exact
                or frame.physical_time_s is not None
            ):
                raise ValueError(
                    "Scope boundary requires exact source PTS and unknown physical clock"
                )
        if self.contact_calibrated is not False:
            raise ValueError("Scope review contact must remain uncalibrated")
        _text(self.reason, "reason")
        _text(self.uncertainty_policy, "uncertainty_policy")
        if self.artifact.metadata:
            raise ValueError("Scope review artifact has unsupported metadata")
        sealed = ArtifactReference(
            self.artifact.artifact_id,
            self.artifact.path,
            self.artifact.hash,
            self.artifact.schema,
            self.artifact.kind,
            cast(dict[str, Any], MappingProxyType({})),
        )
        object.__setattr__(self, "artifact", sealed)

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact": {
                "artifact_id": self.artifact.artifact_id,
                "path": str(self.artifact.path),
                "hash": self.artifact.hash,
                "schema": self.artifact.schema,
                "kind": ArtifactKind.RECEIPT.value,
            },
            "receipt_bytes": self.receipt_bytes,
            "first_identity": self.first_identity.to_dict(),
            "excluded_identity": (
                self.excluded_identity.to_dict()
                if self.excluded_identity is not None
                else None
            ),
            "reason": self.reason,
            "uncertainty_policy": self.uncertainty_policy,
            "contact_calibrated": False,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SourceScopeReview:
        value = _fields(
            record,
            {
                "artifact",
                "receipt_bytes",
                "first_identity",
                "excluded_identity",
                "reason",
                "uncertainty_policy",
                "contact_calibrated",
            },
        )
        artifact = _fields(
            value["artifact"], {"artifact_id", "path", "hash", "schema", "kind"}
        )
        try:
            return cls(
                ArtifactReference(**dict(artifact)),
                value["receipt_bytes"],
                FrameIdentity.from_dict(value["first_identity"]),
                FrameIdentity.from_dict(value["excluded_identity"])
                if value["excluded_identity"] is not None
                else None,
                value["reason"],
                value["uncertainty_policy"],
                value["contact_calibrated"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("Malformed source scope review") from exc


@dataclass(frozen=True)
class SourceFitScope:
    """Approved half-open source frame window, independent of fit support."""

    capture_id: str
    capture_hash: str
    source_clock_sha256: str
    first_frame: int
    end_exclusive_frame: int
    review: SourceScopeReview
    purpose: str = "both_hands_on_club"

    def __post_init__(self) -> None:
        _text(self.capture_id, "capture_id")
        _digest(self.capture_hash, "capture_hash")
        _digest(self.source_clock_sha256, "source_clock_sha256")
        _integer(self.first_frame, "first_frame")
        _integer(self.end_exclusive_frame, "end_exclusive_frame")
        if self.end_exclusive_frame - self.first_frame < 2:
            raise ValueError("Scope bounds must permit at least two source frames")
        if not isinstance(self.review, SourceScopeReview):
            raise ValueError("Scope requires a typed uncalibrated review")
        if self.purpose != "both_hands_on_club":
            raise ValueError("Unsupported source fit scope purpose")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_FIT_SCOPE_SCHEMA,
            "capture_id": self.capture_id,
            "capture_hash": self.capture_hash,
            "source_clock_sha256": self.source_clock_sha256,
            "first_frame": self.first_frame,
            "end_exclusive_frame": self.end_exclusive_frame,
            "review": self.review.to_record(),
            "purpose": self.purpose,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SourceFitScope:
        value = _fields(
            record,
            {
                "schema",
                "capture_id",
                "capture_hash",
                "source_clock_sha256",
                "first_frame",
                "end_exclusive_frame",
                "review",
                "purpose",
            },
        )
        if value["schema"] != SOURCE_FIT_SCOPE_SCHEMA:
            raise ValueError("Unsupported source fit scope schema")
        return cls(
            value["capture_id"],
            value["capture_hash"],
            value["source_clock_sha256"],
            value["first_frame"],
            value["end_exclusive_frame"],
            SourceScopeReview.from_record(value["review"]),
            value["purpose"],
        )

    @classmethod
    def from_review_artifact(
        cls,
        artifact: ArtifactReference,
        receipt_bytes: int,
        artifact_root: Path | str | None = None,
    ) -> SourceFitScope:
        """Read a verified review declaration; capture authentication remains separate."""
        value = _read_review_artifact(artifact, receipt_bytes, artifact_root)
        review = SourceScopeReview(
            artifact,
            receipt_bytes,
            FrameIdentity.from_dict(value["first_identity"]),
            FrameIdentity.from_dict(value["excluded_identity"])
            if value["excluded_identity"] is not None
            else None,
            value["reason"],
            value["uncertainty_policy"],
            value["contact_calibrated"],
        )
        return cls(
            value["capture_id"],
            value["capture_hash"],
            value["source_clock_sha256"],
            value["first_frame"],
            value["end_exclusive_frame"],
            review,
        )


@dataclass(frozen=True)
class SourceSelectedDomain:
    """Actual inclusive support of selected observations, using source PTS only."""

    frame_indices: tuple[int, ...]
    first_pts: Fraction
    last_pts: Fraction


@dataclass(frozen=True)
class BoundSourceFitScope:
    """Canonical identity and verified review bound to an approved frame window."""

    identity: CaptureIdentity
    scope: SourceFitScope

    @property
    def first_pts(self) -> Fraction:
        return self.identity.frames[self.scope.first_frame].presentation_time

    @property
    def end_exclusive_pts(self) -> Fraction | None:
        index = self.scope.end_exclusive_frame
        return (
            self.identity.frames[index].presentation_time
            if index < len(self.identity.frames)
            else None
        )

    def selected_domain(self, indices: tuple[int, ...]) -> SourceSelectedDomain:
        validate_scope_selection(self, indices)
        return SourceSelectedDomain(
            tuple(indices),
            self.identity.frames[indices[0]].presentation_time,
            self.identity.frames[indices[-1]].presentation_time,
        )


def _review_path(
    reference: ArtifactReference, artifact_root: Path | str | None
) -> Path:
    declared = Path(reference.path)
    if not declared.is_absolute() and ".." in declared.parts:
        raise ValueError("Relative scope review cannot escape artifact root")
    path = reference.resolve_path(artifact_root).absolute()
    root = Path(artifact_root).absolute() if artifact_root is not None else None
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("Scope review path contains a link")
        attributes = getattr(ancestor.lstat(), "st_file_attributes", 0)
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Scope review path contains a link/reparse point")
        if ancestor == root:
            break
    return path


def _read_review_artifact(
    reference: ArtifactReference,
    receipt_bytes: int,
    artifact_root: Path | str | None,
) -> dict[str, Any]:
    if not isinstance(reference, ArtifactReference):
        raise ValueError("Typed scope review artifact required")
    if (
        reference.schema != SOURCE_SCOPE_REVIEW_SCHEMA
        or reference.kind != ArtifactKind.RECEIPT
    ):
        raise ValueError("Scope review requires the exact receipt schema")
    _digest(reference.hash, "review artifact hash")
    _integer(receipt_bytes, "receipt_bytes", 1)
    path = _review_path(reference, artifact_root)
    reference.verify_on_disk(artifact_root)
    # Authenticate the exact parsed buffer after the canonical on-disk check.
    raw = path.read_bytes()
    if (
        len(raw) != receipt_bytes
        or "sha256:" + hashlib.sha256(raw).hexdigest() != reference.hash
    ):
        raise ValueError("Scope review bytes/hash changed")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed source scope review JSON") from exc
    value = _fields(
        payload,
        {
            "schema",
            "capture_id",
            "capture_hash",
            "source_clock_sha256",
            "first_frame",
            "end_exclusive_frame",
            "first_identity",
            "excluded_identity",
            "contact_calibrated",
            "review_kind",
            "reason",
            "uncertainty_policy",
        },
    )
    if (
        value["schema"] != SOURCE_SCOPE_REVIEW_SCHEMA
        or value["review_kind"] != "authored_uncalibrated"
        or value["contact_calibrated"] is not False
    ):
        raise ValueError(
            "Scope review schema/content must remain authored and uncalibrated"
        )
    return dict(value)


def _review_payload(
    identity: CaptureIdentity,
    scope: SourceFitScope,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    payload = _read_review_artifact(
        scope.review.artifact, scope.review.receipt_bytes, artifact_root
    )
    expected = {
        "schema": SOURCE_SCOPE_REVIEW_SCHEMA,
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "source_clock_sha256": identity.source_clock_sha256,
        "first_frame": scope.first_frame,
        "end_exclusive_frame": scope.end_exclusive_frame,
        "first_identity": scope.review.first_identity.to_dict(),
        "excluded_identity": scope.review.excluded_identity.to_dict()
        if scope.review.excluded_identity is not None
        else None,
        "contact_calibrated": False,
        "review_kind": "authored_uncalibrated",
        "reason": scope.review.reason,
        "uncertainty_policy": scope.review.uncertainty_policy,
    }
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError("Scope review schema/content differs from declaration")
    # JSON bool/int equality alone would incorrectly admit calibrated=0 or index=True.
    if json.dumps(payload, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Scope review content differs from declaration")
    return payload


def bind_source_fit_scope(
    identity: CaptureIdentity,
    scope: SourceFitScope,
    artifact_root: Path | str | None = None,
) -> BoundSourceFitScope:
    """Authenticate declaration against canonical capture and actual receipt."""
    from .necromatcher_capture_identity import CaptureIdentity

    if not isinstance(identity, CaptureIdentity) or not isinstance(
        scope, SourceFitScope
    ):
        raise ValueError("Typed canonical capture identity and source scope required")
    if (
        identity.capture_id != scope.capture_id
        or identity.capture_hash != scope.capture_hash
    ):
        raise ValueError("Scope capture binding differs")
    if identity.source_clock_sha256 != scope.source_clock_sha256:
        raise ValueError("Scope source clock binding differs")
    if scope.end_exclusive_frame > len(identity.frames):
        raise ValueError("Scope frame bounds exceed capture")
    if identity.frames[scope.first_frame] != scope.review.first_identity:
        raise ValueError("Scope first frame identity differs")
    excluded = (
        identity.frames[scope.end_exclusive_frame]
        if scope.end_exclusive_frame < len(identity.frames)
        else None
    )
    if excluded != scope.review.excluded_identity:
        raise ValueError("Scope excluded boundary frame identity differs")
    _review_payload(identity, scope, artifact_root)
    return BoundSourceFitScope(identity, scope)


def validate_scope_selection(
    bound: BoundSourceFitScope,
    body_indices: tuple[int, ...],
    shaft_evidence: ShaftAxisEvidence | None = None,
) -> None:
    if not isinstance(bound, BoundSourceFitScope):
        raise ValueError("A bound source scope is required")
    indices = tuple(body_indices)
    if len(indices) < 2 or any(
        isinstance(i, bool) or not isinstance(i, int) for i in indices
    ):
        raise ValueError("Scope selection requires at least two integer source frames")
    if any(a >= b for a, b in zip(indices, indices[1:], strict=False)):
        raise ValueError("Scope body selection must strictly increase")
    scope = bound.scope
    if any(not scope.first_frame <= i < scope.end_exclusive_frame for i in indices):
        raise ValueError("Body frame outside source scope")
    if shaft_evidence is not None:
        _validate_scope_shaft(bound, shaft_evidence)


def _validate_scope_shaft(
    bound: BoundSourceFitScope, evidence: ShaftAxisEvidence
) -> None:
    if not isinstance(evidence, ShaftAxisEvidence):
        raise ValueError("Typed shaft evidence required for source scope")
    identity, scope = bound.identity, bound.scope
    if (
        evidence.capture_id != identity.capture_id
        or evidence.capture_sha256 != identity.capture_hash
        or evidence.source_sha256 != identity.source_sha256
    ):
        raise ValueError("Shaft evidence differs from source scope capture")
    for frame in evidence.frames:
        i = frame.frame_index
        if not scope.first_frame <= i < scope.end_exclusive_frame:
            raise ValueError("Shaft frame outside source scope")
        if (
            frame.frame != identity.frames[i]
            or frame.png_sha256 != identity.png_sha256[i]
        ):
            raise ValueError("Shaft frame identity differs from source scope")


def validate_scope_descendant(
    parent: BoundSourceFitScope, child: BoundSourceFitScope
) -> None:
    if not isinstance(parent, BoundSourceFitScope) or not isinstance(
        child, BoundSourceFitScope
    ):
        raise ValueError("Typed bound parent/child source scopes required")
    a, b = parent.scope, child.scope
    if (a.capture_id, a.capture_hash, a.source_clock_sha256) != (
        b.capture_id,
        b.capture_hash,
        b.source_clock_sha256,
    ):
        raise ValueError("Descendant source scope capture/clock differs")
    if b.first_frame < a.first_frame or b.end_exclusive_frame > a.end_exclusive_frame:
        raise ValueError("Descendant cannot widen parent source scope")


def resolve_source_fit_scope(
    identity: CaptureIdentity,
    parent_scope: SourceFitScope | None,
    requested_scope: SourceFitScope | None,
    artifact_root: Path | str | None = None,
) -> BoundSourceFitScope | None:
    """Inherit scope when omitted; retain or narrow after fresh authentication."""
    parent = (
        bind_source_fit_scope(identity, parent_scope, artifact_root)
        if parent_scope is not None
        else None
    )
    child = (
        bind_source_fit_scope(identity, requested_scope, artifact_root)
        if requested_scope is not None
        else parent
    )
    if parent is not None and child is not None:
        validate_scope_descendant(parent, child)
    return child
