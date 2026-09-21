"""Body and club silhouette segmentation, gold-mask evaluation, and occlusion tracking (ST-04).

This module implements:
- `ManualMaskProvider`: Deterministic baseline provider for reviewed gold masks and corrections.
- `ModelSegmentationProvider`: Neural model segmentation adapter with lazy checkpoint validation.
- `track_occlusion_and_identity()`: Occlusion severity and identity loss reporting.
- `compute_mask_iou()` and `compute_mask_dice()`: Valid-pixel aware agreement metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any

from ._validation import (
    check_id,
    check_pos_int,
    check_strict_float,
)
from .contracts import (
    SegmentationRequest,
    SegmentationResult,
    Segmenter,
)
from .mask_records import MaskFrame
from .source_records import FrameIdentity

FrameScopeKey = tuple[str, str, str, str, str]


def frame_scope_key(frame: FrameIdentity) -> FrameScopeKey:
    """Return canonical 5-tuple frame identity scope (asset, shot, swing, camera, frame)."""
    return (
        frame.asset_id,
        frame.shot_id,
        frame.swing_id,
        frame.camera_id,
        frame.frame_id,
    )


def _validate_scope_filters(
    shot_id: str | None,
    asset_id: str | None,
    swing_id: str | None,
    camera_id: str | None,
) -> None:
    for val, name in (
        (shot_id, "shot_id"),
        (asset_id, "asset_id"),
        (swing_id, "swing_id"),
        (camera_id, "camera_id"),
    ):
        if val is not None:
            check_id(val, name)


def _frame_matches_scope(
    frame: FrameIdentity,
    frame_id: str,
    shot_id: str | None,
    asset_id: str | None,
    swing_id: str | None,
    camera_id: str | None,
) -> bool:
    return (
        frame.frame_id == frame_id
        and (shot_id is None or frame.shot_id == shot_id)
        and (asset_id is None or frame.asset_id == asset_id)
        and (swing_id is None or frame.swing_id == swing_id)
        and (camera_id is None or frame.camera_id == camera_id)
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_target = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:12]}.tmp")
    try:
        temp_target.write_text(serialized, encoding="utf-8")
        temp_target.replace(path)
    except BaseException:
        if temp_target.exists():
            temp_target.unlink()
        raise


# ---------------------------------------------------------------------------
# 1. Agreement Metrics (Valid-Pixel Aware)
# ---------------------------------------------------------------------------


def _check_mask_lengths(candidate: bytes, gold: bytes, valid: bytes) -> int:
    length = len(candidate)
    if len(gold) != length or len(valid) != length:
        raise ValueError(
            f"Mask arrays must have equal lengths: candidate={len(candidate)}, "
            f"gold={len(gold)}, valid={len(valid)}"
        )
    return length


def compute_mask_iou(candidate: bytes, gold: bytes, valid: bytes) -> float:
    """Compute Intersection over Union (IoU) evaluated strictly over valid pixels.

    Preconditions:
        - `candidate`, `gold`, and `valid` must have equal lengths.
        - Only indices where `valid[i] != 0` contribute to intersection or union.
    """
    length = _check_mask_lengths(candidate, gold, valid)

    intersection = 0
    union = 0

    for i in range(length):
        if valid[i] != 0:
            c = 1 if candidate[i] != 0 else 0
            g = 1 if gold[i] != 0 else 0
            if c and g:
                intersection += 1
            if c or g:
                union += 1

    if union == 0:
        return 1.0
    return intersection / union


def compute_mask_dice(candidate: bytes, gold: bytes, valid: bytes) -> float:
    """Compute Dice similarity coefficient evaluated strictly over valid pixels.

    ``Dice = 2 * |cand & gold| / (|cand| + |gold|)``
    """
    length = _check_mask_lengths(candidate, gold, valid)

    intersection = 0
    cand_count = 0
    gold_count = 0

    for i in range(length):
        if valid[i] != 0:
            c = 1 if candidate[i] != 0 else 0
            g = 1 if gold[i] != 0 else 0
            if c and g:
                intersection += 1
            if c:
                cand_count += 1
            if g:
                gold_count += 1

    total = cand_count + gold_count
    if total == 0:
        return 1.0
    return (2.0 * intersection) / total


# ---------------------------------------------------------------------------
# 2. Occlusion & Identity Loss Tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class OcclusionReport:
    """Audit report of body visibility, occlusion severity, and identity persistence."""

    frame_id: str
    visible_fraction: float
    is_partially_occluded: bool
    is_identity_lost: bool

    def __post_init__(self) -> None:
        check_id(self.frame_id, "frame_id")
        check_strict_float(self.visible_fraction, "visible_fraction")
        if not 0.0 <= self.visible_fraction <= 10.0:  # Bound allowing reasonable margin
            raise ValueError(
                f"visible_fraction must be non-negative, got {self.visible_fraction}"
            )


def track_occlusion_and_identity(
    mask: MaskFrame,
    *,
    expected_body_area_px: int,
    occlusion_threshold: float = 0.8,
    identity_loss_threshold: float = 0.25,
) -> OcclusionReport:
    """Evaluate body visibility against expected area, tracking partial occlusion and identity loss.

    Preconditions:
        - `expected_body_area_px` must be a positive integer.
        - `occlusion_threshold` > `identity_loss_threshold` >= 0.0.
    """
    if not isinstance(mask, MaskFrame):
        raise TypeError(f"Expected MaskFrame, got {type(mask).__name__}")
    check_pos_int(expected_body_area_px, "expected_body_area_px")

    visible_body_px = mask.body.count(1)
    visible_fraction = visible_body_px / expected_body_area_px

    is_partially_occluded = visible_fraction < occlusion_threshold
    is_identity_lost = visible_fraction < identity_loss_threshold

    return OcclusionReport(
        frame_id=mask.frame.frame_id,
        visible_fraction=visible_fraction,
        is_partially_occluded=is_partially_occluded,
        is_identity_lost=is_identity_lost,
    )


# ---------------------------------------------------------------------------
# 3. Manual Mask Provider (Ground Truth & Correction Authority)
# ---------------------------------------------------------------------------


class ManualMaskProvider:
    """Deterministic segmenter serving reviewed gold masks and corrections with revision tracking.

    Keyed by canonical 5-tuple frame scope `(asset_id, shot_id, swing_id, camera_id, frame_id)`
    to ensure strict shot/camera isolation, parent ownership, and addressable revision lineage.
    """

    __slots__ = ("_masks", "_revisions", "_history", "_callbacks")

    def __init__(self) -> None:
        self._masks: dict[FrameScopeKey, MaskFrame] = {}
        self._revisions: dict[str, MaskFrame] = {}
        self._history: dict[FrameScopeKey, list[MaskFrame]] = {}
        self._callbacks: list[Callable[[FrameScopeKey, MaskFrame], None]] = []

    def _validate_registration(self, mask: MaskFrame) -> bool:
        """Validate revision identity, parent ownership, and acyclic ancestry before mutation.

        Returns True if identical re-registration (idempotent no-op), False if valid new revision.
        """
        if not isinstance(mask, MaskFrame):
            raise TypeError(f"Expected MaskFrame, got {type(mask).__name__}")

        if mask.revision_id in self._revisions:
            existing = self._revisions[mask.revision_id]
            if mask == existing:
                return True
            raise ValueError(
                f"Conflicting duplicate revision_id {mask.revision_id!r}: already registered "
                f"for frame scope {frame_scope_key(existing.frame)}"
            )

        if mask.parent_revision_id is not None:
            parent = self._revisions.get(mask.parent_revision_id)
            if parent is None:
                raise ValueError(
                    f"Parent revision {mask.parent_revision_id!r} not found for revision {mask.revision_id!r}"
                )
            parent_scope = frame_scope_key(parent.frame)
            mask_scope = frame_scope_key(mask.frame)
            if parent_scope != mask_scope:
                raise ValueError(
                    f"Parent revision {mask.parent_revision_id!r} belongs to different frame scope "
                    f"{parent_scope} than child mask {mask_scope}"
                )

            if parent.frame != mask.frame or (parent.width_px, parent.height_px) != (
                mask.width_px,
                mask.height_px,
            ):
                raise ValueError(
                    "Parent and child must describe the same observation and pixel grid"
                )

            curr_id: str | None = mask.parent_revision_id
            seen: set[str] = {mask.revision_id}
            while curr_id is not None:
                if curr_id in seen:
                    raise ValueError(f"Cycle detected in revision lineage: {curr_id!r}")
                seen.add(curr_id)
                ancestor = self._revisions.get(curr_id)
                if ancestor is None:
                    raise ValueError(
                        f"Missing ancestor revision {curr_id!r} in lineage"
                    )
                curr_id = ancestor.parent_revision_id

        return False

    def register_mask(self, mask: MaskFrame) -> None:
        """Register or update a mask frame, enforcing revision identity and transactional lineage."""
        is_idempotent = self._validate_registration(mask)
        if is_idempotent:
            return

        scope = frame_scope_key(mask.frame)
        self._masks[scope] = mask
        self._revisions[mask.revision_id] = mask
        if scope not in self._history:
            self._history[scope] = []
        self._history[scope].append(mask)

        for cb in self._callbacks:
            cb(scope, mask)

    def get_mask(
        self,
        frame_id: str,
        *,
        shot_id: str | None = None,
        asset_id: str | None = None,
        swing_id: str | None = None,
        camera_id: str | None = None,
    ) -> MaskFrame:
        """Retrieve the selected mask for frame_id, optionally filtered by scope."""
        check_id(frame_id, "frame_id")
        _validate_scope_filters(shot_id, asset_id, swing_id, camera_id)

        matching = [
            m
            for m in self._masks.values()
            if _frame_matches_scope(
                m.frame, frame_id, shot_id, asset_id, swing_id, camera_id
            )
        ]
        if not matching:
            raise KeyError(
                f"No mask registered for frame_id {frame_id!r} with specified scope criteria"
            )
        if len(matching) > 1:
            raise ValueError(
                f"Multiple masks ({len(matching)}) match frame_id {frame_id!r}: "
                f"{[frame_scope_key(m.frame) for m in matching]}. "
                "Specify shot_id/asset_id/swing_id/camera_id to disambiguate."
            )
        return matching[0]

    def select_revision(self, revision_id: str) -> None:
        """Select an existing revision without changing history; its cache key becomes current.

        Re-registering an identical record never changes this selection.
        """
        mask = self.get_revision(revision_id)
        scope = frame_scope_key(mask.frame)
        if self._masks[scope] == mask:
            return
        self._masks[scope] = mask
        for cb in self._callbacks:
            cb(scope, mask)

    def get_revision(self, revision_id: str) -> MaskFrame:
        """Retrieve a specific mask revision by revision_id."""
        check_id(revision_id, "revision_id")
        if revision_id not in self._revisions:
            raise KeyError(f"No mask found with revision_id {revision_id!r}")
        return self._revisions[revision_id]

    def has_revision(self, revision_id: str) -> bool:
        """Return whether a specific revision_id exists."""
        return revision_id in self._revisions

    def all_revisions(self) -> tuple[MaskFrame, ...]:
        """Return all registered revisions in registration order."""
        return tuple(self._revisions.values())

    def get_revision_history(
        self,
        frame_id: str,
        *,
        shot_id: str | None = None,
        asset_id: str | None = None,
        swing_id: str | None = None,
        camera_id: str | None = None,
    ) -> tuple[MaskFrame, ...]:
        """Return the complete revision history sequence for a namespaced frame scope."""
        check_id(frame_id, "frame_id")
        _validate_scope_filters(shot_id, asset_id, swing_id, camera_id)

        matching_histories = [
            hist
            for scope, hist in self._history.items()
            if scope[4] == frame_id
            and (asset_id is None or scope[0] == asset_id)
            and (shot_id is None or scope[1] == shot_id)
            and (swing_id is None or scope[2] == swing_id)
            and (camera_id is None or scope[3] == camera_id)
        ]
        if not matching_histories:
            raise KeyError(
                f"No revision history for frame_id {frame_id!r} with specified scope criteria"
            )
        if len(matching_histories) > 1:
            raise ValueError(
                f"Multiple revision histories match frame_id {frame_id!r}: specify scope criteria to disambiguate"
            )
        return tuple(matching_histories[0])

    def has_mask(
        self,
        frame_id: str,
        *,
        shot_id: str | None = None,
        asset_id: str | None = None,
        swing_id: str | None = None,
        camera_id: str | None = None,
    ) -> bool:
        """Return whether a mask is registered for frame_id and optional scope criteria."""
        return any(
            _frame_matches_scope(
                m.frame, frame_id, shot_id, asset_id, swing_id, camera_id
            )
            for m in self._masks.values()
        )

    def get_cache_key(
        self,
        frame_id: str,
        *,
        shot_id: str | None = None,
        asset_id: str | None = None,
        swing_id: str | None = None,
        camera_id: str | None = None,
    ) -> str:
        """Return unique cache key for downstream computation invalidation."""
        mask = self.get_mask(
            frame_id,
            shot_id=shot_id,
            asset_id=asset_id,
            swing_id=swing_id,
            camera_id=camera_id,
        )
        return f"{mask.revision_id}:{mask.observation_hash}"

    def save(self, path: Path | str) -> None:
        """Persist all revisions atomically to a JSON file."""
        target = Path(path)
        payload = {
            "schema_version": "1.1.0",
            "revisions": [m.to_dict() for m in self.all_revisions()],
            "current_revision_ids": [m.revision_id for m in self._masks.values()],
        }
        _atomic_write_json(target, payload)

    def save_revisions(self, path: Path | str) -> None:
        """Alias for save."""
        self.save(path)

    @classmethod
    def load(cls, path: Path | str) -> ManualMaskProvider:
        """Atomically load all revisions from a JSON file, validating lineage."""
        source = Path(path)
        with open(source, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected dict root in {source}, got {type(payload).__name__}"
            )
        version = payload.get("schema_version")
        if version not in ("1.0.0", "1.1.0"):
            raise ValueError(f"Unsupported revision store schema_version: {version!r}")
        expected_keys = {"schema_version", "revisions"}
        if version == "1.1.0":
            expected_keys.add("current_revision_ids")
        if set(payload) != expected_keys:
            raise ValueError("Revision store contains missing or unknown fields")
        raw_revisions = payload.get("revisions")
        if not isinstance(raw_revisions, list):
            raise TypeError(
                f"Expected 'revisions' list in {source}, got {type(raw_revisions).__name__}"
            )

        provider = cls()
        for item in raw_revisions:
            if not isinstance(item, dict):
                raise TypeError(
                    f"Expected revision dict in 'revisions', got {type(item).__name__}"
                )
            mask = MaskFrame.from_dict(item)
            if provider.has_revision(mask.revision_id):
                raise ValueError(f"Duplicate stored revision_id: {mask.revision_id!r}")
            provider.register_mask(mask)
        if version == "1.1.0":
            selected = payload["current_revision_ids"]
            if not isinstance(selected, list):
                raise TypeError("current_revision_ids must be a list")
            current: dict[FrameScopeKey, MaskFrame] = {}
            for revision_id in selected:
                mask = provider.get_revision(revision_id)
                scope = frame_scope_key(mask.frame)
                if scope in current:
                    raise ValueError(
                        "Multiple current revisions for the same observation"
                    )
                current[scope] = mask
            if current.keys() != provider._masks.keys():
                raise ValueError("Current selection must cover every observation")
            provider._masks = current
        return provider

    @classmethod
    def load_revisions(cls, path: Path | str) -> ManualMaskProvider:
        """Alias for load."""
        return cls.load(path)

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """Fulfill a SegmentationRequest using registered reviewed masks for request.shot_id."""
        if not isinstance(request, SegmentationRequest):
            raise TypeError(
                f"Expected SegmentationRequest, got {type(request).__name__}"
            )

        count = 0
        for fid in request.frame_ids:
            if not self.has_mask(fid, shot_id=request.shot_id):
                raise KeyError(
                    f"No mask registered for shot {request.shot_id!r} and frame_id {fid!r}"
                )
            count += 1

        return SegmentationResult(
            shot_id=request.shot_id,
            mask_count=count,
            provenance="manual_gold_review",
        )


# ---------------------------------------------------------------------------
# 4. Model Segmentation Provider (Optional Pinned Architecture Adapter)
# ---------------------------------------------------------------------------


class ModelSegmentationProvider:
    """Adapter for automated neural silhouette segmentation models.

    Enforces lazy missing-provider error reporting and flags results as unreviewed.
    """

    __slots__ = ("_model_name", "_checkpoint_path")

    def __init__(self, model_name: str, checkpoint_path: Path | str) -> None:
        check_id(model_name, "model_name")
        self._model_name = model_name
        self._checkpoint_path = Path(checkpoint_path)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """Run segmentation or raise actionable missing checkpoint/unsupported error.

        Arbitrary or unverified model checkpoints must not report false success.
        Until genuine neural inference execution is integrated, this method raises
        an explicit RuntimeError rather than returning mock mask counts.
        """
        if not isinstance(request, SegmentationRequest):
            raise TypeError(
                f"Expected SegmentationRequest, got {type(request).__name__}"
            )

        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found for {self._model_name}: {self._checkpoint_path}. "
                "Download the pinned model weights or use ManualMaskProvider."
            )

        # Explicitly fail until genuine neural inference is wired and loaded:
        # Never report false segmentation success on an arbitrary file.
        raise RuntimeError(
            f"Automated inference is not supported for checkpoint {self._checkpoint_path.name!r} "
            f"on model {self._model_name!r}. Genuine model execution is not yet integrated; "
            "use ManualMaskProvider for reviewed gold masks."
        )
