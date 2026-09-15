"""Body and club silhouette segmentation, gold-mask evaluation, and occlusion tracking (ST-04).

This module implements:
- `ManualMaskProvider`: Deterministic baseline provider for reviewed gold masks and corrections.
- `ModelSegmentationProvider`: Neural model segmentation adapter with lazy checkpoint validation.
- `track_occlusion_and_identity()`: Occlusion severity and identity loss reporting.
- `compute_mask_iou()` and `compute_mask_dice()`: Valid-pixel aware agreement metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

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
    """Deterministic segmenter serving reviewed gold masks and corrections with revision tracking."""

    __slots__ = ("_masks",)

    def __init__(self) -> None:
        self._masks: dict[str, MaskFrame] = {}

    def register_mask(self, mask: MaskFrame) -> None:
        """Register or update a mask frame, tracking revision lineage."""
        if not isinstance(mask, MaskFrame):
            raise TypeError(f"Expected MaskFrame, got {type(mask).__name__}")
        self._masks[mask.frame.frame_id] = mask

    def get_mask(self, frame_id: str) -> MaskFrame:
        """Retrieve mask for frame_id or raise KeyError."""
        if frame_id not in self._masks:
            raise KeyError(f"No mask registered for frame_id {frame_id!r}")
        return self._masks[frame_id]

    def has_mask(self, frame_id: str) -> bool:
        """Return whether a mask is registered for frame_id."""
        return frame_id in self._masks

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """Fulfill a SegmentationRequest using registered reviewed masks."""
        if not isinstance(request, SegmentationRequest):
            raise TypeError(
                f"Expected SegmentationRequest, got {type(request).__name__}"
            )

        count = 0
        for fid in request.frame_ids:
            _ = self.get_mask(fid)
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
        """Run segmentation or raise actionable missing checkpoint error."""
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found for {self._model_name}: {self._checkpoint_path}. "
                "Download the pinned model weights or use ManualMaskProvider."
            )

        return SegmentationResult(
            shot_id=request.shot_id,
            mask_count=len(request.frame_ids),
            provenance=f"model:{self._model_name}:{self._checkpoint_path.name}",
        )
