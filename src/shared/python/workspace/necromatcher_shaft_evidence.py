"""Read-only admission of reviewed shaft bearings against verified captures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
)
from src.shared.python.shadow_tracker.source_records import FrameIdentity
from src.shared.python.motion_matching.historical_fit.shaft_geometry import (
    resolve_authored_shaft_axis,
)
from src.shared.python.motion_matching.historical_fit.shaft_residuals import (
    AdditionalImageResiduals,
    ImageSourceIdentity,
    ShaftAxisResidualTerm,
)
from .necromatcher_native import NativeFitBinding, load_native_fit_binding
from .necromatcher import NecromatcherLibrary
from .necromatcher_review import CaptureReview


@dataclass(frozen=True)
class BoundShaftEvidence:
    """Verified sparse review with a hash of the complete exact capture clock."""

    evidence: ShaftAxisEvidence
    source_clock_sha256: str
    reviewed_frame_count: int
    observed_segment_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ShaftAxisEvidence):
            raise ValueError("Bound receipt requires typed shaft evidence")
        if not isinstance(self.source_clock_sha256, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.source_clock_sha256
        ):
            raise ValueError("Bound receipt requires a lowercase source clock digest")
        for count in (self.reviewed_frame_count, self.observed_segment_count):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Bound receipt counts must be nonnegative integers")
        observed = sum(
            item.segment.status == "observed" for item in self.evidence.frames
        )
        if (
            self.reviewed_frame_count != len(self.evidence.frames)
            or self.observed_segment_count != observed
        ):
            raise ValueError("Bound receipt counts differ from source evidence")


def bind_shaft_axis_evidence(
    library: NecromatcherLibrary, evidence: ShaftAxisEvidence
) -> BoundShaftEvidence:
    """Verify source identities and original PNG bytes before numerical admission.

    Uses the canonical hash-verifying asset loader and capture reviewer. The
    capture importer verified decoded pixels; its hash remains distinct from
    the PNG-byte hash independently checked here. No original file is mutated.
    """
    if not isinstance(evidence, ShaftAxisEvidence):
        raise ValueError("Typed shaft evidence required")
    asset = library.load_asset(evidence.capture_id)
    source_hash = asset.metadata.get("source_sha256", "")
    if not isinstance(source_hash, str):
        raise ValueError("Capture source hash is malformed")
    source_hash = "sha256:" + source_hash.removeprefix("sha256:")
    if (
        asset.kind != "image_capture"
        or asset.metadata["hash"] != evidence.capture_sha256
        or source_hash != evidence.source_sha256
    ):
        raise ValueError("Shaft evidence differs from verified capture/source hash")
    with CaptureReview(library, evidence.capture_id) as review:
        if review.capture_id != evidence.capture_id:
            raise ValueError("Shaft capture review identity differs")
        clock = []
        for index in range(review.frame_count):
            frame = FrameIdentity.from_dict(review.frame(index)["frame"])
            clock.append(
                {
                    "frame_index": index,
                    "frame_id": frame.frame_id,
                    "pts_ticks": frame.pts_ticks,
                    "timebase_numerator": frame.timebase_numerator,
                    "timebase_denominator": frame.timebase_denominator,
                }
            )
        for item in evidence.frames:
            if item.frame_index >= review.frame_count:
                raise ValueError("Shaft frame index is outside capture")
            row = review.frame(item.frame_index)
            if FrameIdentity.from_dict(row["frame"]) != item.frame:
                raise ValueError(
                    "Shaft frame identity/clock differs from original capture"
                )
            if (row["image_width"], row["image_height"]) != evidence.image_size:
                raise ValueError("Shaft original image dimensions differ")
            digest = (
                "sha256:" + hashlib.sha256(review.image(item.frame_index)).hexdigest()
            )
            if digest != item.png_sha256:
                raise ValueError("Shaft original PNG hash mismatch")
    raw = json.dumps(
        clock, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return BoundShaftEvidence(
        evidence,
        "sha256:" + hashlib.sha256(raw).hexdigest(),
        len(evidence.frames),
        sum(item.segment.status == "observed" for item in evidence.frames),
    )


def load_shaft_image_residuals(
    library: NecromatcherLibrary,
    fit_id: str,
    evidence: ShaftAxisEvidence,
    unknown_visibility_weight: float,
) -> tuple[NativeFitBinding, AdditionalImageResiduals]:
    """Rebind actual source evidence and reuse one canonical native compilation.

    Caller-created typed receipts are not admission capabilities. This boundary
    always reopens the capture, verifies reviewed PNG bytes and exact source
    clocks, and resolves the authored axis from the exact compiled definition.
    Camera calibration, historical geometry and physical time remain unknown.
    """
    if not isinstance(evidence, ShaftAxisEvidence):
        raise ValueError("Typed shaft evidence required")
    binding = load_native_fit_binding(library, fit_id)
    fit = binding.fit
    if (
        fit.get("capture_id") != evidence.capture_id
        or fit.get("capture_hash") != evidence.capture_sha256
    ):
        raise ValueError("Shaft evidence capture differs from canonical fit binding")
    frames = fit.get("frames")
    if (
        not isinstance(frames, list)
        or not frames
        or any(
            not isinstance(frame, dict)
            or frame.get("camera_id") != evidence.frames[0].frame.camera_id
            for frame in frames
        )
    ):
        raise ValueError("Shaft source camera differs from canonical fit frames")
    binding.review_inputs()
    bound = bind_shaft_axis_evidence(library, evidence)
    axis = resolve_authored_shaft_axis(
        binding.definition_bytes, binding.plant.plant_sha
    )
    identity = ImageSourceIdentity(
        evidence.capture_id,
        evidence.capture_sha256,
        evidence.source_sha256,
        evidence.frames[0].frame.camera_id,
        bound.source_clock_sha256,
    )
    term = ShaftAxisResidualTerm(
        evidence, axis, bound.source_clock_sha256, unknown_visibility_weight
    )
    bundle = AdditionalImageResiduals(identity, (term,))
    bundle.validate(binding.plant)
    return binding, bundle
