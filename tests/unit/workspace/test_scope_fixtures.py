"""Canonical SDK-free source fixtures; not an admitted historical review."""

import hashlib
import json
from pathlib import Path
from fractions import Fraction

from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
from src.shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
    FRAME_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from src.shared.python.workspace.necromatcher_capture_identity import CaptureIdentity
from src.shared.python.workspace.artifact_handoff import ArtifactReference, ArtifactKind


def fixture_capture() -> CaptureIdentity:
    source = SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="source-" + "a" * 64,
        source_uri="https://example.invalid/synthetic.mp4",
        content_sha256="a" * 64,
        width_px=32,
        height_px=24,
        rights_status="unknown",
        rights_note="",
    )
    frames = tuple(
        FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id="source-" + "a" * 64,
            shot_id="shot",
            swing_id="swing",
            camera_id="camera",
            frame_id=f"frame-{i}",
            pts_ticks=100 + i,
            timebase_numerator=1,
            timebase_denominator=30,
            physical_time_s=None,
            physical_time_reason="unknown source clock",
            frame_sha256=compute_frame_hash(
                bytes([i]) * (32 * 24 * 3), decoder_name="synthetic-decoder"
            ),
            timing_mode="container_pts",
            is_timing_exact=True,
            clock_evidence="synthetic container record",
            decoder_name="synthetic-decoder",
            decoder_version="1",
            pixel_format="bgr24",
        )
        for i in range(5)
    )
    # Canonical full-clock byte convention; no physical-frame-rate inference.
    clock = [
        {
            "frame_index": i,
            "frame_id": f.frame_id,
            "pts_ticks": f.pts_ticks,
            "timebase_numerator": f.timebase_numerator,
            "timebase_denominator": f.timebase_denominator,
        }
        for i, f in enumerate(frames)
    ]
    digest = hashlib.sha256(
        json.dumps(
            clock, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return CaptureIdentity(
        "capture",
        "sha256:" + "c" * 64,
        source,
        "sha256:" + digest,
        frames,
        tuple("sha256:" + "d" * 64 for _ in frames),
    )


def fixture_review_artifact(
    root: Path, identity: CaptureIdentity, first: int = 0, end: int = 4
) -> ArtifactReference:
    payload = {
        "schema": "necromatcher/source-fit-scope-review/1",
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "source_clock_sha256": identity.source_clock_sha256,
        "first_frame": first,
        "end_exclusive_frame": end,
        "first_identity": identity.frames[first].to_dict(),
        "excluded_identity": identity.frames[end].to_dict()
        if end < len(identity.frames)
        else None,
        "contact_calibrated": False,
        "review_kind": "authored_uncalibrated",
        "reason": "Synthetic conservative window",
        "uncertainty_policy": "exclude transition",
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    path = root / "review.json"
    path.write_bytes(raw)
    return ArtifactReference(
        "review",
        str(path),
        "sha256:" + hashlib.sha256(raw).hexdigest(),
        payload["schema"],
        ArtifactKind.RECEIPT,
    )


def test_canonical_fixture_has_distinct_domains(tmp_path: Path) -> None:
    identity = fixture_capture()
    artifact = fixture_review_artifact(tmp_path, identity)
    artifact.verify_on_disk()
    assert identity.frames[3].presentation_time == Fraction(103, 30)
    assert identity.frames[4].presentation_time == Fraction(104, 30)
    assert identity.frames[0].frame_sha256 != identity.png_sha256[0]
    assert identity.frames[0].physical_time_s is None
